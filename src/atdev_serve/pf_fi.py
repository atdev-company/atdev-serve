"""FlashInfer fa2 paged attention for prefill / mixed steps on the TRITON_ATTN backend (opt-in ATDEV_PF_FI=1, C32MT-fi).

Why: the vllm_flash_attn paged split-KV kernel (what pf_fa2 calls) is badly under-filled for small prefill chunks
(~450-token deltas over 8k context + decode rows: 2.8 ms per layer per call in the C32 v4 profile, ~20 % of the GPU time
of the delta phase). FlashInfer's balanced split-KV with ONE plan per step (shared by all 16 full-attention layers):
microbench results/C32MT/bench_attn_multi.log: 3.30 -> 1.28 ms (1 delta + 20 decode rows), 3.69 -> 1.59, 4.79 -> 2.53,
3.76 -> 2.29, cold 4096-token chunk 1.55 -> 1.39 ms. Rounding-level difference (max abs 2.4e-4 at bf16).

Mechanism (same pattern as fi_attn.py): ``TritonAttentionMetadataBuilder.build`` plans a FlashInfer wrapper on the host
upper bound of the kv lengths for steps whose max_query_len >= ATDEV_PF_FI_MINQ (default 32); in the first target layer
of the forward a tiny Triton kernel (fi_attn._fixup_kernel) rewrites kv_indptr / kv_indices / last_page_len from the GPU
seq_lens and block table (exact lengths, async scheduling). ``TritonAttentionImpl.forward`` runs ``wrapper.run`` for
marked metadata and non-draft layers; everything else keeps the stock path (pf_fa2 / Triton).
Env: ATDEV_PF_FI=1, ATDEV_PF_FI_MINQ (32), ATDEV_PF_FI_MAX_REQS (64), ATDEV_PF_FI_WS_MB (256), ATDEV_PF_FI_CHECK=<n>
(compare the first n calls against flash_attn_varlen_func, log max abs diff).
"""
from __future__ import annotations

import functools
import logging
import os

import torch

OPT_NAME = "pf_fi"
SUPPORTED_VLLM_VERSION = "0.31.0"
log = logging.getLogger("atdev_serve")
MINQ = int(os.environ.get("ATDEV_PF_FI_MINQ", "32"))
MAX_REQS = int(os.environ.get("ATDEV_PF_FI_MAX_REQS", "64"))
DQ = int(os.environ.get("ATDEV_PF_FI_DQ", "4"))  # 0: plan with the CPU query offsets (flashinfer needs qo_indptr[-1] == q.shape[0])
RING = int(os.environ.get("ATDEV_PF_FI_RING", "4"))
CHECK = int(os.environ.get("ATDEV_PF_FI_CHECK", "0"))


class _S:
    ws = None
    wrappers: list = []
    turn = 0
    n_plan = 0
    n_run = 0
    n_chk = 0
    first_layer = None
    logged = False


class _W:
    def __init__(self, device):
        import flashinfer
        if _S.ws is None:
            _S.ws = torch.empty(int(os.environ.get("ATDEV_PF_FI_WS_MB", "256")) << 20, dtype=torch.uint8, device=device)
        self.w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(_S.ws, "NHD", backend="fa2")
        self.seq_lens = self.qsl = self.block_table = None
        self.r = 0
        self.page = 0
        self.fixed = False
        self.ev = None  # recorded after the last run: plan() restages host buffers asynchronously, so wait before re-planning


def _plan(md, cam, cfg) -> None:
    if cam.max_query_len < MINQ or torch.cuda.is_current_stream_capturing():
        return
    ub0 = cam.seq_lens_cpu_upper_bound
    r = cam.num_reqs
    if ub0 is None or r == 0 or r > MAX_REQS:
        return
    page = cfg["page"]
    bt = cam.block_table_tensor
    qsl = cam.query_start_loc_cpu[: r + 1].to(torch.int32)
    qlen = qsl[1:] - qsl[:-1]
    # Batch layout: decode rows first (their per-request draft-row counts are decided on the GPU by DI1 varlen verify, so the
    # CPU query offsets are only an upper-bound layout), then prefill-sized rows whose q (and kv) are exact on the CPU.
    big = qlen > DQ
    if not bool(big.any()):
        return
    rd = int(big.to(torch.int32).argmax())
    if bool((~big[rd:]).any()):  # decode-sized rows interleaved with prefill rows: stock path
        return
    rp = r - rd
    qp_tok = int(qlen[rd:].sum())
    ub = ub0[:r].to(torch.int64).clamp_min(1)
    ubp = ub[rd:]
    npg = (ubp + page - 1) // page
    kv_indptr = torch.zeros(rp + 1, dtype=torch.int32)
    kv_indptr[1:] = npg.cumsum(0).to(torch.int32)
    last = (ubp - (npg - 1) * page).to(torch.int32)
    qsl_p = (qsl[rd:] - qsl[rd]).to(torch.int32)
    if len(_S.wrappers) < RING:
        _S.wrappers.append(_W(bt.device))
    w = _S.wrappers[_S.turn % RING]
    _S.turn += 1
    if w.ev is not None:
        w.ev.synchronize()
    w.w.plan(qsl_p, kv_indptr, torch.zeros(int(kv_indptr[-1]), dtype=torch.int32), last, cfg["qh"], cfg["kvh"], cfg["hs"],
             page, causal=True, sm_scale=cfg["scale"], q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    w.seq_lens, w.qsl, w.block_table, w.page, w.fixed = cam.seq_lens, cam.query_start_loc, bt, page, False
    w.r, w.rd, w.rp, w.qp_tok = r, rd, rp, qp_tok
    w.dmax = int(ub[:rd].max()) if rd else 0
    w.ntok = int(cam.num_actual_tokens)
    w.ub = ub.clone(); w.qlen = qlen.clone(); w.qsl_cpu = qsl.clone()
    if rd:
        import dataclasses
        w.dmd = dataclasses.replace(md, num_actual_tokens=int(cam.num_actual_tokens) - qp_tok, max_query_len=DQ,
                                    query_start_loc=cam.query_start_loc[: rd + 1], max_seq_len=w.dmax,
                                    seq_lens=cam.seq_lens[:rd], block_table=md.block_table[:rd])
    md._atdev_pffi = w
    _S.n_plan += 1


def register() -> None:
    import vllm
    import triton
    from vllm.v1.attention.backends import triton_attn as ta
    from atdev_serve.fi_attn import _fixup_kernel

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written against vLLM {SUPPORTED_VLLM_VERSION}, found {vllm.__version__}")
    B = ta.TritonAttentionMetadataBuilder
    if getattr(B.build, "_atdev_pffi", False):
        return
    orig_build = B.build
    cfg: dict = {}

    @functools.wraps(orig_build)
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = orig_build(self, common_prefix_len, common_attn_metadata, fast_build)
        if cfg and common_prefix_len == 0:
            try:
                _plan(md, common_attn_metadata, cfg)
            except Exception as e:  # never break serving
                if not _S.logged:
                    _S.logged = True
                    log.warning("atdev %s: plan failed (%r) -> stock attention", OPT_NAME, e)
        return md

    build._atdev_pffi = True
    B.build = build
    orig_fwd = ta.TritonAttentionImpl.forward

    @functools.wraps(orig_fwd)
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
                output_block_scale=None):
        if not cfg and attn_metadata is not None and not getattr(layer, "is_draft_layer", False):
            page = kv_cache.shape[2] if kv_cache.dim() == 4 else None
            if page is not None and self.kv_cache_dtype in ("auto", "bfloat16") and self.sinks is None \
                    and self.alibi_slopes is None and not self.logits_soft_cap and self.sliding_window in (None, (-1, -1)):
                cfg.update(page=page, qh=self.num_heads, kvh=self.num_kv_heads, hs=self.head_size, scale=self.scale)
                log.warning("atdev %s: enabled (page %d, %d q / %d kv heads, hd %d, min q %d)", OPT_NAME, page,
                            self.num_heads, self.num_kv_heads, self.head_size, MINQ)
        w = getattr(attn_metadata, "_atdev_pffi", None) if attn_metadata is not None else None
        if w is None or getattr(layer, "is_draft_layer", False) or output_scale is not None:
            return orig_fwd(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                            output_block_scale)
        if not w.fixed:  # once per forward: exact kv description of the prefill rows
            bt = w.block_table
            fw = w.w
            btp = bt[w.rd: w.r]
            _fixup_kernel[(1,)](w.seq_lens[w.rd: w.r], w.qsl[w.rd: w.r + 1], btp, btp.stride(0), fw._paged_kv_indptr_buf,
                                fw._paged_kv_indices_buf, fw._paged_kv_last_page_len_buf, w.rp, w.rp, PAGE=w.page,
                                MAXP=triton.next_power_of_2(bt.shape[1]))
            w.fixed = True
        n = w.ntok
        kc, vc = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        out = output[:n].view(n, self.num_heads, self.head_size)
        nd = n - w.qp_tok
        w.w.run(query[nd:n], (kc, vc), out=out[nd:n])
        if w.rd:  # decode rows (variable q decided on the GPU): the stock path (c32_attn / FA2) on a decode-only metadata view
            orig_fwd(self, layer, query, key, value, kv_cache, w.dmd, output, output_scale, output_block_scale)
        if w.ev is None:
            w.ev = torch.cuda.Event()
        w.ev.record()
        _S.n_run += 1
        if _S.n_run == 1:
            log.warning("atdev %s: FlashInfer prefill-row attention ENGAGED (decode rows %d, prefill rows %d, tokens %d)",
                        OPT_NAME, w.rd, w.rp, n)
        if _S.n_chk < CHECK and _S.n_run % 997 == 5 and n > 64 and _S.n_run > 3000:
            _S.n_chk += 1
            from vllm.vllm_flash_attn import flash_attn_varlen_func
            ref = torch.empty_like(out)
            qq = query[:n].view(n, self.num_heads, self.head_size)
            flash_attn_varlen_func(q=qq, k=kc, v=vc, out=ref,
                                   max_seqlen_q=int(w.qlen.max()), cu_seqlens_q=w.qsl[: w.r + 1],
                                   max_seqlen_k=int(w.seq_lens[: w.r].max()), seqused_k=w.seq_lens[: w.r],
                                   softmax_scale=self.scale, causal=True, block_table=w.block_table[: w.r], fa_version=2)
            d = (ref.float() - out.float()).abs()
            sl = w.seq_lens[: w.r].cpu()
            cu = w.qsl[: w.r + 1].cpu()
            rows = [(i, int(w.qlen[i]), int(sl[i]), int(w.ub[i]), round(d[int(cu[i]):int(cu[i + 1])].max().item(), 4)) for i in range(w.r)]
            fw = w.w
            log.warning("atdev %s: CHECK-DBG qsl_gpu=%s qsl_cpu=%s indptr=%s last=%s sl_gpu=%s ub=%s", OPT_NAME, cu.tolist(),
                        w.qsl_cpu.tolist(), fw._paged_kv_indptr_buf.cpu().tolist(), fw._paged_kv_last_page_len_buf.cpu().tolist(),
                        sl.tolist(), w.ub.tolist())
            log.warning("atdev %s: CHECK n=%d maxdiff %.3g (ref absmax %.3g) rows(i,q,true_kv,ub_kv,maxdiff)=%s", OPT_NAME, n,
                        d.max().item(), ref.float().abs().max().item(), rows)
        return output

    forward._atdev_pffi = True
    ta.TritonAttentionImpl.forward = forward
    log.warning("atdev %s: registered", OPT_NAME)
