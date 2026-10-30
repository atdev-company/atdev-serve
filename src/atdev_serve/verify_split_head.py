"""Split verify head (ATDEV_OPT=verify_split_head) -- accuracy-gated, not exact.

Target logits = exact bf16 for the N most frequent tokens F (draft_vocab
order; contiguous bf16 copy of those rows) + a low-bit Marlin shadow (int4 or
int8 g128) for the remaining tail rows, scattered into the usual
[M, vocab] logits tensor so every downstream consumer (rejection sampler,
sampler, logprobs) is unchanged.

Bytes per verify pass (Qwen3.8, 248320 x 5120): stock 2.54 GB; N=32k + int4
tail 0.33 + 0.55 = 0.88 GB; N=32k + int8 tail 0.33 + 1.11 = 1.44 GB.

Output vs stock: the greedy token differs only if the stock argmax is a tail
token whose shadow logit falls below the best F logit, or a tail token's
shadow logit overtakes the true argmax (shadow error int4 ~ 0.1-1 logit,
int8 ~ 0.01-0.1). ``verify_tail_stats`` measures the argmax match rate on
real verify rows before this is turned on; the GSM8K / long-context gates
decide. Sampling (temperature > 0) sees approximate tail logits.

Env: ATDEV_VSPLIT_SIZE (default 32768), ATDEV_VSPLIT_BITS (4 or 8, default 8),
ATDEV_VSPLIT_MIN_ROWS (default 1: smaller batches use the stock GEMM),
ATDEV_VSPLIT_FBITS (16 default = exact bf16 F rows; 8 = F rows as an int8 g128 Marlin shadow too,
671 -> 336 MB at N=64k; 4 = int4 g128 (168 MB, the whole head low-bit); accuracy-gated like the tail).

ATDEV_VSPLIT_FAST=1 (default off; output bit-identical to the path above): on greedy
dflash_tree verify steps (the only consumer is the tree accept's argmax) skip the
[M, vocab] scatter + torch.argmax and run one fused two-part argmax (split_argmax)
on the two GEMM outputs; F ids are kept ascending (same cuBLAS shape -> same values).
  ATDEV_VSPLIT_STREAMS=1        tail Marlin on a side stream, overlapped with the bf16 GEMM
  ATDEV_VSPLIT_TAIL_CFG=tk,tn,threads,stages,bps,mw
                                tail via marlin_plus with this launch config; checked
                                bit-identical to stock Marlin at load (else ignored).
                                128,128,256,6,1,4 = stock tile, 6 stages (A100, M<=16).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

_ATTR = "_atdev_vsplit"


class W8A8Head:
    """int8 per-channel (symmetric, one fp32 scale per output row) weight x per-token int8
    activation (vLLM per_token_quant_int8, the A8 tail's quant) on CUTLASS int8 tensor cores
    (vllm cutlass_scaled_mm): one weight pass for any M. Rounding-level vs bf16 like the
    shadows; ATDEV_VSPLIT_FBITS=8 + ATDEV_VSPLIT_FA8=1."""

    def __init__(self, weight: torch.Tensor):
        n, k = weight.shape
        assert n % 16 == 0 and k % 16 == 0
        w = weight.float()
        s = (w.abs().amax(1, keepdim=True) / 127.0).clamp_min(1e-12)        # [N, 1]
        q = torch.round(w / s).clamp_(-127, 127).to(torch.int8)            # [N, K]
        del w
        self.qt = q.t()            # [K, N] column-major view of the row-major [N, K] codes
        self.scale = s.contiguous()                                         # [N, 1] fp32
        self.size_n, self.size_k = n, k
        self.dtype = weight.dtype
        self.nbytes = q.numel() + s.numel() * 4

    def logits_q(self, a_q: torch.Tensor, a_s: torch.Tensor) -> torch.Tensor:
        """a_q int8 [M, K], a_s fp32 [M, 1] (per_token_quant_int8 output)."""
        from vllm import _custom_ops as ops

        return ops.cutlass_scaled_mm(a_q, self.qt, a_s, self.scale, self.dtype)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.int8_utils import per_token_quant_int8

        a_q, a_s = per_token_quant_int8(h)
        return self.logits_q(a_q, a_s)

    approx_logits = logits


class SplitHead:
    def __init__(self, weight: torch.Tensor, n_org: int, size: int, bits: int,
                 f_ids: torch.Tensor | None = None, sort_f: bool = False):
        from atdev_serve.shadow_head import ShadowHead, ShadowHeadConfig

        if f_ids is None:
            from atdev_serve import draft_vocab
            f_ids = draft_vocab.load_ids(draft_vocab.DraftVocabConfig(size=size), n_org)
        f_ids = f_ids.to(weight.device)
        if sort_f:  # ATDEV_VSPLIT_FAST: ascending ids (split_argmax tie rule); same values
            f_ids = f_ids.sort().values
        mask = torch.ones(weight.shape[0], dtype=torch.bool, device=weight.device)
        mask[f_ids] = False
        self.f_ids, self.t_ids = f_ids, mask.nonzero().squeeze(1)
        self.n_org = n_org
        self.fw = weight.index_select(0, self.f_ids).contiguous()
        tw = weight.index_select(0, self.t_ids)
        a8 = os.environ.get("ATDEV_VSPLIT_A8", "0") == "1"  # tail GEMM on W4A8 (int8 act)
        self.tail = ShadowHead(tw, ShadowHeadConfig(bits=bits, group_size=128, screen=1, a8=a8))
        del tw
        # ATDEV_VSPLIT_FBITS=8: the F part as an int8 g128 Marlin shadow too (accuracy-gated; the
        # bf16 copy fw stays for dtype / device / tests)
        self.fbits = int(os.environ.get("ATDEV_VSPLIT_FBITS", "16"))
        if self.fbits not in (4, 8, 16):
            raise ValueError(f"ATDEV_VSPLIT_FBITS must be 4, 8 or 16, got {self.fbits}")
        # ATDEV_VSPLIT_FA8=1 (with FBITS=8): F as int8 per-channel W8A8 (per-token int8 activations,
        # CUTLASS int8 GEMM, one launch at any M) instead of the g128 W8A16 Marlin shadow
        # ATDEV_VSPLIT_FA8=2: int8 g128 W8A16 Triton kernel (bf16 activations, one pass at M<=128)
        fa8 = os.environ.get("ATDEV_VSPLIT_FA8", "0") if self.fbits == 8 else "0"
        self.fa8 = fa8 == "1"
        if fa8 == "2":
            from atdev_serve.w8_head import W8G128Head
            self.fq = W8G128Head(self.fw)
        elif self.fa8:
            self.fq = W8A8Head(self.fw)
        else:
            self.fq = (ShadowHead(self.fw, ShadowHeadConfig(bits=self.fbits, group_size=128,
                                                            screen=1))
                       if self.fbits != 16 else None)
        self.nbytes = (self.fq.nbytes if self.fq is not None else self.fw.numel() * 2) + self.tail.nbytes

    def f_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Logits of the F rows: exact bf16 (default) or the int8 / int4 shadow (ATDEV_VSPLIT_FBITS)."""
        if self.fq is not None:
            return self.fq.approx_logits(h)
        return F.linear(h, self.fw)

    def logits(self, h: torch.Tensor, vocab_cols: int) -> torch.Tensor:
        out = torch.empty(h.shape[0], vocab_cols, dtype=self.fw.dtype, device=h.device)
        out.index_copy_(1, self.f_ids, self.f_logits(h))
        out.index_copy_(1, self.t_ids, self.tail.approx_logits(h))
        return out

    # -- ATDEV_VSPLIT_FAST: argmax-only verify (greedy tree steps) -----------------
    # Same two GEMMs (cuBLAS bf16 head part, Marlin int4 tail) -> same bf16 logit
    # values; the [M, vocab] scatter + torch.argmax are replaced by one fused
    # argmax over both parts that reproduces torch.argmax exactly (first = lowest
    # vocab id among maxima, NaN counts as the maximum, -0 == +0, ids >= n_org
    # excluded as the -inf padding / vocab slice does).

    def _fast_init(self) -> None:
        if getattr(self, "_fast_ready", False):
            return
        dev = self.fw.device
        self.f_ids32 = self.f_ids.to(torch.int32).contiguous()
        self.t_ids32 = self.t_ids.to(torch.int32).contiguous()
        # t_ids is ascending (nonzero), so the vocab padding rows are its last entries
        self.nt_valid = int((self.t_ids < self.n_org).sum())
        self.fast_ok = (bool((self.f_ids < self.n_org).all())
                        and bool((self.f_ids[1:] > self.f_ids[:-1]).all()))
        self.streams = os.environ.get("ATDEV_VSPLIT_STREAMS", "0") == "1"
        self.side = torch.cuda.Stream(device=dev) if self.streams else None
        self.tail_cfg = None
        cfg = os.environ.get("ATDEV_VSPLIT_TAIL_CFG", "")
        if cfg and self.tail.backend == "marlin" and not self.tail.a8:
            self.tail_cfg = [int(x) for x in cfg.split(",")]
            if not self._check_tail_cfg():
                logger.warning("atdev verify_split_head: ATDEV_VSPLIT_TAIL_CFG=%s is not "
                               "bit-identical to stock Marlin on this head; ignored", cfg)
                self.tail_cfg = None
        self._fast_ready = True

    def tail_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Tail approx logits; marlin_plus launch config that is bit-identical to stock."""
        if self.tail_cfg is None or not 0 < h.shape[0] <= 16:
            return self.tail.approx_logits(h)
        from atdev_serve import marlin_plus

        t = self.tail
        b = marlin_plus.bufs(h.device)
        out = torch.empty((h.shape[0], t.size_n), dtype=h.dtype, device=h.device)
        return marlin_plus.mod().gemm(h.contiguous(), t.qweight, t.scales, None, out, b["locks"],
                                      b["c_tmp"], t.size_n, t.size_k, list(self.tail_cfg),
                                      True, False)

    @torch.no_grad()
    def _check_tail_cfg(self) -> bool:
        t = self.tail
        if t.padded_n != t.size_n or t.size_k % 128:
            return False
        g = torch.Generator(device=self.fw.device).manual_seed(1234)
        for m in (1, 7, 16):
            for sc in (0.05, 1.0, 8.0):
                h = (torch.randn(m, t.size_k, device=self.fw.device, generator=g) * sc
                     ).to(self.fw.dtype)
                try:
                    if not torch.equal(self.tail_logits(h), t.approx_logits(h)):
                        return False
                except RuntimeError:
                    return False
        return True

    def argmax(self, h: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """int64 [M] == torch.argmax(self.logits(h, .)[:, :n_org], -1), bit-exact.
        Needs ascending f_ids (sort_f=True). out: optional int64 [>= M] buffer (written [:M])."""
        self._fast_init()
        if self.side is not None:
            cur = torch.cuda.current_stream(h.device)
            self.side.wait_stream(cur)
            with torch.cuda.stream(self.side):
                tlog = self.tail_logits(h)
            fl = self.f_logits(h)
            cur.wait_stream(self.side)
            h.record_stream(self.side)
            tlog.record_stream(cur)
        else:
            fl = self.f_logits(h)
            tlog = self.tail_logits(h)
        _post_head_hook()
        return split_argmax(fl, self.f_ids32, tlog, self.t_ids32, None, self.nt_valid, out=out)




def _post_head_hook() -> None:
    """dflash_tree ATDEV_FC_EARLY=2: start the drafter fc on its side stream after the head GEMMs."""
    import sys

    dt = sys.modules.get("atdev_serve.dflash_tree")
    if dt is not None and dt.S.fc_late is not None:
        dt.fc_post_head()


import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _vs_part(x_ptr, ids_ptr, n, stride, row, cols):
    cm = cols < n
    v = tl.load(x_ptr + row * stride + cols, mask=cm, other=float("-inf")).to(tl.float32)
    m = tl.max(v, axis=0)
    # one min-reduction finds the first NaN if any (codes < 2^30: torch.argmax returns
    # the first NaN), else the first max (codes >= 2^30); ids ascending within a part,
    # so the first max = lowest vocab id among the block's ties (v == m: -0 == +0)
    code = tl.where(v != v, cols, tl.where(cm & (v == m), cols + 1073741824, 2147483647))
    code = tl.min(code, axis=0)
    has_nan = code < 1073741824
    vid = tl.load(ids_ptr + (code & 1073741823))
    # block key: order-preserving int32 of the max (-0 -> +0); NaN outranks everything
    mz = tl.where(m == 0.0, 0.0, m)
    bm = mz.to(tl.int32, bitcast=True)
    key = tl.where(bm >= 0, bm, bm ^ 0x7FFFFFFF)
    key = tl.where(has_nan, 0x7FFFFFFF, key)
    return key, vid


@triton.jit
def _vs_stage1(f_ptr, fid_ptr, nf, fs, t_ptr, tid_ptr, nt, ts, pk_ptr, pi_ptr,
               nbf, NB, BLK: tl.constexpr):
    pid = tl.program_id(0)
    row = tl.program_id(1)
    if pid < nbf:
        kmax, imin = _vs_part(f_ptr, fid_ptr, nf, fs, row, pid * BLK + tl.arange(0, BLK))
    else:
        kmax, imin = _vs_part(t_ptr, tid_ptr, nt, ts, row,
                              (pid - nbf) * BLK + tl.arange(0, BLK))
    tl.store(pk_ptr + row * NB + pid, kmax)
    tl.store(pi_ptr + row * NB + pid, imin)


@triton.jit
def _vs_stage2(pk_ptr, pi_ptr, out_ptr, NB, PNB: tl.constexpr):
    r = tl.program_id(0)
    o = tl.arange(0, PNB)
    m = o < NB
    k = tl.load(pk_ptr + r * NB + o, mask=m, other=-2147483648)
    i = tl.load(pi_ptr + r * NB + o, mask=m, other=2147483647)
    kmax = tl.max(k, axis=0)
    best = tl.min(tl.where(k == kmax, i, 2147483647), axis=0)
    tl.store(out_ptr + r, best.to(tl.int64))


_BLK, _NW = 4096, 4


def split_argmax(fl: torch.Tensor, f_ids: torch.Tensor, tl_: torch.Tensor,
                 t_ids: torch.Tensor, nf: int | None = None, nt: int | None = None,
                 out: torch.Tensor | None = None) -> torch.Tensor:
    """torch.argmax over the virtual rows logits[:, f_ids[:nf]] = fl[:, :nf],
    logits[:, t_ids[:nt]] = tl_[:, :nt] (int32 ids, each ASCENDING; columns past nf / nt
    -- e.g. the vocab padding rows -- are ignored). Ties -> lowest vocab id, NaN wins."""
    m = fl.shape[0]
    assert tl_.shape[0] == m and fl.stride(1) == 1 and tl_.stride(1) == 1
    if out is None:
        out = torch.empty(m, dtype=torch.int64, device=fl.device)
    else:  # caller-owned persistent buffer (dflash_tree ATDEV_TREE_ACCEPT_GRAPH)
        assert out.dtype == torch.int64 and out.is_contiguous() and out.numel() >= m
        out = out[:m]
    if m == 0:
        return out
    nf = fl.shape[1] if nf is None else nf
    nt = tl_.shape[1] if nt is None else nt
    nbf = triton.cdiv(nf, _BLK)
    nb = nbf + triton.cdiv(nt, _BLK)
    pk = torch.empty(m, nb, dtype=torch.int32, device=fl.device)
    pi = torch.empty(m, nb, dtype=torch.int32, device=fl.device)
    _vs_stage1[(nb, m)](fl, f_ids, nf, fl.stride(0), tl_, t_ids, nt, tl_.stride(0), pk, pi,
                        nbf, nb, BLK=_BLK, num_warps=_NW)
    _vs_stage2[(m,)](pk, pi, out, nb, PNB=triton.next_power_of_2(nb), num_warps=4)
    return out


def _tree_argmax_only(proc, hidden_states: torch.Tensor) -> bool:
    """True inside dflash_tree's greedy tree accept (its sample() is the only logits
    consumer; _decide already required greedy / no logprobs / no grammar / 1 request)."""
    import sys

    dt = sys.modules.get("atdev_serve.dflash_tree")
    if dt is None:
        return False
    st = dt.S
    return (st.accepting and st.mode == "tree" and st.N is not None
            and hidden_states.shape[0] == st.N * st.batch.num_reqs
            and proc.scale == 1.0 and proc.soft_cap is None
            and not torch.cuda.is_current_stream_capturing())


def register() -> None:
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    fast = os.environ.get("ATDEV_VSPLIT_FAST", "0") == "1"
    size = int(os.environ.get("ATDEV_VSPLIT_SIZE", "32768"))
    bits = int(os.environ.get("ATDEV_VSPLIT_BITS", "8"))
    min_rows = int(os.environ.get("ATDEV_VSPLIT_MIN_ROWS", "1"))
    orig = LogitsProcessor._apply_head

    def _apply_head(self, lm_head, hidden_states, embedding_bias):
        w = getattr(lm_head, "weight", None)
        ok = (
            embedding_bias is None
            and hidden_states.dim() == 2
            and hidden_states.shape[0] >= min_rows
            and hidden_states.is_cuda
            and getattr(lm_head, "tp_size", 1) == 1
            and isinstance(getattr(lm_head, "quant_method", None),
                           (UnquantizedEmbeddingMethod, UnquantizedLinearMethod))
            and isinstance(w, torch.Tensor)
            and w.dtype == hidden_states.dtype
            and (self.head_dtype is None or self.head_dtype == hidden_states.dtype)
        )
        if not ok:
            return orig(self, lm_head, hidden_states, embedding_bias)
        sh = getattr(lm_head, _ATTR, None)
        if sh is None:
            if torch.cuda.is_current_stream_capturing():
                return orig(self, lm_head, hidden_states, embedding_bias)
            n_org = w.shape[0] - lm_head.shard_indices.num_org_vocab_padding
            sh = SplitHead(w.data, n_org, size, bits, sort_f=fast)
            if fast:
                sh._fast_init()
            from atdev_serve import vsplit_a4scr  # C32e: W4A4 screen + exact rescoring (opt-in)
            sh.a4scr = vsplit_a4scr.A4Screen(w.data, n_org) if vsplit_a4scr.enabled() else None
            setattr(lm_head, _ATTR, sh)
            logger.warning("atdev verify_split_head: F=%d exact + %d tail int%d (%.3f GB vs %.2f GB)"
                           "%s", sh.f_ids.numel(), sh.t_ids.numel(), bits, sh.nbytes / 1e9,
                           w.numel() * 2 / 1e9,
                           f" FAST(ok={sh.fast_ok}, streams={sh.streams}, tail_cfg={sh.tail_cfg})"
                           if fast else "")
        if fast and sh.fast_ok and _tree_argmax_only(self, hidden_states):
            from atdev_serve import dflash_tree

            # ATDEV_TREE_ACCEPT_GRAPH: argmax into the accept graph's persistent input buffer
            am = sh.argmax(hidden_states, out=dflash_tree.am_out(hidden_states.shape[0]))
            # the tree accept reads only the argmax; this placeholder is never read
            logits = torch.empty(hidden_states.shape[0], w.shape[0], dtype=hidden_states.dtype,
                                 device=hidden_states.device)
            dflash_tree.S.fast_am = (logits.data_ptr(), logits.shape[0], am)
            return logits
        logits = sh.logits(hidden_states, w.shape[0])
        if w.shape[0] > sh.n_org:
            logits[:, sh.n_org:] = float("-inf")
        return logits

    LogitsProcessor._apply_head = _apply_head
    logger.info("atdev verify_split_head: registered (N=%d, tail int%d)", size, bits)
