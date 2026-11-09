"""FlashInfer fa2 paged attention for uniform spec-decode verify batches on the TRITON_ATTN backend
(ATDEV_OPT=fi_attn). Engine-only (kernel), c=32 target: MTP k=1 -> every request verifies 2 query rows.

Microbench ( page 416 = this model's attention page, 32 req x 2 rows, P6-like context
mix 100-1200): Triton unified_attention 127.6 us/layer -> FlashInfer 99.0 (0.78x) = -0.46 ms/step over the 16
full-attention layers; 1k 0.85x, 2k 0.83x, 4k 0.82x. vLLM's own FLASHINFER backend lost e2e (FI-base) through its
per-layer/per-step overheads; here ONE plan per step is shared by all 16 layers.

Mechanism
* ``TritonAttentionMetadataBuilder.build`` (eager, every step, before the FULL graph replay) is wrapped: for a
  pure verify batch (max_query_len == ATDEV_FI_QLEN, every real request has exactly that many rows) it calls
  ``wrapper.plan`` with the host-side *upper bound* of the kv lengths (async scheduling: exact lengths are only
  on the GPU) and marks the metadata. One CUDA-graph-mode wrapper per (num_reqs, num_tokens) shape (fixed
  buffers, shared 256 MB float workspace).
* Inside the forward (and therefore inside the captured graph) the first FI layer of the step launches a tiny
  Triton kernel that rewrites the wrapper's kv_indptr / kv_indices / last_page_len buffers from the GPU
  ``seq_lens`` and block table (exact lengths). Chunks planned for the upper bound beyond the true length are
  empty, so the result is exact attention over the true context.
* ``TritonAttentionImpl.forward``: target (non-draft) layers with marked metadata run ``wrapper.run(q, (K, V))``
  on the TRITON_ATTN cache views (K/V interleaved in the last dim; FlashInfer takes the strides). Everything else
  (prefill, mixed batches, the MTP draft layer, profiling) runs the stock Triton kernel.

Numerics: a different (equally valid) fp32 reduction order; max rel diff vs Triton 3-6e-3 at bf16 in the microbench.

Env: ATDEV_FI_QLEN (default 2), ATDEV_FI_WS_MB (256), ATDEV_FI_MAX_REQS (64).
"""
from __future__ import annotations

import functools
import logging
import os

import torch

OPT_NAME = "fi_attn"
SUPPORTED_VLLM_VERSION = "0.31.0"
log = logging.getLogger("atdev_serve")


class _S:
    ws = None
    wrappers: dict = {}
    qlen = int(os.environ.get("ATDEV_FI_QLEN", "2"))
    max_reqs = int(os.environ.get("ATDEV_FI_MAX_REQS", "64"))
    n_plan = 0
    n_run = 0
    logged = False
    first_layer = None  # name of the first full-attention target layer (fixup runs there)
    seen: set = set()


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _fixup_kernel(seq_lens_ptr, qsl_ptr, bt_ptr, bt_stride, indptr_ptr, indices_ptr, last_ptr, R, R_META,
                      PAGE: tl.constexpr, MAXP: tl.constexpr):
        """Exact paged-kv description from the GPU seq_lens (one program, R <= 64 sequential requests).
        Requests r >= R_META (graph padding) and zero-query requests get a 1-token dummy context (page of row r)
        so FlashInfer's padded rows stay finite; their outputs are discarded."""
        off = 0
        tl.store(indptr_ptr, 0)
        cols = tl.arange(0, MAXP)
        for r in range(R):
            real = r < R_META
            ln = tl.load(seq_lens_ptr + r, mask=real, other=0)
            ql = tl.load(qsl_ptr + r + 1, mask=real, other=0) - tl.load(qsl_ptr + r, mask=real, other=0)
            ln = tl.where(real & (ql > 0), tl.maximum(ln, 1), 1)
            npg = tl.where(ln > 0, (ln + PAGE - 1) // PAGE, 0)
            last = tl.where(npg > 0, ln - (npg - 1) * PAGE, 0)
            pages = tl.load(bt_ptr + r * bt_stride + cols, mask=cols < npg, other=0)
            tl.store(indices_ptr + off + cols, pages, mask=cols < npg)
            off += npg
            tl.store(indptr_ptr + r + 1, off)
            tl.store(last_ptr + r, last)
except ImportError:  # pragma: no cover
    _fixup_kernel = None


class _Wrapper:
    def __init__(self, num_reqs, num_tokens, max_pages, device):
        import flashinfer
        if _S.ws is None:
            _S.ws = torch.empty(int(os.environ.get("ATDEV_FI_WS_MB", "256")) << 20, dtype=torch.uint8, device=device)
        self.indptr = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        self.indices = torch.zeros(max_pages, dtype=torch.int32, device=device)
        self.last = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        self.w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            _S.ws, "NHD", use_cuda_graph=True, backend="fa2",
            qo_indptr_buf=torch.zeros(num_reqs + 1, dtype=torch.int32, device=device),
            paged_kv_indptr_buf=self.indptr, paged_kv_indices_buf=self.indices,
            paged_kv_last_page_len_buf=self.last)
        self.num_reqs, self.num_tokens, self.max_pages = num_reqs, num_tokens, max_pages
        self.seq_lens = None
        self.qsl = None
        self.r_meta = 0
        self.block_table = None
        self.page = None
        self.fixed = False  # fixup done for the current step


def _plan(md, common, impl_cfg) -> None:
    r_meta, num_tokens = common.num_reqs, common.num_actual_tokens
    ub0 = common.seq_lens_cpu_upper_bound
    Q = _S.qlen
    if ub0 is None or r_meta == 0 or num_tokens % Q:
        return
    num_reqs = num_tokens // Q  # FlashInfer sees every (padded) token: uniform Q rows per request
    if num_reqs > _S.max_reqs or num_reqs < r_meta:
        return
    qsl_m = common.query_start_loc_cpu[: r_meta + 1]
    qlens = qsl_m[1:] - qsl_m[:-1]
    if common.max_query_len != Q or bool((qlens > Q).any()) or bool(((qlens != Q) & (qlens != 0)).any()):
        return
    bt = common.block_table_tensor
    if bt.shape[0] < num_reqs:
        return
    page = impl_cfg["page"]
    ub = torch.ones(num_reqs, dtype=torch.int64)
    real = qlens > 0
    ub[:r_meta][real] = ub0[:r_meta][real].to(torch.int64).clamp_min(1)
    npg = (ub + page - 1) // page
    qsl = torch.arange(0, num_tokens + 1, Q, dtype=torch.int32)
    key = (num_reqs, num_tokens, r_meta)
    w = _S.wrappers.get(key)
    max_pages = num_reqs * bt.shape[1]
    if w is None or w.max_pages < max_pages:
        if torch.cuda.is_current_stream_capturing():
            return
        w = _S.wrappers[key] = _Wrapper(num_reqs, num_tokens, max_pages, bt.device)
    kv_indptr = torch.zeros(num_reqs + 1, dtype=torch.int32)
    kv_indptr[1:] = npg.cumsum(0).to(torch.int32)
    last = (ub - (npg - 1) * page).to(torch.int32)
    # host plan on the upper bound; the indices content is rewritten on the GPU by the fixup kernel
    w.w.plan(qsl, kv_indptr, w.indices[: int(kv_indptr[-1])], last,
             impl_cfg["qh"], impl_cfg["kvh"], impl_cfg["hs"], page, causal=True, sm_scale=impl_cfg["scale"],
             q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    w.seq_lens, w.qsl, w.r_meta, w.block_table, w.page, w.fixed = (common.seq_lens, common.query_start_loc, r_meta,
                                                                   bt, page, False)
    md._atdev_fi = w
    _S.n_plan += 1


def register() -> None:
    import vllm
    from vllm.v1.attention.backends import triton_attn as ta

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written against vLLM {SUPPORTED_VLLM_VERSION}, found {vllm.__version__}")
    B = ta.TritonAttentionMetadataBuilder
    if getattr(B.build, "_atdev_fi", False):
        return
    orig_build = B.build
    cfg: dict = {}

    @functools.wraps(orig_build)
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = orig_build(self, common_prefix_len, common_attn_metadata, fast_build)
        if cfg and common_prefix_len == 0:
            try:
                _plan(md, common_attn_metadata, cfg)
            except Exception as e:  # never break serving: fall back to Triton
                if not _S.logged:
                    _S.logged = True
                    log.warning("atdev %s: plan failed (%s) -> Triton", OPT_NAME, e)
        return md

    build._atdev_fi = True
    B.build = build

    orig_fwd = ta.TritonAttentionImpl.forward

    @functools.wraps(orig_fwd)
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
                output_block_scale=None):
        if not cfg and attn_metadata is not None and not getattr(layer, "is_draft_layer", False):
            # first target call: record the static shape parameters for the planner
            page = kv_cache.shape[2] if kv_cache.dim() == 4 else None
            if page is not None and self.kv_cache_dtype in ("auto", "bfloat16") and self.sinks is None \
                    and self.alibi_slopes is None and not self.logits_soft_cap and self.sliding_window in (None, (-1, -1)):
                cfg.update(page=page, qh=self.num_heads, kvh=self.num_kv_heads, hs=self.head_size, scale=self.scale)
                log.warning("atdev %s: enabled (page %d, %d q / %d kv heads, hd %d, verify rows/request %d)",
                            OPT_NAME, page, self.num_heads, self.num_kv_heads, self.head_size, _S.qlen)
        w = getattr(attn_metadata, "_atdev_fi", None) if attn_metadata is not None else None
        if w is None or getattr(layer, "is_draft_layer", False) or output_scale is not None:
            return orig_fwd(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                            output_block_scale)
        name = getattr(layer, "layer_name", None)
        if _S.first_layer is None:
            _S.first_layer = name
        if name == _S.first_layer:  # once per forward (captured into the graph): exact kv description
            bt = w.block_table
            _fixup_kernel[(1,)](w.seq_lens, w.qsl, bt, bt.stride(0), w.indptr, w.indices, w.last, w.num_reqs,
                                w.r_meta, PAGE=w.page, MAXP=triton.next_power_of_2(bt.shape[1]))
        n = w.num_tokens
        kc, vc = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        w.w.run(query[:n], (kc, vc), out=output[:n].view(n, self.num_heads, self.head_size))
        _S.n_run += 1
        shape = (w.num_reqs, w.num_tokens, w.r_meta)
        if shape not in _S.seen and not torch.cuda.is_current_stream_capturing():
            _S.seen.add(shape)  # one line per batch shape at its first real (non-capture) use
            log.warning("atdev %s: FlashInfer verify attention ENGAGED (reqs %d [meta %d], tokens %d; plans so far %d)",
                        OPT_NAME, w.num_reqs, w.r_meta, w.num_tokens, _S.n_plan)
        return output

    forward._atdev_fi = True
    ta.TritonAttentionImpl.forward = forward
    log.info("atdev %s: registered", OPT_NAME)
