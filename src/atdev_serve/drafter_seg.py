"""Segment-causal draft block for DFlash2 drafters (D16-seg; auto-enabled by dflash_tree).

A drafter whose config has ``dflash_config.segment_causal = s`` (trainer ``--segment-causal s``) was trained with
block rows of offset < s blind to block rows of offset >= s; rows >= s see the whole block. With the 2-tap conv causal
inside the block, rows 0..s-1 of a block-16 forward then equal a block-s forward (s = 8: depths 1..7 behave like the
block-8 drafter).

Serving: every drafter attention call (FLASH_ATTN backend, non-causal, paged KV with the block rows' K/V already in
the cache at the last q_len positions of each request) is rewritten as 2 x num_reqs pseudo-sequences sharing the
request's block-table row:
    pseudo 2r   : query rows [cu[r], cu[r] + min(s, q_len))  seqused_k = seqused_k[r] - q_len + min(s, q_len)
    pseudo 2r+1 : query rows [cu[r] + min(s, q_len), cu[r+1])  seqused_k = seqused_k[r]
FlashAttention aligns non-causal / windowed queries bottom-right, so query row j of pseudo 2r still sits at key
position seqused_k - q_len + j, and the sliding window is unchanged. Requests with q_len <= s are untouched (empty
second pseudo-sequence). The metadata is built by one Triton launch per layer from the step's device tensors (no
host sync) -> CUDA-graph safe. The wrapper sits outside drafter_attn3d's wrapper when both are on.
"""
from __future__ import annotations

import functools
import logging

OPT_NAME = "drafter_seg"
SUPPORTED_VLLM_VERSION = "0.31.0"
log = logging.getLogger("atdev_serve")

_S = {"seg": 0, "draft": False, "installed": False, "engaged": 0, "logged": False}


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def seg_meta_kernel(cu_ptr, sk_ptr, bt_ptr, cu2_ptr, sk2_ptr, bt2_ptr, n_cols, bt_stride, bt2_stride,
                        SEG: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.program_id(0)
        c0 = tl.load(cu_ptr + r)
        c1 = tl.load(cu_ptr + r + 1)
        ql = c1 - c0
        s = tl.minimum(ql, SEG)
        sk = tl.load(sk_ptr + r)
        tl.store(cu2_ptr + 2 * r, c0)
        tl.store(cu2_ptr + 2 * r + 1, c0 + s)
        if r == tl.num_programs(0) - 1:
            tl.store(cu2_ptr + 2 * r + 2, c1)
        tl.store(sk2_ptr + 2 * r, sk - ql + s)
        tl.store(sk2_ptr + 2 * r + 1, sk)
        for off in range(0, n_cols, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            m = cols < n_cols
            v = tl.load(bt_ptr + r * bt_stride + cols, mask=m)
            tl.store(bt2_ptr + (2 * r) * bt2_stride + cols, v, mask=m)
            tl.store(bt2_ptr + (2 * r + 1) * bt2_stride + cols, v, mask=m)

    return seg_meta_kernel


_K = {}


def split_meta(cu, sk, bt, seg: int, use_triton: bool = True):
    """(cu_seqlens_q [b+1], seqused_k [b], block_table [b, n]) -> the 2b pseudo-sequence metadata."""
    import torch
    b = sk.shape[0]
    if not use_triton or not cu.is_cuda:
        ql = cu[1:] - cu[:-1]
        s = ql.clamp(max=seg)
        cu2 = torch.cat([torch.stack([cu[:-1], cu[:-1] + s], 1).flatten(), cu[-1:]])
        sk2 = torch.stack([sk - ql + s, sk], 1).flatten()
        return cu2.to(cu.dtype), sk2.to(sk.dtype), bt.repeat_interleave(2, 0)
    if "k" not in _K:
        _K["k"] = _kernel()
    cu2 = torch.empty(2 * b + 1, dtype=cu.dtype, device=cu.device)
    sk2 = torch.empty(2 * b, dtype=sk.dtype, device=sk.device)
    bt2 = torch.empty(2 * b, bt.shape[1], dtype=bt.dtype, device=bt.device)
    _K["k"][(b,)](cu, sk, bt, cu2, sk2, bt2, bt.shape[1], bt.stride(0), bt2.stride(0), SEG=seg, BLOCK=256)
    return cu2, sk2, bt2


def enable(seg: int) -> None:
    """Called by dflash_tree when the drafter config carries dflash_config.segment_causal."""
    import vllm
    from vllm.v1.attention.backends import flash_attn as fa

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written against vLLM {SUPPORTED_VLLM_VERSION}, found {vllm.__version__}")
    _S["seg"] = int(seg)
    if _S["installed"]:
        return
    inner = fa.flash_attn_varlen_func

    @functools.wraps(inner)
    def varlen(*args, **kw):
        if (_S["draft"] and _S["seg"] and not args and kw.get("seqused_k") is not None
                and kw.get("block_table") is not None and kw.get("cu_seqlens_q") is not None
                and kw.get("scheduler_metadata") is None and kw.get("causal") is False):
            cu2, sk2, bt2 = split_meta(kw["cu_seqlens_q"], kw["seqused_k"], kw["block_table"], _S["seg"])
            kw = dict(kw, cu_seqlens_q=cu2, seqused_k=sk2, block_table=bt2)
            _S["engaged"] += 1
            if not _S["logged"]:
                _S["logged"] = True
                log.warning("atdev %s: segment-causal draft attention ENGAGED (rows < %d see rows < %d; "
                            "%d query rows, max_query_len %s)", OPT_NAME, _S["seg"], _S["seg"],
                            kw["q"].shape[0], kw.get("max_seqlen_q"))
        elif _S["draft"] and _S["seg"] and not _S.get("warned_skip"):
            _S["warned_skip"] = True
            log.warning("atdev %s: a draft attention call did not match the paged non-causal form -> NOT masked "
                        "(causal=%s, seqused_k=%s)", OPT_NAME, kw.get("causal"), kw.get("seqused_k") is not None)
        return inner(*args, **kw)

    varlen._atdev_seg = True
    fa.flash_attn_varlen_func = varlen

    orig_fwd = fa.FlashAttentionImpl.forward

    @functools.wraps(orig_fwd)
    def forward(self, layer, *a, **k):
        if not getattr(layer, "is_draft_layer", False):
            return orig_fwd(self, layer, *a, **k)
        prev = _S["draft"]
        _S["draft"] = True
        try:
            return orig_fwd(self, layer, *a, **k)
        finally:
            _S["draft"] = prev

    fa.FlashAttentionImpl.forward = forward
    _S["installed"] = True
    try:  # JIT-compile the metadata kernel now (never inside a CUDA-graph capture)
        import torch
        if torch.cuda.is_available():
            d = torch.device("cuda", torch.cuda.current_device())
            split_meta(torch.tensor([0, 16], dtype=torch.int32, device=d), torch.tensor([40], dtype=torch.int32, device=d),
                       torch.zeros(1, 4, dtype=torch.int32, device=d), _S["seg"])
            torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        log.warning("atdev %s: kernel warmup failed (%s)", OPT_NAME, e)
    log.warning("atdev %s: enabled (segment %d; wraps %s)", OPT_NAME, seg,
                "drafter_attn3d" if getattr(inner, "_atdev_3d", False) else "flash_attn_varlen_func")


def register() -> None:  # ATDEV_OPT entry: nothing to do until a segment-causal drafter is loaded
    log.info("atdev %s: registered (auto-enabled by dflash_tree for dflash_config.segment_causal)", OPT_NAME)
