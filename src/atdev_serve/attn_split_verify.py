"""Split-KV (3D) TRITON_ATTN for small spec-decode verify batches (ATDEV_OPT=attn_split_verify).

Found by the reframe-B step timeline (docs/research/reframe-B-overlap.md §2): at c=1
the target verify (q_len = 16 tree nodes, or k+1 chain tokens) takes vLLM's *2D*
``kernel_unified_attention`` path because ``max_seqlen_q > 1``. The 2D grid is
(q_blocks = 16 // BLOCK_Q + 1 = 9, kv_heads = 4) = 36 CTAs on 108 SMs, each
streaming the whole context serially: 155 us per layer at ~3k context, 2.5 ms of a
16 ms verify graph (16 full-attention layers).

The 3D path (one more grid axis of NUM_PAR_SOFTMAX_SEGMENTS = 16 KV segments +
``reduce_segments``) is the same kernel with ``IS_3D`` set; its masking is per row
(causal position, ``qq_bias`` tree mask), so it is valid for q_len > 1. vLLM only
gates it on ``max_seqlen_q > 1`` because its softmax scratch buffers have
``seq_threshold_3D`` token rows. We take the 3D path whenever the batch's total
query rows fit the scratch (that is a static property of the captured graph
shape, so CUDA-graph capture and replay agree).

Exactness: a different (but equally valid) fp32 reduction order of the softmax;
outputs are not bit-identical to the 2D path. Gate with GSM8K + the greedy
teacher-forcing check like every kernel change.
"""
from __future__ import annotations

import functools
import logging
import os

OPT_NAME = "attn_split_verify"
SUPPORTED_VLLM_VERSION = "0.31.0"
log = logging.getLogger("atdev_serve")
_STATS = {"split": 0, "logged": False}
_RETILED = None


def retiled_unified_attention():
    """vLLM's ``unified_attention`` launcher re-built with tunable BLOCK_M /
    num_warps / decode tile (module globals ``_ATDEV_BLOCK_M`` etc.; 0 = stock).
    BLOCK_M = 16 holds only 16 // 6 = 2 query tokens per program for Qwen3.8's
    6 q heads per kv head, so a 16-node verify runs 9 q-blocks that each
    re-read the whole KV segment; BLOCK_M = 64 -> 10 tokens, 128 -> 21."""
    global _RETILED
    if _RETILED is not None:
        return _RETILED
    import inspect
    import textwrap

    from vllm.v1.attention.ops import triton_unified_attention as tua

    src = textwrap.dedent(inspect.getsource(tua.unified_attention))
    old_bm = ("    BLOCK_M = (\n        16 if num_queries_per_kv <= 16 else "
              "triton.next_power_of_2(num_queries_per_kv)\n    )\n")
    old_w = "    launch_num_warps: int | None = None\n"
    old_t = ("    TILE_SIZE_DECODE = _get_tile_size(\n        head_size, sliding_window_val, "
             "q.element_size(), is_prefill=False\n    )\n")
    for o in (old_bm, old_w, old_t):
        if o not in src:
            raise RuntimeError(f"atdev {OPT_NAME}: unified_attention source changed; "
                               f"cannot re-tile (missing {o[:40]!r})")
    src = src.replace(old_bm, old_bm + "    if _ATDEV_BLOCK_M:\n        BLOCK_M = _ATDEV_BLOCK_M\n")
    src = src.replace(old_w, "    launch_num_warps: int | None = _ATDEV_WARPS or None\n")
    src = src.replace(old_t, old_t + "    if _ATDEV_TILE:\n        TILE_SIZE_DECODE = _ATDEV_TILE\n")
    src = src.replace("def unified_attention(", "def atdev_retiled_unified_attention(", 1)
    g = dict(vars(tua))
    g.update(_ATDEV_BLOCK_M=int(os.environ.get("ATDEV_ATTN_BLOCK_M", "0")),
             _ATDEV_WARPS=int(os.environ.get("ATDEV_ATTN_WARPS", "0")),
             _ATDEV_TILE=int(os.environ.get("ATDEV_ATTN_TILE", "0")))
    exec(compile(src, tua.__file__ + ":atdev_retiled", "exec"), g)
    _RETILED = (g["atdev_retiled_unified_attention"], g)
    return _RETILED


def register() -> None:
    import vllm
    from vllm.v1.attention.backends import triton_attn as ta

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written against vLLM "
                           f"{SUPPORTED_VLLM_VERSION}, found {vllm.__version__}")
    if getattr(ta.unified_attention, "_atdev_split", False):
        return
    max_rows = int(os.environ.get("ATDEV_ATTN_SPLIT_MAX_ROWS", "64"))
    orig = ta.unified_attention
    retile = any(int(os.environ.get(k, "0")) for k in
                 ("ATDEV_ATTN_BLOCK_M", "ATDEV_ATTN_WARPS", "ATDEV_ATTN_TILE"))
    fast = retiled_unified_attention()[0] if retile else None
    # ATDEV_ATTN_SPLIT_SCRATCH=N (c=32 verify: 64 rows > vLLM's 32-row 3D scratch):
    # own softmax-segment buffers with N token rows so the 3D path also covers
    # batches of up to N query rows. Allocated outside graph capture only.
    scratch_rows = int(os.environ.get("ATDEV_ATTN_SPLIT_SCRATCH", "0"))
    if scratch_rows:
        max_rows = max(max_rows, scratch_rows)
    own = {}

    def _own_scratch(segm, mx, ex):
        key = (segm.shape[1:], segm.device)
        if key not in own:
            import torch
            if torch.cuda.is_current_stream_capturing():
                return None
            own[key] = (segm.new_empty((scratch_rows, *segm.shape[1:])),
                        mx.new_empty((scratch_rows, *mx.shape[1:])),
                        ex.new_empty((scratch_rows, *ex.shape[1:])))
        return own[key]

    @functools.wraps(orig)
    def unified_attention(*args, **kw):
        q = kw.get("q")
        segm = kw.get("softmax_segm_output")
        mq = kw.get("max_seqlen_q")
        if (scratch_rows and q is not None and segm is not None and mq is not None and mq > 1
                and segm.shape[0] < q.shape[0] <= scratch_rows
                and kw.get("num_par_softmax_segments") is not None):
            bufs = _own_scratch(segm, kw["softmax_segm_max"], kw["softmax_segm_expsum"])
            if bufs is not None:
                kw["softmax_segm_output"], kw["softmax_segm_max"], kw["softmax_segm_expsum"] = bufs
                kw["seq_threshold_3D"] = scratch_rows
                segm = bufs[0]
        if (q is not None and segm is not None and mq is not None and mq > 1
                and kw.get("num_par_softmax_segments") is not None
                and kw.get("seq_threshold_3D") is not None
                and q.shape[0] <= min(segm.shape[0], max_rows)
                and kw["seqused_k"].shape[0] <= kw["seq_threshold_3D"]):
            kw["max_seqlen_q"] = 1  # only gates the 2D/3D choice in 0.31.0
            if not _STATS["logged"]:
                _STATS["logged"] = True
                log.warning("atdev %s: split-KV attention ENGAGED for %d query rows "
                            "(max_query_len %d, %d segments)", OPT_NAME, q.shape[0], mq,
                            kw["num_par_softmax_segments"])
            if fast is not None:
                return fast(*args, **kw)
        return orig(*args, **kw)

    unified_attention._atdev_split = True
    ta.unified_attention = unified_attention
    log.info("atdev %s: registered", OPT_NAME)
