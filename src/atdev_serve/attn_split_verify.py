"""Split-KV (3D) TRITON_ATTN for small spec-decode verify batches (ATDEV_OPT=attn_split_verify).

At c=1 the target verify (q_len = 16 tree nodes, or k+1 chain tokens) takes vLLM's *2D*
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
outputs are not bit-identical to the 2D path. On the c=1 tree verify itself the
``tree_attn`` CUDA kernel takes over, bit-identical to this 3D path.
"""
from __future__ import annotations

import functools
import logging

OPT_NAME = "attn_split_verify"
SUPPORTED_VLLM_VERSION = "0.31.0"
log = logging.getLogger("atdev_serve")
_STATS = {"logged": False}
MAX_ROWS = 64  # largest verify batch (query rows) sent to the 3D path


def register() -> None:
    import vllm
    from vllm.v1.attention.backends import triton_attn as ta

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written against vLLM "
                           f"{SUPPORTED_VLLM_VERSION}, found {vllm.__version__}")
    if getattr(ta.unified_attention, "_atdev_split", False):
        return
    orig = ta.unified_attention

    @functools.wraps(orig)
    def unified_attention(*args, **kw):
        q = kw.get("q")
        segm = kw.get("softmax_segm_output")
        mq = kw.get("max_seqlen_q")
        if (q is not None and segm is not None and mq is not None and mq > 1
                and kw.get("num_par_softmax_segments") is not None
                and kw.get("seq_threshold_3D") is not None
                and q.shape[0] <= min(segm.shape[0], MAX_ROWS)
                and kw["seqused_k"].shape[0] <= kw["seq_threshold_3D"]):
            kw["max_seqlen_q"] = 1  # only gates the 2D/3D choice in 0.31.0
            if not _STATS["logged"]:
                _STATS["logged"] = True
                log.warning("atdev %s: split-KV attention ENGAGED for %d query rows "
                            "(max_query_len %d, %d segments)", OPT_NAME, q.shape[0], mq,
                            kw["num_par_softmax_segments"])
        return orig(*args, **kw)

    unified_attention._atdev_split = True
    ta.unified_attention = unified_attention
    log.info("atdev %s: registered", OPT_NAME)
