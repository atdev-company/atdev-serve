"""c=32 varlen decode / MTP-verify full attention as a CUDA kernel (ATDEV_OPT=c32_attn; csrc/c32_attn.cu).

Replaces vLLM 0.31.0 TRITON_ATTN's 2D ``kernel_unified_attention`` on batches whose requests carry <= 32 / G query
rows each (MTP verify: k+1 rows x 6 q heads per (request, kv head)); per-request query lengths come from
cu_seqlens_q, so variable-length verify (di1_varlen) works unchanged. One CTA per (request, kv head) reads each
K / V tile once for all its rows (Triton: several programs per (request, kv head), each reading all of K / V);
requests run longest-first (rank computed on the GPU). Everything is read from the GPU metadata -> CUDA-graph
capturable. Bit-identical to TRITON_ATTN 2D (same per-row fp32 op sequence: 32-key tiles in order,
S = fma(scale, mma chain, 0), ex2.approx softmax, L0 = 1, Triton's row-sum order, acc = mma(P, V, acc * alpha),
out = div.full(acc, L)).

Engaged only where TRITON_ATTN would launch its 2D kernel (max_seqlen_q > 1, or no 3D buffers / too many seqs) for
a bf16 cache, head 256, causal, no sinks / alibi / softcap / window / qq_bias / fp8 / mm-prefix / R-SWA / TD.
"""
from __future__ import annotations

import functools
import logging
import os

import torch

OPT_NAME = "c32_attn"
SUPPORTED_VLLM_VERSION = "0.31.0"
log = logging.getLogger("atdev_serve")
HS = 256
_EXT: list = [None]
_STATS = {"run": 0, "logged": set(), "skip_logged": set()}


def ext():
    if _EXT[0] is None:
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(__file__), "csrc", "c32_attn.cu")
        _EXT[0] = load(name="atdev_c32_attn", sources=[src], verbose=False,
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80", "-std=c++17",
                                          "--expt-relaxed-constexpr"])
    return _EXT[0]


def c32_attention(q, kc, vc, block_table, seqused_k, cu_seqlens_q, out, scale, max_ql):
    """q [N, QH, 256] bf16, kc / vc [blocks, page, kvh, 256] views, block_table [R, P] int32, seqused_k [R] int32,
    cu_seqlens_q [>= R + 1] int32, out [N, QH, 256]; max_ql = upper bound of the per-request query length."""
    ext().c32_attn_fwd(q, kc, vc, block_table, seqused_k, cu_seqlens_q, out, float(scale), int(max_ql))
    return out


# ------------------------------------------------------------------------------------------------ engine hook
def _eligible(kw) -> str | None:
    from vllm.v1.kv_cache_interface import KVQuantMode
    q, k, v, o = kw.get("q"), kw.get("k"), kw.get("v"), kw.get("out")
    if q is None or k is None or v is None or o is None:
        return "args"
    if q.dim() != 3 or q.shape[2] != HS or k.dim() != 4 or k.shape[3] != HS:
        return f"shape q{tuple(q.shape)}"
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16 or o.dtype != torch.bfloat16:
        return "dtype"
    if kw.get("kv_quant_mode", KVQuantMode.NONE) != KVQuantMode.NONE:
        return "kv quant"
    qh, kvh = q.shape[1], k.shape[2]
    if qh % kvh:
        return "gqa"
    G = qh // kvh
    msq = int(kw["max_seqlen_q"])
    if msq < 1 or msq * G > 32:
        return f"max_seqlen_q {msq} x G {G} > 32"
    if kw.get("causal", True) is not True:
        return "non-causal / per-seq causal"
    if any(kw.get(n) is not None for n in ("alibi_slopes", "sinks", "output_scale", "qq_bias", "mm_prefix_range",
                                           "k_scale_cache", "rswa_prefix_lens")):
        return "unsupported feature"
    if kw.get("rswa_window") is not None or kw.get("use_td") or kw.get("mm_prefix_clamp_sliding_window"):
        return "rswa/td"
    if kw.get("softcap") or (tuple(kw.get("window_size") or (-1, -1)) != (-1, -1)):
        return "softcap/window"
    # exactly the cases where TRITON_ATTN launches its 2D kernel (TILE_SIZE_PREFILL = 32)
    from vllm.v1.attention.ops import triton_unified_attention as tua
    use_3d = not (kw.get("seq_threshold_3D") is None or kw.get("num_par_softmax_segments") is None
                  or kw.get("softmax_segm_output") is None or kw.get("softmax_segm_max") is None
                  or kw.get("softmax_segm_expsum") is None or msq > 1
                  or kw["seqused_k"].shape[0] > kw["seq_threshold_3D"] or tua.is_batch_invariant)
    if use_3d:
        return "3D path (single-row decode)"
    if k.stride(3) != 1 or v.stride(3) != 1 or q.stride(2) != 1 or o.stride(2) != 1:
        return "strides"
    bt = kw["block_table"]
    if bt.dtype != torch.int32 or bt.stride(1) != 1 or kw["seqused_k"].dtype != torch.int32 \
            or kw["cu_seqlens_q"].dtype != torch.int32:
        return "metadata dtype"
    if kw["seqused_k"].shape[0] > 256:
        return "too many requests"
    if q.shape[0] != o.shape[0]:
        return "out rows"
    return None


def register() -> None:
    import vllm
    from vllm.v1.attention.backends import triton_attn as ta

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written against vLLM {SUPPORTED_VLLM_VERSION}, found {vllm.__version__}")
    if getattr(ta.unified_attention, "_atdev_c32_attn", False):
        return
    ext()  # build / load before any capture
    inner = ta.unified_attention

    @functools.wraps(inner)
    def unified_attention(*args, **kw):
        if args:
            return inner(*args, **kw)
        why = _eligible(kw)
        if why is not None:
            if why not in _STATS["skip_logged"]:
                _STATS["skip_logged"].add(why)
                log.warning("atdev %s: NOT engaged (%s) -> TRITON_ATTN", OPT_NAME, why)
            return inner(*args, **kw)
        q = kw["q"]
        c32_attention(q, kw["k"], kw["v"], kw["block_table"], kw["seqused_k"], kw["cu_seqlens_q"], kw["out"],
                      kw["softmax_scale"], int(kw["max_seqlen_q"]))
        _STATS["run"] += 1
        sig = (kw["seqused_k"].shape[0], int(kw["max_seqlen_q"]))
        if len(_STATS["logged"]) < 8 and sig not in _STATS["logged"]:
            _STATS["logged"].add(sig)
            log.warning("atdev %s: CUDA attention ENGAGED (%d reqs, max q %d, %d tokens)",
                        OPT_NAME, sig[0], sig[1], q.shape[0])
        return None

    unified_attention._atdev_c32_attn = True
    ta.unified_attention = unified_attention

    from atdev_serve import compile_cache
    compile_cache._SALTS[OPT_NAME] = compile_cache._SALTS.get(OPT_NAME, "") + "+s0st2"
    log.warning("atdev %s: registered", OPT_NAME)
