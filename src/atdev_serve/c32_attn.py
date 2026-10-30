"""c=32 varlen decode / MTP-verify full attention as a CUDA kernel (ATDEV_OPT=c32_attn; csrc/c32_attn.cu).

Replaces vLLM 0.31.0 TRITON_ATTN's 2D ``kernel_unified_attention`` on batches whose requests carry <= 32 / G query
rows each (c=32 MTP k=2 verify: 3 rows x 6 q heads = 18 rows per (request, kv head)); per-request query lengths
come from cu_seqlens_q, so varlen verify (DI1) works unchanged. One CTA per (request, kv head) reads each K / V tile
once for all its rows (Triton: 2 programs of BLOCK_Q 2 per (request, kv head), each reading all of K / V); requests
run longest-first (rank computed on the GPU). Everything is read from the GPU metadata -> CUDA-graph capturable.

Modes:
  ATDEV_C32_ATTN_SPLIT=0 (default) EXACT: bit-identical to TRITON_ATTN 2D (same per-row fp32 op sequence: 32-key
      tiles in order, S = fma(scale, mma chain, 0), ex2.approx softmax, L0 = 1, Triton's row-sum order, acc =
      mma(P, V, acc * alpha), out = div.full(acc, L)); tests/test_c32_attn.py checks torch.equal.
  ATDEV_C32_ATTN_SPLIT=N > 1: split-KV over <= N pieces of >= C32_TPS_MIN tiles + in-kernel merge by the last
      arriving CTA (rounding-level, accuracy-gated).
  ATDEV_C32_ATTN_ST: cp.async ring stages (2: 2 CTAs / SM, default; 3: 1 CTA / SM).
  ATDEV_C32_ATTN_DEFS: extra -D defines of csrc/c32_attn.cu (A/B only).

  ATDEV_C32_ATTN_DECODE=1: also replace the 3D split-KV single-row decode launches (drafter decode passes).

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
SPLIT = int(os.environ.get("ATDEV_C32_ATTN_SPLIT", "0"))
STAGES = int(os.environ.get("ATDEV_C32_ATTN_ST", "2"))
MAXR_SPLIT = int(os.environ.get("ATDEV_C32_ATTN_MAXR", "64"))  # split scratch sized for this many requests
# C32n: also take TRITON_ATTN's 3D (split-KV) single-row decode launches (MTP drafter decode passes); NOT
# bit-identical to the 3D path (exact = the 2D op sequence) -> drafter-only rounding, accuracy-gated
DECODE = os.environ.get("ATDEV_C32_ATTN_DECODE", "0") == "1"
_EXT = {}
_BUF: dict = {}
_EMPTY: dict = {}
_STATS = {"run": 0, "logged": set(), "skip_logged": set()}


def build(defines: str = ""):
    from torch.utils.cpp_extension import load
    defs = [d for d in defines.split(",") if d]
    key = tuple(defs)
    if key not in _EXT:
        src = os.path.join(os.path.dirname(__file__), "csrc", "c32_attn.cu")
        name = "atdev_c32_attn" + "".join("_" + d.replace("=", "") for d in defs)
        _EXT[key] = load(name=name, sources=[src], verbose=False,
                         extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80", "-std=c++17",
                                            "--expt-relaxed-constexpr"] + [f"-D{d}" for d in defs])
    return _EXT[key]


def ext():
    return build(os.environ.get("ATDEV_C32_ATTN_DEFS", ""))


def _empty(dev):
    e = _EMPTY.get(dev)
    if e is None:
        e = _EMPTY[dev] = torch.empty(0, device=dev)
    return e


def split_bufs(dev, kvh: int, nsplit: int, maxr: int):
    key = (dev, kvh, nsplit, maxr)
    b = _BUF.get(key)
    if b is None:
        b = _BUF[key] = (torch.empty(maxr * kvh * nsplit * 32 * HS, dtype=torch.float32, device=dev),
                         torch.empty(maxr * kvh * nsplit * 32 * 2, dtype=torch.float32, device=dev),
                         torch.zeros(maxr * kvh, dtype=torch.int32, device=dev))
    return b


def c32_attention(q, kc, vc, block_table, seqused_k, cu_seqlens_q, out, scale, max_ql, nsplit=None, stages=None,
                  defines=None):
    """q [N, QH, 256] bf16, kc / vc [blocks, page, kvh, 256] views, block_table [R, P] int32, seqused_k [R] int32,
    cu_seqlens_q [>= R + 1] int32, out [N, QH, 256]; max_ql = upper bound of the per-request query length."""
    ns = SPLIT if nsplit is None else int(nsplit)
    st = STAGES if stages is None else int(stages)
    e = ext() if defines is None else build(defines)
    dev = q.device
    if ns > 1:
        R = seqused_k.shape[0]
        maxr = max(MAXR_SPLIT, R)
        if (dev, kc.shape[2], ns, maxr) not in _BUF and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("c32_attn: split scratch must be allocated before graph capture")
        op, ml, cnt = split_bufs(dev, kc.shape[2], ns, maxr)
    else:
        op = ml = cnt = _empty(dev)
    e.c32_attn_fwd(q, kc, vc, block_table, seqused_k, cu_seqlens_q, out, float(scale), int(max_ql), ns, st,
                   op, ml, cnt)
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
    if use_3d and not (DECODE and msq == 1):
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
        if SPLIT > 1:
            R = kw["seqused_k"].shape[0]
            key = (q.device, kw["k"].shape[2], SPLIT, max(MAXR_SPLIT, R))
            if key not in _BUF and torch.cuda.is_current_stream_capturing():
                if "nobuf" not in _STATS["skip_logged"]:
                    _STATS["skip_logged"].add("nobuf")
                    log.warning("atdev %s: split scratch for %d requests not allocated before capture -> TRITON_ATTN",
                                OPT_NAME, R)
                return inner(*args, **kw)
        c32_attention(q, kw["k"], kw["v"], kw["block_table"], kw["seqused_k"], kw["cu_seqlens_q"], kw["out"],
                      kw["softmax_scale"], int(kw["max_seqlen_q"]))
        _STATS["run"] += 1
        sig = (kw["seqused_k"].shape[0], int(kw["max_seqlen_q"]))
        if len(_STATS["logged"]) < 8 and sig not in _STATS["logged"]:
            _STATS["logged"].add(sig)
            log.warning("atdev %s: CUDA attention ENGAGED (%d reqs, max q %d, %d tokens, split %d, stages %d)",
                        OPT_NAME, sig[0], sig[1], q.shape[0], SPLIT, STAGES)
        return None

    unified_attention._atdev_c32_attn = True
    ta.unified_attention = unified_attention

    if SPLIT > 1:  # scratch before capture (KV-cache sizing sees it)
        orig_init = ta.TritonAttentionImpl.__init__

        @functools.wraps(orig_init)
        def __init__(self, num_heads, head_size, scale, num_kv_heads, *a, **k):
            orig_init(self, num_heads, head_size, scale, num_kv_heads, *a, **k)
            try:
                if head_size == HS and torch.cuda.is_available():
                    split_bufs(torch.device("cuda", torch.cuda.current_device()), num_kv_heads, SPLIT, MAXR_SPLIT)
            except Exception as e:  # pragma: no cover
                log.warning("atdev %s: scratch prealloc failed (%s)", OPT_NAME, e)

        ta.TritonAttentionImpl.__init__ = __init__
    from atdev_serve import compile_cache
    compile_cache._SALTS[OPT_NAME] = compile_cache._SALTS.get(OPT_NAME, "") + f"+s{SPLIT}st{STAGES}"
    log.warning("atdev %s: registered (split %d, stages %d)", OPT_NAME, SPLIT, STAGES)
