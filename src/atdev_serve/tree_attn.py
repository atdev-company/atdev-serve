"""c=1 tree-verify full attention as a dedicated CUDA split-KV kernel (ATDEV_OPT=tree_attn; csrc/tree_attn.cu).

Replaces, on the 16 target full-attention layers of a c=1 DFlash2 tree verify, vLLM's TRITON_ATTN 3D path
(kernel_unified_attention 24.8 us + reduce_segments 5.5 us per layer, results/prof-DG-stack) by

  split   : grid (kv heads x hspl, nsplit); a CTA holds G / hspl q heads (default hspl 2 -> 3 heads = 48 query rows,
            sharing one K/V read), warp pair per head (Q in registers, keys split for QK^T, dims split for PV),
            32-key tiles in a 3-stage cp.async ring, mma.sync bf16 + register online softmax, fp32 partials (O, m, l);
            the tile range per CTA comes from the GPU seq len (graph-capturable, idle splits exit)
  combine : CTA per (token, q head): merges the active splits, rounds to bf16; optionally the output gate and the
            tree K/V cache write (below).

Engaged inside vLLM's ``unified_attention`` (the split-point op) when the call is dflash_tree's tree verify
(qq_bias, one request, T <= 16 query rows, bf16 cache, no sinks / alibi / softcap / window / fp8). Anything else
runs the original launcher. Microbench (scripts/bench_tree_attn.py, 16 layers, distinct caches, 1k ctx):
attention 585 -> 324 us/step; with gate + KV folds the whole block 654 -> 340 us/step.

Gate fold (ATDEV_TREE_ATTN_GATE=1, needs c1_fuse): c1_fuse's attention layer calls ``vllm::atdev_attn_gated``
(registered as a piecewise splitting op like unified_attention_with_output) with the gate tensor; the gate is
applied in the combine kernel (tree path) or by one Triton kernel after the stock attention (any other batch).
Removes the Inductor ``mul_sigmoid`` launch (16 x 2 us).

KV fuse (ATDEV_TREE_ATTN_KV=1, with ATDEV_TREE_ATTN_GATE=1): atdev_attn_gated also owns the KV-cache update; on the
tree path the split kernel reads the 16 tree K/V rows straight from k / v and the combine kernel stores them to their
cache slots (removes reshape_and_cache_kernel_flash, 16 x 2.2 us); any other batch runs impl.do_kv_cache_update
first. Cache contents and attention inputs are the same values -> exact w.r.t. tree_attn without it.

Numerics: NOT bit-identical to TRITON_ATTN -- a different (equally valid) fp32 softmax / split order, exp2 via
ex2.approx; output rounding to bf16 as before (tests/test_tree_attn.py: max error vs an fp32 reference within
1.07x of Triton 3D's at every context). Gate with GSM8K like attn_split_verify. The gate fold keeps the two bf16
roundings of the unfused path (bf16 attention output, then bf16 of o * sigmoid(gate)); its sigmoid (__expf) differs
from Inductor's by 1 bf16 ulp on ~1 in 5000 elements.

EXACT mode (ATDEV_TREE_ATTN_EXACT=1|2, default off): bit-identical to the path it replaces (TRITON_ATTN 3D of
attn_split_verify: 16 segments of cdiv(L, 256) 16-key tiles, per-row tile ranges of the Triton q block, the compiled
kernel's fma / ex2.approx / butterfly-sum order, reduce_segments' per-lane fma chains and div.full, and Inductor's
mul_sigmoid for the gate fold); tests/test_tree_attn_exact.py checks torch.equal at ctx 0..8176. 1 = 3-stage KV ring
(default hspl 1: 64 CTAs), 2 = 2-stage ring (2 CTAs / SM, default hspl 3: 192 CTAs). Microbench (16 layers, gate +
KV fused, 1k ctx): stock 654.8 us, approximate tree_attn 340.5, exact 2 / hspl 3 370.0 (8k: 2938.8 / 1043.6 / 1132.1).

Env: ATDEV_TREE_ATTN_EXACT (0), ATDEV_TREE_ATTN_HSPL (2; exact 1 -> 1, exact 2 -> 3), ATDEV_TREE_ATTN_NSPLIT (default SMs // (kv heads * hspl)), ATDEV_TREE_ATTN_GATE (0),
ATDEV_TREE_ATTN_KV (0), ATDEV_TREE_ATTN_DEFS (extra -D defines of csrc/tree_attn.cu, A/B only).
"""
from __future__ import annotations

import functools
import logging
import os

import torch

try:
    from vllm.utils.torch_utils import LayerNameType
except Exception:  # pragma: no cover
    LayerNameType = str

OPT_NAME = "tree_attn"
SUPPORTED_VLLM_VERSION = "0.31.0"
log = logging.getLogger("atdev_serve")
HS = 256
_EXT = None
_BUF: dict = {}
_STATS = {"run": 0, "logged": False, "skip_logged": set()}
GATE_FOLD = os.environ.get("ATDEV_TREE_ATTN_GATE", "0") == "1"
# fused KV-cache write (needs the gate-fold op path): the tree K/V go to the cache from tree_attn's combine kernel
KV_FUSE = GATE_FOLD and os.environ.get("ATDEV_TREE_ATTN_KV", "0") == "1"
# bit-identical mode: TRITON_ATTN 3D's segments / tiles / reduction order / roundings (tree_attn.cu "EXACT mode")
# 1: 3-stage KV ring, 2: 2-stage ring (2 CTAs / SM; needs G / hspl <= 4); same results
EXACT = int(os.environ.get("ATDEV_TREE_ATTN_EXACT", "0"))
NSEG = 16  # vllm.v1.attention.backends.triton_attn.NUM_PAR_SOFTMAX_SEGMENTS (checked per call)


def build(defines: str = ""):
    """csrc/tree_attn.cu with comma-separated -D defines (A/B variants get their own extension name)."""
    from torch.utils.cpp_extension import load
    src = os.path.join(os.path.dirname(__file__), "csrc", "tree_attn.cu")
    defs = [d for d in defines.split(",") if d]
    name = "atdev_tree_attn" + "".join("_" + d.replace("=", "") for d in defs)
    return load(name=name, sources=[src], verbose=False,
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80", "-std=c++17",
                                   "--expt-relaxed-constexpr"] + [f"-D{d}" for d in defs])


def ext():
    global _EXT
    if _EXT is None:
        _EXT = build(os.environ.get("ATDEV_TREE_ATTN_DEFS", ""))
    return _EXT


def nsplit_for(dev: torch.device, kvh: int, hspl: int = 1) -> int:
    """KV splits per (kv head, head group): the grid kvh * hspl * nsplit must fit one wave (the fused merge needs
    every CTA co-resident; 1 CTA / SM)."""
    if EXACT:
        return NSEG
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    cap = max(1, min(64, sms // (kvh * hspl)))
    env = int(os.environ.get("ATDEV_TREE_ATTN_NSPLIT", "0"))
    return max(1, min(cap, env)) if env else cap


def bufs(dev: torch.device, qh: int, kvh: int, nsplit: int):
    """fp32 partial scratch + the fused-merge arrival counters (zeroed once, self-resetting); shared by all layers
    (launches are stream-ordered). Must exist before capture."""
    key = (dev, qh, kvh, nsplit)
    b = _BUF.get(key)
    if b is None:
        g = qh // kvh
        b = _BUF[key] = (torch.empty(kvh * nsplit * 16 * g * HS, dtype=torch.float32, device=dev),
                         torch.empty(kvh * nsplit * 16 * g * 2, dtype=torch.float32, device=dev),
                         torch.zeros(2 * kvh * g, dtype=torch.int32, device=dev))
    return b


def hspl_for(g: int) -> int:
    """CTAs per kv head along its q heads (default 2: 3 heads / 48 query rows per CTA for Qwen3.8; KS-c1 sweep:
    1k ctx 20.3 us vs 22.4 (hspl 1) / 22.0 (hspl 3) per layer)."""
    # exact mode (16 segments -> grid kvh x hspl x 16): 1 -> hspl 1 (64 CTAs, 1 wave), 2 -> hspl 3 (192 CTAs,
    # 2 / SM); TA-exact microbench (scripts/bench_tree_attn.py --exact): 3-stage hspl 2 needs 2 waves above 1k ctx
    dflt = {1: "1", 2: "3"}.get(EXACT, "2")
    env = int(os.environ.get("ATDEV_TREE_ATTN_HSPL", dflt))
    return env if env >= 1 and g % env == 0 else 1


def tree_attention(q, kc, vc, block_table_row, seqlen, qq, out, scale, gate=None, nsplit=None, kv_new=None,
                   hspl=None, exact=None):
    """q [T, QH, 256] bf16, kc / vc [blocks, page, kvh, 256] views (contiguous head dim), block_table_row [>= pages]
    int32, seqlen [1] int32 (incl. the T tree tokens), qq [T, T] fp32 additive bias or None, out [T, QH, 256].
    kv_new = (k [T, kvh, 256], v [T, kvh, 256], slot_mapping [>= T] int64): the tree K/V are NOT yet in the cache;
    the kernels read them from k / v and write them to the cache slots (fused reshape_and_cache)."""
    kvh = kc.shape[2]
    exact = EXACT if exact is None else int(exact)
    hs = hspl or hspl_for(q.shape[1] // kvh)
    ns = NSEG if exact else (nsplit or nsplit_for(q.device, kvh, hs))
    if (q.device, q.shape[1], kvh, ns) not in _BUF and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("tree_attn: scratch must be allocated before graph capture")
    op, ml, sync = bufs(q.device, q.shape[1], kvh, ns)
    empty = q.new_empty(0)
    kn, vn, slot = kv_new if kv_new is not None else (empty, empty, empty)
    ext().tree_attn_fwd(q, kc, vc, block_table_row, seqlen, empty if qq is None else qq, op, ml, out,
                        empty if gate is None else gate, ns, float(scale), kn, vn, slot, hs, sync, exact)
    return out


# ------------------------------------------------------------------------------------------------ engine hook
_GATE_CTX: dict = {}  # set by atdev_attn_gated around impl.forward: the gate of the current call


def _eligible(kw) -> str | None:
    q, k = kw.get("q"), kw.get("k")
    if q is None or k is None or kw.get("qq_bias") is None:
        return "no tree bias"
    if q.dim() != 3 or q.shape[2] != HS or q.shape[0] > 16 or q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        return f"shape/dtype q{tuple(q.shape)} {q.dtype} k {k.dtype}"
    if kw["cu_seqlens_q"].shape[0] != 2 or kw["seqused_k"].shape[0] < 1:
        return "batch != 1 request"
    if q.shape[1] % k.shape[2] or (q.shape[1] // k.shape[2]) not in (1, 2, 4, 6, 8):
        return "gqa group"
    if any(kw.get(n) is not None for n in ("alibi_slopes", "sinks", "output_scale", "q_descale", "mm_prefix_range",
                                           "k_scale_cache", "rswa_prefix_lens")):
        return "unsupported feature"
    if kw.get("softcap") or (kw.get("window_size") not in (None, (-1, -1), [-1, -1])):
        return "softcap/window"
    if not kw.get("causal", True):
        return "non-causal"
    if EXACT and (kw.get("num_par_softmax_segments") != NSEG or kw.get("seq_threshold_3D") is None):
        return "exact mode needs the 16-segment 3D path"
    if k.stride(3) != 1 or kw["v"].stride(3) != 1 or q.stride(2) != 1 or kw["out"].stride(2) != 1:
        return "strides"
    if q.shape[0] != kw["out"].shape[0]:
        return "out rows"
    return None


def register() -> None:
    import vllm
    from vllm.v1.attention.backends import triton_attn as ta

    from atdev_serve.register import enabled

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written against vLLM {SUPPORTED_VLLM_VERSION}, found {vllm.__version__}")
    if "dflash_tree" not in enabled():
        raise RuntimeError(f"atdev {OPT_NAME}: needs dflash_tree (the tree verify qq_bias)")
    if getattr(ta.unified_attention, "_atdev_tree_attn", False):
        return
    ext()  # build / load before any capture
    from atdev_serve import dflash_tree as dt
    inner = ta.unified_attention

    def run_inner(*args, **kw):
        kvc = _GATE_CTX.get("kv")
        if kvc is not None:
            _kv_update(kvc)  # stock order: cache update before the attention reads it
        return inner(*args, **kw)

    @functools.wraps(inner)
    def unified_attention(*args, **kw):
        if args or not dt.S.attn_on:
            return run_inner(*args, **kw)
        kw2 = dict(kw)
        if kw2.get("qq_bias") is None:
            kw2["qq_bias"] = dt.S.qq
        why = _eligible(kw2)
        if why is not None:
            if why not in _STATS["skip_logged"]:
                _STATS["skip_logged"].add(why)
                log.warning("atdev %s: tree attention NOT engaged (%s) -> TRITON_ATTN", OPT_NAME, why)
            return run_inner(*args, **kw)
        q = kw2["q"]
        dev = q.device
        kvh_ = kw2["k"].shape[2]
        ns = nsplit_for(dev, kvh_, hspl_for(q.shape[1] // kvh_))
        if (dev, q.shape[1], kw2["k"].shape[2], ns) not in _BUF:
            if torch.cuda.is_current_stream_capturing():
                if "nobuf" not in _STATS["skip_logged"]:
                    _STATS["skip_logged"].add("nobuf")
                    log.warning("atdev %s: scratch not allocated before capture -> TRITON_ATTN", OPT_NAME)
                return run_inner(*args, **kw)
            bufs(dev, q.shape[1], kw2["k"].shape[2], ns)
        gate = _GATE_CTX.get("gate")
        if gate is not None:
            _GATE_CTX["used"] = True
        kvc = _GATE_CTX.get("kv")
        kv_new = None
        if kvc is not None and not kvc["done"]:
            T = q.shape[0]
            kn, vn, slot = kvc["key"][:T], kvc["value"][:T], kvc["slot"]
            if (slot is not None and slot.dtype == torch.int64 and slot.dim() == 1 and slot.stride(0) == 1
                    and slot.shape[0] >= T and kn.stride(-1) == 1 and vn.stride(-1) == 1 and kn.dim() == 3
                    and all((x.stride(0) * 2) % 16 == 0 and (x.stride(1) * 2) % 16 == 0 and x.data_ptr() % 16 == 0
                            for x in (kn, vn))):
                kv_new = (kn, vn, slot)
                kvc["done"] = True
            else:
                _kv_update(kvc)
        tree_attention(q, kw2["k"], kw2["v"], kw2["block_table"][0], kw2["seqused_k"], kw2["qq_bias"], kw2["out"],
                       kw2["softmax_scale"], gate=gate, nsplit=ns, kv_new=kv_new)
        _STATS["run"] += 1
        if not _STATS["logged"]:
            _STATS["logged"] = True
            log.warning("atdev %s: CUDA tree attention ENGAGED (%d q rows, %d/%d heads, nsplit %d, gate fold %s)",
                        OPT_NAME, q.shape[0], q.shape[1], kw2["k"].shape[2], ns, gate is not None)
        return None

    unified_attention._atdev_tree_attn = True
    ta.unified_attention = unified_attention

    # scratch before capture: allocate at the first TritonAttentionImpl construction's device / shape
    orig_init = ta.TritonAttentionImpl.__init__

    @functools.wraps(orig_init)
    def __init__(self, num_heads, head_size, scale, num_kv_heads, *a, **k):
        orig_init(self, num_heads, head_size, scale, num_kv_heads, *a, **k)
        try:
            if head_size == HS and torch.cuda.is_available():
                dev = torch.device("cuda", torch.cuda.current_device())
                bufs(dev, num_heads, num_kv_heads,
                     nsplit_for(dev, num_kv_heads, hspl_for(num_heads // num_kv_heads)))
        except Exception as e:  # pragma: no cover
            log.warning("atdev %s: scratch prealloc failed (%s)", OPT_NAME, e)

    ta.TritonAttentionImpl.__init__ = __init__
    if GATE_FOLD:
        _register_gated_op()
    log.warning("atdev %s: registered (gate fold %s)", OPT_NAME, GATE_FOLD)


# ------------------------------------------------------------------------------------------------ gate fold op
def atdev_attn_gated(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, gate: torch.Tensor,
                     output: torch.Tensor, layer_name: LayerNameType, kv_cache_dummy_dep: torch.Tensor | None = None) -> None:
    """unified_attention_with_output + the attention output gate: output = bf16(o * sigmoid(gate)) (fp32 math)."""
    from vllm.model_executor.layers.attention.attention import (get_attention_context,
                                                                unified_attention_with_output)
    from vllm.utils.torch_utils import _resolve_layer_name
    _GATE_CTX["gate"] = gate
    _GATE_CTX["used"] = False
    kvc = None
    if KV_FUSE and kv_cache_dummy_dep is None:  # gated_attention skipped unified_kv_cache_update: we own it
        _, layer, kv_cache, slot = get_attention_context(_resolve_layer_name(layer_name))
        kvc = _GATE_CTX["kv"] = dict(key=key, value=value, slot=slot, layer=layer, kv_cache=kv_cache, done=False)
    try:
        unified_attention_with_output(query, key, value, output, layer_name, kv_cache_dummy_dep=kv_cache_dummy_dep)
        used = _GATE_CTX["used"]
        if kvc is not None:
            _kv_update(kvc)  # attention never ran (profile run): keep the cache write
    finally:
        _GATE_CTX.clear()
    if not used:
        _apply_gate(output.view(output.shape[0], -1), gate)


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _gate_kernel(o_ptr, so, g_ptr, sg, N, BLOCK: tl.constexpr):
        # o = bf16(o * sigmoid(gate)) in fp32 (= Inductor's fused mul_sigmoid of the unfused path)
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = cols < N
        o = tl.load(o_ptr + row * so + cols, mask=m, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + row * sg + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(o_ptr + row * so + cols, (o * tl.sigmoid(g)).to(o_ptr.dtype.element_ty), mask=m)
except ImportError:  # pragma: no cover
    triton = None


def _apply_gate(o2: torch.Tensor, gate: torch.Tensor) -> None:
    if triton is None or o2.stride(-1) != 1 or gate.stride(-1) != 1:
        o2.copy_((o2.float() * torch.sigmoid(gate.float())).to(o2.dtype))
        return
    n = o2.shape[1]
    _gate_kernel[(o2.shape[0], triton.cdiv(n, 1024))](o2, o2.stride(0), gate, gate.stride(0), n, BLOCK=1024)


def _kv_update(kvc) -> None:
    if kvc["done"]:
        return
    kvc["done"] = True
    if kvc["slot"] is not None:
        layer = kvc["layer"]
        layer.impl.do_kv_cache_update(layer, kvc["key"], kvc["value"], kvc["kv_cache"], kvc["slot"])


def _atdev_attn_gated_fake(query, key, value, gate, output, layer_name, kv_cache_dummy_dep=None) -> None:
    return None


_GATED_REG = False


def _register_gated_op() -> None:
    global _GATED_REG
    if _GATED_REG:
        return
    _GATED_REG = True
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture
    from vllm.config import compilation as cc
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op("atdev_attn_gated", eager_break_during_capture(atdev_attn_gated),
                              mutates_args=["output"], fake_impl=_atdev_attn_gated_fake)
    if "vllm::atdev_attn_gated" not in cc.CompilationConfig._attention_ops:
        cc.CompilationConfig._attention_ops.append("vllm::atdev_attn_gated")
    from atdev_serve import compile_cache
    compile_cache._SALTS[OPT_NAME] = compile_cache._SALTS.get(OPT_NAME, "") + "+gate"


def gated_attention(at, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """c1_fuse: Attention.forward (use_direct_call False path) with the gate passed into the attention op."""
    from vllm.model_executor.layers.attention.attention import _encode_layer_name
    a = at.attn
    n = q.shape[0]
    out = torch.empty((n, a.num_heads * a.head_size_v), dtype=q.dtype, device=q.device)
    qv = q.view(-1, a.num_heads, a.head_size)
    ov = out.view(-1, a.num_heads, a.head_size_v)
    kv_ = k.view(-1, a.num_kv_heads, a.head_size)
    vv = v.view(-1, a.num_kv_heads, a.head_size_v)
    enc = _encode_layer_name(a.layer_name)
    dep = None
    if not a.attn_backend.forward_includes_kv_cache_update and a.kv_sharing_target_layer_name is None \
            and not KV_FUSE:
        dep = torch.ops.vllm.unified_kv_cache_update(kv_, vv, enc)
    torch.ops.vllm.atdev_attn_gated(qv, kv_, vv, gate.reshape(n, -1), ov, enc, kv_cache_dummy_dep=dep)
    return out


def gated_ok(at, gate) -> bool:
    a = getattr(at, "attn", None)
    return (GATE_FOLD and _GATED_REG and gate is not None and a is not None and not a.use_direct_call
            and a.query_quant is None and a.head_size == HS
            and (not KV_FUSE or (not a.attn_backend.forward_includes_kv_cache_update
                                 and a.kv_sharing_target_layer_name is None)))
