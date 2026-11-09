"""PFX-s9: prefill / mixed-prefill attention through FlashAttention-2 instead of TRITON_ATTN's 2D unified kernel.

Finding (measured): on the Qwen3.8-27B full-attention shape (24 q heads, 4 kv heads,
d=256, paged block 416) vLLM's Triton ``kernel_unified_attention`` runs prefill at 31-40 TFLOP/s (10-13 % of the A100
bf16 peak); vllm_flash_attn FA2 runs 104-143 TFLOP/s on the same paged cache (3.1-3.8x faster), outputs equal to
bf16 rounding (mean rel diff 2e-3, a different online-softmax order, NOT bit-identical).

Engages only for batches with ``max_seqlen_q >= ATDEV_PF_FA2_MINQ`` (default 32: cold/delta prefill chunks and mixed
steps); decode / verify batches (q <= 4 per request, including CUDA-graph captures) keep the inner path (TRITON_ATTN or
c32_attn). Anything the FA2 call cannot express (sinks, alibi, softcap, window, qq_bias, mm prefix, quantised KV,
fused output scale) falls through. Opt-in: ATDEV_OPT=...,pf_fa2 or ATDEV_PF_FA2=1.
"""
from __future__ import annotations

import functools
import logging
import os

log = logging.getLogger("atdev_serve")
OPT_NAME = "pf_fa2"
SUPPORTED_VLLM_VERSION = "0.31.0"
MINQ = int(os.environ.get("ATDEV_PF_FA2_MINQ", "32"))
# C1R2: FlashInfer fa2 paged prefill for small single-sequence steps (q <= FI_MAXQ): non-split-KV tiles are badly under-filled
# at q<~300, the vllm_flash_attn paged split-KV path is 4.6-9 ms/layer at 100k ctx where FlashInfer's balanced split-KV needs 0.3-4 ms.
FI = os.environ.get("ATDEV_PF_FA2_FI", "0") == "1"
FI_MAXQ = int(os.environ.get("ATDEV_PF_FA2_FI_MAXQ", "299"))
FI_MINQ = int(os.environ.get("ATDEV_PF_FA2_FI_MINQ", str(MINQ)))
CHECK = int(os.environ.get("ATDEV_PF_FA2_CHECK", "0"))
_FIS = {"w": None, "key": None, "n": 0, "chk": 0}
# IMAFIX: flashinfer plan() builds its work list in ONE pinned host buffer and cudaMemcpyAsync's it into the device int workspace.
# Planning step B before the GPU executed step A's copy (CPU runs ahead: async scheduling, a heavy 2048-token chunk still on the
# GPU) lets A's kernels read B's work list -> wrong tiles / illegal memory access.  Rotate NPIN pinned buffers and make the CPU wait
# (event) before it rewrites a buffer whose copy may still be pending.
PINRING = os.environ.get("ATDEV_PF_FA2_FI_PINRING", "1") == "1"
NPIN = 8


def _fi_run(kw, kl, qn):
    import torch
    import flashinfer
    q, k, v = kw["q"], kw["k"], kw["v"]
    bs = k.shape[1]; nbk = (kl + bs - 1) // bs
    if _FIS["w"] is None:
        _FIS["ws"] = torch.empty(128 << 20, dtype=torch.uint8, device=q.device)
        _FIS["w"] = flashinfer.BatchPrefillWithPagedKVCacheWrapper(_FIS["ws"], "NHD", backend="fa2")
    w = _FIS["w"]
    if _FIS["key"] != (qn, kl):
        if PINRING:
            if "pins" not in _FIS:
                p0 = w._pin_memory_int_workspace_buffer
                _FIS["pins"] = [p0] + [torch.empty(p0.shape, dtype=p0.dtype, device="cpu", pin_memory=True) for _ in range(NPIN - 1)]
                _FIS["evs"] = [None] * NPIN
                _FIS["slot"] = 0
            slot = _FIS["slot"] = (_FIS["slot"] + 1) % NPIN
            ev = _FIS["evs"][slot]
            if ev is not None:
                ev.synchronize()  # the H2D copy that read this buffer last has completed (no-op after NPIN plans)
            w._pin_memory_int_workspace_buffer = _FIS["pins"][slot]
        w.plan(torch.tensor([0, qn], dtype=torch.int32), torch.tensor([0, nbk], dtype=torch.int32),
               torch.arange(nbk, dtype=torch.int32), torch.tensor([kl - (nbk - 1) * bs], dtype=torch.int32),
               q.shape[1], k.shape[2], q.shape[2], bs, causal=True, sm_scale=kw["softmax_scale"],
               q_data_type=q.dtype, kv_data_type=k.dtype)
        _FIS["key"] = (qn, kl)
        if PINRING:
            ev = torch.cuda.Event()
            ev.record()  # after plan(): the copy out of pins[slot] is enqueued before this point
            _FIS["evs"][slot] = ev
    w._paged_kv_indices_buf.copy_(kw["block_table"][0, :nbk])  # plan only needs lengths; real page ids are read at run
    w.run(q, (k, v), out=kw["out"])

_ST = {"run": 0, "skip": set(), "gather": 0, "buf": {}, "cuk": {}}
# C1R2: gather the paged K/V of a single-sequence prefill step into contiguous scratch and call the NON-split FA2 kernel
# (hdim256 -> <256,128,64,8> tile, ~181-193 TFLOP/s vs 113-137 for the forced split-KV <256,64,64,4> paged kernel at 100k ctx).
GATHER = os.environ.get("ATDEV_PF_FA2_GATHER", "0") == "1"
GMINQ = int(os.environ.get("ATDEV_PF_FA2_GATHER_MINQ", "300"))


def _why_not(kw) -> str | None:
    for k in ("alibi_slopes", "sinks", "qq_bias", "mm_prefix_range", "rswa_prefix_lens", "output_scale"):
        if kw.get(k) is not None:
            return k
    if kw.get("softcap"):
        return "softcap"
    ws = kw.get("window_size")
    if ws is not None and tuple(ws) not in ((-1, -1), (-1, 0)):
        return "window"
    if kw.get("use_alibi_sqrt") or kw.get("rswa_window") is not None or kw.get("k_scale_cache") is not None:
        return "extra"
    if int(kw.get("kv_quant_mode", 0) or 0) != 0:  # KVQuantMode is an IntEnum, NONE = 0
        return "kv_quant"
    if not kw.get("causal", True):
        return "noncausal"
    q, k = kw["q"], kw["k"]
    import torch
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or q.shape[-1] > 256 or k.stride(-1) != 1:
        return "dtype/shape"
    return None


def register() -> None:
    import vllm
    from vllm.v1.attention.backends import triton_attn as ta

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written against vLLM {SUPPORTED_VLLM_VERSION}, found {vllm.__version__}")
    if getattr(ta.unified_attention, "_atdev_pf_fa2", False):
        return
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    inner = ta.unified_attention

    @functools.wraps(inner)
    def unified_attention(*args, **kw):
        if args or int(kw["max_seqlen_q"]) < min(MINQ, FI_MINQ if FI else MINQ):
            return inner(*args, **kw)
        why = _why_not(kw)
        if why is not None:
            if why not in _ST["skip"]:
                _ST["skip"].add(why)
                log.warning("atdev %s: NOT engaged (%s) -> inner attention", OPT_NAME, why)
            return inner(*args, **kw)
        if FI and FI_MINQ <= kw["q"].shape[0] <= FI_MAXQ and kw["cu_seqlens_q"].shape[0] == 2 and kw["block_table"].shape[0] == 1:
            kl = int(kw["max_seqlen_k"]); qn = int(kw["q"].shape[0])
            _fi_run(kw, kl, qn)
            _FIS["n"] += 1
            if _FIS["n"] == 1:
                log.warning("atdev %s: FlashInfer small-q prefill ENGAGED (first: q %d, k %d)", OPT_NAME, qn, kl)
            if _FIS["chk"] < CHECK and kl > 1000:
                import torch
                _FIS["chk"] += 1
                ref = torch.empty_like(kw["out"])
                flash_attn_varlen_func(q=kw["q"], k=kw["k"], v=kw["v"], out=ref, max_seqlen_q=int(kw["max_seqlen_q"]),
                    cu_seqlens_q=kw["cu_seqlens_q"], max_seqlen_k=kl, seqused_k=kw["seqused_k"], softmax_scale=kw["softmax_scale"],
                    causal=True, block_table=kw["block_table"], fa_version=2)
                d = (ref.float() - kw["out"].float()).abs().max().item()
                log.warning("atdev %s: CHECK FI q %d k %d maxdiff %.3g (ref absmax %.3g)", OPT_NAME, qn, kl, d, ref.float().abs().max().item())
            return None
        if int(kw["max_seqlen_q"]) < MINQ:
            return inner(*args, **kw)
        if GATHER and kw["q"].shape[0] >= GMINQ and kw["cu_seqlens_q"].shape[0] == 2 and kw["block_table"].shape[0] == 1:
            import torch
            kl = int(kw["max_seqlen_k"]); k, v = kw["k"], kw["v"]; bs = k.shape[1]; nbk = (kl + bs - 1) // bs
            ent = _ST["buf"].get(k.device)
            if ent is None or ent[0].shape[0] < nbk:
                ent = _ST["buf"][k.device] = (torch.empty((nbk + 64,) + tuple(k.shape[1:]), dtype=k.dtype, device=k.device),
                                              torch.empty((nbk + 64,) + tuple(v.shape[1:]), dtype=v.dtype, device=v.device))
            idx = kw["block_table"][0, :nbk]
            kb = torch.index_select(k, 0, idx, out=ent[0][:nbk]).view(-1, k.shape[2], k.shape[3])[:kl]
            vb = torch.index_select(v, 0, idx, out=ent[1][:nbk]).view(-1, v.shape[2], v.shape[3])[:kl]
            cuk = _ST["cuk"].get(kl)
            if cuk is None:
                if len(_ST["cuk"]) > 4096:
                    _ST["cuk"].clear()
                cuk = _ST["cuk"][kl] = torch.tensor([0, kl], dtype=torch.int32, device=k.device)
            flash_attn_varlen_func(
                q=kw["q"], k=kb, v=vb, out=kw["out"], max_seqlen_q=int(kw["max_seqlen_q"]), cu_seqlens_q=kw["cu_seqlens_q"],
                max_seqlen_k=kl, cu_seqlens_k=cuk, softmax_scale=kw["softmax_scale"], causal=True, fa_version=2)
            _ST["gather"] += 1
            if _ST["gather"] == 1:
                log.warning("atdev %s: gather+non-split FA2 ENGAGED (first: q %d, k %d)", OPT_NAME, kw["q"].shape[0], kl)
            return None
        flash_attn_varlen_func(
            q=kw["q"], k=kw["k"], v=kw["v"], out=kw["out"], max_seqlen_q=int(kw["max_seqlen_q"]),
            cu_seqlens_q=kw["cu_seqlens_q"], max_seqlen_k=int(kw["max_seqlen_k"]), seqused_k=kw["seqused_k"],
            softmax_scale=kw["softmax_scale"], causal=True, block_table=kw["block_table"], fa_version=2)
        if _ST["run"] == 0:
            log.warning("atdev %s: FA2 prefill attention ENGAGED (first call: %d reqs, max q %d, %d tokens)", OPT_NAME,
                        kw["seqused_k"].shape[0], int(kw["max_seqlen_q"]), kw["q"].shape[0])
        _ST["run"] += 1
        return None

    unified_attention._atdev_pf_fa2 = True
    ta.unified_attention = unified_attention
    from atdev_serve import compile_cache
    compile_cache._SALTS[OPT_NAME] = compile_cache._SALTS.get(OPT_NAME, "") + f"+minq{MINQ}+g{int(GATHER)}{GMINQ}+fi{int(FI)}{FI_MINQ}-{FI_MAXQ}"
    log.warning("atdev %s: registered (min max_seqlen_q %d)", OPT_NAME, MINQ)
