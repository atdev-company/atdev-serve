"""GDN out_proj W4A4 quant folded into the single-state GDN verify kernel's epilogue (ATDEV_OPT=gdn_a4_fold;
c=32 stack with gdn_single_state + ATDEV_GDN_IMPL=cuda). Bit-identical.

Stock per GDN layer: the verify kernel writes the gated-RMSNorm output core [M, 48, 128] bf16, then
out_proj (marlin_plus W4A4) runs marlin_plus_a4 quant_a4 on it before gemm_a4. One s4 group = 128
columns = exactly one value head, which one warp of the CUDA verify kernel already holds for its token,
so gdn_ss_a4 writes the s4 fragments + group scales of its bf16 output rows itself (same formula /
layout as quant_a4) and out_proj skips quant_a4:

* QwenGatedDeltaNetAttention._forward_core_fused_norm_packed: remembers (layer, core) for the call;
* gdn_single_state.gdn_single_state_verify_cuda: if out_proj runs the a4 path for M = core rows,
  calls gdn_ss_a4 (marlin_plus' shared a4 buffers, out_proj's input_global_scale, the bucket's
  tmb) and tags the buffers with (core ptr, M, K, tmb);
* marlin_plus.mod_a4(): proxy whose quant_a4 skips the quantisation when the tag matches its
  input (the tag is consumed by the first quant_a4 call either way).

The GDN core output is allocated uninitialised (fused_quant); it is zero-filled here unless the folded
epilogue covers every row out_proj reads. Rows the GDN kernel does not write (CUDA-graph padding tokens
past num_actual_tokens) keep stale fragments instead of the fragments of the zero rows; they only
reach the padded output rows of out_proj (row-independent GEMM), never a real token.
"""
from __future__ import annotations

import logging
import os

import torch

OPT_NAME = "gdn_a4_fold"
SUPPORTED_VLLM_VERSION = "0.31.0"
log = logging.getLogger("atdev_serve")

_REQ: list = [None]  # (layer, full core tensor) of the running fused-norm GDN core call
_STATS = {"fold": 0, "skip": 0}


def _plan(layer, core):
    """(a4_pk, a4_s, gs, tmb, tag) if out_proj (marlin_plus W4A8 kernel) takes the W4A4 path
    for M = core rows, else None -- the same decision as marlin_plus.mp_linear."""
    from atdev_serve import marlin_plus as mp

    op = getattr(layer, "out_proj", None)
    k = getattr(getattr(op, "scheme", None), "kernel", None)
    if k is None or type(k) is not mp.kernel_class() or getattr(k, "_atdev_stock", True):
        return None
    if k.config.act_type != torch.int8 or getattr(op, "bias", None) is not None:
        return None
    gs = getattr(op, "input_global_scale", None)
    if gs is None or core.dim() != 3 or not core.is_contiguous():
        return None
    kk, n = k.config.partition_weight_shape
    m = core.shape[0]
    if m <= 0 or kk != core.shape[1] * core.shape[2]:
        return None
    ent = mp.lookup(n, kk, m, True)
    if ent is None or not ent.get("a4"):
        return None
    tmb = ent["cfg"][4]
    if m > 16 * tmb or kk > mp.A4_MAX_K or tmb > mp.A4_MAX_TMB:
        return None
    b = mp.bufs(core.device)
    gsv = gs.reshape(1).float()
    return b["a4_pk"], b["a4_s"], gsv, tmb, (core.data_ptr(), m, kk, tmb, b["a4_pk"].data_ptr())


def will_fold(layer, core) -> bool:
    """True iff this call takes the folded gdn_ss4 path (single-state CUDA verify, k <= 3, native bf16
    state, out_proj on W4A4 for M = core rows): then out_proj reads only the s4 fragments, never core,
    so core may stay uninitialised. Mirrors the dispatch in
    QwenGatedDeltaNetAttention._forward_core_fused_norm, SingleStateGDNAttention and gdn_ss_impl."""
    from vllm.forward_context import get_forward_context

    md = get_forward_context().attn_metadata
    md = md.get(layer.prefix) if isinstance(md, dict) else None
    if md is None or getattr(md, "num_prefills", 1) != 0:
        return False
    if getattr(layer, "atdev_impl", None) != "cuda":
        return False
    try:
        if not layer._can_use_fused_gdn_mtp_decode(md):
            return False
    except Exception:  # noqa: BLE001
        return False
    si = md.spec_state_indices_tensor
    if layer.kv_cache[1].dtype != torch.bfloat16 or si.size(1) > 4 or layer.num_v_heads // layer.tp_size > 64:
        return False
    return _plan(layer, core) is not None


def verify_cuda_a4(orig, **kw):
    req = _REQ[0]
    plan = _plan(*req) if req is not None else None
    if plan is None:
        return orig(**kw)
    from atdev_serve import gdn_single_state as gss
    from atdev_serve import marlin_plus as mp

    ext = gss.cuda_ext()
    state, si, out, og = kw["state"], kw["state_indices"], kw["out"], kw["output_gate"]
    slots, hv, vdim, kdim = state.shape
    assert state.dtype == torch.bfloat16 and vdim == 128 and kdim == 128
    assert state.stride()[1:] == (vdim * kdim, kdim, 1)
    assert si.size(1) <= 8, "k <= 7"
    assert og.stride(-1) == 1 and og.stride(-2) == vdim
    lay = gss.log_layout(kw["num_k_heads"], hv, kdim, vdim, 32, si.size(1) - 1)
    if lay["total"] > gss.slot_capacity_f32(state):
        raise ValueError("gdn_single_state: log does not fit in one state slot")
    pk, s, gs, tmb, tag = plan
    args = (kw["mixed_qkv"], kw["a"], kw["b"], kw["A_log"], kw["dt_bias"], state, si, kw["cu_seqlens"],
            kw["num_accepted"], og, kw["norm_weight"], out, kw["num_k_heads"], kw["num_requests"],
            float(kw["scale"]), float(kw["norm_eps"]), bool(kw["sigmoid_gate"]),
            lay["log_base"], lay["buf"], lay["v_off"], lay["d_off"], lay["b_off"], lay["nv"], pk, s, gs, tmb)
    ok = ext.gdn_ss_a4(*args)
    if ok:
        mp.bufs(out.device)["a4_for"] = tag
        _STATS["fold"] += 1
        if _STATS["fold"] == 1:
            log.warning("atdev %s: out_proj s4 fragments from the GDN epilogue (M=%d, tmb=%d)",
                        OPT_NAME, tag[1], tmb)
    return None


class _A4Proxy:
    """marlin_plus_a4 extension proxy: quant_a4 is skipped for the tagged (already folded) input."""

    def __init__(self, mod):
        self._mod = mod

    def __getattr__(self, name):
        return getattr(self._mod, name)

    def quant_a4(self, x, gs, a_pk, a_s, tmb):
        from atdev_serve import marlin_plus as mp

        tag = mp.bufs(x.device).pop("a4_for", None)
        if tag is not None and tag == (x.data_ptr(), x.shape[0], x.shape[1], tmb, a_pk.data_ptr()):
            _STATS["skip"] += 1
            return None
        return self._mod.quant_a4(x, gs, a_pk, a_s, tmb)


def register() -> None:
    import vllm
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn_mod

    from atdev_serve import gdn_single_state as gss
    from atdev_serve import marlin_plus as mp
    from atdev_serve.register import enabled

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written for vLLM {SUPPORTED_VLLM_VERSION}")
    need = {"gdn_single_state", "marlin_plus"}
    if not need <= enabled() or os.environ.get("ATDEV_GDN_IMPL") != "cuda":
        raise RuntimeError(f"atdev {OPT_NAME}: needs ATDEV_OPT gdn_single_state,marlin_plus and "
                           "ATDEV_GDN_IMPL=cuda")
    if getattr(gss.gdn_single_state_verify_cuda, "_atdev_a4", False):
        return
    o_verify = gss.gdn_single_state_verify_cuda

    def gdn_single_state_verify_cuda(**kw):
        return verify_cuda_a4(o_verify, **kw)

    gdn_single_state_verify_cuda._atdev_a4 = True
    gss.gdn_single_state_verify_cuda = gdn_single_state_verify_cuda

    cls = gdn_mod.QwenGatedDeltaNetAttention
    o_packed = cls._forward_core_fused_norm_packed

    def _forward_core_fused_norm_packed(self, mixed_qkvz, ba, core_attn_out):
        if not will_fold(self, core_attn_out):
            core_attn_out.zero_()
        _REQ[0] = (self, core_attn_out)
        try:
            return o_packed(self, mixed_qkvz, ba, core_attn_out)
        finally:
            _REQ[0] = None

    cls._forward_core_fused_norm_packed = _forward_core_fused_norm_packed

    o_mod_a4 = mp.mod_a4
    proxy: list = [None]

    def mod_a4():
        if proxy[0] is None:
            proxy[0] = _A4Proxy(o_mod_a4())
        return proxy[0]

    mp.mod_a4 = mod_a4
    log.warning("atdev %s: registered (GDN epilogue -> out_proj s4 fragments)", OPT_NAME)
