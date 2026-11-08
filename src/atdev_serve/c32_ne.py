"""C32-NE (ATDEV_C32_NE=1 with ATDEV_OPT ...,fused_quant on the c=32 W4A4 stack, A4_NE=1 build): the
residual add + Gemma RMSNorm + s4 quant producer (_norm_a4_chunk, 128 launches/step) is folded into the
epilogue of the W4A4 GEMM that produces its input (out_proj / o_proj / down_proj, N = 5120), and the
row scale rstd is applied to the fp32 accumulator of the consuming GEMM (qkvz / qkv / gate_up):

  producer epilogue: x = bf16(A W) + residual (fp32); ro = bf16(x); a = x (1 + w) (fp32);
                     s4(a) per (row, 128-group) -> the consumer's A buffers (a4_pk2 / a4_s2),
                     P[row, tile] = sum over the 64-column tile of x^2, rstd = rsqrt(sum P / N + eps)
                     (pair / global tickets, csrc/marlin_plus_a4/mp_a4_ne.h)
  consumer:          y = rstd[row] * (s4(a) W)
  in_proj_ba (GDN):  ba = rstd * (bf16(a) W_ba)

Numerics (not bit-identical, needs the gates): the s4 groups are quantised from a = x (1 + w) instead of
bf16(rstd x (1 + w)) (same codes up to the bf16 rounding and fp32 order), rstd scales the consumer output,
sum x^2 in a different fp32 order. Layer 0's input norm and every M the fused path does not cover
(prefill, buckets with tn != 64 / 128, a non-a4 consumer) run the deployed kernels with rstd = 1.
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("atdev_serve")
H = 5120
_B: dict = {}
_STATS = {"fused": 0, "fallback": 0}


def enabled() -> bool:
    return os.environ.get("ATDEV_C32_NE", "0") == "1"


def _bufs(dev):
    from atdev_serve import marlin_plus as mp
    b = _B.get(dev)
    if b is None:
        tmb = mp.A4_MAX_TMB
        b = _B[dev] = dict(
            S=torch.empty(H * 16 * tmb, dtype=torch.float32, device=dev),
            P=torch.empty(16 * tmb * (H // 16), dtype=torch.float32, device=dev),
            cnt=torch.zeros(H // 64 + 8, dtype=torch.int32, device=dev),
            pk=torch.zeros(H // 64 * tmb * 32 * 4, dtype=torch.int32, device=dev),
            s=torch.zeros(H // 128 * 16 * tmb, dtype=torch.float32, device=dev),
            h=torch.empty(16 * tmb * H, dtype=torch.bfloat16, device=dev),
            rstd=torch.empty(128, dtype=torch.float32, device=dev),
        )
    return b


def _mod(mode):
    from atdev_serve import marlin_plus_a4_build
    return marlin_plus_a4_build.load_ne(mode)


def _ent(n, k, m):
    from atdev_serve import fused_quant as fq
    return fq._a4_ent(n, k, m)


def _plan(n, k, n_next, m):
    """(producer cfg, consumer tmb) of the fused path, else None; consumer-a4 flag."""
    ec = _ent(n_next, n, m)
    ep = _ent(n, k, m)
    if ec is None:
        return None, False
    if ep is None or ep["cfg"][0] not in (64, 128) or n != H:
        return None, True
    return (list(ep["cfg"]), ec["cfg"][4]), True


def _ones(m, dev):
    return torch.ones(max(m, 128), dtype=torch.float32, device=dev)


def _norm_into(x, residual, weight, eps, gs, want_bf16, n_next, has_res=True):
    """deployed norm (rstd folded in) -> s4 into the C32-NE consumer buffers, or the int8 path for a
    non-a4 consumer; returns (xb, xq, xs, ro, rstd = ones)."""
    import triton
    from atdev_serve import fused_quant as fq
    m, n = x.shape
    dev = x.device
    ec = _ent(n_next, n, m)
    if ec is None:
        xb, xq, xs, ro = fq.norm_quant(x, residual if has_res else None, weight, eps, gs, want_bf16, n_next)
        return xb, xq, xs, ro, _ones(m, dev)
    b = _bufs(dev)
    tmb = ec["cfg"][4]
    xb = torch.empty((m, n) if want_bf16 else (0, n), dtype=torch.bfloat16, device=dev)
    xq = torch.empty((m, n), dtype=torch.int8, device=dev)
    xs = torch.empty((m,), dtype=torch.float32, device=dev)
    ro = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    r = residual if has_res else x
    fq._norm_a4_chunk_kernel[(16 * tmb, triton.cdiv(n // 128, 8))](
        x, x.stride(0), r, r.stride(0), weight, gs, xb, xb.stride(0) if want_bf16 else 0,
        ro, ro.stride(0), b["pk"], b["s"], m, n, eps, 16 * tmb, tmb,
        HAS_RES=has_res, WANT_BF16=want_bf16, BLOCK=triton.next_power_of_2(n), NGB=8, num_warps=8)
    return xb, xq, xs, ro, _ones(m, dev)


def ne_prod(xq: torch.Tensor, xs: torch.Tensor, x_bf: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
            gs_self: torch.Tensor, n: int, k: int, residual: torch.Tensor, weight: torch.Tensor, eps: float,
            gs_next: torch.Tensor, n_next: int, want_bf16: bool
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """producer linear (pre-quantised s4/int8 input xq/xs, or bf16 x_bf when x_bf has rows: GDN out_proj)
    + the next linear's input norm. Returns (xb, xq', xs', residual_out, rstd)."""
    from atdev_serve import fused_quant as fq
    from atdev_serve import marlin_plus as mp
    m = residual.shape[0]
    dev = residual.device
    plan, _ = _plan(n, k, n_next, m) if m > 0 else (None, False)
    bf_in = x_bf.shape[0] > 0
    if plan is None:
        _STATS["fallback"] += 1
        if bf_in:
            out = mp.mp_linear(x_bf, w_q, w_s, gs_self, n, k, True)
        else:
            out = fq.fq_linear(xq, xs, w_q, w_s, n, k)
        return _norm_into(out, residual, weight, eps, gs_next, want_bf16, n_next)
    cfg_p, tmb_c = plan
    b = mp.bufs(dev)
    nb = _bufs(dev)
    if bf_in:  # quant_a4 of the GDN core (skipped by the gdn_a4_fold proxy when its epilogue wrote it)
        mp.mod_a4().quant_a4(x_bf, gs_self.reshape(1).float(), b["a4_pk"], b["a4_s"], cfg_p[4])
    xb = torch.empty((m, n) if want_bf16 else (0, n), dtype=torch.bfloat16, device=dev)
    ro = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    rstd = torch.empty(max(16 * tmb_c, 128), dtype=torch.float32, device=dev)
    _mod(1).gemm_a4_ne(b["a4_pk"], b["a4_s"], w_q, w_s, b["locks"], b["c_tmp"], m, n, k, cfg_p, residual, ro,
                  weight, xb if want_bf16 else None, nb["S"], nb["P"], nb["cnt"], nb["pk"], nb["s"],
                  gs_next.reshape(1).float(), rstd, float(eps), tmb_c, nb["h"])
    _STATS["fused"] += 1
    if _STATS["fused"] == 1:
        log.warning("atdev c32_ne: norm folded into the W4A4 epilogue (N=%d K=%d M=%d cfg %s, consumer tmb %d)",
                    n, k, m, cfg_p, tmb_c)
    return (xb, torch.empty((m, n), dtype=torch.int8, device=dev),
            torch.empty((m,), dtype=torch.float32, device=dev), ro, rstd)


GB = os.environ.get("ATDEV_C32_NE_MODE", "gb") == "gb"


def ne_prod_gb(xq: torch.Tensor, xs: torch.Tensor, x_bf: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
               gs_self: torch.Tensor, n: int, k: int, residual: torch.Tensor, weight: torch.Tensor, eps: float,
               gs_next: torch.Tensor, n_next: int, want_bf16: bool
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """GB mode (A4_NE=3): producer linear + grid barrier + the deployed norm of the next linear's input, in
    one kernel; the s4 result lands in marlin_plus' shared a4 buffers (the producer has finished reading its
    own A by the barrier), the consumer is the plain fq_linear. Returns norm_quant's (xb, xq, xs, ro)."""
    from atdev_serve import fused_quant as fq
    from atdev_serve import marlin_plus as mp
    m = residual.shape[0]
    dev = residual.device
    ec = _ent(n_next, n, m) if m > 0 else None
    ep = _ent(n, k, m) if m > 0 else None
    bf_in = x_bf.shape[0] > 0
    if ec is None or ep is None or n != H or ep["cfg"][1] != 128 or m > 16 * ec["cfg"][4]:
        _STATS["fallback"] += 1
        out = mp.mp_linear(x_bf, w_q, w_s, gs_self, n, k, True) if bf_in else fq.fq_linear(xq, xs, w_q, w_s, n, k)
        return fq.norm_quant(out, residual, weight, eps, gs_next, want_bf16, n_next)
    cfg_p, tmb_c = list(ep["cfg"]), ec["cfg"][4]
    b = mp.bufs(dev)
    nb = _bufs(dev)
    if bf_in:
        mp.mod_a4().quant_a4(x_bf, gs_self.reshape(1).float(), b["a4_pk"], b["a4_s"], cfg_p[4])
    xb = torch.empty((m, n) if want_bf16 else (0, n), dtype=torch.bfloat16, device=dev)
    ro = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    _mod(3).gemm_a4_ne(b["a4_pk"], b["a4_s"], w_q, w_s, b["locks"], b["c_tmp"], m, n, k, cfg_p, residual, ro,
                       weight, xb if want_bf16 else None, nb["S"], nb["P"], nb["cnt"], b["a4_pk"], b["a4_s"],
                       gs_next.reshape(1).float(), nb["rstd"], float(eps), tmb_c, nb["h"])
    _STATS["fused"] += 1
    if _STATS["fused"] == 1:
        log.warning("atdev c32_ne(gb): norm in the W4A4 producer after a grid barrier (N=%d K=%d M=%d cfg %s, "
                    "consumer tmb %d)", n, k, m, cfg_p, tmb_c)
    return (xb, torch.empty((m, n), dtype=torch.int8, device=dev),
            torch.empty((m,), dtype=torch.float32, device=dev), ro)


def _ne_prod_gb_fake(xq, xs, x_bf, w_q, w_s, gs_self, n, k, residual, weight, eps, gs_next, n_next, want_bf16):
    return _ne_prod_fake(xq, xs, x_bf, w_q, w_s, gs_self, n, k, residual, weight, eps, gs_next, n_next,
                         want_bf16)[:4]


def _ne_prod_fake(xq, xs, x_bf, w_q, w_s, gs_self, n, k, residual, weight, eps, gs_next, n_next, want_bf16):
    m = residual.shape[0]
    dev = residual.device
    return (torch.empty((m, n) if want_bf16 else (0, n), dtype=torch.bfloat16, device=dev),
            torch.empty((m, n), dtype=torch.int8, device=dev),
            torch.empty((m,), dtype=torch.float32, device=dev),
            torch.empty((m, n), dtype=torch.bfloat16, device=dev),
            torch.empty((max(m, 128),), dtype=torch.float32, device=dev))


def ne_first(x: torch.Tensor, weight: torch.Tensor, eps: float, gs_next: torch.Tensor, n_next: int,
             want_bf16: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """layer 0 input norm (no residual): deployed norm into the C32-NE consumer buffers, rstd = 1."""
    return _norm_into(x, None, weight, eps, gs_next, want_bf16, n_next, has_res=False)


def _ne_first_fake(x, weight, eps, gs_next, n_next, want_bf16):
    m, n = x.shape
    return _ne_prod_fake(x, x, x, x, x, x, n, n, x, weight, eps, gs_next, n_next, want_bf16)


def ne_cons(xq: torch.Tensor, xs: torch.Tensor, rstd: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
            n: int, k: int) -> torch.Tensor:
    """consumer linear of a C32-NE norm: W4A4 on the C32-NE buffers with the rows scaled by rstd, or the
    int8 path (normalised input, rstd = 1) when the bucket is not a4."""
    from atdev_serve import marlin_plus as mp
    m = xq.shape[0]
    ent = _ent(n, k, m) if m > 0 else None
    if ent is None:
        return mp.mp_linear_q(xq, xs, w_q, w_s, n, k)
    b = mp.bufs(xq.device)
    nb = _bufs(xq.device)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=xq.device)
    return _mod(2).gemm_a4_rs(nb["pk"], nb["s"], w_q, w_s, out, b["locks"], b["c_tmp"], m, n, k,
                                  list(ent["cfg"]), rstd)


def _ne_cons_fake(xq, xs, rstd, w_q, w_s, n, k):
    return torch.empty((xq.shape[0], n), dtype=torch.bfloat16, device=xq.device)


# ---------------------------------------------------------------- model plumbing

def _w(linear):
    from atdev_serve import fused_quant as fq
    kern = fq._mp_kernel(linear)
    kk, n = kern.config.partition_weight_shape
    w_q, w_s, _ = kern._get_weight_params(linear)
    return w_q, w_s, n, kk


def _gsl(linear):
    from atdev_serve import fused_quant as fq
    return fq._gs(linear)


def _first(layer):
    return layer.linear_attn.in_proj_qkvz if layer.layer_type == "linear_attention" else layer.self_attn.qkv_proj


def _empty_bf(dev, n):
    return torch.empty((0, n), dtype=torch.bfloat16, device=dev)


def _prod(lin, xq, xs, x_bf, residual, nxt_norm, nxt_first, want_bf16):
    w_q, w_s, n, k = _w(lin)
    if GB:
        xb, xq2, xs2, ro = torch.ops.vllm.atdev_ne_prod_gb(
            xq, xs, x_bf, w_q, w_s, _gsl(lin), n, k, residual, nxt_norm.weight, nxt_norm.variance_epsilon,
            _gsl(nxt_first), _w(nxt_first)[2], want_bf16)
        return xb, xq2, xs2, ro, None
    return torch.ops.vllm.atdev_ne_prod(xq, xs, x_bf, w_q, w_s, _gsl(lin), n, k, residual, nxt_norm.weight,
                                        nxt_norm.variance_epsilon, _gsl(nxt_first), _w(nxt_first)[2],
                                        want_bf16)


def _cons(lin, xq, xs, rstd):
    if rstd is None:  # GB mode: the norm is already applied
        from atdev_serve import fused_quant as fq
        return fq.linear_q(lin, xq, xs)
    w_q, w_s, n, k = _w(lin)
    return torch.ops.vllm.atdev_ne_cons(xq, xs, rstd, w_q, w_s, n, k)


def _layer(layer, positions, st, nxt):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import _encode_layer_name
    from atdev_serve import fused_quant as fq
    xb, xq, xs, residual, rstd = st
    m = xq.shape[0]
    gu, dn = layer.mlp.gate_up_proj, layer.mlp.down_proj
    if layer.layer_type == "linear_attention":
        g = layer.linear_attn
        qkvz = _cons(g.in_proj_qkvz, xq, xs, rstd)
        ba, _ = g.in_proj_ba(xb)
        if rstd is not None:
            ba = (ba.float() * rstd[:m, None]).to(ba.dtype)
        core = torch.zeros((m, g.num_v_heads // g.tp_size, g.head_v_dim), dtype=torch.bfloat16,
                           device=xq.device)
        torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(qkvz, ba, core,
                                                                 layer_name=_encode_layer_name(g.prefix))
        dummy_q = torch.empty((0, 1), dtype=torch.int8, device=xq.device)
        dummy_s = torch.empty((0,), dtype=torch.float32, device=xq.device)
        st2 = _prod(g.out_proj, dummy_q, dummy_s, core.flatten(-2), residual, layer.post_attention_layernorm,
                    gu, False)
    else:
        at = layer.self_attn
        qkv = _cons(at.qkv_proj, xq, xs, rstd)
        q, k, v, gate = at._project_qkv_gate(qkv, positions)
        o = at.attn(q, k, v)
        oq, os_ = torch.ops.vllm.atdev_gate_quant(o, gate, fq._gs(at.o_proj), fq._n(at.o_proj))
        st2 = _prod(at.o_proj, oq, os_, _empty_bf(xq.device, at.o_proj.input_size_per_partition), residual,
                    layer.post_attention_layernorm, gu, False)
    _, xq2, xs2, res2, rstd2 = st2
    y = _cons(gu, xq2, xs2, rstd2)
    q, s = torch.ops.vllm.atdev_silu_quant(y, fq._gs(dn), fq._n(dn))
    if nxt is None:
        return fq.linear_q(dn, q, s), res2
    nf = _first(nxt)
    return _prod(dn, q, s, _empty_bf(xq.device, dn.input_size_per_partition), res2, nxt.input_layernorm, nf,
                 nxt.layer_type == "linear_attention")


_ELIG: dict = {}


def _layer_ok(layer) -> bool:
    from atdev_serve import fused_quant as fq
    e = _ELIG.get(id(layer))
    if e is None:
        e = fq._layer_eligible(layer) and not getattr(layer, "layer_scale", False) and \
            not getattr(layer, "use_attn_reduce_scatter_for_moe", False)
        if e and layer.layer_type != "linear_attention":
            e = getattr(layer.self_attn, "attn_output_gate", True)
        if e and layer.layer_type == "linear_attention":
            g = layer.linear_attn
            e = g.enable_fused_gdn_decode and g.norm.weight.dtype in (torch.bfloat16, torch.float32) and \
                getattr(g.out_proj, "bias", None) is None
        _ELIG[id(layer)] = e
    return e


def model_ok(model) -> bool:
    from vllm.distributed import get_pp_group
    key = ("model", id(model))
    e = _ELIG.get(key)
    if e is None:
        pp = get_pp_group()
        layers = list(model.layers[model.start_layer:model.end_layer])
        e = (pp.is_first_rank and pp.is_last_rank and not model.use_sequence_parallel and len(layers) > 1
             and not getattr(model, "aux_hidden_state_layers", ()) and all(_layer_ok(l) for l in layers))
        _ELIG[key] = e
    return e


def model_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    h = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
    layers = list(self.layers[self.start_layer:self.end_layer])
    l0 = layers[0]
    nf = _first(l0)
    if GB:
        xb, xq, xs, res = torch.ops.vllm.atdev_norm_quant(
            h, None, l0.input_layernorm.weight, l0.input_layernorm.variance_epsilon, _gsl(nf),
            l0.layer_type == "linear_attention", _w(nf)[2])
        st = (xb, xq, xs, res, None)
    else:
        st = torch.ops.vllm.atdev_ne_first(h, l0.input_layernorm.weight, l0.input_layernorm.variance_epsilon,
                                           _gsl(nf), _w(nf)[2], l0.layer_type == "linear_attention")
    out = None
    for i, layer in enumerate(layers):
        nxt = layers[i + 1] if i + 1 < len(layers) else None
        out = _layer(layer, positions, st, nxt)
        if nxt is not None:
            st = out
    hidden_states, _ = self.norm(out[0], out[1])
    return hidden_states


_REGISTERED = False


def register() -> None:
    """called from fused_quant.register when ATDEV_C32_NE=1"""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    from vllm.model_executor.models import qwen3_next as qn
    from vllm.utils.torch_utils import direct_register_custom_op
    from atdev_serve import compile_cache
    from atdev_serve import marlin_plus as mp
    mp.mod_a4()
    modes = (3,) if GB else (1, 2)
    if any(_mod(md).ne != md for md in modes):  # build / load the variants before capture
        raise RuntimeError("atdev c32_ne: C32-NE variant builds missing")
    direct_register_custom_op("atdev_ne_prod", ne_prod, fake_impl=_ne_prod_fake)
    direct_register_custom_op("atdev_ne_first", ne_first, fake_impl=_ne_first_fake)
    direct_register_custom_op("atdev_ne_cons", ne_cons, fake_impl=_ne_cons_fake)
    direct_register_custom_op("atdev_ne_prod_gb", ne_prod_gb, fake_impl=_ne_prod_gb_fake)
    compile_cache.salt_aot_cache("c32_ne", __file__)  # the traced model forward differs
    orig = qn.Qwen3NextModel.forward

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None or not model_ok(self):
            return orig(self, input_ids, positions, intermediate_tensors, inputs_embeds)
        return model_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)

    qn.Qwen3NextModel.forward = forward
    log.warning("atdev c32_ne: Qwen3NextModel.forward -> norm folded into the W4A4 GEMM epilogues")
