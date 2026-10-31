"""c1_fuse: at the c=1 verify batch (8 < M <= 16, bf16 activations, 3-bit W3 weights of w3_mixed) the residual-add
+ Gemma RMSNorm and the SiLU*mul run inside the marlin_c1 W3 GEMM epilogues (csrc/marlin_c1, MpEpi;
marlin_c1_w3_build, tile table w3_c1_cfg.json):

  producer (out_proj / o_proj / down_proj): h = x @ W; x' = h + residual (fp32)
      -> residual_out = bf16(x'), a = bf16(x' * (1 + w_norm)), P[row, tile] = sum over the tile of x'^2
  consumer (qkvz / qkv / gate_up): y = rstd[row] * (a @ W), rstd = rsqrt(sum_t P[row, t] / 5120 + eps)
      gate_up: weight columns interleaved per 64 at load, y = bf16(silu(g) * u) in the epilogue
  in_proj_ba: split-K GEMV on a with the rows scaled by the qkvz consumer's rstd

so the 128 norm and 64 SiLU*mul kernels of a step disappear. Numerics: rstd scales the fp32 GEMM accumulator
instead of the bf16 GEMM input (one bf16 rounding of x' (1 + w) instead of x' rstd (1 + w)); the fp32 order of
sum(x'^2) differs from Inductor. Other M (prefill, M <= 8) run the same ops through an unfused fallback
(atdev_w3_linear + torch epilogue math). Qwen3NextModel.forward is replaced (the decoder layers are driven from
it so a layer's down_proj can apply the next layer's input norm).

Two folds on the GDN / attention side (needs dflash_tree with the CUDA tree kernel, and tree_attn):
  * in_proj_ba runs on a side stream forked inside the captured graph, concurrently with the qkvz consumer GEMM
    (op vllm::atdev_e6_cons_bat): the side stream computes the unscaled fp32 split-K accumulator (c1_norm.ba_tot),
    and the CUDA tree kernel rounds bf16(tot * rstd) itself (vllm::atdev_gdn_core_bap splitting op) --
    bit-identical to the serial GEMV with the rstd row scale.
  * full-attention layers pass the output gate into vllm::atdev_attn_gated (tree_attn.py); the gate is applied in
    tree_attn's combine kernel.
Both are used only if the run's splitting ops contain the new ops whenever they contain the stock ones (checked at
load; otherwise the unfused path is kept and a warning logged)."""
from __future__ import annotations

import logging

import torch

from atdev_serve import c1_norm as cn

try:
    from vllm.utils.torch_utils import LayerNameType
except Exception:  # pragma: no cover
    LayerNameType = str

OPT_NAME = "c1_fuse"
SUPPORTED_VLLM_VERSION = "0.31.0"
log = logging.getLogger("atdev_serve")
H = 5120


def _tn16(n: int, k: int) -> int:
    cfg = cn.mp_cfg(n, k, 16)
    return cfg[1] if cfg is not None else 64


def _w3_tn16(n: int, k: int) -> int:
    cfg = cn.w3c1_cfg(n, k, 16)
    return cfg[1] if cfg is not None else 64


def _w3_fused_ok(m: int, n: int, k: int) -> bool:
    return cn.w3c1_cfg(n, k, m) is not None


def _w3lin(x, q3, ws, n, k):
    from atdev_serve import w3_mixed
    return w3_mixed.w3_linear(x, q3, ws, n, k)


# ---------------------------------------------------------------- custom ops (real + fake)
_LOGGED = False


def _log_once():
    global _LOGGED
    if not _LOGGED:
        _LOGGED = True
        log.warning("atdev %s: fused norm/SiLU epilogues ENGAGED (marlin_c1, M=9..16)", OPT_NAME)


def _unperm_silu(gu: torch.Tensor) -> torch.Tensor:
    m, n = gu.shape
    g4 = gu.view(m, n // 128, 2, 64).float()
    g, u = g4[:, :, 0].reshape(m, -1), g4[:, :, 1].reshape(m, -1)
    return (g / (1.0 + torch.exp(-g)) * u).to(torch.bfloat16)


def _e6_cons_fake(a, wq, ws, p, n, k, n_norm, eps, silu):
    m = a.shape[0]
    return a.new_empty((m, n // 2 if silu else n)), a.new_empty((m,), dtype=torch.float32)


def _e6_cons_r_fake(a, wq, ws, p, n, k, n_norm, eps):
    m = a.shape[0]
    return a.new_empty((m, n)), a.new_empty((m,), dtype=torch.float32)


def e6w3_prod(x: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor, res: torch.Tensor, nw: torch.Tensor,
              n: int, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = x.shape[0]
    t = n // _w3_tn16(n, k)
    if _w3_fused_ok(m, n, k):
        _log_once()
        return cn.lin_prod_w3(x.contiguous(), wq, ws, n, k, res.contiguous(), nw)
    h = _w3lin(x, wq, ws, n, k)
    xf = h.float() + res.float()
    p = torch.zeros((m, t), dtype=torch.float32, device=x.device)
    p[:, 0] = (xf * xf).sum(1)
    return (xf * (nw.float() + 1.0)).to(torch.bfloat16), xf.to(torch.bfloat16), p


def _e6w3_prod_fake(x, wq, ws, res, nw, n, k):
    m = x.shape[0]
    t = n // _w3_tn16(n, k)
    e = x.new_empty((m, n))
    return e, torch.empty_like(e), x.new_empty((m, t), dtype=torch.float32)


def e6w3_cons(a: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor, p: torch.Tensor, n: int, k: int,
              n_norm: int, eps: float, silu: bool) -> tuple[torch.Tensor, torch.Tensor]:
    m = a.shape[0]
    if _w3_fused_ok(m, n, k) and (not silu or cn.w3c1_cfg(n, k, m)[1] == 128):
        out, r = cn.lin_cons_w3(a.contiguous(), wq, ws, n, k, p.contiguous(), n_norm, eps, silu=silu, want_r=False)
        return out, torch.empty((m,), dtype=torch.float32, device=a.device) if r is None else r
    r = torch.rsqrt(p.sum(1) / n_norm + eps)
    y = _w3lin((a.float() * r[:, None]).to(torch.bfloat16), wq, ws, n, k)
    return (_unperm_silu(y) if silu else y), r


def e6w3_cons_r(a: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor, p: torch.Tensor, n: int, k: int,
                n_norm: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """consumer that also returns the row rstd (for in_proj_ba)."""
    m = a.shape[0]
    if _w3_fused_ok(m, n, k):
        return cn.lin_cons_w3(a.contiguous(), wq, ws, n, k, p.contiguous(), n_norm, eps, silu=False, want_r=True)
    return e6w3_cons(a, wq, ws, p, n, k, n_norm, eps, False)


def e6_ba(a: torch.Tensor, w: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    m = a.shape[0]
    if m <= 16 and a.stride(-1) == 1 and w.stride(-1) == 1 and w.shape[0] % 16 == 0 and a.shape[1] % 1024 == 0:
        return cn.ba_gemv_r(a, w, r.contiguous())
    return torch.nn.functional.linear((a.float() * r[:, None]).to(torch.bfloat16), w)


def _e6_ba_fake(a, w, r):
    return a.new_empty((a.shape[0], w.shape[0]))


# ---- in_proj_ba on a side stream concurrent with the qkvz consumer GEMM
_SIDE: dict = {}


def _side(dev) -> torch.cuda.Stream:
    s = _SIDE.get(dev.index)
    if s is None:
        s = _SIDE[dev.index] = torch.cuda.Stream(device=dev)
    return s


def _cons_tot(a, wq, ws, p, n, k, n_norm, eps, wba):
    """(qkvz, r, tot | None): fork -> side: tot = ba_tot(a, W_ba); main: qkvz consumer (+ rstd); join."""
    if not (a.is_cuda and cn.ba_tot_ok(a, wba)):
        qkvz, r = e6w3_cons_r(a, wq, ws, p, n, k, n_norm, eps)
        return qkvz, r, None
    m = a.shape[0]
    tot = torch.empty((m, wba.shape[0]), dtype=torch.float32, device=a.device)
    cur = torch.cuda.current_stream(a.device)
    side = _side(a.device)
    # fork point recorded BEFORE the GEMM, side branch issued AFTER it: the graph launches the GEMM node first
    # (issuing the side node first delays the GEMM launch in a replay)
    ev = torch.cuda.Event()
    ev.record(cur)
    qkvz, r = e6w3_cons_r(a, wq, ws, p, n, k, n_norm, eps)
    side.wait_event(ev)
    with torch.cuda.stream(side):
        cn.ba_tot(a, wba, out=tot)
    cur.wait_stream(side)  # every tensor above is allocated on (and outlives the join on) the main stream
    return qkvz, r, tot


def e6_cons_bat(a: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor, p: torch.Tensor, n: int, k: int,
                n_norm: int, eps: float, wba: torch.Tensor, w3: bool
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(qkvz, r', tot) with bf16(tot * r') == e6_ba(a, wba, r) (fallback shapes: tot = the bf16 ba in fp32,
    r' = 1 -> exact)."""
    qkvz, r, tot = _cons_tot(a, wq, ws, p, n, k, n_norm, eps, wba)
    if tot is None:
        ba = e6_ba(a, wba, r)
        r1, tot = torch.ones((a.shape[0],), dtype=torch.float32, device=a.device), ba.float()
    else:
        r1 = r.contiguous()
    return qkvz, r1, tot


def _e6_cons_bat_fake(a, wq, ws, p, n, k, n_norm, eps, wba, w3):
    m = a.shape[0]
    return (a.new_empty((m, n)), a.new_empty((m,), dtype=torch.float32),
            a.new_empty((m, wba.shape[0]), dtype=torch.float32))


def gdn_core_bap(qkvz: torch.Tensor, tot: torch.Tensor, r: torch.Tensor, core: torch.Tensor,
                 layer_name: LayerNameType) -> None:
    from vllm.forward_context import get_forward_context
    from vllm.utils.torch_utils import _resolve_layer_name
    from atdev_serve import gdn_tree_cuda as gtc
    layer = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    defer = getattr(layer, "_atdev_ba_defer_ok", None)
    if defer is not None and defer(qkvz):
        ba = torch.empty(tot.shape, dtype=qkvz.dtype, device=qkvz.device)
        gtc.BA_CTX.update(kind="tot", tot=tot, r=r, ba=ba, done=False)
        try:
            layer._forward_core_fused_norm_packed(mixed_qkvz=qkvz, ba=ba, core_attn_out=core)
        finally:
            gtc.BA_CTX.clear()
        return
    layer._forward_core_fused_norm_packed(mixed_qkvz=qkvz, ba=cn.ba_scale(tot, r, tot.shape[0]),
                                          core_attn_out=core)


def _gdn_core_bap_fake(qkvz, tot, r, core, layer_name) -> None:
    return None


_FOLD_OK = {"gate": False, "bap": False}  # set at load from the run's splitting ops


def _split_ops_ok(stock: str, new: str) -> bool | None:
    """True if `new` is a splitting op whenever `stock` is (None: no vLLM config to check)."""
    try:
        from vllm.config import get_current_vllm_config_or_none
        vc = get_current_vllm_config_or_none()
        if vc is None or vc.model_config is None:
            return None
        so = vc.compilation_config.splitting_ops or []
    except Exception:
        return None
    return (stock not in so) or (new in so)


def e6_first(x: torch.Tensor, nw: torch.Tensor, n_t: int) -> tuple[torch.Tensor, torch.Tensor]:
    """layer 0 (no residual): a = bf16(x * (1 + w)), P[:, 0] = sum(x^2)."""
    xf = x.float()
    p = torch.zeros((x.shape[0], n_t), dtype=torch.float32, device=x.device)
    p[:, 0] = (xf * xf).sum(1)
    return (xf * (nw.float() + 1.0)).to(torch.bfloat16), p


def _e6_first_fake(x, nw, n_t):
    return torch.empty_like(x), x.new_empty((x.shape[0], n_t), dtype=torch.float32)


# ---------------------------------------------------------------- model forward

def _kern(linear):
    from atdev_serve.marlin_plus import kernel_class
    kk = getattr(getattr(linear, "scheme", None), "kernel", None)
    if kk is None or type(kk) is not kernel_class() or getattr(kk, "_atdev_stock", True) \
            or kk.config.act_type != torch.bfloat16:
        return None
    return kk


def _is_w3(linear) -> bool:
    return bool(getattr(linear, "_atdev_w3", False))


def _w(linear):
    kk = _kern(linear)
    kdim, n = kk.config.partition_weight_shape
    wq, ws, _ = kk._get_weight_params(linear)
    return wq, ws, n, kdim


_ELIG: dict = {}


def _layer_ok(layer) -> bool:
    key = id(layer)
    e = _ELIG.get(key)
    if e is None:
        gdn = layer.layer_type == "linear_attention"
        lins = ([layer.linear_attn.in_proj_qkvz, layer.linear_attn.out_proj] if gdn
                else [layer.self_attn.qkv_proj, layer.self_attn.o_proj])
        lins += [layer.mlp.gate_up_proj, layer.mlp.down_proj]
        e = all(_kern(l) is not None and _is_w3(l) for l in lins) and not getattr(layer, "layer_scale", False) \
            and getattr(layer.mlp, "expert_gate", None) is None \
            and getattr(layer.mlp.gate_up_proj, "_atdev_gu_perm", False) \
            and layer.input_layernorm.weight.shape[0] == H
        if gdn:
            g = layer.linear_attn
            e = e and g.enable_fused_gdn_decode and g.in_proj_ba.weight.dtype == torch.bfloat16
        _ELIG[key] = e
    return e


def _model_ok(model) -> bool:
    from vllm.distributed import get_pp_group
    key = ("model", id(model))
    e = _ELIG.get(key)
    if e is None:
        pp = get_pp_group()
        layers = list(model.layers[model.start_layer:model.end_layer])
        e = (pp.is_first_rank and pp.is_last_rank and not model.use_sequence_parallel and len(layers) > 1
             and all(_layer_ok(l) for l in layers))
        _ELIG[key] = e  # (no logging: runs under Dynamo)
    return e


def _consume(layer, positions, a, p, eps):
    """attention / GDN part of a layer from the prepared input (a, p); returns the input of o_proj/out_proj."""
    ops = torch.ops.vllm
    if layer.layer_type == "linear_attention":
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import _encode_layer_name
        g = layer.linear_attn
        wq, ws, n, k = _w(g.in_proj_qkvz)
        shape = (a.shape[0], g.num_v_heads // g.tp_size, g.head_v_dim)
        # no zero fill: the core op writes every real row; padded rows stay row-local through the bf16 path
        if _FOLD_OK["bap"]:
            core = torch.empty(shape, dtype=a.dtype, device=a.device)
            qkvz, r, tot = ops.atdev_e6_cons_bat(a, wq, ws, p, n, k, H, eps, g.in_proj_ba.weight, True)
            ops.atdev_gdn_core_bap(qkvz, tot, r, core, layer_name=_encode_layer_name(g.prefix))
            return core.flatten(-2), g.out_proj
        qkvz, r = ops.atdev_e6w3_cons_r(a, wq, ws, p, n, k, H, eps)
        core = torch.empty(shape, dtype=a.dtype, device=a.device)
        ba = ops.atdev_e6_ba(a, g.in_proj_ba.weight, r)
        ops.qwen_gdn_attention_core_fused_norm_packed(qkvz, ba, core, layer_name=_encode_layer_name(g.prefix))
        return core.flatten(-2), g.out_proj
    at = layer.self_attn
    wq, ws, n, k = _w(at.qkv_proj)
    qkv, _ = ops.atdev_e6w3_cons(a, wq, ws, p, n, k, H, eps, False)
    q, kk, v, gate = at._project_qkv_gate(qkv, positions)
    if gate is not None and _FOLD_OK["gate"]:
        from atdev_serve import tree_attn
        if tree_attn.gated_ok(at, gate):
            return tree_attn.gated_attention(at, q, kk, v, gate), at.o_proj
    o = at.attn(q, kk, v)
    if gate is not None:
        o = o * torch.sigmoid(gate)
    return o, at.o_proj


def _layer(layer, positions, a, p, residual, nxt):
    ops = torch.ops.vllm
    eps_in = layer.input_layernorm.variance_epsilon
    x, lin = _consume(layer, positions, a, p, eps_in)
    wq, ws, n, k = _w(lin)
    ln2 = layer.post_attention_layernorm
    a2, res2, p2 = ops.atdev_e6w3_prod(x, wq, ws, residual, ln2.weight, n, k)
    gu, dn = layer.mlp.gate_up_proj, layer.mlp.down_proj
    wq, ws, n, k = _w(gu)
    y, _ = ops.atdev_e6w3_cons(a2, wq, ws, p2, n, k, H, ln2.variance_epsilon, True)
    wq, ws, n, k = _w(dn)
    if nxt is None:
        return ops.atdev_w3_linear(y, wq, ws, n, k), res2
    return ops.atdev_e6w3_prod(y, wq, ws, res2, nxt.input_layernorm.weight, n, k)


def model_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    h = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
    aux = self._maybe_add_hidden_state([], 0, h, None)
    layers = list(self.layers[self.start_layer:self.end_layer])
    l0 = layers[0]
    n_t = _NT  # partial count of the first consumer's P input (set at load)
    a, p = torch.ops.vllm.atdev_e6_first(h, l0.input_layernorm.weight, n_t)
    residual = h
    out = None
    for i, layer in enumerate(layers):
        nxt = layers[i + 1] if i + 1 < len(layers) else None
        out = _layer(layer, positions, a, p, residual, nxt)
        idx = self.start_layer + i + 1
        if nxt is not None:
            a, residual, p = out
            if idx in self.aux_hidden_state_layers:
                aux.append(residual)  # == hidden + residual of the stock loop (bf16 of the fp32 sum)
        else:
            self._maybe_add_hidden_state(aux, idx, out[0], out[1])
    hidden_states, _ = self.norm(out[0], out[1])
    if aux:
        return hidden_states, aux
    return hidden_states


# ---------------------------------------------------------------- weights

_NT = 80


def _permute_gate_up(model) -> int:
    from vllm.model_executor.models import qwen3_next as qn
    global _NT
    _NT = H // _tn16(H, 6144)  # (outside Dynamo: reads the marlin_plus config table)
    cnt = 0
    for mod in model.modules():
        if not isinstance(mod, qn.Qwen3NextDecoderLayer):
            continue
        gu = mod.mlp.gate_up_proj
        kk = _kern(gu)
        if kk is None or getattr(gu, "_atdev_gu_perm", False) or not _is_w3(gu):
            continue
        wq, ws, n, k = _w(gu)
        # 3-bit planar (w3_mixed): the fused W3 SiLU consumer needs tn 128
        c3 = cn.w3c1_cfg(n, k, 16)
        if n % 128 or (c3 is not None and c3[1] != 128):
            continue
        wq2, ws2 = cn.permute_gu_w3(wq, ws, n)
        wq.data.copy_(wq2)
        ws.data.copy_(ws2)
        gu._atdev_gu_perm = True
        cnt += 1
    return cnt


def _prepare_w3(model) -> None:
    """build / load the W3 marlin_c1 modes the model's 3-bit layers need (outside any capture) and log the plan."""
    from vllm.model_executor.models import qwen3_next as qn
    modes, n3, nf, slow, fused_shapes = set(), 0, 0, {}, set()
    for mod in model.modules():
        if not isinstance(mod, qn.Qwen3NextDecoderLayer):
            continue
        roles = [(mod.mlp.gate_up_proj, 3), (mod.mlp.down_proj, 1)]
        if mod.layer_type == "linear_attention":
            roles += [(mod.linear_attn.in_proj_qkvz, 2), (mod.linear_attn.out_proj, 1)]
        else:
            roles += [(mod.self_attn.qkv_proj, 2), (mod.self_attn.o_proj, 1)]
        for lin, md in roles:
            if not _is_w3(lin) or _kern(lin) is None:
                continue
            n3 += 1
            _, _, n, k = _w(lin)
            if _w3_fused_ok(16, n, k) and (md != 3 or cn.w3c1_cfg(n, k, 16)[1] == 128):
                modes.add(md)
                fused_shapes.add((n, k, md))
                nf += 1
            else:
                why = "no w3_c1_cfg entry" if cn.w3c1_cfg(n, k, 16) is None else "SiLU consumer needs thread_n 128"
                key = (f"N={n} K={k} mode {md}", why)
                slow[key] = slow.get(key, 0) + 1
    if not n3:
        return
    for md in sorted(modes):
        cn.mc1w3(md)
    for (n, k, md) in sorted(fused_shapes):  # a stale / hand-edited table entry must fail at load, not at capture
        c = cn.w3c1_cfg(n, k, 16)
        inst = {tuple(i[3:]) for i in cn.mc1w3(md).instantiated()}
        if (c[0], c[1], c[2], c[3]) not in inst:
            raise RuntimeError(f"atdev {OPT_NAME}: w3_c1_cfg N={n} K={k} {c} is not an instantiated W3 marlin_c1 "
                               f"mode-{md} kernel")
    log.warning("atdev %s: %d W3 (3-bit planar) linears -> %d fused W3 marlin_c1 (modes %s), %d unfused W3 "
                "fallback", OPT_NAME, n3, nf, sorted(modes), n3 - nf)
    for (shape, why), c in sorted(slow.items()):
        log.warning("atdev %s: W3 UNFUSED at M=9..16 for %d linears %s: %s (fallback = atdev_w3_linear + torch "
                    "epilogue)", OPT_NAME, c, shape, why)


def _check_folds() -> None:
    """enable the folds only if their ops split the piecewise graphs like the stock ops they replace."""
    ok = _split_ops_ok("vllm::qwen_gdn_attention_core_fused_norm_packed", "vllm::atdev_gdn_core_bap")
    _FOLD_OK["bap"] = bool(ok)
    log.warning("atdev %s: in_proj_ba side stream + fp32 ba / rstd into the GDN core op %s", OPT_NAME,
                "ENABLED" if ok else f"DISABLED (splitting-op check {ok})")
    dev = torch.device("cuda", torch.cuda.current_device())
    _side(dev)  # create outside any capture
    cn._sbufs(dev)  # scratch + tick outside any capture
    cn.ba_side_ext()  # build / load before any capture
    ok = _split_ops_ok("vllm::unified_attention_with_output", "vllm::atdev_attn_gated")
    _FOLD_OK["gate"] = bool(ok)
    log.warning("atdev %s: attention output gate fold %s", OPT_NAME,
                "ENABLED" if ok else f"DISABLED (splitting-op check {ok})")


_REGISTERED = False


def register() -> None:
    global _REGISTERED
    import hashlib
    import json
    import os

    import vllm
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture
    from vllm.config import compilation as vcc
    from vllm.model_executor.model_loader import base_loader
    from vllm.model_executor.models import qwen3_next as qn
    from vllm.utils.torch_utils import direct_register_custom_op
    from atdev_serve import compile_cache, tree_attn
    from atdev_serve.register import enabled

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written for vLLM {SUPPORTED_VLLM_VERSION}, found {vllm.__version__}")
    need = {"marlin_plus", "w3_mixed", "dflash_tree", "tree_attn"}
    if not need <= enabled() or os.environ.get("VLLM_MARLIN_INPUT_DTYPE") == "int8" or "fused_quant" in enabled():
        raise RuntimeError(f"atdev {OPT_NAME}: needs {sorted(need)} on the bf16 path (no fused_quant / int8)")
    if _REGISTERED:
        return
    _REGISTERED = True
    # the W3 tile table sets the traced partial-count shapes; the traced attention block differs (gated op, KV
    # update inside it) -> separate AOT artifact
    h = hashlib.sha256(json.dumps(cn.w3c1_table(), sort_keys=True).encode()).hexdigest()[:10]
    compile_cache._SALTS[OPT_NAME] = compile_cache._SALTS.get(OPT_NAME, "") + f"+w3c1={h}"
    direct_register_custom_op("atdev_e6_ba", e6_ba, fake_impl=_e6_ba_fake)
    direct_register_custom_op("atdev_e6_first", e6_first, fake_impl=_e6_first_fake)
    direct_register_custom_op("atdev_e6w3_prod", e6w3_prod, fake_impl=_e6w3_prod_fake)
    direct_register_custom_op("atdev_e6w3_cons", e6w3_cons, fake_impl=_e6_cons_fake)
    direct_register_custom_op("atdev_e6w3_cons_r", e6w3_cons_r, fake_impl=_e6_cons_r_fake)
    direct_register_custom_op("atdev_e6_cons_bat", e6_cons_bat, fake_impl=_e6_cons_bat_fake)
    direct_register_custom_op("atdev_gdn_core_bap", eager_break_during_capture(gdn_core_bap),
                              mutates_args=["core"], fake_impl=_gdn_core_bap_fake)
    if "vllm::atdev_gdn_core_bap" not in vcc.CompilationConfig._attention_ops:
        vcc.CompilationConfig._attention_ops.append("vllm::atdev_gdn_core_bap")
    tree_attn._register_gated_op()

    orig_pwal = base_loader.process_weights_after_loading

    def process_weights_after_loading(model, model_config, target_device):
        orig_pwal(model, model_config, target_device)
        c = _permute_gate_up(model)
        if c:
            log.warning("atdev %s: gate_up weights interleaved per 64 columns on %d layers", OPT_NAME, c)
        _prepare_w3(model)
        _check_folds()

    base_loader.process_weights_after_loading = process_weights_after_loading

    orig_fwd = qn.Qwen3NextModel.forward

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None or not _model_ok(self):
            if any(getattr(l.mlp.gate_up_proj, "_atdev_gu_perm", False) for l in self.layers
                   if hasattr(l, "mlp") and hasattr(l.mlp, "gate_up_proj")):
                raise RuntimeError(f"atdev {OPT_NAME}: gate_up weights are interleaved but the fused "
                                   "forward is not eligible")
            return orig_fwd(self, input_ids, positions, intermediate_tensors, inputs_embeds)
        return model_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)

    qn.Qwen3NextModel.forward = forward
    log.warning("atdev %s: Qwen3NextModel.forward -> norm / SiLU folded into the marlin_c1 GEMM epilogues", OPT_NAME)
