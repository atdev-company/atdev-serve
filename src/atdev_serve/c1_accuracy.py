"""Measured no-W3 W4A16 c1_fuse path; same DFlash, BA sidefork and attention folds.

Only selected W4 code is restored from the measured implementation. Kernel builds
are deferred until weights are loaded (outside capture), so CPU CLI help stays lazy.
"""
from __future__ import annotations
import logging
import hashlib
import os
import torch
from atdev_serve import c1_fuse as fast
from atdev_serve import c1_norm as cn

log=logging.getLogger("atdev_serve")
H=fast.H
OPT_NAME="c1_fuse_accuracy"
_SUPPORTED="0.31.0"
_kern=fast._kern
_w=fast._w
_tn16=fast._tn16
_unperm_silu=fast._unperm_silu
_log_once=fast._log_once
_side=fast._side
_ELIG={}
_NT=80
_REGISTERED=False

def _fused_ok(m: int, n: int, k: int) -> bool:
    return 8 < m <= 16 and cn.mp_cfg(n, k, m) is not None

def _mp(x, wq, ws, n, k):
    from atdev_serve import marlin_plus
    return marlin_plus.mp_linear(x, wq, ws, ws, n, k, False)

def e6_prod(x: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor, res: torch.Tensor, nw: torch.Tensor,
            n: int, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = x.shape[0]
    t = n // _tn16(n, k)
    if _fused_ok(m, n, k) and _tn16(n, k) == cn.mp_cfg(n, k, m)[1]:
        _log_once()
        return cn.lin_prod(x.contiguous(), wq, ws, n, k, res.contiguous(), nw)
    h = _mp(x, wq, ws, n, k)
    xf = h.float() + res.float()
    p = torch.zeros((m, t), dtype=torch.float32, device=x.device)
    p[:, 0] = (xf * xf).sum(1)
    return (xf * (nw.float() + 1.0)).to(torch.bfloat16), xf.to(torch.bfloat16), p

def _e6_prod_fake(x, wq, ws, res, nw, n, k):
    m = x.shape[0]
    t = n // _tn16(n, k)
    e = x.new_empty((m, n))
    return e, torch.empty_like(e), x.new_empty((m, t), dtype=torch.float32)

def e6_cons(a: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor, p: torch.Tensor, n: int, k: int,
            n_norm: int, eps: float, silu: bool) -> tuple[torch.Tensor, torch.Tensor]:
    m = a.shape[0]
    if _fused_ok(m, n, k) and (not silu or cn.mp_cfg(n, k, m)[1] == 128):
        out, r = cn.lin_cons(a.contiguous(), wq, ws, n, k, p.contiguous(), n_norm, eps, silu=silu,
                             want_r=False)
        r = torch.empty((m,), dtype=torch.float32, device=a.device) if r is None else r
        # rstd for a second consumer: the Marlin consumer only exports it on request (want_r)
        return out, r
    r = torch.rsqrt(p.sum(1) / n_norm + eps)
    y = _mp((a.float() * r[:, None]).to(torch.bfloat16), wq, ws, n, k)
    return (_unperm_silu(y) if silu else y), r

def e6_cons_r(a: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor, p: torch.Tensor, n: int, k: int,
              n_norm: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """consumer that also returns the row rstd (for in_proj_ba)."""
    m = a.shape[0]
    if _fused_ok(m, n, k):
        return cn.lin_cons(a.contiguous(), wq, ws, n, k, p.contiguous(), n_norm, eps, silu=False, want_r=True)
    return e6_cons(a, wq, ws, p, n, k, n_norm, eps, False)

def _e6_cons_fake(a, wq, ws, p, n, k, n_norm, eps, silu):
    m = a.shape[0]
    return a.new_empty((m, n // 2 if silu else n)), a.new_empty((m,), dtype=torch.float32)

def _e6_cons_r_fake(a, wq, ws, p, n, k, n_norm, eps):
    m = a.shape[0]
    return a.new_empty((m, n)), a.new_empty((m,), dtype=torch.float32)

def _layer_ok(layer) -> bool:
    key = id(layer)
    e = _ELIG.get(key)
    if e is None:
        gdn = layer.layer_type == "linear_attention"
        lins = ([layer.linear_attn.in_proj_qkvz, layer.linear_attn.out_proj] if gdn
                else [layer.self_attn.qkv_proj, layer.self_attn.o_proj])
        lins += [layer.mlp.gate_up_proj, layer.mlp.down_proj]
        e = all(_kern(l) is not None and not fast._is_w3(l) for l in lins) and not getattr(layer, "layer_scale", False) \
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

def model_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    h = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
    aux = self._maybe_add_hidden_state([], 0, h, None)
    layers = list(self.layers[self.start_layer:self.end_layer])
    l0 = layers[0]
    n_t = _NT  # producer partial count (all producers share N = 5120 and thread_n; set at load)
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

def _cons_tot(a, wq, ws, p, n, k, n_norm, eps, wba):
    """(qkvz, r, tot | None): fork -> side: tot = ba_tot(a, W_ba); main: qkvz consumer (+ rstd); join."""
    if not (a.is_cuda and cn.ba_tot_ok(a, wba)):
        qkvz, r = e6_cons_r(a, wq, ws, p, n, k, n_norm, eps)
        return qkvz, r, None
    m = a.shape[0]
    tot = torch.empty((m, wba.shape[0]), dtype=torch.float32, device=a.device)
    cur = torch.cuda.current_stream(a.device)
    side = _side(a.device)
    # fork point recorded BEFORE the GEMM, side branch issued AFTER it: the graph launches the GEMM node first
    # (issuing the side node first delays the GEMM launch in a replay)
    ev = torch.cuda.Event()
    ev.record(cur)
    qkvz, r = e6_cons_r(a, wq, ws, p, n, k, n_norm, eps)
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
    if w3:
        raise ValueError("accuracy path requires W4 weights")
    qkvz, r, tot = _cons_tot(a, wq, ws, p, n, k, n_norm, eps, wba)
    if tot is None:
        ba = fast.e6_ba(a, wba, r)
        r1, tot = torch.ones((a.shape[0],), dtype=torch.float32, device=a.device), ba.float()
    else:
        r1 = r.contiguous()
    return qkvz, r1, tot

def _consume(layer, positions, a, p, eps):
    """attention / GDN part of a layer from the prepared input (a, p); returns the input of o_proj/out_proj."""
    ops = torch.ops.vllm
    if layer.layer_type == "linear_attention":
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import _encode_layer_name
        g = layer.linear_attn
        wq, ws, n, k = _w(g.in_proj_qkvz)
        shape = (a.shape[0], g.num_v_heads // g.tp_size, g.head_v_dim)
        # no zero fill: the core op writes every real row; padded rows stay row-local through the bf16 path
        if fast._FOLD_OK["bap"]:
            core = torch.empty(shape, dtype=a.dtype, device=a.device)
            qkvz, r, tot = ops.atdev_e6_cons_bat(a, wq, ws, p, n, k, H, eps, g.in_proj_ba.weight, False)
            ops.atdev_gdn_core_bap(qkvz, tot, r, core, layer_name=_encode_layer_name(g.prefix))
            return core.flatten(-2), g.out_proj
        qkvz, r = ops.atdev_e6_cons_r(a, wq, ws, p, n, k, H, eps)
        core = torch.empty(shape, dtype=a.dtype, device=a.device)
        ba = ops.atdev_e6_ba(a, g.in_proj_ba.weight, r)
        ops.qwen_gdn_attention_core_fused_norm_packed(qkvz, ba, core, layer_name=_encode_layer_name(g.prefix))
        return core.flatten(-2), g.out_proj
    at = layer.self_attn
    wq, ws, n, k = _w(at.qkv_proj)
    qkv, _ = ops.atdev_e6_cons(a, wq, ws, p, n, k, H, eps, False)
    q, kk, v, gate = at._project_qkv_gate(qkv, positions)
    if gate is not None and fast._FOLD_OK["gate"]:
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
    a2, res2, p2 = ops.atdev_e6_prod(x, wq, ws, residual, ln2.weight, n, k)
    gu, dn = layer.mlp.gate_up_proj, layer.mlp.down_proj
    wq, ws, n, k = _w(gu)
    y, _ = ops.atdev_e6_cons(a2, wq, ws, p2, n, k, H, ln2.variance_epsilon, True)
    wq, ws, n, k = _w(dn)
    if nxt is None:
        return ops.atdev_mp_linear(y, wq, ws, ws, n, k, False), res2
    return ops.atdev_e6_prod(y, wq, ws, res2, nxt.input_layernorm.weight, n, k)

def _permute_gate_up(model):
    from vllm.model_executor.models import qwen3_next as qn
    global _NT
    _NT=H//_tn16(H,6144)
    count=0
    for layer in model.modules():
        if not isinstance(layer,qn.Qwen3NextDecoderLayer):continue
        gu=layer.mlp.gate_up_proj
        if fast._is_w3(gu):raise ValueError("accuracy profile cannot interleave W3 weights")
        if _kern(gu) is None or getattr(gu,"_atdev_gu_perm",False):continue
        wq,ws,n,k=_w(gu)
        cfg=cn.mp_cfg(n,k,16)
        if n%128 or cfg is None or cfg[1]!=128:continue
        q2,s2=cn.permute_gu_marlin(wq,ws,n)
        wq.data.copy_(q2);ws.data.copy_(s2);gu._atdev_gu_perm=True;count+=1
    return count


def register():
    global _REGISTERED
    import vllm
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture
    from vllm.config import compilation as vcc
    from vllm.model_executor.model_loader import base_loader
    from vllm.model_executor.models import qwen3_next as qn
    from vllm.utils.torch_utils import direct_register_custom_op
    from atdev_serve import compile_cache,tree_attn
    from atdev_serve.register import enabled
    need={"marlin_plus","dflash_tree","tree_attn"}
    if vllm.__version__!=_SUPPORTED or not need<=enabled() or enabled()&{"w3_mixed","fused_quant","c1_norm"} or os.environ.get("VLLM_MARLIN_INPUT_DTYPE")=="int8":
        raise RuntimeError("c1 accuracy needs the measured W4A16 DFlash/tree stack, without W3/fused_quant/int8")
    if _REGISTERED:return
    _REGISTERED=True
    with open(__file__,"rb") as source:
        compile_cache._SALTS[OPT_NAME]=hashlib.sha256(source.read()).hexdigest()[:16]
    for name,fn,fake in (("atdev_e6_prod",e6_prod,_e6_prod_fake),("atdev_e6_cons",e6_cons,_e6_cons_fake),("atdev_e6_cons_r",e6_cons_r,_e6_cons_r_fake),
        ("atdev_e6_ba",fast.e6_ba,fast._e6_ba_fake),("atdev_e6_first",fast.e6_first,fast._e6_first_fake),
        ("atdev_e6_cons_bat",e6_cons_bat,fast._e6_cons_bat_fake)):
        direct_register_custom_op(name,fn,fake_impl=fake)
    direct_register_custom_op("atdev_gdn_core_bap",eager_break_during_capture(fast.gdn_core_bap),mutates_args=["core"],fake_impl=fast._gdn_core_bap_fake)
    if "vllm::atdev_gdn_core_bap" not in vcc.CompilationConfig._attention_ops:vcc.CompilationConfig._attention_ops.append("vllm::atdev_gdn_core_bap")
    tree_attn._register_gated_op()
    original_load=base_loader.process_weights_after_loading
    def after_loading(model,model_config,target_device):
        original_load(model,model_config,target_device)
        for mode in (1,2,3):cn.mc1(mode)
        count=_permute_gate_up(model)
        log.warning("atdev c1 accuracy: interleaved %d W4 gate/up projections",count)
        fast._check_folds()
    base_loader.process_weights_after_loading=after_loading
    original_forward=qn.Qwen3NextModel.forward
    def forward(self,input_ids,positions,intermediate_tensors=None,inputs_embeds=None):
        if intermediate_tensors is not None or not _model_ok(self):
            if any(getattr(l.mlp.gate_up_proj,"_atdev_gu_perm",False) for l in self.layers if hasattr(l,"mlp") and hasattr(l.mlp,"gate_up_proj")):
                raise RuntimeError("c1 accuracy interleaved W4 weights require fused forward eligibility")
            return original_forward(self,input_ids,positions,intermediate_tensors,inputs_embeds)
        return model_forward(self,input_ids,positions,intermediate_tensors,inputs_embeds)
    qn.Qwen3NextModel.forward=forward
    log.warning("atdev c1 accuracy: measured W4 norm/SiLU fusions registered (no W3 overlay)")
