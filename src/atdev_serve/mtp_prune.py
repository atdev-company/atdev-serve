"""MTP first-pass row pruning (ATDEV_OPT=mtp_prune; c=32 MTP k>=2 stack).

The first MTP draft pass (AutoRegressiveSpeculator._prefill) runs the MTP decoder layer on every
verified token (M = 96 at c=32 MTP k=2: 32 requests x 3) because every position's K/V must be
written to the MTP attention cache. Only the last accepted row of each request is used afterwards
(`last_hidden_states[last_token_indices]` -> draft logits and the hidden state of the next pass),
so after attention everything is row-wasted work: o_proj, post-attention norm, gate_up, down and
the final norm run on 96 rows instead of 32 (stock W4A16 Marlin also splits M=96 into a 64-row
and a 32-row launch, each streaming the full weight). Measured: o 66 -> 24, gate_up 210 -> 82,
down 145 -> 53 us per step.

This module, for the first pass only:
* `_prefill` (patched) publishes `last_token_indices[:num_reqs]` (a persistent buffer, so the
  gather is graph-safe in the FULL prefill graph) and treats the model output as already gathered;
* `Qwen3_5MTP.__call__` (patched; bypasses the torch.compile'd forward) then runs an eager copy of the stock forward
  (pre-fc norms, fc, input norm, qkv, fused qk-norm/rope/gate, attention over all rows) and
  index_selects the attention output, gate and residual rows before o_proj; the rest runs on
  num_reqs rows.

The decode passes (M = num_reqs) keep the compiled stock path. Numerics: the per-row math is the
stock math; GEMM tiling / Inductor fusion differ, so the drafter hidden states may differ at
rounding level -> only the acceptance rate can change, never the generated text (the target
verifies every draft token). PIECEWISE (prefill-mixed) steps and unusual configs (PCP, multiple
MTP layers, layer_scale, sequence-parallel MoE) fall back to stock.
"""
from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

OPT_NAME = "mtp_prune"
SUPPORTED_VLLM_VERSION = "0.31.0"
log = logging.getLogger("atdev_serve")


class _State:
    rows: torch.Tensor | None = None
    hits: int = 0


S = _State()


# ---------------------------------------------------------------------------------------------
# small Triton kernels for the eager pruned path (the stock path gets these fused by Inductor)
# Gemma RMSNorm: y = bf16(x * rsqrt(mean(x^2) + eps) * (1 + w)), x in fp32 (incl. the residual add)
# ---------------------------------------------------------------------------------------------

@triton.jit
def _pre_fc_kernel(e_ptr, se, h_ptr, sh, we_ptr, wh_ptr, o_ptr, so, H, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < H
    e = tl.load(e_ptr + row * se + c, mask=m, other=0.0).to(tl.float32)
    h = tl.load(h_ptr + row * sh + c, mask=m, other=0.0).to(tl.float32)
    we = tl.load(we_ptr + c, mask=m, other=0.0).to(tl.float32) + 1.0
    wh = tl.load(wh_ptr + c, mask=m, other=0.0).to(tl.float32) + 1.0
    re = tl.rsqrt(tl.sum(e * e, axis=0) / H + eps)
    rh = tl.rsqrt(tl.sum(h * h, axis=0) / H + eps)
    tl.store(o_ptr + row * so + c, (e * re * we).to(tl.bfloat16), mask=m)
    tl.store(o_ptr + row * so + H + c, (h * rh * wh).to(tl.bfloat16), mask=m)


@triton.jit
def _norm_kernel(x_ptr, sx, r_ptr, sr, rows_ptr, w_ptr, o_ptr, so, ro_ptr, sro, H, eps,
                 HAS_RES: tl.constexpr, GATHER: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < H
    x = tl.load(x_ptr + row * sx + c, mask=m, other=0.0).to(tl.float32)
    if HAS_RES:
        rr = row
        if GATHER:
            rr = tl.load(rows_ptr + row).to(tl.int64)
        x = x + tl.load(r_ptr + rr * sr + c, mask=m, other=0.0).to(tl.float32)
        tl.store(ro_ptr + row * sro + c, x.to(tl.bfloat16), mask=m)
    w = tl.load(w_ptr + c, mask=m, other=0.0).to(tl.float32) + 1.0
    r = tl.rsqrt(tl.sum(x * x, axis=0) / H + eps)
    tl.store(o_ptr + row * so + c, (x * r * w).to(tl.bfloat16), mask=m)


@triton.jit
def _gate_gather_kernel(a_ptr, sa, g_ptr, sg, rows_ptr, o_ptr, so, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < N
    rr = tl.load(rows_ptr + row).to(tl.int64)
    a = tl.load(a_ptr + rr * sa + c, mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + rr * sg + c, mask=m, other=0.0).to(tl.float32)
    sg_ = tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)
    tl.store(o_ptr + row * so + c, (a * sg_).to(tl.bfloat16), mask=m)


@triton.jit
def _silu_mul_kernel(x_ptr, sx, o_ptr, so, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < N
    g = tl.load(x_ptr + row * sx + c, mask=m, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + row * sx + N + c, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + row * so + c, (g * tl.sigmoid(g) * u).to(tl.bfloat16), mask=m)


def _ok2(*ts) -> bool:
    return all(t.dim() == 2 and t.stride(-1) == 1 and t.dtype == torch.bfloat16 for t in ts)


def pre_fc(e, h, ne, nh):
    """cat([ne(e), nh(h)], -1)"""
    if not _ok2(e, h):
        return torch.cat([ne(e), nh(h)], dim=-1)
    m, H = e.shape
    out = torch.empty((m, 2 * H), dtype=e.dtype, device=e.device)
    if m:
        _pre_fc_kernel[(m,)](e, e.stride(0), h, h.stride(0), ne.weight, nh.weight, out, out.stride(0),
                             H, ne.variance_epsilon, BLOCK=triton.next_power_of_2(H), num_warps=8)
    return out


def norm(mod, x, residual=None, rows=None):
    """mod(x) / mod(x, residual[rows]) -> y or (y, new residual)"""
    if not (_ok2(x) and (residual is None or _ok2(residual))):
        if residual is None:
            return mod(x)
        return mod(x, residual if rows is None else residual.index_select(0, rows))
    m, H = x.shape
    y = torch.empty_like(x)
    has_res = residual is not None
    ro = torch.empty_like(x) if has_res else y
    r = residual if has_res else x
    if m:
        _norm_kernel[(m,)](x, x.stride(0), r, r.stride(0), rows if rows is not None else x,
                           mod.weight, y, y.stride(0), ro, ro.stride(0), H, mod.variance_epsilon,
                           HAS_RES=has_res, GATHER=rows is not None,
                           BLOCK=triton.next_power_of_2(H), num_warps=8)
    return (y, ro) if has_res else y


def gate_gather(o, gate, rows):
    """o[rows] * sigmoid(gate[rows]) (gate None -> o[rows])"""
    if gate is None:
        return o.index_select(0, rows)
    if not (_ok2(o, gate) and o.shape == gate.shape):
        return o.index_select(0, rows) * torch.sigmoid(gate.index_select(0, rows))
    n = o.shape[1]
    out = torch.empty((rows.shape[0], n), dtype=o.dtype, device=o.device)
    if rows.shape[0]:
        _gate_gather_kernel[(rows.shape[0], triton.cdiv(n, 1024))](
            o, o.stride(0), gate, gate.stride(0), rows, out, out.stride(0), n, BLOCK=1024,
            num_warps=4)
    return out


def silu_mul(mlp, gu):
    if not _ok2(gu):
        return mlp.act_fn(gu)
    m, n2 = gu.shape
    n = n2 // 2
    out = torch.empty((m, n), dtype=gu.dtype, device=gu.device)
    if m:
        _silu_mul_kernel[(m, triton.cdiv(n, 2048))](gu, gu.stride(0), out, out.stride(0), n,
                                                    BLOCK=2048, num_warps=4)
    return out


def layer_ok(model) -> bool:
    """model = Qwen3_5MultiTokenPredictor"""
    if getattr(model, "num_mtp_layers", 1) != 1 or len(model.layers) != 1:
        return False
    layer = model.layers[0]
    return (getattr(layer, "layer_type", "") == "full_attention"
            and not getattr(layer, "layer_scale", False)
            and not getattr(layer, "use_attn_reduce_scatter_for_moe", False)
            and hasattr(layer, "self_attn")
            and hasattr(getattr(layer, "mlp", None), "gate_up_proj")
            and hasattr(layer.mlp, "down_proj"))


def pruned_forward(model, input_ids, positions, hidden_states, inputs_embeds, rows):
    """Eager stock Qwen3_5MultiTokenPredictor.forward whose post-attention part runs on `rows`."""
    if inputs_embeds is None:
        inputs_embeds = model.embed_input_ids(input_ids)
    hidden_states = model.fc(pre_fc(inputs_embeds, hidden_states, model.pre_fc_norm_embedding,
                                    model.pre_fc_norm_hidden))
    layer = model.layers[0]
    residual = hidden_states
    x = norm(layer.input_layernorm, hidden_states)
    att = layer.self_attn
    qkv, _ = att.qkv_proj(x)
    q, k, v, gate = att._project_qkv_gate(qkv, positions)
    o = att.attn(q, k, v)
    o, _ = att.o_proj(gate_gather(o, gate, rows))
    x, residual = norm(layer.post_attention_layernorm, o, residual, rows)
    mlp = layer.mlp
    gu, _ = mlp.gate_up_proj(x)
    x, _ = mlp.down_proj(silu_mul(mlp, gu))
    x, _ = norm(model.norm, x, residual)
    return x


def _install() -> None:
    from vllm.config.compilation import CUDAGraphMode
    from vllm.model_executor.models import qwen3_5_mtp
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

    cls = qwen3_5_mtp.Qwen3_5MTP
    if getattr(cls, "_atdev_mtp_prune", False):
        return
    # Qwen3_5MTP itself is the torch.compile'd module (support_torch_compile wraps __call__): the
    # pruned call must bypass the compiled artifact, the stock call must not change at all.
    o_call = cls.__call__

    def __call__(self, *args, **kwargs):
        rows = S.rows
        if rows is None:
            return o_call(self, *args, **kwargs)
        S.rows = None  # consumed (one model call per _prefill)
        if args or kwargs.get("intermediate_tensors") is not None:
            raise RuntimeError("atdev mtp_prune: unexpected MTP call signature")
        return pruned_forward(self.model, kwargs["input_ids"], kwargs["positions"],
                              kwargs["hidden_states"], kwargs.get("inputs_embeds"), rows)

    cls.__call__ = __call__
    cls._atdev_mtp_prune = True

    o_prefill = MTPSpeculator._prefill

    def _prefill(self, num_reqs, num_tokens, attn_metadata, slot_mappings,
                 num_tokens_across_dp, cudagraph_runtime_mode=CUDAGraphMode.NONE,
                 mm_inputs=None):
        ok = getattr(self, "_atdev_prune_ok", None)
        if ok is None:
            ok = self._atdev_prune_ok = (
                isinstance(self.model, cls) and self.pcp_manager is None
                and not self.supports_mm_inputs and layer_ok(self.model.model))
            log.warning("atdev %s: first-pass row pruning %s", OPT_NAME,
                        "ENABLED" if ok else "unsupported here -> stock")
        if not ok or cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return o_prefill(self, num_reqs, num_tokens, attn_metadata, slot_mappings,
                             num_tokens_across_dp, cudagraph_runtime_mode, mm_inputs)
        # == stock _prefill with `last_hidden_states` already gathered by the model ==
        last_token_indices = self.last_token_indices[:num_reqs]
        positions = self.input_buffers.positions[last_token_indices]
        sample_src_positions = positions + 1
        idx_mapping = self.idx_mapping[:num_reqs]
        S.rows = last_token_indices
        try:
            last_hidden_states, _ = self._run_model(
                num_tokens, attn_metadata, slot_mappings,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode, mm_inputs=mm_inputs)
        finally:
            S.rows = None
        assert last_hidden_states.shape[0] == num_reqs, last_hidden_states.shape
        S.hits += 1
        self.draft_tokens[:num_reqs, 0] = self.sample_draft(
            last_hidden_states, sample_src_positions, idx_mapping, self.temperature,
            self.seeds, self.current_draft_step, self.draft_logits)
        self.hidden_states[:num_reqs] = last_hidden_states
        self.input_buffers.positions[:num_reqs] = positions
        self.sample_src_positions[:num_reqs] = sample_src_positions

    MTPSpeculator._prefill = _prefill


def register() -> None:
    import vllm

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written for vLLM {SUPPORTED_VLLM_VERSION}, "
                           f"found {vllm.__version__}")
    _install()
    log.info("atdev %s: registered", OPT_NAME)
