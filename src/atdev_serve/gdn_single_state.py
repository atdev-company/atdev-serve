"""Single-state GDN spec-decode verify (ATDEV_OPT=gdn_single_state).

Stock vLLM 0.31 (Qwen3.5/3.8 GDN layers, spec decode, mamba_cache_mode
"none") gives each request ``S = num_speculative_tokens + 1`` recurrent-state
slots (``spec_state_indices_tensor[n, 0..S-1]``). The verify kernel
(``fused_gdn_decode_post_conv_mtp`` CUDA, or FLA's Triton
``fused_sigmoid_gating_delta_rule_update``) reads slot ``a - 1`` (``a`` =
tokens accepted last step, >= 1) and writes the full fp32 state after *every*
verify token t into slot t, so rollback is free: next step simply reads slot
``a' - 1``. Per GDN layer and request that is 1 read + (k+1) writes of a
48 x 128 x 128 fp32 state (3.1 MB each): 1.36 GB per step at k = 7.

This module keeps exactly one state write per step:

* slot ``s0 = indices[n, 0]`` holds the state *after token 0 of the previous
  verify chunk*. That is the same content stock leaves in slot 0, so every
  stock code path that reads slot 0 (prefill with initial state, non-spec
  decode, conv) sees what it sees today.
* The ssm region of slot ``s1 = indices[n, 1]`` (unused otherwise) holds a
  small *token log*: for verify tokens 1..T-1 the l2-normalised key, the
  value, the decay ``exp(g)`` and ``beta`` exactly as the kernel used them
  (fp32, ~32 KB per token per layer for Qwen3.8), double-buffered, plus a
  private parity word per CTA.
* Verify kernel, per (request, value head, V-row block):
    1. load the slot-0 state,
    2. *replay* the ``a - 1`` accepted tokens 1..a-1 of the previous chunk
       from the log (rank-1 delta-rule updates; identical arithmetic to the
       step that first processed them, so the state is the one stock would
       read from slot a-1),
    3. process the T current tokens, emitting outputs; store the state after
       token 0 into slot 0 (the single state write); log tokens 1..T-1 into
       the other log buffer; flip the parity.
  After a prefill / non-spec step vLLM's ``num_accepted_tokens`` is 1, so
  nothing is replayed and the parity is reset: no extra bookkeeping hooks.

Bytes per GDN layer per request: 1 state read + 1 state write + log
(<= 2 x 0.23 MB) instead of 1 read + (k+1) writes. The grid is
(requests, 48 heads, 128 / BV) = 192 CTAs at c=1 with BV=32 (stock CUDA:
48), and k+1 is not capped at 8.

Race freedom: the log is written into buffer ``1 - p`` while this launch reads
buffer ``p``; ``p`` is per-CTA private state that all CTAs of a request flip
in lockstep (each launch runs all of them) and reset to 0 whenever nothing is
replayed, so they always agree. Keys are shared by the GQA group and by all
V-row blocks; one CTA per key head writes them, others only read the other
buffer. Values are per-CTA rows.

Integration (no vLLM source edits):
* ``QwenGatedDeltaNetAttention`` is a ``PluggableLayer``; ``register()``
  installs ``SingleStateGDNAttention`` via ``register_oot``. It overrides
  ``_can_use_fused_gdn_mtp_decode`` (pure spec-decode batches: any k, fp32
  or bf16 state) and ``_forward_core_decode_spec_post_conv_fused_norm``
  (our kernel instead of the CUDA op; conv stays stock).
* Mixed batches (spec rows + prefills) go through ``_forward_core``, which
  calls the module-global ``fused_sigmoid_gating_delta_rule_update`` for the
  spec rows. ``register()`` swaps that global in
  ``vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn`` for a
  dispatcher that sends spec-verify calls to our kernel (every other call is
  forwarded unchanged), so every spec computation of every GDN layer uses the
  single-state slot semantics. Asserts ``vllm.__version__ == "0.31.0"``.

Requirements: ``mamba_cache_mode == "none"`` (``--no-enable-prefix-caching``):
the "align" pre-copy kernel copies the state from spec slot ``a - 1``, which
is no longer materialised. Checked at layer construction (raises). K, V,
BV powers of two; num_v_heads % num_k_heads == 0.

Not done (needs MambaSpec / metadata-builder changes): shrinking the
allocation from k+1 to 2 slots per request. Bandwidth, not capacity, is the
c=1 bottleneck; at c=32 with k=7 it would free ~32 x 6 x 151 MB.

Env knobs:
  ATDEV_GDN_BV   V rows per CTA (default 32; 128 = one CTA per head with the
                 gated RMSNorm fused into the kernel, otherwise vLLM's
                 gated-norm kernel runs after ours, like stock's Triton path)
"""

from __future__ import annotations

import functools
import os

import torch

from vllm.logger import init_logger

try:  # Triton is needed on CUDA (or under TRITON_INTERPRET=1 in tests).
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

logger = init_logger(__name__)

OPT_NAME = "gdn_single_state"
SUPPORTED_VLLM_VERSION = "0.31.0"
L2NORM_EPS = 1e-6
# ATDEV_GDN_STATE=int8: tag word marking an int8-coded CTA region of slot 0.
# As two bf16 halves it is (valid number, NaN) and as fp32 a NaN, so a real
# native state never contains it; any stock write of the slot (prefill,
# non-spec decode) overwrites it, which makes the region native again.
QSTATE_MAGIC = 0x7FC1A3B7


# ---------------------------------------------------------------------------
# Log layout inside the ssm region of slot s1 (offsets in fp32 elements)
# ---------------------------------------------------------------------------


def log_layout(h: int, hv: int, k: int, v: int, bv: int, max_log: int) -> dict:
    """Parity words, then two buffers of [keys | values | decays | betas]."""
    nv = v // bv
    log_base = -(-hv * nv // 32) * 32
    buf = max_log * (h * k + hv * v + 2 * hv)
    return {
        "nv": nv,
        "log_base": log_base,
        "buf": buf,
        "k_off": 0,
        "v_off": max_log * h * k,
        "d_off": max_log * (h * k + hv * v),
        "b_off": max_log * (h * k + hv * v) + max_log * hv,
        "total": log_base + 2 * buf,
    }


def slot_capacity_f32(state: torch.Tensor) -> int:
    """fp32 elements available per slot in the ssm state [slots, HV, V, K]."""
    return state[0].numel() * state.element_size() // 4


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if triton is not None:
    QMAGIC = tl.constexpr(QSTATE_MAGIC)

    @triton.jit
    def _gdn_rank1_step(h, k, v, decay, beta):
        # Same op order as FLA's fused_sigmoid_gating_delta_rule_update and
        # vLLM's fused CUDA MTP kernel: decay, delta = (v - h k) * beta,
        # h += delta k^T. Used by both the replay and the live loop so a
        # replayed state is computed by exactly the same instructions.
        h = h * decay
        delta = (v - tl.sum(h * k[None, :], axis=1)) * beta
        h = h + delta[:, None] * k[None, :]
        return h

    @triton.jit
    def _gdn_single_state_kernel(
        q_ptr, k_ptr, v_ptr, stride_q, stride_k, stride_v,
        a_ptr, b_ptr, stride_a, stride_b,
        A_log_ptr, dt_bias_ptr,
        state_ptr, log_ptr, par_ptr,
        slot_state, slot_log, slot_par,
        si_ptr, stride_si, cu_ptr, acc_ptr,
        o_ptr, stride_o,
        gate_ptr, stride_gate, nw_ptr,
        scale, sp_beta, sp_threshold, norm_eps,
        H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BV: tl.constexpr, MAX_LOG: tl.constexpr,
        LOG_BASE: tl.constexpr, BUF: tl.constexpr,
        V_OFF: tl.constexpr, D_OFF: tl.constexpr, B_OFF: tl.constexpr,
        FUSE_NORM: tl.constexpr, SIGMOID_GATE: tl.constexpr,
        QSTATE: tl.constexpr = False, NG: tl.constexpr = 1, ES: tl.constexpr = 2,
        st8_ptr=None, slot_bytes=0,
    ):
        n = tl.program_id(0)
        hv = tl.program_id(1)
        iv = tl.program_id(2)
        NV: tl.constexpr = V // BV
        G: tl.constexpr = HV // H
        hk = hv // G

        bos = tl.load(cu_ptr + n).to(tl.int64)
        eos = tl.load(cu_ptr + n + 1).to(tl.int64)
        T = eos - bos
        if T <= 0:
            return
        offs_v = iv * BV + tl.arange(0, BV)
        offs_k = tl.arange(0, K)
        s0 = tl.load(si_ptr + n * stride_si).to(tl.int64)
        s1 = tl.load(si_ptr + n * stride_si + 1).to(tl.int64)
        if (s0 <= 0) | (s1 <= 0) | (T > MAX_LOG + 1):
            # Padded / NULL rows: zero output, like the stock kernels.
            zero = tl.zeros([BV], dtype=tl.float32)
            for t in range(0, T):
                tl.store(o_ptr + (bos + t) * stride_o + hv * V + offs_v,
                         zero.to(o_ptr.dtype.element_ty))
            return

        n_acc = tl.load(acc_ptr + n)
        n_rep = tl.minimum(tl.maximum(n_acc - 1, 0), MAX_LOG)
        par_addr = par_ptr + s1 * slot_par + hv * NV + iv
        p = tl.load(par_addr) & 1
        p = tl.where(n_rep > 0, p, 0)
        lbase = log_ptr + s1 * slot_log + LOG_BASE
        rbuf = lbase + p * BUF
        wbuf = lbase + (1 - p) * BUF

        h_ptrs = (state_ptr + s0 * slot_state + hv * V * K
                  + offs_v[:, None] * K + offs_k[None, :])
        if QSTATE:
            # int8 region of this CTA inside its own native rows (see
            # QSTATE_MAGIC): [BV*K codes][BV*NG fp32 scales][tag]
            rb = s0 * slot_bytes + (hv * V + iv * BV) * K * ES
            offs_r = tl.arange(0, BV)
            c_ptrs = st8_ptr + rb + offs_r[:, None] * K + offs_k[None, :]
            sc_ptrs = log_ptr + (rb + BV * K) // 4 + offs_r[:, None] * NG + offs_k[None, :] // (K // NG)
            tag_ptr = par_ptr + (rb + BV * K + BV * NG * 4) // 4
            if tl.load(tag_ptr) == QMAGIC:
                h = tl.load(c_ptrs).to(tl.float32) * tl.load(sc_ptrs)
            else:  # fresh native state (prefill / non-spec wrote the slot)
                h = tl.load(h_ptrs).to(tl.float32)
        else:
            h = tl.load(h_ptrs).to(tl.float32)

        # 1) replay the accepted tail of the previous chunk
        for t in range(0, n_rep):
            kk = tl.load(rbuf + (t * H + hk) * K + offs_k)
            vv = tl.load(rbuf + V_OFF + (t * HV + hv) * V + offs_v)
            dd = tl.load(rbuf + D_OFF + t * HV + hv)
            bb = tl.load(rbuf + B_OFF + t * HV + hv)
            h = _gdn_rank1_step(h, kk, vv, dd, bb)

        # 2) current chunk
        A = tl.exp(tl.load(A_log_ptr + hv).to(tl.float32))
        dt_b = tl.load(dt_bias_ptr + hv).to(tl.float32)
        for t in range(0, T):
            tok = bos + t
            q = tl.load(q_ptr + tok * stride_q + hk * K + offs_k).to(tl.float32)
            k = tl.load(k_ptr + tok * stride_k + hk * K + offs_k).to(tl.float32)
            v = tl.load(v_ptr + tok * stride_v + hv * V + offs_v).to(tl.float32)
            q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6)
            k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
            q = q * scale
            x = tl.load(a_ptr + tok * stride_a + hv).to(tl.float32) + dt_b
            sp = tl.where(sp_beta * x <= sp_threshold,
                          (1 / sp_beta) * tl.log(1 + tl.exp(sp_beta * x)), x)
            decay = tl.exp(-A * sp)
            beta = tl.sigmoid(tl.load(b_ptr + tok * stride_b + hv).to(tl.float32))
            h = _gdn_rank1_step(h, k, v, decay, beta)
            o = tl.sum(h * q[None, :], axis=1)
            o_dst = o_ptr + tok * stride_o + hv * V + offs_v
            if FUSE_NORM:
                ob = o.to(tl.bfloat16).to(tl.float32)
                rstd = tl.rsqrt(tl.sum(ob * ob, axis=0) / V + norm_eps)
                z = tl.load(gate_ptr + tok * stride_gate + hv * V + offs_v).to(tl.float32)
                if SIGMOID_GATE:
                    act = tl.sigmoid(z)
                else:
                    act = z * tl.sigmoid(z)
                w = tl.load(nw_ptr + offs_v).to(tl.float32)
                tl.store(o_dst, (ob * rstd * w * act).to(o_ptr.dtype.element_ty))
            else:
                tl.store(o_dst, o.to(o_ptr.dtype.element_ty))
            if t == 0:
                if QSTATE:
                    amax = tl.max(tl.abs(tl.reshape(h, (BV, NG, K // NG))), axis=2)
                    sc = tl.maximum(amax, 1e-30) * (1.0 / 127.0)
                    scf = tl.reshape(tl.broadcast_to(sc[:, :, None], (BV, NG, K // NG)), (BV, K))
                    qv = tl.floor(h / scf + 0.5)
                    qv = tl.minimum(tl.maximum(qv, -127.0), 127.0)
                    tl.debug_barrier()  # all native reads of the region done
                    tl.store(c_ptrs, qv.to(tl.int8))
                    offs_g = tl.arange(0, NG)
                    tl.store(log_ptr + (rb + BV * K) // 4 + offs_r[:, None] * NG + offs_g[None, :], sc)
                    tl.store(tag_ptr, QMAGIC)
                else:
                    tl.store(h_ptrs, h.to(state_ptr.dtype.element_ty))
            else:
                if (iv == 0) & (hv % G == 0):
                    tl.store(wbuf + ((t - 1) * H + hk) * K + offs_k, k)
                tl.store(wbuf + V_OFF + ((t - 1) * HV + hv) * V + offs_v, v)
                if iv == 0:
                    tl.store(wbuf + D_OFF + (t - 1) * HV + hv, decay)
                    tl.store(wbuf + B_OFF + (t - 1) * HV + hv, beta)
        tl.store(par_addr, 1 - p)


if triton is not None:

    @triton.jit
    def _gdn_untag_kernel(state_ptr, st8_ptr, f32_ptr, i32_ptr, idx_ptr, slot_bytes,
                          slot_state, V: tl.constexpr, K: tl.constexpr, BV: tl.constexpr,
                          NG: tl.constexpr, ES: tl.constexpr):
        # Convert int8-tagged slot regions back to the native dtype in place
        # (before a stock kernel reads the slot).
        n = tl.program_id(0)
        hv = tl.program_id(1)
        iv = tl.program_id(2)
        s0 = tl.load(idx_ptr + n).to(tl.int64)
        if s0 <= 0:
            return
        offs_r = tl.arange(0, BV)
        offs_k = tl.arange(0, K)
        rb = s0 * slot_bytes + (hv * V + iv * BV) * K * ES
        tag_ptr = i32_ptr + (rb + BV * K + BV * NG * 4) // 4
        if tl.load(tag_ptr) == QMAGIC:
            h = (tl.load(st8_ptr + rb + offs_r[:, None] * K + offs_k[None, :]).to(tl.float32)
                 * tl.load(f32_ptr + (rb + BV * K) // 4 + offs_r[:, None] * NG
                           + offs_k[None, :] // (K // NG)))
            tl.debug_barrier()
            tl.store(state_ptr + s0 * slot_state + (hv * V + iv * BV) * K
                     + offs_r[:, None] * K + offs_k[None, :],
                     h.to(state_ptr.dtype.element_ty))


def gdn_untag_slots(state: torch.Tensor, slot_ids: torch.Tensor, block_v: int,
                    ngroups: int = 1) -> None:
    """In-place: int8-tagged regions of ``state[slot_ids]`` -> native dtype."""
    if slot_ids.numel() == 0:
        return
    slots, hv, vd, kd = state.shape
    es = state.element_size()
    _gdn_untag_kernel[(slot_ids.numel(), hv, vd // block_v)](
        state, state.view(torch.int8), state.view(torch.float32), state.view(torch.int32),
        slot_ids, state.stride(0) * es, state.stride(0), V=vd, K=kd, BV=block_v,
        NG=ngroups, ES=es, num_warps=1)


def read_slot_state(state: torch.Tensor, s0: int, block_v: int, ngroups: int = 1):
    """Decode slot ``s0`` ([HV, V, K] fp32), int8-tagged regions or native."""
    slots, hv, vd, kd = state.shape
    es = state.element_size()
    slot_bytes = state.stride(0) * es
    base_all = torch.as_strided(state.view(torch.int8), (slot_bytes,), (1,),
                                state.view(torch.int8).storage_offset() + s0 * slot_bytes)
    out = state[s0].float().clone()
    for h_ in range(hv):
        for iv in range(vd // block_v):
            rb = (h_ * vd + iv * block_v) * kd * es
            reg = base_all[rb: rb + block_v * kd * es]
            tag = reg[block_v * kd + block_v * ngroups * 4:][:4].view(torch.int32)
            if int(tag[0]) == QSTATE_MAGIC:
                codes = reg[: block_v * kd].view(block_v, kd).float()
                sc = reg[block_v * kd: block_v * kd + block_v * ngroups * 4].view(
                    torch.float32).view(block_v, ngroups)
                out[h_, iv * block_v:(iv + 1) * block_v] = (
                    codes * sc.repeat_interleave(kd // ngroups, 1))
    return out


def ref_quantize_rows(h: torch.Tensor, ngroups: int = 1) -> torch.Tensor:
    """What the int8 store + load does to an fp32 [..., V, K] state."""
    *lead, vd, kd = h.shape
    hg = h.float().reshape(*lead, vd, ngroups, kd // ngroups)
    sc = hg.abs().amax(-1, keepdim=True).clamp(min=1e-30) * (1.0 / 127.0)
    q = torch.floor(hg / sc + 0.5).clamp(-127, 127)
    return (q * sc).reshape(*lead, vd, kd)


def _rows2d(x: torch.Tensor, width: int) -> torch.Tensor:
    """View a [L, ..., width-total] tensor as rows of `width` contiguous cols."""
    x2 = x.reshape(x.shape[0], -1) if x.dim() > 2 else x
    assert x2.shape[1] == width and x2.stride(1) == 1, (x2.shape, x2.stride())
    return x2


def gdn_single_state_verify(
    *,
    q: torch.Tensor,  # [L, H*K] (row-strided view ok)
    k: torch.Tensor,  # [L, H*K]
    v: torch.Tensor,  # [L, HV*V]
    a: torch.Tensor,  # [L, HV]
    b: torch.Tensor,  # [L, HV]
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,  # [slots, HV, V, K] fp32 or bf16
    state_indices: torch.Tensor,  # [N, S] int32, S >= 2
    cu_seqlens: torch.Tensor,  # [N + 1] int32
    num_accepted: torch.Tensor,  # [N] int32
    out: torch.Tensor,  # [L, HV, V] bf16
    num_k_heads: int,
    scale: float,
    output_gate: torch.Tensor | None = None,  # [L, HV, V] (fused norm only)
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    sigmoid_gate: bool = False,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    block_v: int = 32,
    fuse_norm: bool = False,
    num_requests: int | None = None,
    num_warps: int = 0,
    qstate: bool = False,
    qgroups: int = 1,
) -> None:
    slots, hv, vdim, kdim = state.shape
    h = num_k_heads
    assert state.dtype in (torch.float32, torch.bfloat16)
    # Slots may be strided (vLLM binds them as views into paged int8 KV
    # pages: stride(0) = page size); each slot's [HV, V, K] must be dense.
    assert state.stride()[1:] == (vdim * kdim, kdim, 1), state.stride()
    assert state_indices.dim() == 2 and state_indices.size(1) >= 2
    assert hv % h == 0 and vdim % block_v == 0
    max_log = state_indices.size(1) - 1
    lay = log_layout(h, hv, kdim, vdim, block_v, max_log)
    cap = slot_capacity_f32(state)
    if lay["total"] > cap:
        raise ValueError(f"gdn_single_state: log needs {lay['total']} fp32 per "
                         f"slot, slot has {cap}")
    n = state_indices.size(0) if num_requests is None else num_requests
    if n == 0:
        return
    fuse = fuse_norm
    if fuse:
        # the RMSNorm spans the whole head: one CTA must own all V rows
        assert block_v == vdim, "fuse_norm needs block_v == head_v_dim"
        assert output_gate is not None and norm_weight is not None
    q2, k2, v2 = _rows2d(q, h * kdim), _rows2d(k, h * kdim), _rows2d(v, hv * vdim)
    assert out.stride(-1) == 1 and out.stride(-2) == vdim
    gate = output_gate if fuse else out
    assert gate.stride(-1) == 1 and gate.stride(-2) == vdim
    nw = norm_weight if fuse else A_log
    log_view = state.view(torch.float32)
    par_view = state.view(torch.int32)
    _gdn_single_state_kernel[(n, hv, vdim // block_v)](
        q2, k2, v2, q2.stride(0), k2.stride(0), v2.stride(0),
        a, b, a.stride(0), b.stride(0),
        A_log, dt_bias,
        state, log_view, par_view,
        state.stride(0), log_view.stride(0), par_view.stride(0),
        state_indices, state_indices.stride(0), cu_seqlens, num_accepted,
        out, out.stride(0),
        gate, gate.stride(0), nw,
        float(scale), float(softplus_beta), float(softplus_threshold), float(norm_eps),
        H=h, HV=hv, K=kdim, V=vdim, BV=block_v, MAX_LOG=max_log,
        LOG_BASE=lay["log_base"], BUF=lay["buf"],
        V_OFF=lay["v_off"], D_OFF=lay["d_off"], B_OFF=lay["b_off"],
        FUSE_NORM=fuse, SIGMOID_GATE=sigmoid_gate,
        QSTATE=qstate, NG=qgroups, ES=state.element_size(),
        st8_ptr=state.view(torch.int8) if qstate else None,
        slot_bytes=state.stride(0) * state.element_size(),
        num_warps=num_warps or int(os.environ.get("ATDEV_GDN_WARPS", "0"))
        or (8 if block_v >= 64 else 4),
    )


# ---------------------------------------------------------------------------
# PyTorch references (CPU tests; numerically careful: same op order)
# ---------------------------------------------------------------------------


def ref_token_params(q, k, a, b, A_log, dt_bias, scale, sp_beta=1.0, sp_thr=20.0):
    """Per-token q, k (l2-normed), decay, beta in the computation dtype of q."""
    dt = q.dtype
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + L2NORM_EPS)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + L2NORM_EPS)
    q = q * scale
    x = a.to(dt) + dt_bias.to(dt)
    sp = torch.where(sp_beta * x <= sp_thr, torch.log1p(torch.exp(sp_beta * x)) / sp_beta, x)
    decay = torch.exp(-torch.exp(A_log.to(dt)) * sp)
    beta = torch.sigmoid(b.to(dt))
    return q, k, decay, beta


def ref_rank1_step(h, k, v, decay, beta):
    """h [..., V, K]; k [..., K]; v [..., V]; decay, beta [...]."""
    h = h * decay[..., None, None]
    delta = (v - (h * k[..., None, :]).sum(-1)) * beta[..., None]
    return h + delta[..., :, None] * k[..., None, :]


def _split_qkv(qkv, h, hv, kd, vd):
    L = qkv.shape[0]
    q = qkv[:, : h * kd].reshape(L, h, kd)
    k = qkv[:, h * kd: 2 * h * kd].reshape(L, h, kd)
    v = qkv[:, 2 * h * kd:].reshape(L, hv, vd)
    return q, k, v


def ref_stock_verify(qkv, a, b, A_log, dt_bias, state, state_indices, cu_seqlens,
                     num_accepted, num_k_heads, scale, dtype=torch.float64):
    """Stock semantics: read slot a-1, write the state after every token t
    into slot t. Returns o [L, HV, V] (``dtype``); mutates ``state``."""
    slots, hv, vd, kd = state.shape
    h = num_k_heads
    g = hv // h
    q_all, k_all, v_all = _split_qkv(qkv.to(dtype), h, hv, kd, vd)
    o = torch.zeros(qkv.shape[0], hv, vd, dtype=dtype)
    for n in range(state_indices.shape[0]):
        bos, eos = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        if eos <= bos:
            continue
        idx = state_indices[n].tolist()
        acc = int(num_accepted[n])
        if not (0 < acc <= len(idx)) or idx[acc - 1] <= 0:
            continue
        hstate = state[idx[acc - 1]].to(dtype)
        for t in range(eos - bos):
            tok = bos + t
            q, k, dec, bet = ref_token_params(
                q_all[tok], k_all[tok], a[tok], b[tok], A_log, dt_bias, scale)
            qh = q.repeat_interleave(g, 0)
            kh = k.repeat_interleave(g, 0)
            hstate = ref_rank1_step(hstate, kh, v_all[tok], dec, bet)
            o[tok] = (hstate * qh[:, None, :]).sum(-1)
            if idx[t] > 0:
                state[idx[t]] = hstate.to(state.dtype)
    return o


def ref_single_state_verify(qkv, a, b, A_log, dt_bias, state, state_indices,
                            cu_seqlens, num_accepted, num_k_heads, scale,
                            block_v, dtype=torch.float32):
    """Exactly the kernel's algorithm and memory layout, in torch.

    Returns o [L, HV, V] (``dtype``, un-normed); mutates ``state`` (slot 0
    state, log and parity words of slot 1).
    """
    slots, hv, vd, kd = state.shape
    h = num_k_heads
    g = hv // h
    max_log = state_indices.shape[1] - 1
    lay = log_layout(h, hv, kd, vd, block_v, max_log)
    if lay["total"] > slot_capacity_f32(state):
        raise ValueError("gdn_single_state: log does not fit in one state slot")
    nv = lay["nv"]
    log_v = state.view(torch.float32)
    par_v = state.view(torch.int32)
    q_all, k_all, v_all = _split_qkv(qkv.to(dtype), h, hv, kd, vd)
    o = torch.zeros(qkv.shape[0], hv, vd, dtype=dtype)
    kof, vof, dof, bof = lay["k_off"], lay["v_off"], lay["d_off"], lay["b_off"]

    def buf_view(s1, p):
        base = lay["log_base"] + p * lay["buf"]
        flat = log_v[s1].reshape(-1)[base: base + lay["buf"]]
        return (
            flat[kof:vof].view(max_log, h, kd),
            flat[vof:dof].view(max_log, hv, vd),
            flat[dof:bof].view(max_log, hv),
            flat[bof:].view(max_log, hv),
        )

    for n in range(state_indices.shape[0]):
        bos, eos = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        T = eos - bos
        if T <= 0:
            continue
        s0, s1 = int(state_indices[n, 0]), int(state_indices[n, 1])
        if s0 <= 0 or s1 <= 0 or T > max_log + 1:
            continue
        n_rep = min(max(int(num_accepted[n]) - 1, 0), max_log)
        # all CTAs of a request hold the same parity (see module docstring)
        par = par_v[s1].reshape(-1)  # a view: slot regions are dense
        p = int(par[0]) & 1 if n_rep > 0 else 0
        hstate = state[s0].to(dtype)  # [HV, V, K]
        rk, rv, rd, rb = buf_view(s1, p)
        for t in range(n_rep):
            kh = rk[t].to(dtype).repeat_interleave(g, 0)
            hstate = ref_rank1_step(hstate, kh, rv[t].to(dtype), rd[t].to(dtype),
                                    rb[t].to(dtype))
        wk, wv, wd, wb = buf_view(s1, 1 - p)
        for t in range(T):
            tok = bos + t
            q, k, dec, bet = ref_token_params(
                q_all[tok], k_all[tok], a[tok], b[tok], A_log, dt_bias, scale)
            qh = q.repeat_interleave(g, 0)
            kh = k.repeat_interleave(g, 0)
            hstate = ref_rank1_step(hstate, kh, v_all[tok], dec, bet)
            o[tok] = (hstate * qh[:, None, :]).sum(-1)
            if t == 0:
                state[s0] = hstate.to(state.dtype)
            else:
                wk[t - 1] = k.to(torch.float32)
                wv[t - 1] = v_all[tok].to(torch.float32)
                wd[t - 1] = dec.to(torch.float32)
                wb[t - 1] = bet.to(torch.float32)
        par[: hv * nv] = 1 - p
    return o


def ref_gated_rms_norm(o_bf16, gate, weight, eps, sigmoid_gate=False):
    x = o_bf16.float()
    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    z = gate.float()
    act = torch.sigmoid(z) if sigmoid_gate else z * torch.sigmoid(z)
    return (x * rstd * weight.float() * act).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# vLLM integration
# ---------------------------------------------------------------------------


def _impl() -> str:
    impl = os.environ.get("ATDEV_GDN_IMPL", "triton")
    if impl not in ("triton", "cuda"):
        raise ValueError(f"ATDEV_GDN_IMPL must be triton or cuda, got {impl!r}")
    return impl


def _block_v() -> int:
    # the CUDA kernel works on 32-row chunks; Triton calls (mixed batches)
    # must use the same int8 region / parity layout
    if _impl() == "cuda":
        return 32
    return int(os.environ.get("ATDEV_GDN_BV", "32"))


_EXT = None


def cuda_ext():
    """JIT-built csrc/gdn_ss.cu (cached in TORCH_EXTENSIONS_DIR)."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        src = os.path.join(os.path.dirname(__file__), "csrc", "gdn_ss.cu")
        name = "atdev_gdn_ss"
        alt = os.environ.get("ATDEV_GDN_SRC")  # P3: kernel variant (e.g. csrc/gdn_var/gdn_ss_v5.cu)
        if alt:
            import hashlib
            src = alt if os.path.isabs(alt) else os.path.join(os.path.dirname(__file__), alt)
            name = "atdev_gdn_ss_var_" + hashlib.sha256(open(src, "rb").read()).hexdigest()[:8]
        _EXT = load(name=name, sources=[src], verbose=False,
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80",
                                       "-std=c++17", "--expt-relaxed-constexpr"])
    return _EXT


def gdn_single_state_verify_cuda(*, mixed_qkv, a, b, A_log, dt_bias, state, state_indices,
                                 cu_seqlens, num_accepted, out, output_gate, norm_weight,
                                 num_k_heads, scale, norm_eps, sigmoid_gate, num_requests,
                                 qstate=False, qgroups=1, dbg=0):
    """CUDA single-state verify with fused gated RMSNorm (bf16 state, K=V=128)."""
    slots, hv, vdim, kdim = state.shape
    assert state.dtype == torch.bfloat16 and vdim == 128 and kdim == 128
    assert state.stride()[1:] == (vdim * kdim, kdim, 1)
    assert state_indices.size(1) <= 8, "k <= 7"
    assert output_gate.stride(-1) == 1 and output_gate.stride(-2) == vdim
    lay = log_layout(num_k_heads, hv, kdim, vdim, 32, state_indices.size(1) - 1)
    if lay["total"] > slot_capacity_f32(state):
        raise ValueError("gdn_single_state: log does not fit in one state slot")
    cuda_ext().gdn_ss(
        mixed_qkv, a, b, A_log, dt_bias, state, state_indices, cu_seqlens, num_accepted,
        output_gate, norm_weight, out, num_k_heads, num_requests, float(scale),
        float(norm_eps), bool(sigmoid_gate), bool(qstate), int(qgroups),
        lay["log_base"], lay["buf"], lay["v_off"], lay["d_off"], lay["b_off"], lay["nv"], int(dbg))


def _qstate() -> bool:
    mode = os.environ.get("ATDEV_GDN_STATE", "native")
    if mode not in ("native", "int8"):
        raise ValueError(f"ATDEV_GDN_STATE must be native or int8, got {mode!r}")
    return mode == "int8"


def _qgroups() -> int:
    return int(os.environ.get("ATDEV_GDN_QGROUPS", "1"))


def _make_layer_class():
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    class SingleStateGDNAttention(QwenGatedDeltaNetAttention):
        """GDN layer whose spec-decode verify writes one state per step."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            mode = self.cache_config.mamba_cache_mode
            if self.num_spec > 0 and mode != "none":
                raise ValueError(
                    f"atdev {OPT_NAME}: mamba_cache_mode={mode!r} is not "
                    "supported (the align pre-copy reads spec slot a-1, which "
                    "this optimization no longer writes). Run with "
                    "--no-enable-prefix-caching (mamba_cache_mode='none')."
                )
            self.atdev_block_v = _block_v()
            if self.head_v_dim % self.atdev_block_v:
                raise ValueError(f"ATDEV_GDN_BV must divide {self.head_v_dim}")
            self.atdev_impl = _impl()
            if self.atdev_impl == "cuda":
                cuda_ext()  # build / load before any graph capture
            self.atdev_qstate = _qstate()
            self.atdev_qgroups = _qgroups()
            if self.atdev_qstate and self.head_k_dim % (4 * self.atdev_qgroups):
                raise ValueError("ATDEV_GDN_QGROUPS must divide head_k_dim / 4")
            if self.num_spec > 0:
                logger.info_once(
                    "atdev %s: single-state GDN verify on (k=%d, BV=%d, %s norm)",
                    OPT_NAME, self.num_spec, self.atdev_block_v,
                    "fused" if self.atdev_block_v == self.head_v_dim else "separate",
                )

        def _can_use_fused_gdn_mtp_decode(self, attn_metadata) -> bool:
            si = attn_metadata.spec_state_indices_tensor
            return (
                attn_metadata.spec_sequence_masks is not None
                and attn_metadata.num_decodes == 0
                and attn_metadata.num_spec_decodes > 0
                and si is not None
                and si.size(1) >= 2
                and self.kv_cache[1].dtype in (torch.float32, torch.bfloat16)
                and self.num_v_heads % self.num_k_heads == 0
            )

        def _forward_core_decode_spec_post_conv_fused_norm(
            self, mixed_qkv, b, a, output_gate, core_attn_out, attn_metadata
        ) -> None:
            num_requests = attn_metadata.num_spec_decodes
            if not getattr(type(self), "_atdev_logged", False):
                type(self)._atdev_logged = True
                import logging
                logging.getLogger("atdev_serve").warning(
                    "atdev gdn_single_state: spec verify path ENGAGED "
                    "(state dtype=%s, block_v=%d)", self.kv_cache[1].dtype,
                    self.atdev_block_v)
            hk = self.num_k_heads // self.tp_size
            key_w = hk * self.head_k_dim
            if (self.atdev_impl == "cuda" and self.kv_cache[1].dtype == torch.bfloat16
                    and attn_metadata.spec_state_indices_tensor.size(1) <= 8):
                gdn_single_state_verify_cuda(
                    mixed_qkv=mixed_qkv, a=a, b=b, A_log=self.A_log, dt_bias=self.dt_bias,
                    state=self.kv_cache[1],
                    state_indices=attn_metadata.spec_state_indices_tensor[:num_requests],
                    cu_seqlens=attn_metadata.spec_query_start_loc[: num_requests + 1],
                    num_accepted=attn_metadata.num_accepted_tokens[:num_requests],
                    out=core_attn_out, output_gate=output_gate,
                    norm_weight=self.norm.weight, num_k_heads=hk,
                    scale=self.head_k_dim**-0.5, norm_eps=self.layer_norm_epsilon,
                    sigmoid_gate=self.norm.activation == "sigmoid",
                    num_requests=num_requests, qstate=self.atdev_qstate,
                    qgroups=self.atdev_qgroups)
                return
            bv = self.atdev_block_v
            fuse = bv == self.head_v_dim
            gdn_single_state_verify(
                q=mixed_qkv[:, :key_w],
                k=mixed_qkv[:, key_w: 2 * key_w],
                v=mixed_qkv[:, 2 * key_w:],
                a=a,
                b=b,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                state=self.kv_cache[1],
                state_indices=attn_metadata.spec_state_indices_tensor[:num_requests],
                cu_seqlens=attn_metadata.spec_query_start_loc[: num_requests + 1],
                num_accepted=attn_metadata.num_accepted_tokens[:num_requests],
                out=core_attn_out,
                num_k_heads=hk,
                scale=self.head_k_dim**-0.5,
                output_gate=output_gate if fuse else None,
                norm_weight=self.norm.weight if fuse else None,
                norm_eps=self.layer_norm_epsilon,
                sigmoid_gate=self.norm.activation == "sigmoid",
                block_v=bv,
                fuse_norm=fuse,
                num_requests=num_requests,
                qstate=self.atdev_qstate,
                qgroups=self.atdev_qgroups,
            )
            if not fuse:
                self._rms_norm_gated_cuda(core_attn_out, output_gate, core_attn_out)

        def _atdev_untag_decodes(self, attn_metadata) -> None:
            # Stock non-spec decode kernels read slot 0 natively: convert any
            # int8-coded regions of those slots back first (rare path).
            if not self.atdev_qstate or attn_metadata.num_decodes <= 0:
                return
            idx = attn_metadata.non_spec_state_indices_tensor
            if idx is None:
                return
            if not getattr(type(self), "_atdev_untag_logged", False):
                type(self)._atdev_untag_logged = True
                logger.warning("atdev %s: int8 state: non-spec decode rows seen; "
                               "untagging their slots for the stock kernel", OPT_NAME)
            gdn_untag_slots(self.kv_cache[1], idx[: attn_metadata.num_decodes],
                            self.atdev_block_v, self.atdev_qgroups)

        def _forward_core(self, mixed_qkv, b, a, core_attn_out):
            if self.atdev_qstate:
                from vllm.forward_context import get_forward_context

                md = get_forward_context().attn_metadata
                md = md.get(self.prefix) if isinstance(md, dict) else None
                if md is not None:
                    self._atdev_untag_decodes(md)
            return super()._forward_core(mixed_qkv, b, a, core_attn_out)

        def _forward_core_decode_non_spec(self, *args, attn_metadata=None, **kwargs):
            if attn_metadata is not None:
                self._atdev_untag_decodes(attn_metadata)
            return super()._forward_core_decode_non_spec(
                *args, attn_metadata=attn_metadata, **kwargs)

    return SingleStateGDNAttention


def _spec_dispatcher(orig):
    """Replacement for qwen_gdn_linear_attn.fused_sigmoid_gating_delta_rule_update."""

    @functools.wraps(orig)
    def fused_sigmoid_gating_delta_rule_update(
        A_log, a, b, dt_bias, q, k, v, beta=1.0, threshold=20.0, scale=None,
        initial_state=None, inplace_final_state=True, cu_seqlens=None,
        ssm_state_indices=None, num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=False, is_kda=False,
    ):
        is_spec_verify = (
            num_accepted_tokens is not None
            and ssm_state_indices is not None
            and ssm_state_indices.dim() == 2
            and ssm_state_indices.size(1) >= 2
            and cu_seqlens is not None
            and initial_state is not None
            and inplace_final_state
            and use_qk_l2norm_in_kernel
            and not is_kda
            and q.dim() == 4 and q.shape[0] == 1
        )
        if not is_spec_verify:
            return orig(A_log, a, b, dt_bias, q, k, v, beta=beta, threshold=threshold,
                        scale=scale, initial_state=initial_state,
                        inplace_final_state=inplace_final_state,
                        cu_seqlens=cu_seqlens, ssm_state_indices=ssm_state_indices,
                        num_accepted_tokens=num_accepted_tokens,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        is_kda=is_kda)
        _, L, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        if scale is None:
            scale = K**-0.5
        o = q.new_empty(1, L, HV, V)
        # The caller applies the gated norm afterwards: un-normed output.
        gdn_single_state_verify(
            q=q.reshape(L, H * K), k=k.reshape(L, H * K), v=v.reshape(L, HV * V),
            a=a, b=b, A_log=A_log, dt_bias=dt_bias, state=initial_state,
            state_indices=ssm_state_indices, cu_seqlens=cu_seqlens,
            num_accepted=num_accepted_tokens, out=o[0], num_k_heads=H,
            scale=scale, softplus_beta=beta, softplus_threshold=threshold,
            block_v=min(_block_v(), V), fuse_norm=False,
            num_requests=cu_seqlens.numel() - 1,
            qstate=_qstate(), qgroups=_qgroups(),
        )
        return o, initial_state

    fused_sigmoid_gating_delta_rule_update._atdev_single_state = True
    return fused_sigmoid_gating_delta_rule_update


_LAYER_CLS = None


def register() -> None:
    global _LAYER_CLS
    import vllm
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn_mod

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"atdev {OPT_NAME}: written against vLLM {SUPPORTED_VLLM_VERSION}, "
            f"found {vllm.__version__}"
        )
    if triton is None:
        raise RuntimeError(f"atdev {OPT_NAME}: Triton is required")
    if _LAYER_CLS is None:
        from atdev_serve.register import enabled

        if _qstate() and "dflash_tree" in enabled():
            raise ValueError("ATDEV_GDN_STATE=int8 is not supported with dflash_tree")

        _LAYER_CLS = _make_layer_class()
        # dflash_tree registers a subclass of this class instead (only one
        # OOT class per layer name is allowed).
        if "dflash_tree" not in enabled():
            gdn_mod.QwenGatedDeltaNetAttention.register_oot(_LAYER_CLS)
    fn = gdn_mod.fused_sigmoid_gating_delta_rule_update
    if not getattr(fn, "_atdev_single_state", False):
        gdn_mod.fused_sigmoid_gating_delta_rule_update = _spec_dispatcher(fn)
    logger.info("atdev %s: registered (BV=%d)", OPT_NAME, _block_v())
