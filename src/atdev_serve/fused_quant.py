"""fused_quant (P1-F2, c=32 W4A8): the per-token int8 activation quantisation
is folded into the kernel that produces the activation, and the W4A8 GEMMs
take pre-quantised A (ATDEV_OPT=fused_quant; requires marlin_plus +
VLLM_MARLIN_INPUT_DTYPE=int8).

Per decoder layer today (docs/research/P1-fusion.md section 1.2, c=32):
  residual+RMSNorm (5 us) -> GEMM with in-kernel quant prologue + grid barrier
  (`fq`, ~ one small launch worth), SiLU.mul (4.4) -> `_per_token_quant_int8`
  (5.3) -> `a_s * gs` (1.9) -> down GEMM, attention: sigmoid-gate mul (3.0) ->
  in-GEMM quant.
Here:
  atdev_norm_quant : residual add + GemmaRMSNorm -> bf16 x (optional, for
                     in_proj_ba) + int8 x_q + fp32 scale (incl. input_global_scale)
                     + bf16 residual; one launch, one program per row
  atdev_silu_quant : SiLU(g).u -> int8 + scale
  atdev_gate_quant : attn * sigmoid(gate) -> int8 + scale
and Qwen3NextDecoderLayer.forward is replaced by a version that feeds the
quantised activations to `vllm::atdev_mp_linear_q` (marlin_plus, no prologue).

Numerics: the int8 values are computed from the bf16-rounded activation with
vLLM's formula (absmax clamp 1e-10, round half away from zero, s = absmax/127 *
input_global_scale), the norm follows vllm.ir.ops.fused_add_rms_norm in fp32
with the Gemma (1+w) weight in fp32. Only the fp32 reduction order differs
from the Inductor kernels.
"""
from __future__ import annotations

import logging
import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

OPT_NAME = "fused_quant"
SUPPORTED_VLLM_VERSION = "0.31.0"
log = logging.getLogger("atdev_serve")

if triton is not None:
    @triton.jit
    def _round_i8(v):
        return tl.extra.cuda.libdevice.round(v).to(tl.int8)

    @triton.jit
    def _store_sa(q, row, g0, sa_ptr, NG, NGB: tl.constexpr):
        # P3 U8: exact int32 sum of the int8 codes of each 128-column group (groups g0 .. g0+NGB-1)
        qs = tl.sum(tl.reshape(q.to(tl.int32), (NGB, 128)), axis=1)
        g = g0 + tl.arange(0, NGB)
        tl.store(sa_ptr + g * 256 + row, qs, mask=g < NG)  # group-major [K/128][256] (marlin_plus U8)

    @triton.jit
    def _norm_quant_kernel(x_ptr, stride_x, r_ptr, stride_r, w_ptr, gs_ptr,
                           xb_ptr, stride_xb, xq_ptr, stride_xq, xs_ptr, ro_ptr, stride_ro,
                           N, eps,
                           HAS_RES: tl.constexpr, WANT_BF16: tl.constexpr, BLOCK: tl.constexpr,
                           sa_ptr=None, WANT_SA: tl.constexpr = False):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        m = cols < N
        x = tl.load(x_ptr + row * stride_x + cols, mask=m, other=0.0).to(tl.float32)
        if HAS_RES:
            x = x + tl.load(r_ptr + row * stride_r + cols, mask=m, other=0.0).to(tl.float32)
        # the residual stream handed to the next layer (bf16 of the fp32 sum; == x if no residual)
        tl.store(ro_ptr + row * stride_ro + cols, x.to(tl.bfloat16), mask=m)
        var = tl.sum(x * x, axis=0) / N
        xn = x * tl.rsqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32) + 1.0
        xb = (xn * w).to(tl.bfloat16)          # == the bf16 GEMM input of the stock path
        if WANT_BF16:
            tl.store(xb_ptr + row * stride_xb + cols, xb, mask=m)
        xf = xb.to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(xf), axis=0), 1e-10)
        q = _round_i8(xf * (127.0 / amax))
        tl.store(xq_ptr + row * stride_xq + cols, q, mask=m)
        if WANT_SA:
            _store_sa(q, row, 0, sa_ptr, N // 128, BLOCK // 128)
        gs = tl.load(gs_ptr).to(tl.float32)
        tl.store(xs_ptr + row, (amax / 127.0) * gs)

    @triton.jit
    def _silu_quant_kernel(gu_ptr, stride_gu, gs_ptr, q_ptr, stride_q, s_ptr, N,
                           BLOCK: tl.constexpr, sa_ptr=None, WANT_SA: tl.constexpr = False):
        # row = [g (N) | u (N)]; y = bf16(silu(g) * u); two passes over N (amax, quant)
        row = tl.program_id(0)
        base = gu_ptr + row * stride_gu
        amax = tl.full([BLOCK], 1e-10, dtype=tl.float32)
        for c0 in range(0, N, BLOCK):
            cols = c0 + tl.arange(0, BLOCK)
            m = cols < N
            g = tl.load(base + cols, mask=m, other=0.0).to(tl.float32)
            u = tl.load(base + N + cols, mask=m, other=0.0).to(tl.float32)
            y = (g * tl.sigmoid(g) * u).to(tl.bfloat16).to(tl.float32)
            amax = tl.maximum(amax, tl.abs(y))
        am = tl.max(amax, axis=0)
        inv = 127.0 / am
        for c0 in range(0, N, BLOCK):
            cols = c0 + tl.arange(0, BLOCK)
            m = cols < N
            g = tl.load(base + cols, mask=m, other=0.0).to(tl.float32)
            u = tl.load(base + N + cols, mask=m, other=0.0).to(tl.float32)
            y = (g * tl.sigmoid(g) * u).to(tl.bfloat16).to(tl.float32)
            qv = _round_i8(y * inv)
            tl.store(q_ptr + row * stride_q + cols, qv, mask=m)
            if WANT_SA:
                _store_sa(qv, row, c0 // 128, sa_ptr, N // 128, BLOCK // 128)
        tl.store(s_ptr + row, (am / 127.0) * tl.load(gs_ptr).to(tl.float32))

    @triton.jit
    def _silu_quant_1p_kernel(gu_ptr, stride_gu, gs_ptr, q_ptr, stride_q, s_ptr, N,
                              BLOCK: tl.constexpr, sa_ptr=None, WANT_SA: tl.constexpr = False):
        # single pass: the whole row (N <= BLOCK) stays in registers
        row = tl.program_id(0)
        base = gu_ptr + row * stride_gu
        cols = tl.arange(0, BLOCK)
        m = cols < N
        g = tl.load(base + cols, mask=m, other=0.0).to(tl.float32)
        u = tl.load(base + N + cols, mask=m, other=0.0).to(tl.float32)
        y = (g * tl.sigmoid(g) * u).to(tl.bfloat16).to(tl.float32)
        am = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-10)
        qv = _round_i8(y * (127.0 / am))
        tl.store(q_ptr + row * stride_q + cols, qv, mask=m)
        if WANT_SA:
            _store_sa(qv, row, 0, sa_ptr, N // 128, BLOCK // 128)
        tl.store(s_ptr + row, (am / 127.0) * tl.load(gs_ptr).to(tl.float32))

    @triton.jit
    def _silu_quant_ch_kernel(gu_ptr, stride_gu, gs_ptr, q_ptr, stride_q, s_ptr, amax_ptr, tick_ptr,
                              N, CH: tl.constexpr, BLOCK: tl.constexpr, sa_ptr=None,
                              WANT_SA: tl.constexpr = False):
        # grid (rows, CH): program (row, c) handles cols [c*BLOCK, (c+1)*BLOCK); the row
        # absmax is combined through an atomic max (fp32 >= 0 compares as int bits), every
        # program waits for the row's CH arrivals, quantises its chunk, and the last reader
        # resets the slots. All CH programs of a row must be co-resident (small grid).
        row = tl.program_id(0)
        c = tl.program_id(1)
        cols = c * BLOCK + tl.arange(0, BLOCK)
        m = cols < N
        base = gu_ptr + row * stride_gu
        g = tl.load(base + cols, mask=m, other=0.0).to(tl.float32)
        u = tl.load(base + N + cols, mask=m, other=0.0).to(tl.float32)
        y = (g * tl.sigmoid(g) * u).to(tl.bfloat16).to(tl.float32)
        am = tl.max(tl.abs(y), axis=0)
        tl.atomic_max(amax_ptr + row, am.to(tl.int32, bitcast=True), sem="release")
        tl.atomic_add(tick_ptr + row, 1, sem="acq_rel")
        while tl.atomic_add(tick_ptr + row, 0, sem="acquire") < CH:
            pass
        amr = tl.atomic_add(amax_ptr + row, 0, sem="acquire").to(tl.float32, bitcast=True)
        amr = tl.maximum(amr, 1e-10)
        qv = _round_i8(y * (127.0 / amr))
        tl.store(q_ptr + row * stride_q + cols, qv, mask=m)
        if WANT_SA:
            _store_sa(qv, row, c * (BLOCK // 128), sa_ptr, N // 128, BLOCK // 128)
        if c == 0:
            tl.store(s_ptr + row, (amr / 127.0) * tl.load(gs_ptr).to(tl.float32))
        # second count: last reader resets both slots
        done = tl.atomic_add(tick_ptr + row, 1, sem="acq_rel")
        if done == 2 * CH - 1:
            tl.atomic_xchg(tick_ptr + row, 0)
            tl.atomic_xchg(amax_ptr + row, 0)

    @triton.jit
    def _gate_quant_kernel(a_ptr, stride_a, g_ptr, stride_g, gs_ptr, q_ptr, stride_q, s_ptr, N,
                           BLOCK: tl.constexpr, sa_ptr=None, WANT_SA: tl.constexpr = False):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        m = cols < N
        a = tl.load(a_ptr + row * stride_a + cols, mask=m, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + row * stride_g + cols, mask=m, other=0.0).to(tl.float32)
        y = (a * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
        am = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-10)
        qv = _round_i8(y * (127.0 / am))
        tl.store(q_ptr + row * stride_q + cols, qv, mask=m)
        if WANT_SA:
            _store_sa(qv, row, 0, sa_ptr, N // 128, BLOCK // 128)
        tl.store(s_ptr + row, (am / 127.0) * tl.load(gs_ptr).to(tl.float32))


    # ---------------- W4A4 (P3 marlin_plus_a4) producers ----------------
    # Same quantisation and fragment layout as csrc/marlin_plus_a4/a4_host.cu
    # a4_quant_kernel (bit-equal): per (row, 128-group) s = max(amax, 1e-8) * (1/7),
    # q = clamp(rint(x / s), -8, 7) (IEEE division), lane L holds k = 128g + 4L + e,
    # word (kb, mb, l, r), kb = 2g + L/16, l = (row%8)*4 + L%4, r = (row%16)/8 + 2*((L%16)/8),
    # nibble 2e + ((L/4)&1); A_s[g * m_pad + row] = s * gs.
    @triton.jit
    def _a4_pack_store(y, row, g0, NGB: tl.constexpr, ngroups, apk_ptr, as_ptr, gs, m_pad, tmb,
                       MK: tl.constexpr = False):
        yy = tl.reshape(y, [NGB, 128])
        amax = tl.max(tl.abs(yy), axis=1)
        sc = tl.maximum(amax, 1e-8) * (1.0 / 7.0)
        if MK:  # C32h-mk: branch-free quotient, nibbles == div_rn for every bf16 y (scripts/c32hmk_exhaust.cu)
            rc = tl.extra.cuda.libdevice.div_rn(1.0, sc)
            scb = tl.broadcast_to(sc[:, None], yy.shape)
            rcb = tl.broadcast_to(rc[:, None], yy.shape)
            q1 = tl.inline_asm_elementwise(
                "{ .reg .f32 q0, rr, ns; mul.rn.ftz.f32 q0, $1, $3; neg.f32 ns, $2; "
                "fma.rn.ftz.f32 rr, ns, q0, $1; fma.rn.ftz.f32 $0, rr, $3, q0; }",
                "=r,r,r,r", [yy, scb, rcb], dtype=tl.float32, is_pure=True, pack=1)
            q = tl.extra.cuda.libdevice.rint(q1)
        else:
            q = tl.extra.cuda.libdevice.rint(tl.extra.cuda.libdevice.div_rn(yy, sc[:, None]))
        q = tl.minimum(tl.maximum(q, -8.0), 7.0).to(tl.int32) & 15
        q5 = tl.reshape(q, [NGB, 4, 2, 4, 4])                      # (g, a, p, b, e): L = 8a + 4p + b
        p_i = tl.arange(0, 2)[None, None, :, None, None]
        e_i = tl.arange(0, 4)[None, None, None, None, :]
        w = tl.sum(tl.sum(q5 << (8 * e_i + 4 * p_i), axis=4), axis=2)  # [NGB, 4(a), 4(b)], disjoint bits
        g = g0 + tl.arange(0, NGB)[:, None, None]
        a_i = tl.arange(0, 4)[None, :, None]
        b_i = tl.arange(0, 4)[None, None, :]
        kb = 2 * g + a_i // 2
        l_i = (row % 8) * 4 + b_i
        r_i = (row % 16) // 8 + 2 * (a_i % 2)
        idx = ((kb * tmb + row // 16) * 32 + l_i) * 4 + r_i
        tl.store(apk_ptr + idx, w, mask=g < ngroups)
        gv = g0 + tl.arange(0, NGB)
        tl.store(as_ptr + gv * m_pad + row, sc * gs, mask=gv < ngroups)

    @triton.jit
    def _norm_a4_kernel(x_ptr, stride_x, r_ptr, stride_r, w_ptr, gs_ptr,
                        xb_ptr, stride_xb, ro_ptr, stride_ro, apk_ptr, as_ptr,
                        M, N, eps, m_pad, tmb,
                        HAS_RES: tl.constexpr, WANT_BF16: tl.constexpr, BLOCK: tl.constexpr,
                        MK: tl.constexpr = False):
        row = tl.program_id(0)
        valid = row < M
        cols = tl.arange(0, BLOCK)
        m = (cols < N) & valid
        x = tl.load(x_ptr + row * stride_x + cols, mask=m, other=0.0).to(tl.float32)
        if HAS_RES:
            x = x + tl.load(r_ptr + row * stride_r + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(ro_ptr + row * stride_ro + cols, x.to(tl.bfloat16), mask=m)
        var = tl.sum(x * x, axis=0) / N
        xn = x * tl.rsqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=cols < N, other=0.0).to(tl.float32) + 1.0
        xb = (xn * w).to(tl.bfloat16)
        if WANT_BF16:
            tl.store(xb_ptr + row * stride_xb + cols, xb, mask=m)
        y = tl.where(m, xb.to(tl.float32), 0.0)
        _a4_pack_store(y, row, 0, BLOCK // 128, N // 128, apk_ptr, as_ptr,
                       tl.load(gs_ptr).to(tl.float32), m_pad, tmb, MK)

    @triton.jit
    def _norm_a4_chunk_kernel(x_ptr, stride_x, r_ptr, stride_r, w_ptr, gs_ptr,
                              xb_ptr, stride_xb, ro_ptr, stride_ro, apk_ptr, as_ptr,
                              M, N, eps, m_pad, tmb,
                              HAS_RES: tl.constexpr, WANT_BF16: tl.constexpr, BLOCK: tl.constexpr,
                              NGB: tl.constexpr, MK: tl.constexpr = False):
        # grid (m_pad, N / (128*NGB)): same math as _norm_a4_kernel; the full-row variance is
        # recomputed by every chunk program (identical reduction -> identical rstd)
        row = tl.program_id(0)
        g0 = tl.program_id(1) * NGB
        valid = row < M
        cols = tl.arange(0, BLOCK)
        mf = (cols < N) & valid
        xf = tl.load(x_ptr + row * stride_x + cols, mask=mf, other=0.0).to(tl.float32)
        if HAS_RES:
            xf = xf + tl.load(r_ptr + row * stride_r + cols, mask=mf, other=0.0).to(tl.float32)
        var = tl.sum(xf * xf, axis=0) / N
        rstd = tl.rsqrt(var + eps)
        cc = g0 * 128 + tl.arange(0, NGB * 128)
        mc = (cc < N) & valid
        x = tl.load(x_ptr + row * stride_x + cc, mask=mc, other=0.0).to(tl.float32)
        if HAS_RES:
            x = x + tl.load(r_ptr + row * stride_r + cc, mask=mc, other=0.0).to(tl.float32)
        tl.store(ro_ptr + row * stride_ro + cc, x.to(tl.bfloat16), mask=mc)
        w = tl.load(w_ptr + cc, mask=cc < N, other=0.0).to(tl.float32) + 1.0
        xb = (x * rstd * w).to(tl.bfloat16)
        if WANT_BF16:
            tl.store(xb_ptr + row * stride_xb + cc, xb, mask=mc)
        y = tl.where(mc, xb.to(tl.float32), 0.0)
        _a4_pack_store(y, row, g0, NGB, N // 128, apk_ptr, as_ptr, tl.load(gs_ptr).to(tl.float32),
                       m_pad, tmb, MK)

    @triton.jit
    def _silu_a4_kernel(gu_ptr, stride_gu, gs_ptr, apk_ptr, as_ptr, M, N, m_pad, tmb,
                        NGB: tl.constexpr, MK: tl.constexpr = False):
        row = tl.program_id(0)
        g0 = tl.program_id(1) * NGB
        cols = g0 * 128 + tl.arange(0, NGB * 128)
        m = (cols < N) & (row < M)
        base = gu_ptr + row * stride_gu
        g = tl.load(base + cols, mask=m, other=0.0).to(tl.float32)
        u = tl.load(base + N + cols, mask=m, other=0.0).to(tl.float32)
        y = (g * tl.sigmoid(g) * u).to(tl.bfloat16).to(tl.float32)
        _a4_pack_store(y, row, g0, NGB, N // 128, apk_ptr, as_ptr, tl.load(gs_ptr).to(tl.float32),
                       m_pad, tmb, MK)

    @triton.jit
    def _gate_a4_kernel(a_ptr, stride_a, g_ptr, stride_g, gs_ptr, apk_ptr, as_ptr, M, N, m_pad, tmb,
                        NGB: tl.constexpr, MK: tl.constexpr = False):
        row = tl.program_id(0)
        g0 = tl.program_id(1) * NGB
        cols = g0 * 128 + tl.arange(0, NGB * 128)
        m = (cols < N) & (row < M)
        a = tl.load(a_ptr + row * stride_a + cols, mask=m, other=0.0).to(tl.float32)
        gg = tl.load(g_ptr + row * stride_g + cols, mask=m, other=0.0).to(tl.float32)
        y = (a * tl.sigmoid(gg)).to(tl.bfloat16).to(tl.float32)
        _a4_pack_store(y, row, g0, NGB, N // 128, apk_ptr, as_ptr, tl.load(gs_ptr).to(tl.float32),
                       m_pad, tmb, MK)


# ---------------------------------------------------------------------------
# W4A4 mode: decided at run time (inside the opaque ops) from the consumer GEMM's
# marlin_plus table entry, exactly as marlin_plus.mp_linear does. In a4 mode the
# producer writes the s4 fragments into marlin_plus' shared a4 buffers (the same
# buffers its own quant_a4 + gemm_a4 pair uses) and the consumer runs gemm_a4 on
# them; the returned int8 tensors are then placeholders that only carry the data
# dependency (producer and consumer are adjacent on the stream).
# ---------------------------------------------------------------------------

def _a4_ent(n: int, k: int, m: int):
    from atdev_serve import marlin_plus as mp
    if m <= 0:
        return None
    ent = mp.lookup(n, k, m, True)
    if ent is None or not ent.get("a4"):
        return None
    tmb = ent["cfg"][4]
    if m > 16 * tmb or k > mp.A4_MAX_K or tmb > mp.A4_MAX_TMB:
        return None
    return ent


def fq_linear(xq: torch.Tensor, xs: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
              n: int, k: int) -> torch.Tensor:
    from atdev_serve import marlin_plus as mp
    m = xq.shape[0]
    ent = _a4_ent(n, k, m)
    if ent is None:
        return mp.mp_linear_q(xq, xs, w_q, w_s, n, k)
    b = mp.bufs(xq.device)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=xq.device)
    cfg = list(ent["cfg"])
    a4 = mp.mod_a4()
    if "A4_PAR_REDUCE" in os.environ.get("ATDEV_A4_DEFINES", ""):
        if "a4_locks" not in b:
            b["a4_locks"] = torch.zeros(1024, dtype=torch.int32, device=xq.device)
            b["a4_ctmp"] = torch.empty(64 << 18, dtype=torch.float32, device=xq.device)
        return a4.gemm_a4(b["a4_pk"], b["a4_s"], w_q, w_s, out, b["a4_locks"], b["a4_ctmp"], m, n, k, cfg)
    return a4.gemm_a4(b["a4_pk"], b["a4_s"], w_q, w_s, out, b["locks"], b["c_tmp"], m, n, k, cfg)


_CS_ON = [False]  # ATDEV_C32C_CSUM=1: attn o_proj + GDN out_proj (set in register)
_CS_DOWN = [False]  # ATDEV_C32C_CSUM=2: + MLP down (in-situ loss: LEDGER C32c-prof)


def fq_linear_cs(xq: torch.Tensor, xs: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
                 n: int, k: int) -> torch.Tensor:
    """fq_linear whose output only the next norm_quant reads: csum GEMM (C32c) when eligible."""
    from atdev_serve import c32c_csum, marlin_plus as mp
    m = xq.shape[0]
    ent = _a4_ent(n, k, m)
    if not c32c_csum.eligible(ent, m, n, k, xq.device):
        c32c_csum.clear()
        return fq_linear(xq, xs, w_q, w_s, n, k)
    b = mp.bufs(xq.device)
    return c32c_csum.gemm(b["a4_pk"], b["a4_s"], w_q, w_s, m, n, k, ent, xq.device)


def cs_mp_linear(x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor, gs: torch.Tensor,
                 n: int, k: int) -> torch.Tensor:
    """marlin_plus.mp_linear (W4A8 kernel, a4 entry) for GDN out_proj with a csum GEMM (C32c)."""
    from atdev_serve import c32c_csum, marlin_plus as mp
    m = x.shape[0]
    ent = mp.lookup(n, k, m, True) if m > 0 else None
    if (not c32c_csum.eligible(ent, m, n, k, x.device) or k > mp.A4_MAX_K
            or ent["cfg"][4] > mp.A4_MAX_TMB):
        c32c_csum.clear()
        return mp.mp_linear(x, w_q, w_s, gs, n, k, True)
    if x.stride(-1) != 1 or x.stride(0) % 8:
        x = x.contiguous()
    b = mp.bufs(x.device)
    tmb = ent["cfg"][4]
    mp.mod_a4().quant_a4(x, gs.reshape(1).float(), b["a4_pk"], b["a4_s"], tmb)  # gdn_a4_fold proxy may skip
    return c32c_csum.gemm(b["a4_pk"], b["a4_s"], w_q, w_s, m, n, k, ent, x.device)


def _cs_mp_linear_fake(x, w_q, w_s, gs, n, k):
    return torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)


def _fq_linear_fake(xq, xs, w_q, w_s, n, k):
    return torch.empty((xq.shape[0], n), dtype=torch.bfloat16, device=xq.device)


# ---------------------------------------------------------------------------
# custom ops
# ---------------------------------------------------------------------------

# C32h-mk (LEDGER C32hmk-*): ATDEV_C32_MK=1 -> every s4 producer uses the bit-identical branch-free quotient
# (Triton: MK constexpr of _a4_pack_store; CUDA: c32_prod / c32c_csum MK builds); ATDEV_C32_MKCHK=<file>: shadow-run
# the other variant after every a4 producer call (in the CUDA graphs) and count s4 word / scale mismatches on device
_MK = os.environ.get("ATDEV_C32_MK", "0") == "1"
_MKCHK = os.environ.get("ATDEV_C32_MKCHK", "") != ""
_NORM_WARPS = int(os.environ.get("ATDEV_NORMQ_WARPS", "8"))  # int8 norm_quant program width (P3 microbench)


def _sa_buf(dev, xq):
    """P3 U8: marlin_plus' shared Sa buffer, tagged with the int8 tensor it describes (consumed by the
    adjacent mp_linear_q); None when U8 is off."""
    from atdev_serve import marlin_plus as mp
    if not mp.U8 or xq.shape[0] == 0 or xq.shape[0] > mp.FQ_MAX_M:
        return None
    b = mp.bufs(dev)
    b["sa_for"] = xq.data_ptr()
    return b["a_sa"]


def _cuda_prod() -> bool:
    """ATDEV_FQ_A4_CUDA=1: the SiLU.mul / gate W4A4 producers run the bit-identical CUDA kernels
    (c32_prod.py; the norm->s4 CUDA kernel is not bit-identical and is not used)."""
    return os.environ.get("ATDEV_FQ_A4_CUDA", "0") == "1"


def _al16(*ts: torch.Tensor) -> bool:
    """unit inner stride, 16-byte aligned base and rows (uint4 loads/stores in c32_prod.cu)"""
    return all(t.stride(-1) == 1 and t.data_ptr() % 16 == 0
               and (t.dim() == 1 or (t.stride(0) * t.element_size()) % 16 == 0) for t in ts)


def norm_quant(x: torch.Tensor, residual: torch.Tensor | None, weight: torch.Tensor, eps: float,
               gs: torch.Tensor, want_bf16: bool, n_next: int
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = x.shape
    dev = x.device
    xb = torch.empty((m, n) if want_bf16 else (0, n), dtype=torch.bfloat16, device=dev)
    xq = torch.empty((m, n), dtype=torch.int8, device=dev)
    xs = torch.empty((m,), dtype=torch.float32, device=dev)
    has_res = residual is not None
    ro = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    _log_once()
    if m == 0:
        return xb, xq, xs, ro
    r = residual if has_res else x
    ent = _a4_ent(n_next, n, m)
    if _CS_ON[0]:  # C32c consumer-sum: x may be the placeholder of a csum GEMM (c32c_csum.py)
        from atdev_serve import c32c_csum
        if c32c_csum.tagged(x):
            if ent is not None and has_res:
                from atdev_serve import marlin_plus as mp
                b = mp.bufs(dev)
                tmb = ent["cfg"][4]
                c32c_csum.norm_a4(x, r, weight, gs, xb if want_bf16 else None, ro, b["a4_pk"], b["a4_s"], m, eps,
                                  16 * tmb, tmb)
                return xb, xq, xs, ro
            c32c_csum.materialise(x)
    if ent is not None:
        from atdev_serve import marlin_plus as mp
        b = mp.bufs(dev)
        tmb = ent["cfg"][4]
        ngb, warps = (int(v) for v in os.environ.get("ATDEV_FQ_A4_NORM", "8,8").split(","))
        if ngb == 0:  # one program per row
            _norm_a4_kernel[(16 * tmb,)](
                x, x.stride(0), r, r.stride(0), weight, gs, xb, xb.stride(0) if want_bf16 else 0,
                ro, ro.stride(0), b["a4_pk"], b["a4_s"], m, n, eps, 16 * tmb, tmb,
                HAS_RES=has_res, WANT_BF16=want_bf16, BLOCK=triton.next_power_of_2(n), MK=_MK, num_warps=warps)
        else:
            _norm_a4_chunk_kernel[(16 * tmb, triton.cdiv(n // 128, ngb))](
                x, x.stride(0), r, r.stride(0), weight, gs, xb, xb.stride(0) if want_bf16 else 0,
                ro, ro.stride(0), b["a4_pk"], b["a4_s"], m, n, eps, 16 * tmb, tmb,
                HAS_RES=has_res, WANT_BF16=want_bf16, BLOCK=triton.next_power_of_2(n), NGB=ngb, MK=_MK,
                num_warps=warps)
            if _MKCHK:  # C32h-mk shadow check: the other quotient variant into scratch, compared on device
                from atdev_serve import c32hmk_check as ck
                ck.check("tchunk", lambda pk2, s2: _norm_a4_chunk_kernel[(16 * tmb, triton.cdiv(n // 128, ngb))](
                    x, x.stride(0), r, r.stride(0), weight, gs, xb, 0, torch.empty_like(ro), ro.stride(0), pk2, s2,
                    m, n, eps, 16 * tmb, tmb, HAS_RES=has_res, WANT_BF16=False, BLOCK=triton.next_power_of_2(n),
                    NGB=ngb, MK=not _MK, num_warps=warps), b["a4_pk"], b["a4_s"], n, 16 * tmb, tmb)
        return xb, xq, xs, ro
    block = triton.next_power_of_2(n)
    sa = _sa_buf(dev, xq)
    _norm_quant_kernel[(m,)](
        x, x.stride(0), r, r.stride(0), weight, gs, xb, xb.stride(0) if want_bf16 else 0,
        xq, xq.stride(0), xs, ro, ro.stride(0), n, eps,
        HAS_RES=has_res, WANT_BF16=want_bf16, BLOCK=block, num_warps=_NORM_WARPS,
        sa_ptr=sa, WANT_SA=sa is not None)
    return xb, xq, xs, ro


def _norm_quant_fake(x, residual, weight, eps, gs, want_bf16, n_next):
    m, n = x.shape
    return (torch.empty((m, n) if want_bf16 else (0, n), dtype=torch.bfloat16, device=x.device),
            torch.empty((m, n), dtype=torch.int8, device=x.device),
            torch.empty((m,), dtype=torch.float32, device=x.device),
            torch.empty((m, n), dtype=torch.bfloat16, device=x.device))


# (variant, block, num_warps); ATDEV_SILUQ_CFG="ch,1024,4" | "1p,32768,16" | "2p,2048,8" (bench sweep)
_SILU_CFG = os.environ.get("ATDEV_SILUQ_CFG", "ch,1024,4").split(",")
_SLOTS: dict = {}


def _slots(dev: torch.device) -> dict:
    b = _SLOTS.get(dev.index)
    if b is None:
        b = _SLOTS[dev.index] = dict(amax=torch.zeros(256, dtype=torch.int32, device=dev),
                                     tick=torch.zeros(256, dtype=torch.int32, device=dev))
    return b


def silu_quant(gu: torch.Tensor, gs: torch.Tensor, cfg=None, n_next: int = 0
               ) -> tuple[torch.Tensor, torch.Tensor]:
    m, n2 = gu.shape
    n = n2 // 2
    q = torch.empty((m, n), dtype=torch.int8, device=gu.device)
    s = torch.empty((m,), dtype=torch.float32, device=gu.device)
    ent = _a4_ent(n_next, n, m) if n_next else None
    if ent is not None:
        from atdev_serve import marlin_plus as mp
        b = mp.bufs(gu.device)
        tmb = ent["cfg"][4]
        if _cuda_prod() and _al16(gu) and n % 256 == 0:
            from atdev_serve import c32_prod
            c32_prod.silu_a4(gu, gs, b["a4_pk"], b["a4_s"], m, n, 16 * tmb, tmb)
            if _MKCHK:
                from atdev_serve import c32hmk_check as ck
                ck.check("silu", lambda pk2, s2: c32_prod.silu_a4(gu, gs, pk2, s2, m, n, 16 * tmb, tmb, mk=not _MK),
                         b["a4_pk"], b["a4_s"], n, 16 * tmb, tmb)
            return q, s
        ngb, warps = (int(v) for v in os.environ.get("ATDEV_FQ_A4_SILU", "8,4").split(","))
        _silu_a4_kernel[(16 * tmb, triton.cdiv(n // 128, ngb))](
            gu, gu.stride(0), gs, b["a4_pk"], b["a4_s"], m, n, 16 * tmb, tmb, NGB=ngb, MK=_MK, num_warps=warps)
        return q, s
    if cfg is None and "ATDEV_SILUQ_CFG" not in os.environ:
        # microbench (P1-fusion.md section 2): chunked spin-wait wins at small M (8.2 us vs 12 at M=16),
        # the single-pass row kernel at M=64 (12.2 vs 17; 1088 spinning programs hurt)
        cfg = ("ch", 1024, 4) if m <= 32 else ("1p", 32768, 32)
    var, block, warps = cfg or _SILU_CFG
    block, warps = int(block), int(warps)
    sa = _sa_buf(gu.device, q) if m else None
    sak = dict(sa_ptr=sa, WANT_SA=sa is not None)
    if m:
        if var == "ch" and block % 128 == 0 and m <= 256:
            # P3: masked last chunk (block need not divide n), e.g. 17408 cols in 3 x 8192 / 5 x 4096
            b = _slots(gu.device)
            ch = triton.cdiv(n, block)
            _silu_quant_ch_kernel[(m, ch)](gu, gu.stride(0), gs, q, q.stride(0), s, b["amax"],
                                           b["tick"], n, CH=ch, BLOCK=block,
                                           num_warps=warps, **sak)
        elif var == "1p" and n <= block:
            _silu_quant_1p_kernel[(m,)](gu, gu.stride(0), gs, q, q.stride(0), s, n, BLOCK=block,
                                        num_warps=warps, **sak)
        else:
            _silu_quant_kernel[(m,)](gu, gu.stride(0), gs, q, q.stride(0), s, n, BLOCK=min(block, 4096),
                                     num_warps=warps, **sak)
    return q, s


def _silu_quant_op(gu: torch.Tensor, gs: torch.Tensor, n_next: int) -> tuple[torch.Tensor, torch.Tensor]:
    return silu_quant(gu, gs, None, n_next)


def _silu_quant_fake(gu, gs, n_next):
    m, n2 = gu.shape
    return (torch.empty((m, n2 // 2), dtype=torch.int8, device=gu.device),
            torch.empty((m,), dtype=torch.float32, device=gu.device))


def gate_quant(a: torch.Tensor, g: torch.Tensor, gs: torch.Tensor, n_next: int = 0
               ) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = a.shape
    q = torch.empty((m, n), dtype=torch.int8, device=a.device)
    s = torch.empty((m,), dtype=torch.float32, device=a.device)
    ent = _a4_ent(n_next, n, m) if n_next else None
    if ent is not None:
        from atdev_serve import marlin_plus as mp
        b = mp.bufs(a.device)
        tmb = ent["cfg"][4]
        if _cuda_prod() and _al16(a, g) and n % 256 == 0:
            from atdev_serve import c32_prod
            c32_prod.gate_a4(a, g, gs, b["a4_pk"], b["a4_s"], m, n, 16 * tmb, tmb)
            if _MKCHK:
                from atdev_serve import c32hmk_check as ck
                ck.check("gate", lambda pk2, s2: c32_prod.gate_a4(a, g, gs, pk2, s2, m, n, 16 * tmb, tmb, mk=not _MK),
                         b["a4_pk"], b["a4_s"], n, 16 * tmb, tmb)
            return q, s
        _gate_a4_kernel[(16 * tmb, triton.cdiv(n // 128, 8))](
            a, a.stride(0), g, g.stride(0), gs, b["a4_pk"], b["a4_s"], m, n, 16 * tmb, tmb,
            NGB=8, MK=_MK, num_warps=4)
        return q, s
    if m:
        sa = _sa_buf(a.device, q)
        _gate_quant_kernel[(m,)](a, a.stride(0), g, g.stride(0), gs, q, q.stride(0), s, n,
                                 BLOCK=triton.next_power_of_2(n), num_warps=8,
                                 sa_ptr=sa, WANT_SA=sa is not None)
    return q, s


def _gate_quant_fake(a, g, gs, n_next=0):
    m, n = a.shape
    return (torch.empty((m, n), dtype=torch.int8, device=a.device),
            torch.empty((m,), dtype=torch.float32, device=a.device))


# ---------------------------------------------------------------------------
# layer forward with pre-quantised GEMM inputs
# ---------------------------------------------------------------------------

def _is_mp_a8(linear) -> bool:
    from atdev_serve.marlin_plus import kernel_class
    k = getattr(getattr(linear, "scheme", None), "kernel", None)
    return (k is not None and type(k) is kernel_class() and k.config.act_type == torch.int8
            and getattr(linear, "input_global_scale", None) is not None)


_ELIG: dict = {}


def _layer_eligible(layer) -> bool:
    """All GEMMs of this decoder layer on marlin_plus W4A8 (the MTP drafter layer, bf16 or
    mtp_w4 int4 Marlin, is not) -- decided once per layer, no module attributes touched."""
    key = id(layer)
    e = _ELIG.get(key)
    if e is None:
        gdn = layer.layer_type == "linear_attention"
        lins = ([layer.linear_attn.in_proj_qkvz, layer.linear_attn.out_proj] if gdn
                else [layer.self_attn.qkv_proj, layer.self_attn.o_proj])
        lins += [layer.mlp.gate_up_proj, layer.mlp.down_proj]
        e = _ELIG[key] = all(_is_mp_a8(l) for l in lins) and getattr(layer.mlp, "expert_gate", None) is None
        # (no logging here: this runs under Dynamo; summarised from the first op call instead)
        _ELIG.setdefault("_names", {})[getattr(layer, "prefix", str(key))] = e
    return e


_LOGGED = False


def _log_once():
    # (the per-layer eligibility dict is filled under Dynamo tracing and is not visible here)
    global _LOGGED
    if not _LOGGED:
        _LOGGED = True
        log.warning("atdev %s: fused norm/act quant producers ENGAGED (int8 or W4A4 fragments per "
                    "consumer GEMM entry)", OPT_NAME)


def _mp_kernel(linear):
    from atdev_serve.marlin_plus import kernel_class
    k = getattr(getattr(linear, "scheme", None), "kernel", None)
    if k is None or type(k) is not kernel_class() or k.config.act_type != torch.int8:
        raise RuntimeError(
            f"atdev {OPT_NAME}: {getattr(linear, 'prefix', linear)} is not a marlin_plus W4A8 "
            f"linear (kernel {type(k).__name__}); needs ATDEV_OPT=marlin_plus + "
            "VLLM_MARLIN_INPUT_DTYPE=int8")
    return k


def _gs(linear) -> torch.Tensor:
    # input_global_scale: 0-dim fp32 parameter set by Marlin's int8 weight processing
    _mp_kernel(linear)
    return linear.input_global_scale.reshape(1)


def linear_q(linear, xq: torch.Tensor, xs: torch.Tensor, cs: bool = False) -> torch.Tensor:
    k = _mp_kernel(linear)
    kk, n = k.config.partition_weight_shape
    w_q, w_s, _ = k._get_weight_params(linear)
    if cs:
        return torch.ops.vllm.atdev_fq_linear_cs(xq, xs, w_q, w_s, n, kk)
    return torch.ops.vllm.atdev_fq_linear(xq, xs, w_q, w_s, n, kk)


def _n(linear) -> int:
    return _mp_kernel(linear).config.partition_weight_shape[1]


def _core_empty() -> bool:
    from atdev_serve.register import enabled
    return os.environ.get("ATDEV_GDN_CORE_EMPTY", "0") == "1" and "gdn_a4_fold" in enabled()


def _gdn_forward(gdn, xb, xq, xs):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import _encode_layer_name
    if not (gdn.enable_fused_gdn_decode and gdn.norm.weight.dtype in (torch.bfloat16, torch.float32)):
        return gdn(hidden_states=xb)  # stock path (bf16 input)
    qkvz = linear_q(gdn.in_proj_qkvz, xq, xs)
    ba, _ = gdn.in_proj_ba(xb)
    # C32-gdn2 (ATDEV_GDN_CORE_EMPTY=1, needs gdn_a4_fold): uninitialised core; gdn_a4_fold zero-fills it
    # inside the op unless the folded GDN epilogue writes every row out_proj reads (it then reads none)
    alloc = torch.empty if _core_empty() else torch.zeros
    core = alloc((xq.shape[0], gdn.num_v_heads // gdn.tp_size, gdn.head_v_dim),
                 dtype=torch.bfloat16, device=xq.device)
    torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
        qkvz, ba, core, layer_name=_encode_layer_name(gdn.prefix))
    if _CS_ON[0]:
        k_ = _mp_kernel(gdn.out_proj)
        kk, n = k_.config.partition_weight_shape
        w_q, w_s, _ = k_._get_weight_params(gdn.out_proj)
        return torch.ops.vllm.atdev_cs_mp_linear(core.flatten(-2), w_q, w_s, gdn.out_proj.input_global_scale,
                                                  n, kk)
    out, _ = gdn.out_proj(core.flatten(-2))
    return out


def _attn_forward(attn, positions, xq, xs):
    qkv = linear_q(attn.qkv_proj, xq, xs)
    q, k, v, gate = attn._project_qkv_gate(qkv, positions)
    o = attn.attn(q, k, v)
    if gate is None:
        out, _ = attn.o_proj(o)
        return out
    oq, os_ = torch.ops.vllm.atdev_gate_quant(o, gate, _gs(attn.o_proj), _n(attn.o_proj))
    return linear_q(attn.o_proj, oq, os_, cs=_CS_ON[0])


def _mlp_forward(mlp, xq, xs, cs=False):
    if getattr(mlp.gate_up_proj, "_atdev_gu_perm", False):  # C32h: interleaved gate_up, SiLU in its epilogue
        kg, kd = _mp_kernel(mlp.gate_up_proj), _mp_kernel(mlp.down_proj)
        k_gu, n_gu = kg.config.partition_weight_shape
        k_dn, n_dn = kd.config.partition_weight_shape
        gq, gsw, _ = kg._get_weight_params(mlp.gate_up_proj)
        dq, dsw, _ = kd._get_weight_params(mlp.down_proj)
        return torch.ops.vllm.atdev_c32h_mlp(xq, xs, gq, gsw, dq, dsw, _gs(mlp.down_proj), n_gu, k_gu, n_dn, k_dn)
    gu = linear_q(mlp.gate_up_proj, xq, xs)
    q, s = torch.ops.vllm.atdev_silu_quant(gu, _gs(mlp.down_proj), _n(mlp.down_proj))
    return linear_q(mlp.down_proj, q, s, cs=cs)


def layer_forward(self, hidden_states, residual, positions, **kwargs):
    gdn = self.layer_type == "linear_attention"
    first = self.linear_attn.in_proj_qkvz if gdn else self.self_attn.qkv_proj
    ln = self.input_layernorm
    xb, xq, xs, residual = torch.ops.vllm.atdev_norm_quant(
        hidden_states, residual, ln.weight, ln.variance_epsilon, _gs(first), gdn, _n(first))
    if gdn:
        h = _gdn_forward(self.linear_attn, xb, xq, xs)
    else:
        h = _attn_forward(self.self_attn, positions, xq, xs)
    ln2 = self.post_attention_layernorm
    _, xq, xs, residual = torch.ops.vllm.atdev_norm_quant(
        h, residual, ln2.weight, ln2.variance_epsilon, _gs(self.mlp.gate_up_proj), False,
        _n(self.mlp.gate_up_proj))
    h = _mlp_forward(self.mlp, xq, xs, cs=_CS_DOWN[0] and not getattr(self, "_atdev_last", True))
    return h, residual


_REGISTERED = False


def register() -> None:
    global _REGISTERED
    import vllm
    from vllm.model_executor.models import qwen3_next as m
    from vllm.utils.torch_utils import direct_register_custom_op
    from atdev_serve.register import enabled

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written for vLLM {SUPPORTED_VLLM_VERSION}, "
                           f"found {vllm.__version__}")
    if triton is None:
        raise RuntimeError(f"atdev {OPT_NAME}: Triton required")
    if "marlin_plus" not in enabled() or os.environ.get("VLLM_MARLIN_INPUT_DTYPE") != "int8":
        raise RuntimeError(f"atdev {OPT_NAME}: needs ATDEV_OPT=marlin_plus and "
                           "VLLM_MARLIN_INPUT_DTYPE=int8 (W4A8 path only)")
    if _REGISTERED:
        return
    _REGISTERED = True
    direct_register_custom_op("atdev_norm_quant", norm_quant, fake_impl=_norm_quant_fake)
    direct_register_custom_op("atdev_silu_quant", _silu_quant_op, fake_impl=_silu_quant_fake)
    direct_register_custom_op("atdev_gate_quant", gate_quant, fake_impl=_gate_quant_fake)
    direct_register_custom_op("atdev_fq_linear", fq_linear, fake_impl=_fq_linear_fake)
    from atdev_serve import c32c_csum
    if c32c_csum.enabled():  # C32c consumer-sum (rounding-level: CUDA norm producer)
        _CS_ON[0] = True
        _CS_DOWN[0] = os.environ.get("ATDEV_C32C_CSUM") == "2"
        direct_register_custom_op("atdev_fq_linear_cs", fq_linear_cs, fake_impl=_fq_linear_fake)
        direct_register_custom_op("atdev_cs_mp_linear", cs_mp_linear, fake_impl=_cs_mp_linear_fake)
        # the last decoder layer's down output goes to the stock final norm -> never a csum placeholder.
        # Qwen3_5DecoderLayer (the Qwen3.8 checkpoints) overrides __init__ without calling ours: patch both.
        from vllm.model_executor.models import qwen3_5 as m35
        for cls_ in (m.Qwen3NextDecoderLayer, m35.Qwen3_5DecoderLayer):
            def _mk(oinit):
                def __init__(self, vllm_config, *args, **kwargs):
                    oinit(self, vllm_config, *args, **kwargs)
                    nl = vllm_config.model_config.hf_text_config.num_hidden_layers
                    self._atdev_last = self.layer_idx >= nl - 1
                return __init__
            cls_.__init__ = _mk(cls_.__init__)
        from atdev_serve import compile_cache
        compile_cache._SALTS["c32c_csum"] = "1"
        log.warning("atdev %s: C32c consumer-sum ON (o_proj / out_proj%s -> slice partials -> norm)",
                    OPT_NAME, " / down" if _CS_DOWN[0] else "")
    orig = m.Qwen3NextDecoderLayer.forward

    def forward(self, hidden_states, residual, positions, **kwargs):
        if self.layer_scale or getattr(self, "use_attn_reduce_scatter_for_moe", False) or \
                not _layer_eligible(self):
            return orig(self, hidden_states, residual, positions, **kwargs)
        return layer_forward(self, hidden_states, residual, positions, **kwargs)

    m.Qwen3NextDecoderLayer.forward = forward
    from atdev_serve import c32h_silu
    if c32h_silu.enabled():  # C32h: SiLU.mul -> s4 folded into the gate_up GEMM (weights interleaved at load)
        if _CS_DOWN[0]:
            raise RuntimeError(f"atdev {OPT_NAME}: ATDEV_C32H_SILU=1 does not support ATDEV_C32C_CSUM=2")
        c32h_silu.register()
    if _core_empty():  # the traced GDN core allocation differs (zeros -> empty): separate AOT artifact
        from atdev_serve import compile_cache
        compile_cache._SALTS["gdn_core_empty"] = "1"
    from atdev_serve import c32_ne
    if c32_ne.enabled():  # C32-NE: norm folded into the W4A4 GEMM epilogues (model-level forward)
        c32_ne.register()
    if os.environ.get("ATDEV_CONV_UPDATE_CFG"):  # bit-identical GDN conv launch config (conv_tune.py)
        from atdev_serve import conv_tune
        conv_tune.install()
        log.warning("atdev %s: causal_conv1d_update launch BLOCK_N,num_warps = %s", OPT_NAME,
                    conv_tune.CFG)
    log.warning("atdev %s: Qwen3NextDecoderLayer.forward -> fused norm/act + int8 quant, "
                "pre-quantised marlin_plus GEMMs", OPT_NAME)
