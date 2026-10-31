"""Kernel wrappers for c1_fuse: the norm / SiLU fused epilogues of the marlin_c1 W3 GEMMs,
the in_proj_ba split-K GEMV with the row rstd applied, and its side-stream variant (csrc/ba_side.cu)."""
from __future__ import annotations

import json
import os

import torch


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

if triton is not None:

    @triton.jit
    def _ba_gemv_r_kernel(x_ptr, stride_x, w_ptr, stride_w, o_ptr, stride_o, ws_ptr, tick_ptr, r_ptr,
                          M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                          SPLIT: tl.constexpr):
        # split-K GEMV (as ba_gemv._ba_gemv_kernel) with the output rows scaled by rstd r[m] (fp32) before rounding
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = tl.arange(0, BM)
        cols = pid_n * BN + tl.arange(0, BN)
        rmask = rows < M
        kk = tl.arange(0, BK)
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        k0 = pid_k * (K // SPLIT)
        for ks in range(0, K // SPLIT, BK):
            kidx = k0 + ks + kk
            xk = tl.load(x_ptr + rows[:, None] * stride_x + kidx[None, :], mask=rmask[:, None], other=0.0)
            wk = tl.load(w_ptr + cols[None, :] * stride_w + kidx[:, None])
            acc = tl.dot(xk, wk, acc)
        ws = ws_ptr + (pid_k * BM + rows)[:, None] * N + cols[None, :]
        tl.store(ws, acc)
        tl.debug_barrier()
        t = tl.atomic_add(tick_ptr + pid_n, 1, sem="acq_rel")
        if t == SPLIT - 1:
            tot = tl.zeros([BM, BN], dtype=tl.float32)
            for s in tl.static_range(SPLIT):
                tot += tl.load(ws_ptr + (s * BM + rows)[:, None] * N + cols[None, :])
            r = tl.load(r_ptr + rows, mask=rmask, other=0.0)
            tot = tot * r[:, None]
            tl.store(o_ptr + rows[:, None] * stride_o + cols[None, :], tot.to(o_ptr.dtype.element_ty),
                     mask=rmask[:, None])
            tl.atomic_xchg(tick_ptr + pid_n, 0)


_FB: dict = {}


def _fbufs(dev):
    b = _FB.get(dev.index)
    if b is None:
        from atdev_serve import marlin_plus
        mb = marlin_plus.bufs(dev)
        b = _FB[dev.index] = dict(
            locks=torch.zeros(mb["locks"].numel(), dtype=torch.int32, device=dev),
            c_tmp=mb["c_tmp"],  # fp32 split-K scratch: stream-ordered reuse is safe
            ws=torch.empty(32 * 64 * 256, dtype=torch.float32, device=dev),
            tick=torch.zeros(64, dtype=torch.int32, device=dev),
        )
    return b


def mp_cfg(n: int, k: int, m: int):
    from atdev_serve import marlin_plus
    ent = marlin_plus.lookup(n, k, m, False)
    return None if ent is None else list(ent["cfg"])


def ba_gemv_r(x: torch.Tensor, w: torch.Tensor, r: torch.Tensor, cfg=(8, 128, 4), out=None) -> torch.Tensor:
    split, bk, warps = cfg
    m, k = x.shape
    n = w.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    b = _fbufs(x.device)
    _ba_gemv_r_kernel[(n // 16, split)](x, x.stride(0), w, w.stride(0), out, out.stride(0), b["ws"], b["tick"], r,
                                        m, n, k, BM=16, BN=16, BK=bk, SPLIT=split, num_warps=warps)
    return out


# ---- ba_gemv_r split in two so the GEMV can run on a side stream before the qkvz consumer has produced rstd
# (c1_fuse): ba_tot = the _ba_gemv_r_kernel accumulation (same tiles, same split-K partials, same in-order
# 0..SPLIT-1 sum; csrc/ba_side.cu) stored as fp32 WITHOUT the rstd row scale;  ba_scale = bf16(tot * r).
# fp32 mul + RN round of the same fp32 operands -> bit-identical to ba_gemv_r.
if triton is not None:

    @triton.jit
    def _ba_scale_kernel(t_ptr, r_ptr, o_ptr, stride_o, M, N, BM: tl.constexpr, BN: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.arange(0, BN)
        msk = (rows < M)[:, None] & (cols < N)[None, :]
        tot = tl.load(t_ptr + rows[:, None] * N + cols[None, :], mask=msk, other=0.0)
        r = tl.load(r_ptr + rows, mask=rows < M, other=0.0)
        tot = tot * r[:, None]
        tl.store(o_ptr + rows[:, None] * stride_o + cols[None, :], tot.to(o_ptr.dtype.element_ty), mask=msk)


_SB: dict = {}


def _sbufs(dev):
    """side-stream ba_tot scratch (separate from _fbufs: may run concurrently with a main-stream ba_gemv_r)."""
    b = _SB.get(dev.index)
    if b is None:
        b = _SB[dev.index] = dict(ws=torch.empty(32 * 64 * 256, dtype=torch.float32, device=dev),
                                  tick=torch.zeros(64, dtype=torch.int32, device=dev))
    return b


def ba_tot_ok(x: torch.Tensor, w: torch.Tensor) -> bool:
    """shapes on which ba_gemv_r's kernel (and so ba_tot) is the production path (c1_fuse.e6_ba condition)."""
    return (x.shape[0] <= 16 and x.stride(-1) == 1 and w.stride(-1) == 1 and w.shape[0] % 16 == 0
            and x.shape[1] % 1024 == 0 and w.shape[0] <= 256)


_BA_SIDE_EXT = None


def ba_side_ext():
    """csrc/ba_side.cu (smem-free one-warp-per-CTA ba_tot, co-resides with the Marlin consumer)."""
    global _BA_SIDE_EXT
    if _BA_SIDE_EXT is None:
        from torch.utils.cpp_extension import load
        _BA_SIDE_EXT = load(name="atdev_ba_side", sources=[os.path.join(os.path.dirname(__file__), "csrc",
                                                                         "ba_side.cu")],
                            extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80", "-std=c++17"],
                            verbose=False)
    return _BA_SIDE_EXT


def ba_tot(x: torch.Tensor, w: torch.Tensor, out=None) -> torch.Tensor:
    """fp32 [M, N] unscaled ba accumulator of ba_gemv_r(x, w, r); out must be contiguous."""
    m = x.shape[0]
    n = w.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    assert out.is_contiguous() and out.shape == (m, n)
    b = _sbufs(x.device)
    ba_side_ext().ba_tot(x, w, out, b["ws"], b["tick"])
    return out


def ba_scale(tot: torch.Tensor, r: torch.Tensor, m: int, out=None) -> torch.Tensor:
    """bf16(tot[:m] * r[:, None]) == ba_gemv_r's epilogue."""
    n = tot.shape[1]
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=tot.device)
    assert tot.is_contiguous() and tot.shape[0] >= m and r.shape[0] >= m
    _ba_scale_kernel[(triton.cdiv(m, 16),)](tot, r, out, out.stride(0), m, n, BM=16, BN=triton.next_power_of_2(n), num_warps=4)
    return out


# ---- W3 (3-bit planar weights, w3_mixed) variants: csrc/marlin_c1 + the W3 B stream (marlin_c1_w3_build)
W3C1_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w3_c1_cfg.json")
_W3C1_TABLE: dict | None = None


def mc1w3(mode: int):
    from atdev_serve import marlin_c1_w3_build
    return marlin_c1_w3_build.load(mode)


def w3c1_table() -> dict:
    """{"N,K": {"16": [tk, tn, threads, stages, bps, max_tmb]}} (w3_c1_cfg.json)"""
    global _W3C1_TABLE
    if _W3C1_TABLE is None:
        _W3C1_TABLE = json.load(open(W3C1_CONFIG))
    return _W3C1_TABLE


def w3c1_cfg(n: int, k: int, m: int):
    if not 8 < m <= 16:
        return None
    ent = w3c1_table().get(f"{n},{k}", {}).get("16")
    if ent is None:
        return None
    return list(ent["cfg"] if isinstance(ent, dict) else ent)


def lin_prod_w3(x, q3, ws, n, k, res, nw):
    """lin_prod on W3 planar weights (q3 = w3_mixed.pack_w3 layout)."""
    m = x.shape[0]
    cfg = w3c1_cfg(n, k, m)
    t = n // cfg[1]
    a = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    ro = torch.empty_like(a)
    p = torch.empty((m, t), dtype=torch.float32, device=x.device)
    b = _fbufs(x.device)
    mc1w3(1).gemm_epi(x, q3, ws, a, b["locks"], b["c_tmp"], n, k, cfg, 1, R=res, W=nw, C2=ro, P=p)
    return a, ro, p


def lin_cons_w3(a, q3, ws, n, k, p, n_norm, eps, silu=False, want_r=False):
    """lin_cons on W3 planar weights."""
    m = a.shape[0]
    cfg = w3c1_cfg(n, k, m)
    out = torch.empty((m, n // 2 if silu else n), dtype=torch.bfloat16, device=a.device)
    r = torch.empty((m,), dtype=torch.float32, device=a.device) if want_r else None
    b = _fbufs(a.device)
    mc1w3(3 if silu else 2).gemm_epi(a, q3, ws, out, b["locks"], b["c_tmp"], n, k, cfg, 3 if silu else 2, Pin=p,
                                     T_in=p.shape[1], n_norm=n_norm, eps=eps, Rout=r)
    return out, r


def permute_gu_w3(q3: torch.Tensor, ws: torch.Tensor, n: int):
    """W3 planar gate_up [K/16, 3, N/2] int32 (+ scales) -> n64-tile interleaved copies. Plane word j of a k16
    row <-> int4 chunk j of the Marlin layout, an n64 tile = 32 chunks, so the tiles move as 32-word blocks."""
    idx = gu_perm_index(n // 2, q3.device)
    kt = q3.shape[0]
    q2 = q3.view(kt, 3, n // 64, 32)[:, :, idx].reshape(q3.shape).contiguous()
    g = ws.shape[0]
    ws2 = ws.view(g, n // 64, 64)[:, idx].reshape(ws.shape).contiguous()
    return q2, ws2


def gu_perm_index(n_half: int, dev) -> torch.Tensor:
    """n64-tile order of the interleaved gate_up weight: tile 2t = gate tile t, 2t + 1 = up tile t."""
    t = n_half // 64
    return torch.stack([torch.arange(t, device=dev), torch.arange(t, device=dev) + t], 1).reshape(-1)
