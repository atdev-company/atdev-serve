"""W3 stream-K GEMM (csrc/w3_sk/w3_sk.cu) for the c=1 c1_fuse x w3_mixed path (8 < M <= 16 target linears).

Opt-in: ATDEV_C1_W3_SK=1 -> c1_norm.lin_prod_w3 / lin_cons_w3 (and c1_fuse's P-partial count) route the supported
(N, K, mode) shapes here instead of the marlin_c1 W3 V2 kernels. Same planar weights (w3_mixed.pack_w3), the same
Marlin-permuted scales and gate_up interleave; persistent stream-K schedule, offset dequant with fp32 per-group
scaling (see the .cu header). Producer partials: T = N / 128 per row (the W3 V2 tables use N / tn).
Output vs V2: rounding-level (exact (u-4)*s products instead of bf16-rounded ones + other fp32 order), deterministic.
STATUS (2026-10-20): correct but SLOWER than V2 (256-set ~1.5x V2) -- dev / research path, default off.

ATDEV_W3_SK_DEPTH (2 | 3, default 3): cp.async ring depth (stages of 4 slabs per warp); ATDEV_W3_SK_GRID (default #SMs): persistent grid.
"""
from __future__ import annotations

import hashlib
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "csrc", "w3_sk", "w3_sk.cu")
TILE_N = 128
_MOD = None
_BUF: dict = {}


def enabled() -> bool:
    return os.environ.get("ATDEV_C1_W3_SK", "0") == "1"


def mod():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
        abl = os.environ.get("ATDEV_W3_SK_ABL", "0")  # dev-only ablation builds (wrong results)
        prof = os.environ.get("ATDEV_W3_SK_PROF", "0")  # dev-only per-CTA timeline (mod().prof())
        h = hashlib.sha256(open(SRC, "rb").read() + abl.encode() + prof.encode()).hexdigest()[:10]
        _MOD = load(name=f"atdev_w3_sk_{h}", sources=[SRC], verbose=False,
                    extra_cuda_cflags=["-O3", f"-DW3SK_ABL={abl}", f"-DW3SK_PROF={prof}", "-std=c++17", "-gencode=arch=compute_80,code=sm_80",
                                       "--expt-relaxed-constexpr", "-lineinfo",
                                       "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                                       "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                                       "-U__CUDA_NO_HALF2_OPERATORS__"])
    return _MOD


def supported(n: int, k: int, mode: int, m: int = 16) -> bool:
    return 8 < m <= 16 and (n, k, mode) in _SUPPORTED


_SUPPORTED = {(34816, 5120, 3), (5120, 17408, 1), (5120, 6144, 1), (16384, 5120, 2), (14336, 5120, 2)} | {
    (n, k, 0) for n, k in ((34816, 5120), (5120, 17408), (5120, 6144), (16384, 5120), (14336, 5120))}


def grid(dev) -> int:
    g = int(os.environ.get("ATDEV_W3_SK_GRID", "0"))
    return g if g > 0 else torch.cuda.get_device_properties(dev).multi_processor_count


def depth() -> int:
    return int(os.environ.get("ATDEV_W3_SK_DEPTH", "3"))


def bufs(dev) -> dict:
    b = _BUF.get(dev.index)
    if b is None:
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        # flags are reset by their consumer inside every launch (graph-replay safe); zero once here
        b = _BUF[dev.index] = dict(ws=torch.empty(sms * 16 * TILE_N, dtype=torch.float32, device=dev),
                                   flags=torch.zeros(sms, dtype=torch.int32, device=dev))
    return b


def gemm(a, q3, ws, n, k, mode, out, g=None, d=None, **kw):
    b = bufs(a.device)
    mod().gemm(a, q3, ws, out, n, k, mode, ws=b["ws"], flags=b["flags"], grid=g or grid(a.device),
               depth=d or depth(), **kw)
    return out


def lin_plain(a, q3, ws, n, k, **kw):
    out = torch.empty((a.shape[0], n), dtype=torch.bfloat16, device=a.device)
    return gemm(a, q3, ws, n, k, 0, out, **kw)


def lin_prod(x, q3, ws, n, k, res, nw, **kw):
    """c1_norm.lin_prod_w3 contract: (a = bf16(x' (1 + nw)), res_out = bf16(x'), P [M, N / 128])."""
    m = x.shape[0]
    a = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    ro = torch.empty_like(a)
    p = torch.empty((m, n // TILE_N), dtype=torch.float32, device=x.device)
    gemm(x, q3, ws, n, k, 1, a, R=res, W=nw, C2=ro, P=p, **kw)
    return a, ro, p


def lin_cons(a, q3, ws, n, k, p, n_norm, eps, silu=False, want_r=False, **kw):
    """c1_norm.lin_cons_w3 contract."""
    m = a.shape[0]
    out = torch.empty((m, n // 2 if silu else n), dtype=torch.bfloat16, device=a.device)
    r = torch.empty((m,), dtype=torch.float32, device=a.device) if want_r else None
    gemm(a, q3, ws, n, k, 3 if silu else 2, out, Pin=p, T_in=p.shape[1], n_norm=n_norm, eps=eps, Rout=r, **kw)
    return out, r
