"""ATDEV_PF_FUSE=1: Triton kernels for the large-M (prefill, M > 16) fallbacks of c1_fuse.

At M beyond the marlin_c1 epilogue range c1_fuse runs eager torch fp32 math around the W3 GEMM:
  * _unperm_silu   : g4.float(); g / (1 + exp(-g)) * u  -> bf16   (gate/up un-permute + SiLU*mul)
  * e6w3_prod tail : xf = h.float() + res.float(); sum(xf^2); xf * (nw + 1) -> bf16; xf -> bf16
  * e6w3_cons head : (a.float() * r[:, None]) -> bf16
These are ~380 ms of the 1.56 s of a 3.8k-token prefill (profiled). Each is
replaced by one kernel with the same elementwise fp32 operations in the same order (exp = libdevice
expf like torch, division IEEE round-to-nearest); the only difference is the reduction order of
sum(xf^2) (fp32), which feeds the next layer's rstd.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


def enabled() -> bool:
    return os.environ.get("ATDEV_PF_FUSE", "0") == "1"


@triton.jit
def _silu_unperm_kernel(gu_ptr, out_ptr, n_half, BLK: tl.constexpr):
    # gu row = [n/128 groups][gate 64 | up 64]; out row = [n/128 groups][64]
    row = tl.program_id(0)
    grp = tl.program_id(1)
    c = tl.arange(0, 64)
    base = gu_ptr + row.to(tl.int64) * (n_half * 2) + grp * 128
    g = tl.load(base + c).to(tl.float32)
    u = tl.load(base + 64 + c).to(tl.float32)
    e = libdevice.exp(-g)
    s = libdevice.div_rn(g, 1.0 + e)
    y = s * u
    tl.store(out_ptr + row.to(tl.int64) * n_half + grp * 64 + c, y.to(tl.bfloat16))


def unperm_silu(gu: torch.Tensor) -> torch.Tensor:
    m, n = gu.shape
    gu = gu.contiguous()
    out = torch.empty((m, n // 2), dtype=torch.bfloat16, device=gu.device)
    _silu_unperm_kernel[(m, n // 128)](gu, out, n // 2, BLK=64, num_warps=2)
    return out


@triton.jit
def _prod_kernel(h_ptr, r_ptr, nw_ptr, o1_ptr, o2_ptr, p_ptr, p_stride, n, BLK: tl.constexpr):
    row = tl.program_id(0)
    off = tl.arange(0, BLK)
    msk = off < n
    b = row.to(tl.int64) * n
    xf = tl.load(h_ptr + b + off, mask=msk, other=0.0).to(tl.float32) + tl.load(r_ptr + b + off, mask=msk, other=0.0).to(tl.float32)
    ss = tl.sum(xf * xf, axis=0)
    nw = tl.load(nw_ptr + off, mask=msk, other=0.0).to(tl.float32)
    tl.store(o1_ptr + b + off, (xf * (nw + 1.0)).to(tl.bfloat16), mask=msk)
    tl.store(o2_ptr + b + off, xf.to(tl.bfloat16), mask=msk)
    tl.store(p_ptr + row.to(tl.int64) * p_stride, ss)


def prod_tail(h: torch.Tensor, res: torch.Tensor, nw: torch.Tensor, t: int):
    m, n = h.shape
    h = h.contiguous(); res = res.contiguous()
    o1 = torch.empty_like(h); o2 = torch.empty_like(h)
    p = torch.zeros((m, t), dtype=torch.float32, device=h.device)
    _prod_kernel[(m,)](h, res, nw, o1, o2, p, p.stride(0), n, BLK=triton.next_power_of_2(n), num_warps=8)
    return o1, o2, p


@triton.jit
def _scale_kernel(a_ptr, r_ptr, o_ptr, n, BLK: tl.constexpr):
    row = tl.program_id(0)
    off = tl.arange(0, BLK)
    msk = off < n
    b = row.to(tl.int64) * n
    r = tl.load(r_ptr + row)
    x = tl.load(a_ptr + b + off, mask=msk, other=0.0).to(tl.float32)
    tl.store(o_ptr + b + off, (x * r).to(tl.bfloat16), mask=msk)


def row_scale(a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    m, n = a.shape
    a = a.contiguous()
    o = torch.empty_like(a)
    _scale_kernel[(m,)](a, r.contiguous(), o, n, BLK=triton.next_power_of_2(n), num_warps=8)
    return o
