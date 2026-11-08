"""C1R2 ATDEV_PF_GLUE=1: prefill glue fusion around the int8 persistent GEMM (bit-identical to the unfused sequence).

  * rs_quant   : (a * rstd[:,None]).to(bf16) -> per-row int8 quant   (was prefill_fuse.row_scale + w3_dq.quant_rows)
  * silu_quant : un-permute + SiLU*mul -> bf16 y, plus int8 quant of y (was prefill_fuse.unperm_silu + quant_rows);
                 y is still written (any consumer that does not find the quantised copy re-quantises it), the quantised copy is
                 handed to the down projection through a one-slot side channel keyed by the y buffer address.
Same fp32 elementwise ops, same rounding points (bf16 round before the amax / quant), same libdevice exp / div_rn.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


def enabled() -> bool:
    return os.environ.get("ATDEV_PF_GLUE", "0") == "1"


@triton.jit
def _rs_quant(a_ptr, r_ptr, q_ptr, s_ptr, K, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    r = tl.load(r_ptr + row)
    offs = tl.arange(0, BLOCK)
    mx = tl.zeros([BLOCK], dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        v = (tl.load(a_ptr + row * K + k0 + offs, mask=k0 + offs < K, other=0.0).to(tl.float32) * r).to(tl.bfloat16).to(tl.float32)
        mx = tl.maximum(mx, tl.abs(v))
    amax = tl.maximum(tl.max(mx, 0), 1e-8)
    tl.store(s_ptr + row, amax / 127.0)
    inv = 127.0 / amax
    for k0 in range(0, K, BLOCK):
        v = (tl.load(a_ptr + row * K + k0 + offs, mask=k0 + offs < K, other=0.0).to(tl.float32) * r).to(tl.bfloat16).to(tl.float32)
        tl.store(q_ptr + row * K + k0 + offs, tl.floor(v * inv + 0.5).to(tl.int8), mask=k0 + offs < K)


def rs_quant(a: torch.Tensor, r: torch.Tensor):
    m, k = a.shape
    a = a.contiguous()
    xq = torch.empty((m, k), dtype=torch.int8, device=a.device)
    sx = torch.empty(m, dtype=torch.float32, device=a.device)
    _rs_quant[(m,)](a, r.contiguous(), xq, sx, k, BLOCK=4096)
    return xq, sx


@triton.jit
def _silu_quant(gu_ptr, y_ptr, q_ptr, s_ptr, n_half, BLOCK: tl.constexpr):
    # gu row = [n/128 groups][gate 64 | up 64]; y / q row = [n/128 groups][64]
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mx = tl.zeros([BLOCK], dtype=tl.float32)
    for o0 in range(0, n_half, BLOCK):
        o = o0 + offs
        msk = o < n_half
        src = gu_ptr + row * (n_half * 2) + (o // 64) * 128 + (o % 64)
        g = tl.load(src, mask=msk, other=0.0).to(tl.float32)
        u = tl.load(src + 64, mask=msk, other=0.0).to(tl.float32)
        y = (libdevice.div_rn(g, 1.0 + libdevice.exp(-g)) * u).to(tl.bfloat16)
        mx = tl.maximum(mx, tl.abs(y.to(tl.float32)))
    amax = tl.maximum(tl.max(mx, 0), 1e-8)
    tl.store(s_ptr + row, amax / 127.0)
    inv = 127.0 / amax
    for o0 in range(0, n_half, BLOCK):
        o = o0 + offs
        msk = o < n_half
        src = gu_ptr + row * (n_half * 2) + (o // 64) * 128 + (o % 64)
        g = tl.load(src, mask=msk, other=0.0).to(tl.float32)
        u = tl.load(src + 64, mask=msk, other=0.0).to(tl.float32)
        y = (libdevice.div_rn(g, 1.0 + libdevice.exp(-g)) * u).to(tl.bfloat16)
        tl.store(y_ptr + row * n_half + o, y, mask=msk)
        tl.store(q_ptr + row * n_half + o, tl.floor(y.to(tl.float32) * inv + 0.5).to(tl.int8), mask=msk)


_SLOT: dict = {}


def silu_quant(gu: torch.Tensor) -> torch.Tensor:
    m, n = gu.shape
    gu = gu.contiguous()
    nh = n // 2
    y = torch.empty((m, nh), dtype=torch.bfloat16, device=gu.device)
    xq = torch.empty((m, nh), dtype=torch.int8, device=gu.device)
    sx = torch.empty(m, dtype=torch.float32, device=gu.device)
    _silu_quant[(m,)](gu, y, xq, sx, nh, BLOCK=2048, num_warps=8)
    _SLOT["e"] = (y.data_ptr(), tuple(y.shape), y, xq, sx)  # keeps y alive so the address cannot be reused while the slot holds it
    return y


def take(x: torch.Tensor):
    """quantised copy for activation x if it is the buffer produced by the last silu_quant (consumed once), else None"""
    e = _SLOT.get("e")
    if e is None:
        return None
    if x.data_ptr() == e[0] and tuple(x.shape) == e[1] and x.is_contiguous():
        _SLOT.pop("e")
        return e[3], e[4]
    return None
