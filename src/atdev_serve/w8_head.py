"""int8 g128 W8A16 head GEMM in Triton for the c=32 verify head F part (ATDEV_VSPLIT_FBITS=8 +
ATDEV_VSPLIT_FA8=2). ROUNDING-LEVEL vs bf16 like the Marlin shadow (symmetric int8, one fp32
scale per (row, 128-column group)); activations stay bf16 (no per-token int8 error).

Why: stock W8A16 Marlin (ShadowHead bits=8) runs M=96 as 64 + 32-row launches, each streaming
the 336 MB F weight (484 us at M=96, slower than bf16 cuBLAS 471 us); CUTLASS W8A8 is one pass
(265 us) but per-token int8 activations put F logit rel-err at 5e-2 (scripts/bench_vhead_f8_c32.py).
This kernel is one weight pass for M <= BM (128) with bf16 activations.

out[m, n] = sum_g s[n, g] * sum_{k in g} h[m, k] * q[n, k]
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _w8_kernel(h_ptr, q_ptr, s_ptr, o_ptr, M, N, K, sh, sq, ss, so,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    rk = tl.arange(0, BK)
    mm = rm < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for g in range(0, K // BK):
        k0 = g * BK
        a = tl.load(h_ptr + rm[:, None] * sh + (k0 + rk)[None, :], mask=mm[:, None], other=0.0)
        w = tl.load(q_ptr + rn[None, :] * sq + (k0 + rk)[:, None])          # [BK, BN] int8
        s = tl.load(s_ptr + rn * ss + g)                                     # [BN] fp32
        p = tl.dot(a, w.to(tl.bfloat16))                                     # exact ints in bf16
        acc += p * s[None, :]
    tl.store(o_ptr + rm[:, None] * so + rn[None, :], acc.to(tl.bfloat16), mask=mm[:, None])


class W8G128Head:
    def __init__(self, weight: torch.Tensor, group: int = 128):
        n, k = weight.shape
        assert k % group == 0 and n % 128 == 0
        w = weight.float().view(n, k // group, group)
        s = (w.abs().amax(-1) / 127.0).clamp_min(1e-12)                      # [N, G]
        self.q = torch.round(w / s[..., None]).clamp_(-127, 127).to(torch.int8).view(n, k)
        del w
        self.s = s.contiguous()
        self.size_n, self.size_k, self.group = n, k, group
        self.nbytes = self.q.numel() + self.s.numel() * 4
        self.cfg = dict(BN=64, num_warps=4, num_stages=4)

    def dequant(self) -> torch.Tensor:
        n, k = self.q.shape
        return (self.q.float().view(n, -1, self.group) * self.s[..., None]).view(n, k)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        m = h.shape[0]
        out = torch.empty((m, self.size_n), dtype=torch.bfloat16, device=h.device)
        if m == 0:
            return out
        h = h.contiguous()
        c = self.cfg
        for m0 in range(0, m, 128):   # one weight pass per 128 rows (c=32 verify: M <= 128)
            mb = min(128, m - m0)
            bm = max(16, triton.next_power_of_2(mb))
            hs, os_ = h[m0:m0 + mb], out[m0:m0 + mb]
            _w8_kernel[(self.size_n // c["BN"],)](
                hs, self.q, self.s, os_, mb, self.size_n, self.size_k, hs.stride(0),
                self.q.stride(0), self.s.stride(0), os_.stride(0), BM=bm, BN=c["BN"],
                BK=self.group, num_warps=c["num_warps"], num_stages=c["num_stages"])
        return out

    approx_logits = logits
