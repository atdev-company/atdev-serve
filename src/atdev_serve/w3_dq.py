"""PFX-s10 (opt-in ATDEV_W3_DQ=1): large-M W3 GEMM as dequant-to-bf16 scratch + cuBLAS.

The 3-bit planar Marlin weights (w3_mixed.pack_w3) of one layer are expanded to a natural-order bf16 [K, N]
scratch (w = bf16(q * s_group), one rounding) by a bandwidth-bound CUDA kernel, then torch.mm (cuBLAS) runs
the GEMM. Marlin W3 reaches ~195 TFLOP/s at M >= 512 on A100, cuBLAS bf16 ~270-290.  Decode (M < ATDEV_W3_DQ_MIN_M)
is untouched.  Not bit-identical to the Marlin path (weights rounded to bf16 before the MMA).
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("atdev_serve")
_MOD = None
_T: dict = {}
_SCR: dict = {}

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

// grid (ceil(N/64/8), K/16), block 256 = 8 warps; one warp = one 16x64 Marlin tile
__global__ void w3_dq_kernel(const int32_t* __restrict__ q3, const __nv_bfloat16* __restrict__ ms,
                             __nv_bfloat16* __restrict__ out, const uint16_t* __restrict__ tab,
                             const uint8_t* __restrict__ inv, int N, int K) {
  __shared__ uint16_t s_tab[1024];
  __shared__ uint8_t s_inv[64];
  __shared__ int8_t s_tile[8][1024];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // lane-major table: entry (t*8+j)*32+lane = stream position (lane*4+t)*8+j  (conflict-free reads)
  for (int i = tid; i < 1024; i += 256) { int l = i & 31, tj = i >> 5; s_tab[i] = tab[(l * 4 + (tj >> 3)) * 8 + (tj & 7)]; }
  if (tid < 64) s_inv[tid] = inv[tid];
  __syncthreads();
  const int r = blockIdx.y;
  const int tn = blockIdx.x * 8 + warp;
  if (tn * 64 >= N) return;
  const int64_t base = ((int64_t)r * 3) * (N / 2) + (int64_t)tn * 32 + lane;
  const uint32_t p0 = (uint32_t)q3[base], p1 = (uint32_t)q3[base + N / 2], p2 = (uint32_t)q3[base + N];
  int8_t* tile = s_tile[warp];
#pragma unroll
  for (int t = 0; t < 4; t++) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
      int u;
      if (t == 0) u = (p0 >> (4 * j)) & 7;
      else if (t == 1) u = (p1 >> (4 * j)) & 7;
      else if (t == 2) u = (p2 >> (4 * j)) & 7;
      else u = ((p0 >> (4 * j + 3)) & 1) | (((p1 >> (4 * j + 3)) & 1) << 1) | (((p2 >> (4 * j + 3)) & 1) << 2);
      tile[s_tab[(t * 8 + j) * 32 + lane]] = (int8_t)(u - 4);
    }
  }
  __syncwarp();
  const int g = r >> 3;  // 16 rows per tile row, 128 per group
  const int n = 2 * lane;
  const __nv_bfloat16* sp = ms + (int64_t)g * N + (int64_t)tn * 64;
  const float s0 = __bfloat162float(sp[s_inv[n]]), s1 = __bfloat162float(sp[s_inv[n + 1]]);
#pragma unroll
  for (int kk = 0; kk < 16; kk++) {
    float a = (float)tile[kk * 64 + n] * s0, b = (float)tile[kk * 64 + n + 1] * s1;
    __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    *reinterpret_cast<__nv_bfloat162*>(out + (int64_t)(r * 16 + kk) * N + (int64_t)tn * 64 + n) = v;
  }
}


// PFX-s14 int8 variant: out8[(c)*K + k] = round((q-4) * s_g[c] / smax[c] * 127/4), c = output column. One 16-byte store per column.
__global__ void w3_dq8_kernel(const int32_t* __restrict__ q3, const __nv_bfloat16* __restrict__ ms,
                              int8_t* __restrict__ out, const uint16_t* __restrict__ tab,
                              const uint8_t* __restrict__ inv, const float* __restrict__ smax, int N, int K) {
  __shared__ uint16_t s_tab[1024];
  __shared__ uint8_t s_inv[64];
  __shared__ int8_t s_tile[8][1024];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < 1024; i += 256) { int l = i & 31, tj = i >> 5; s_tab[i] = tab[(l * 4 + (tj >> 3)) * 8 + (tj & 7)]; }
  if (tid < 64) s_inv[tid] = inv[tid];
  __syncthreads();
  const int r = blockIdx.y;
  const int tn = blockIdx.x * 8 + warp;
  if (tn * 64 >= N) return;
  const int64_t base = ((int64_t)r * 3) * (N / 2) + (int64_t)tn * 32 + lane;
  const uint32_t p0 = (uint32_t)q3[base], p1 = (uint32_t)q3[base + N / 2], p2 = (uint32_t)q3[base + N];
  int8_t* tile = s_tile[warp];
#pragma unroll
  for (int t = 0; t < 4; t++) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
      int u;
      if (t == 0) u = (p0 >> (4 * j)) & 7;
      else if (t == 1) u = (p1 >> (4 * j)) & 7;
      else if (t == 2) u = (p2 >> (4 * j)) & 7;
      else u = ((p0 >> (4 * j + 3)) & 1) | (((p1 >> (4 * j + 3)) & 1) << 1) | (((p2 >> (4 * j + 3)) & 1) << 2);
      tile[s_tab[(t * 8 + j) * 32 + lane]] = (int8_t)(u - 4);
    }
  }
  __syncwarp();
  const int g = r >> 3;
  const int n = 2 * lane;
  const __nv_bfloat16* sp = ms + (int64_t)g * N + (int64_t)tn * 64;
  const int c0 = tn * 64 + n;
  const float f0 = __bfloat162float(sp[s_inv[n]]) / smax[c0] * 31.75f, f1 = __bfloat162float(sp[s_inv[n + 1]]) / smax[c0 + 1] * 31.75f;
  int8_t o0[16], o1[16];
#pragma unroll
  for (int kk = 0; kk < 16; kk++) {
    o0[kk] = (int8_t)__float2int_rn((float)tile[kk * 64 + n] * f0);
    o1[kk] = (int8_t)__float2int_rn((float)tile[kk * 64 + n + 1] * f1);
  }
  *reinterpret_cast<uint4*>(out + (int64_t)c0 * K + r * 16) = *reinterpret_cast<uint4*>(o0);
  *reinterpret_cast<uint4*>(out + (int64_t)(c0 + 1) * K + r * 16) = *reinterpret_cast<uint4*>(o1);
}

void w3_dq8(torch::Tensor q3, torch::Tensor ms, torch::Tensor out, torch::Tensor tab, torch::Tensor inv, torch::Tensor smax,
            int64_t N, int64_t K) {
  TORCH_CHECK(q3.is_contiguous() && ms.is_contiguous() && out.is_contiguous() && smax.is_contiguous());
  TORCH_CHECK(N % 64 == 0 && K % 128 == 0);
  dim3 grid((unsigned)((N / 64 + 7) / 8), (unsigned)(K / 16));
  w3_dq8_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const int32_t*)q3.data_ptr(), (const __nv_bfloat16*)ms.data_ptr(), (int8_t*)out.data_ptr(),
      (const uint16_t*)tab.data_ptr(), (const uint8_t*)inv.data_ptr(), (const float*)smax.data_ptr(), (int)N, (int)K);
}

void w3_dq(torch::Tensor q3, torch::Tensor ms, torch::Tensor out, torch::Tensor tab, torch::Tensor inv,
           int64_t N, int64_t K) {
  TORCH_CHECK(q3.is_contiguous() && ms.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(N % 64 == 0 && K % 128 == 0);
  dim3 grid((unsigned)((N / 64 + 7) / 8), (unsigned)(K / 16));
  w3_dq_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const int32_t*)q3.data_ptr(), (const __nv_bfloat16*)ms.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
      (const uint16_t*)tab.data_ptr(), (const uint8_t*)inv.data_ptr(), (int)N, (int)K);
}
"""
CPP_SRC = "void w3_dq8(torch::Tensor q3, torch::Tensor ms, torch::Tensor out, torch::Tensor tab, torch::Tensor inv, torch::Tensor smax, int64_t N, int64_t K);\nvoid w3_dq(torch::Tensor q3, torch::Tensor ms, torch::Tensor out, torch::Tensor tab, torch::Tensor inv, int64_t N, int64_t K);"


def enabled() -> bool:
    return os.environ.get("ATDEV_W3_DQ", "0") == "1"


def min_m() -> int:
    return int(os.environ.get("ATDEV_W3_DQ_MIN_M", "256"))


def mod():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
        _MOD = load_inline(name="atdev_w3_dq_v3", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                           functions=["w3_dq", "w3_dq8"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"],
                           verbose=False)
    return _MOD


def tables(dev):
    t = _T.get(dev)
    if t is None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import get_scale_perms
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (get_weight_perm,
                                                                                      marlin_permute_weights)
        idx = torch.arange(16 * 64, dtype=torch.int64).reshape(16, 64)
        p = marlin_permute_weights(idx, 16, 64, get_weight_perm(4))  # [1, 1024]: stream position -> k*64+n
        tab = p.reshape(-1).to(torch.int16)
        tab = tab.to(dev).contiguous()
        sp, _ = get_scale_perms()
        inv = torch.empty(64, dtype=torch.uint8)
        for j, n in enumerate(sp):
            inv[n] = j
        t = _T[dev] = (tab, inv.to(dev))
    return t


def scratch(dev, numel: int) -> torch.Tensor:
    b = _SCR.get(dev)
    if b is None or b.numel() < numel:
        b = _SCR[dev] = torch.empty(numel, dtype=torch.bfloat16, device=dev)
    return b[:numel]


def prealloc(dev, numel: int) -> None:
    """Called at weight-load time so the scratch never gets allocated inside a CUDA-graph capture."""
    scratch(dev, max(numel, 34816 * 5120))  # largest W3 layer (gate_up); never regrow
    tables(dev)
    mod()


def dq_linear(x: torch.Tensor, q3: torch.Tensor, ms: torch.Tensor, n: int, k: int) -> torch.Tensor:
    tab, inv = tables(x.device)
    w = scratch(x.device, n * k).view(k, n)
    mod().w3_dq(q3, ms, w, tab, inv, n, k)
    return torch.mm(x, w)


# ---------------- PFX-s14: ATDEV_W3_I8=1 -> prefill GEMM as int8 tensor-core matmul (per-token act quant, per-channel requantised weight) -------
_SMAX: dict = {}
_S8: dict = {}


def i8_enabled() -> bool:
    return os.environ.get("ATDEV_W3_I8", "0") == "1"


def i8_min_m() -> int:
    return int(os.environ.get("ATDEV_W3_I8_MIN_M", "1024"))


_TRI = None


def _tri():
    global _TRI
    if _TRI is not None:
        return _TRI
    import triton
    import triton.language as tl

    @triton.jit
    def quant_rows(x_ptr, q_ptr, s_ptr, K, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        m = tl.zeros([BLOCK], dtype=tl.float32)
        for k0 in range(0, K, BLOCK):
            v = tl.load(x_ptr + row * K + k0 + offs, mask=k0 + offs < K, other=0.0).to(tl.float32)
            m = tl.maximum(m, tl.abs(v))
        amax = tl.maximum(tl.max(m, 0), 1e-8)
        sc = amax / 127.0
        tl.store(s_ptr + row, sc)
        inv = 127.0 / amax
        for k0 in range(0, K, BLOCK):
            v = tl.load(x_ptr + row * K + k0 + offs, mask=k0 + offs < K, other=0.0).to(tl.float32)
            q = tl.floor(v * inv + 0.5)
            tl.store(q_ptr + row * K + k0 + offs, q.to(tl.int8), mask=k0 + offs < K)

    @triton.jit
    def epi(y_ptr, sx_ptr, cs_ptr, o_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
        pm = tl.program_id(0); pn = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN)
        mk = (rm[:, None] < M) & (rn[None, :] < N)
        y = tl.load(y_ptr + rm[:, None] * N + rn[None, :], mask=mk, other=0).to(tl.float32)
        sx = tl.load(sx_ptr + rm, mask=rm < M, other=0.0)
        cs = tl.load(cs_ptr + rn, mask=rn < N, other=0.0)
        tl.store(o_ptr + rm[:, None] * N + rn[None, :], (y * sx[:, None] * cs[None, :]).to(tl.bfloat16), mask=mk)
    _TRI = (quant_rows, epi, triton)
    return _TRI


_W8: dict = {}   # q3.data_ptr() -> (w8 [n,k] int8, cs [n] f32), built lazily from a preallocated slot
_FREE: dict = {}  # (n,k) -> [preallocated int8 slots]  (allocated at weight-load time so the memory profiler sees them)


def i8_reserve(dev, n: int, k: int) -> None:
    if os.environ.get("ATDEV_W3_I8_PERSIST", "1") == "1":
        _FREE.setdefault((n, k), []).append(torch.empty((n, k), dtype=torch.int8, device=dev))


def _build(q3, ms, n, k, slot):
    dev = q3.device
    tab, inv = tables(dev)
    w = scratch(dev, n * k).view(k, n)
    mod().w3_dq(q3, ms, w, tab, inv, n, k)
    cs = torch.empty(n, dtype=torch.float32, device=dev)
    for c0 in range(0, n, 4096):
        blk = w[:, c0:c0 + 4096].float()
        c = (blk.abs().amax(0) / 127.0).clamp_min(1e-8)
        cs[c0:c0 + 4096] = c
        slot[c0:c0 + 4096].copy_(torch.round(blk.t() / c[:, None]).to(torch.int8))
    return slot, cs


def i8_linear(x: torch.Tensor, q3: torch.Tensor, ms: torch.Tensor, n: int, k: int, pre=None, dt=None) -> torch.Tensor:
    """pre = (xq int8 [m,k], sx f32 [m]) skips the activation quantisation (C1R2 ATDEV_PF_GLUE); then x may be None and dt is the out dtype."""
    dev = (x if x is not None else pre[0]).device
    dt = dt or x.dtype
    quant_rows, epi, triton = _tri()
    ent = _W8.get(q3.data_ptr())
    if ent is None and _FREE.get((n, k)):
        ent = _W8[q3.data_ptr()] = _build(q3, ms, n, k, _FREE[(n, k)].pop())
    if ent is not None:
        w8v, cs = ent
    else:  # no persistent slot: dequantise to int8 on the fly (slower)
        tab, inv = tables(dev)
        key = (ms.data_ptr(), n)
        e2 = _SMAX.get(key)
        if e2 is None:
            smax = ms.float().amax(0).contiguous()
            e2 = _SMAX[key] = (smax, (smax * (4.0 / 127.0)).contiguous())
        smax, cs = e2
        w8 = _S8.get(dev)
        if w8 is None or w8.numel() < n * k:
            w8 = _S8[dev] = torch.empty(max(n * k, 34816 * 5120), dtype=torch.int8, device=dev)
        w8v = w8[: n * k].view(n, k)
        mod().w3_dq8(q3, ms, w8v, tab, inv, smax, n, k)
    if pre is not None:
        xq, sx = pre
        m = xq.shape[0]
    else:
        m = x.shape[0]
        xq = torch.empty((m, k), dtype=torch.int8, device=dev)
        sx = torch.empty(m, dtype=torch.float32, device=dev)
        quant_rows[(m,)](x, xq, sx, k, BLOCK=4096)
    global _CUT
    if _CUT:
        try:
            from vllm import _custom_ops as ops
            return ops.cutlass_scaled_mm(xq, w8v.t(), sx.view(m, 1), cs.view(n, 1), dt)  # fused scale epilogue
        except Exception as e:  # noqa: BLE001
            log.warning("atdev w3 i8: cutlass_scaled_mm failed (%s) -> _int_mm + epi", e)
            _CUT = False
    y = torch._int_mm(xq, w8v.t())
    out = torch.empty((m, n), dtype=dt, device=dev)
    epi[(triton.cdiv(m, 64), triton.cdiv(n, 128))](y, sx, cs, out, m, n, BM=64, BN=128)
    return out


_CUT = os.environ.get("ATDEV_W3_I8_CUTLASS", "1") == "1"
