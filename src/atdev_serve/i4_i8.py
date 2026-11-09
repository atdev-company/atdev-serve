"""I4I8 (ATDEV_I4_I8=1): large-M prefill GEMMs of the stock INT4 (uint4b8 g128) target without the W3 overlay.

Decode / small M (M < ATDEV_I4_MIN_M) keeps the marlin_plus W4A16 kernels untouched. For M >= MIN_M the layer runs

  mode i8  (default): persistent per-output-channel int8 copy of the weights (built once at load time, 1 byte/weight),
                      per-token int8 activation quantisation (w3_dq.quant_rows, amax/127, no clip), cutlass_scaled_mm with the
                      scale epilogue (the same GEMM as the W3 ATDEV_W3_I8 path, w3_dq.i8_linear).
  mode dq             : INT4 -> bf16 scratch dequant (w = bf16(code * s_g), same rounding point as Marlin) per call + cuBLAS bf16
                      GEMM. No persistent memory (one shared 356 MB scratch), bf16 activations.

Weight expansion (decision): int4 code q in [-8, 7] times the bf16 group scale s_g (g128 along K). A plain int8 GEMM can
only apply one scale per output channel, so the group scales must be folded into the integers:
  w8[c, k] = round(q[c, k] * s_g(c, k) / cs[c]),  cs[c] = max_k |q * s_g| / 127   (identical to the W3 path, w3_dq._build).
"code << 4 with scale / 16" is exact per group but still needs one scale per group, i.e. a group-scaled int8 GEMM (not
available in cutlass_scaled_mm), so it is not used. The per-channel fold is NOT lossless when the group scales of a channel
differ.  The error is about 0.9% of the weights (prefill only); the end effect is covered by the accuracy gates.
"""
from __future__ import annotations

import logging
import os

import torch

OPT_NAME = "i4_i8"
log = logging.getLogger("atdev_serve")
MODE = os.environ.get("ATDEV_I4_MODE", "i8")
MIN_M = int(os.environ.get("ATDEV_I4_MIN_M", "64"))
_MOD = None
_STATE = {"n": 0, "bytes": 0}
_PTRS: set = set()   # w_q.data_ptr() of the converted target layers
_ORIG_MP = None
_PG = os.environ.get("ATDEV_PF_GLUE", "0") == "1"   # fused rstd-rowscale/SiLU*mul + int8 activation quant (pf_glue.py)

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Marlin uint4b8 layout: int32 [K/16, 2N]; tile (r, tn) = 128 consecutive int32 = 1024 nibbles (q + 8), stream position
// (lane*4 + t)*8 + j holds tile element tab[...] (k*64 + n).  grid (ceil(N/64/8), K/16), block 256 = 8 warps, one warp = one tile.
__global__ void i4_dq_kernel(const int32_t* __restrict__ qw, const __nv_bfloat16* __restrict__ ms,
                             __nv_bfloat16* __restrict__ out, const uint16_t* __restrict__ tab,
                             const uint8_t* __restrict__ inv, int N, int K) {
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
  const int64_t base = (int64_t)r * (2 * (int64_t)N) + (int64_t)tn * 128 + lane * 4;
  uint32_t p[4];
  *reinterpret_cast<uint4*>(p) = *reinterpret_cast<const uint4*>(qw + base);
  int8_t* tile = s_tile[warp];
#pragma unroll
  for (int t = 0; t < 4; t++) {
#pragma unroll
    for (int j = 0; j < 8; j++) tile[s_tab[(t * 8 + j) * 32 + lane]] = (int8_t)(((p[t] >> (4 * j)) & 15) - 8);
  }
  __syncwarp();
  const int g = r >> 3;
  const int n = 2 * lane;
  const __nv_bfloat16* sp = ms + (int64_t)g * N + (int64_t)tn * 64;
  const float s0 = __bfloat162float(sp[s_inv[n]]), s1 = __bfloat162float(sp[s_inv[n + 1]]);
#pragma unroll
  for (int kk = 0; kk < 16; kk++) {
    float a = (float)tile[kk * 64 + n] * s0, b = (float)tile[kk * 64 + n + 1] * s1;
    *reinterpret_cast<__nv_bfloat162*>(out + (int64_t)(r * 16 + kk) * N + (int64_t)tn * 64 + n) = __floats2bfloat162_rn(a, b);
  }
}

void i4_dq(torch::Tensor qw, torch::Tensor ms, torch::Tensor out, torch::Tensor tab, torch::Tensor inv, int64_t N, int64_t K) {
  TORCH_CHECK(qw.is_contiguous() && ms.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(N % 64 == 0 && K % 128 == 0);
  dim3 grid((unsigned)((N / 64 + 7) / 8), (unsigned)(K / 16));
  i4_dq_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const int32_t*)qw.data_ptr(), (const __nv_bfloat16*)ms.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
      (const uint16_t*)tab.data_ptr(), (const uint8_t*)inv.data_ptr(), (int)N, (int)K);
}
"""
CPP_SRC = "void i4_dq(torch::Tensor qw, torch::Tensor ms, torch::Tensor out, torch::Tensor tab, torch::Tensor inv, int64_t N, int64_t K);"


def mod():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
        _MOD = load_inline(name="atdev_i4_dq_v1", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC], functions=["i4_dq"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"], verbose=False)
    return _MOD


def dq_bf16(qw, ms, n, k, out=None):
    from atdev_serve import w3_dq
    tab, inv = w3_dq.tables(qw.device)
    if out is None:
        out = torch.empty((k, n), dtype=torch.bfloat16, device=qw.device)
    mod().i4_dq(qw, ms, out, tab, inv, n, k)
    return out


def build_i8(qw, ms, n, k):
    """persistent int8 copy [n, k] + per-channel scale [n] f32 (registered in w3_dq._W8 under qw.data_ptr())"""
    from atdev_serve import w3_dq
    dev = qw.device
    w = w3_dq.scratch(dev, n * k).view(k, n)
    dq_bf16(qw, ms, n, k, out=w)
    slot = torch.empty((n, k), dtype=torch.int8, device=dev)
    cs = torch.empty(n, dtype=torch.float32, device=dev)
    for c0 in range(0, n, 4096):
        blk = w[:, c0:c0 + 4096].float()
        c = (blk.abs().amax(0) / 127.0).clamp_min(1e-8)
        cs[c0:c0 + 4096] = c
        slot[c0:c0 + 4096].copy_(torch.round(blk.t() / c[:, None]).to(torch.int8))
    return slot, cs


def rebuild(w_q, w_s, n, k):
    """weights were changed in place after load (c1_accuracy gate_up column permutation): refresh the int8 copy"""
    from atdev_serve import w3_dq
    if MODE == "i8" and w_q.data_ptr() in _PTRS:
        w3_dq._W8[w_q.data_ptr()] = build_i8(w_q.data, w_s.data, n, k)  # old slot freed, new one allocated (same size)


def i4_linear(x: torch.Tensor, qw: torch.Tensor, ms: torch.Tensor, gs: torch.Tensor, n: int, k: int, a8: bool) -> torch.Tensor:
    m = x.shape[0]
    if m < MIN_M:
        return _ORIG_MP(x, qw, ms, gs, n, k, a8)
    from atdev_serve import w3_dq
    if not x.is_contiguous():
        x = x.contiguous()
    if MODE == "dq":
        w = w3_dq.scratch(x.device, n * k).view(k, n)
        dq_bf16(qw, ms, n, k, out=w)
        return torch.mm(x, w)
    if _PG:
        from atdev_serve import pf_glue
        pre = pf_glue.take(x)
        if pre is not None:
            return w3_dq.i8_linear(x, qw, ms, n, k, pre=pre)
    return w3_dq.i8_linear(x, qw, ms, n, k)


def _fake(x: torch.Tensor, qw: torch.Tensor, ms: torch.Tensor, gs: torch.Tensor, n: int, k: int, a8: bool) -> torch.Tensor:
    return torch.empty((x.shape[0], n), dtype=x.dtype, device=x.device)


def mp_linear_i4(x: torch.Tensor, mq: torch.Tensor, ms: torch.Tensor, gs: torch.Tensor, n: int, k: int, a8: bool) -> torch.Tensor:
    """replaces marlin_plus.mp_linear for direct callers (c1_accuracy._mp: the fused no-W3 path bypasses apply_weights)"""
    if not a8 and x.shape[0] >= MIN_M and mq.data_ptr() in _PTRS:
        return i4_linear(x, mq, ms, gs, n, k, a8)
    return _ORIG_MP(x, mq, ms, gs, n, k, a8)


def register() -> None:
    global _ORIG_MP
    from vllm.utils.torch_utils import direct_register_custom_op
    from vllm.scalar_type import scalar_types
    from atdev_serve.compile_cache import salt_aot_cache
    from atdev_serve import compile_cache, marlin_plus
    salt_aot_cache(OPT_NAME, __file__)
    compile_cache._SALTS[OPT_NAME] += f"+{MODE}+{MIN_M}"
    if not hasattr(torch.ops.vllm, "atdev_i4_linear"):
        direct_register_custom_op("atdev_i4_linear", i4_linear, fake_impl=_fake)
    _ORIG_MP = marlin_plus.mp_linear
    marlin_plus.mp_linear = mp_linear_i4
    cls = marlin_plus.kernel_class()
    orig_pw, orig_apply = cls.process_weights_after_loading, cls.apply_weights

    def process_weights_after_loading(self, layer):
        r = orig_pw(self, layer)
        c = self.config
        if self._atdev_stock or c.weight_type != scalar_types.uint4b8 or c.group_size != 128 or c.act_type != torch.bfloat16:
            return r
        k, n = c.partition_weight_shape
        w_q, w_s, _ = self._get_weight_params(layer)
        if tuple(w_q.shape) != (k // 16, 2 * n) or w_q.dtype != torch.int32:
            log.warning("atdev i4_i8: %s unexpected shape %s -> stock", getattr(layer, "prefix", ""), tuple(w_q.shape))
            return r
        from atdev_serve import w3_dq
        w3_dq.prealloc(w_q.device, n * k)  # shared bf16 scratch + tables (+ the W3 dq module, unused)
        mod()
        if MODE == "i8":
            w3_dq._W8[w_q.data_ptr()] = build_i8(w_q.data, w_s.data, n, k)
            _STATE["bytes"] += n * k + 4 * n
        layer._atdev_i4 = True
        _PTRS.add(w_q.data_ptr())
        _STATE["n"] += 1
        return r

    def apply_weights(self, layer, x, bias=None):
        if not getattr(layer, "_atdev_i4", False) or bias is not None or x.dtype != torch.bfloat16:
            return orig_apply(self, layer, x, bias)
        k, n = self.config.partition_weight_shape
        w_q, w_s, _ = self._get_weight_params(layer)
        x2 = x.reshape(-1, x.shape[-1])
        out = torch.ops.vllm.atdev_i4_linear(x2, w_q, w_s, w_s, n, k, False)  # w_s: unused gs placeholder (bf16 path)
        return out.reshape(x.shape[:-1] + (n,))

    cls.process_weights_after_loading = process_weights_after_loading
    cls.apply_weights = apply_weights
    log.warning("atdev %s: mode %s, M >= %d -> %s", OPT_NAME, MODE, MIN_M, "persistent int8 + cutlass" if MODE == "i8" else "bf16 dequant + cuBLAS")
