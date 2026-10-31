// c32_prod: CUDA versions of the fused_quant W4A4 activation producers (c=32): act_a4 (SiLU.mul / gate
// -> s4) is bit-identical to fused_quant's Triton _silu_a4_kernel / _gate_a4_kernel. Every floating-point
// step follows the PTX Triton 3.x emits:
//   * sigmoid(g) = div.full.f32(1, ex2.approx.ftz.f32((0 - g) * log2e) + 1)
//   * s4 quant: s = max(amax, 1e-8) * fp32(1/7), q = clamp(rint(q1), -8, 7) with the branch-free
//     quotient q1 = fma(fma(-s, y*rc, y), rc, y*rc), rc = div.rn(1, s), which yields the same nibble as
//     div.rn.ftz(y, s) for every bf16 input
// and the s4 fragment layout of fused_quant._a4_pack_store / marlin_plus_a4 quant_a4.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float div_rn_ftz(float a, float b) {
  float r;
  asm("div.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float ex2_approx(float a) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float sigmoid_tl(float g) {
  float e = ex2_approx(__fmul_rn(__fsub_rn(0.0f, g), 1.44269502162933349609375f));
  return div_full(1.0f, __fadd_rn(e, 1.0f));
}
__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// 8 consecutive values y[0..7] of a 128-group held by lane s = (thread % 16) of a 16-lane group
// (values 8s .. 8s+7). Writes the s4 fragments + the group scale (fused_quant._a4_pack_store).
__device__ __forceinline__ void a4_pack8(const float (&y)[8], int s, int g, int row, int m_pad,
                                         int tmb, float gs, uint32_t* __restrict__ apk,
                                         float* __restrict__ as_) {
  float am = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; i++) am = fmaxf(am, fabsf(y[i]));
#pragma unroll
  for (int o = 1; o < 16; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
  const float sc = __fmul_rn(fmaxf(am, 9.99999993922529029077850282192230224609375e-9f),
                             0.142857149243354797363281250f);
  const int p = (s >> 1) & 1, a = s >> 2;
  uint32_t part[2] = {0u, 0u};
  const float rc = div_rn_ftz(1.0f, sc);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    float q0, r, q1;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(q0) : "f"(y[i]), "f"(rc));
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(-sc), "f"(q0), "f"(y[i]));
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(q1) : "f"(r), "f"(rc), "f"(q0));
    float q = rintf(q1);
    q = fminf(fmaxf(q, -8.0f), 7.0f);
    const uint32_t nib = (uint32_t)((int)q) & 15u;
    part[i >> 2] |= nib << (8 * (i & 3) + 4 * p);
  }
  const uint32_t o0 = __shfl_xor_sync(0xffffffffu, part[0], 2);
  const uint32_t o1 = __shfl_xor_sync(0xffffffffu, part[1], 2);
  if (p == 0) {
    const int b0 = 2 * (s & 1);
    const int kb = 2 * g + (a >> 1);
    const int rr = (row & 15) / 8 + 2 * (a & 1);
    const int base = (kb * tmb + row / 16) * 32 + (row & 7) * 4;
    apk[(base + b0) * 4 + rr] = part[0] | o0;
    apk[(base + b0 + 1) * 4 + rr] = part[1] | o1;
  }
  if (s == 0) as_[g * m_pad + row] = __fmul_rn(sc, gs);
}

// SiLU(g) * u (row = [g | u], N each) or a * sigmoid(gate) -> s4; CTA = GPC 128-groups of a row
template <int MODE, int GPC>  // MODE 0: silu-mul, 1: gate
__global__ void __launch_bounds__(GPC * 16) act_a4_kernel(
    const __nv_bfloat16* __restrict__ a, int sa, const __nv_bfloat16* __restrict__ b, int sb,
    const float* __restrict__ gs_ptr, uint32_t* __restrict__ apk, float* __restrict__ as_, int M,
    int N, int m_pad, int tmb) {
  const int row = blockIdx.y, t = threadIdx.x;
  const int c = blockIdx.x * GPC * 128 + 8 * t;
  if (c >= N) return;  // whole 16-lane groups drop out together (N % 128 == 0)
  const bool m = row < M;
  uint4 xa = make_uint4(0, 0, 0, 0), xb = make_uint4(0, 0, 0, 0);
  if (m) {
    if (MODE == 0) {
      xa = *reinterpret_cast<const uint4*>(a + (size_t)row * sa + c);
      xb = *reinterpret_cast<const uint4*>(a + (size_t)row * sa + N + c);
    } else {
      xa = *reinterpret_cast<const uint4*>(a + (size_t)row * sa + c);
      xb = *reinterpret_cast<const uint4*>(b + (size_t)row * sb + c);
    }
  }
  const __nv_bfloat16* ah = reinterpret_cast<const __nv_bfloat16*>(&xa);
  const __nv_bfloat16* bh = reinterpret_cast<const __nv_bfloat16*>(&xb);
  float y[8];
#pragma unroll
  for (int j = 0; j < 8; j++) {
    if (MODE == 0) {
      const float g = bf(ah[j]);
      y[j] = rbf(__fmul_rn(__fmul_rn(sigmoid_tl(g), g), bf(bh[j])));
    } else {
      y[j] = rbf(__fmul_rn(bf(ah[j]), sigmoid_tl(bf(bh[j]))));
    }
  }
  a4_pack8(y, t & 15, c / 128, row, m_pad, tmb, *gs_ptr, apk, as_);
}

}  // namespace

void act_a4(int64_t mode, torch::Tensor a, c10::optional<torch::Tensor> b, torch::Tensor gs,
            torch::Tensor apk, torch::Tensor as_, int64_t M, int64_t N, int64_t m_pad, int64_t tmb,
            int64_t gpc) {
  TORCH_CHECK(N % 256 == 0 && a.stride(1) == 1 && a.stride(0) % 8 == 0 && m_pad <= 16 * tmb);
  const at::cuda::OptionalCUDAGuard guard(a.device());
  auto st = at::cuda::getCurrentCUDAStream();
  const __nv_bfloat16* bp = nullptr;
  int sb = 0;
  if (mode == 1) {
    TORCH_CHECK(b.has_value() && b->stride(1) == 1 && b->stride(0) % 8 == 0);
    bp = (const __nv_bfloat16*)b->data_ptr();
    sb = b->stride(0);
  }
#define AA_LAUNCH(MD, G)                                                                        \
  act_a4_kernel<MD, G><<<dim3((N / 128 + G - 1) / G, m_pad), G * 16, 0, st>>>(                \
      (const __nv_bfloat16*)a.data_ptr(), a.stride(0), bp, sb, gs.data_ptr<float>(),            \
      (uint32_t*)apk.data_ptr(), as_.data_ptr<float>(), M, N, m_pad, tmb)
  TORCH_CHECK(gpc == 8, "act_a4: gpc must be 8");
  if (mode == 0) { AA_LAUNCH(0, 8); } else { AA_LAUNCH(1, 8); }
#undef AA_LAUNCH
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("act_a4", &act_a4);
}
