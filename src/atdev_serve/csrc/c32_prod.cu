// c32_prod: CUDA versions of the fused_quant W4A4 producers (c=32, MTP k=2, M=96). act_a4
// (SiLU.mul / gate -> s4) is bit-identical to fused_quant's Triton _silu_a4_kernel / _gate_a4_kernel
// (tests/test_c32_prod.py). norm_a4 matches _norm_a4_chunk_kernel's variance / rstd bit for bit but
// NOT the bf16 rounding of (rstd*x)*(w+1) on rare exact fp32 ties -> not used (fused_quant keeps
// Triton for the norm). Every floating-point step follows the PTX Triton 3.x emits:
//   * norm: fp32 x(+r); sum of squares per thread in the Triton blocked-layout element order
//     (sizePerThread 8, 8 warps: thread t holds k*2048 + 8t + j, k-major) as
//     fma(e0,e0, e1*e1), fma(e2,e2,.), ...; warp butterfly xor 16..1 (add own + other), lane-0
//     partials to smem, threads 0..7 butterfly xor 4,2,1; div.full.f32 by N, + eps,
//     rsqrt.approx.ftz.f32; xb = bf16((rstd * x) * (w + 1))
//   * sigmoid(g) = div.full.f32(1, ex2.approx.f32((0 - g) * log2e) + 1)
//   * s4 quant: s = max(amax, 1e-8) * fp32(1/7), q = clamp(rint(div.rn.ftz(y, s)), -8, 7)
// and the s4 fragment layout of fused_quant._a4_pack_store / marlin_plus_a4 quant_a4.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cstdint>

// C32h-mk (MK=1, ATDEV_C32_MK=1): ex2.approx.ftz.f32 in the sigmoid and the branch-free quotient
// q1 = fma(fma(-sc, y*rc, y), rc, y*rc), rc = div.rn(1, sc) in place of div.rn.ftz(y, sc): s4 nibbles proven equal
// for every bf16 amax / bf16 y and sigmoid bits equal for every bf16 g (scripts/c32hmk_exhaust.cu).
#ifndef MK
#define MK 0
#endif

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
__device__ __forceinline__ float rsqrt_approx_ftz(float a) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float ex2_approx(float a) {
  float r;
#if MK
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
#else
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
#endif
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
#if MK
  const float rc = div_rn_ftz(1.0f, sc);
#endif
#pragma unroll
  for (int i = 0; i < 8; i++) {
#if MK
    float q0, r, q1;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(q0) : "f"(y[i]), "f"(rc));
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(-sc), "f"(q0), "f"(y[i]));
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(q1) : "f"(r), "f"(rc), "f"(q0));
    float q = rintf(q1);
#else
    float q = rintf(div_rn_ftz(y[i], sc));
#endif
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

// one CTA (256 threads = 8 warps, Triton num_warps=8) per row of the padded batch (m_pad rows)
template <int NR, bool HAS_RES, bool WANT_BF16>
__global__ void __launch_bounds__(256) norm_a4_kernel(
    const __nv_bfloat16* __restrict__ x, int sx, const __nv_bfloat16* __restrict__ r, int sr,
    const __nv_bfloat16* __restrict__ w, const float* __restrict__ gs_ptr,
    __nv_bfloat16* __restrict__ xb, int sxb, __nv_bfloat16* __restrict__ ro, int sro,
    uint32_t* __restrict__ apk, float* __restrict__ as_, int M, int N, float eps, int m_pad,
    int tmb, float* __restrict__ dbg) {
  __shared__ float red[8];
  const int row = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const bool valid = row < M;
  float v[NR][8];
#pragma unroll
  for (int k = 0; k < NR; k++) {
    const int c = k * 2048 + 8 * t;
    const bool m = valid && c < N;
    uint4 xa = make_uint4(0, 0, 0, 0), ra = make_uint4(0, 0, 0, 0);
    if (m) {
      xa = *reinterpret_cast<const uint4*>(x + (size_t)row * sx + c);
      if (HAS_RES) ra = *reinterpret_cast<const uint4*>(r + (size_t)row * sr + c);
    }
    const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xa);
    const __nv_bfloat16* rh = reinterpret_cast<const __nv_bfloat16*>(&ra);
#pragma unroll
    for (int j = 0; j < 8; j++) v[k][j] = HAS_RES ? __fadd_rn(bf(xh[j]), bf(rh[j])) : bf(xh[j]);
  }
  // per-thread sum of squares in Triton's order: fma(e0, e0, e1*e1), then e2, e3, ...
  float acc = __fmul_rn(v[0][1], v[0][1]);
  acc = __fmaf_rn(v[0][0], v[0][0], acc);
#pragma unroll
  for (int j = 2; j < 8; j++) acc = __fmaf_rn(v[0][j], v[0][j], acc);
#pragma unroll
  for (int k = 1; k < NR; k++)
#pragma unroll
    for (int j = 0; j < 8; j++) acc = __fmaf_rn(v[k][j], v[k][j], acc);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
  if (lane == 0) red[warp] = acc;
  __syncthreads();
  if (warp == 0) {
    float s = lane < 8 ? red[lane] : 0.0f;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
    __syncwarp();
    if (lane == 0) red[0] = s;
  }
  __syncthreads();
  const float rstd = rsqrt_approx_ftz(__fadd_rn(eps, div_full(red[0], (float)N)));
  if (dbg != nullptr && t == 0) { dbg[2 * row] = red[0]; dbg[2 * row + 1] = rstd; }
  const float gs = *gs_ptr;
#pragma unroll
  for (int k = 0; k < NR; k++) {
    const int c = k * 2048 + 8 * t;
    if (k * 2048 >= N) break;  // whole repetition past the row (uniform)
    const bool m = valid && c < N;
    float y[8];
    __nv_bfloat16 rob[8], xbb[8];
    uint4 wa = make_uint4(0, 0, 0, 0);
    if (c < N) wa = *reinterpret_cast<const uint4*>(w + c);
    const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wa);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      rob[j] = __float2bfloat16_rn(v[k][j]);
      const float w1 = __fadd_rn(bf(wh[j]), 1.0f);
      xbb[j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(rstd, v[k][j]), w1));
      y[j] = m ? __bfloat162float(xbb[j]) : 0.0f;
    }
    if (m) {
      *reinterpret_cast<uint4*>(ro + (size_t)row * sro + c) = *reinterpret_cast<uint4*>(rob);
      if (WANT_BF16)
        *reinterpret_cast<uint4*>(xb + (size_t)row * sxb + c) = *reinterpret_cast<uint4*>(xbb);
    }
    // a 128-group = 16 consecutive threads; N % 128 == 0 so a group is all-in or all-out
    if (c < N) a4_pack8(y, t & 15, c / 128, row, m_pad, tmb, gs, apk, as_);
  }
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

void norm_a4(torch::Tensor x, c10::optional<torch::Tensor> r, torch::Tensor w, torch::Tensor gs,
             c10::optional<torch::Tensor> xb, torch::Tensor ro, torch::Tensor apk, torch::Tensor as_,
             int64_t M, double eps, int64_t m_pad, int64_t tmb, c10::optional<torch::Tensor> dbg) {
  const int N = x.size(1);
  TORCH_CHECK(N % 256 == 0 && N <= 8192 && x.stride(1) == 1 && x.stride(0) % 8 == 0);  // whole warps per group pair
  TORCH_CHECK(w.is_contiguous() && ro.stride(0) % 8 == 0 && m_pad <= 16 * tmb);
  int block = 2048;
  while (block < N) block <<= 1;
  const int nr = block / 2048;
  const bool has_res = r.has_value(), want_bf16 = xb.has_value();
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  const __nv_bfloat16* rp = has_res ? (const __nv_bfloat16*)r->data_ptr() : nullptr;
  const int srr = has_res ? r->stride(0) : 0;
  __nv_bfloat16* xbp = want_bf16 ? (__nv_bfloat16*)xb->data_ptr() : nullptr;
  const int sxb = want_bf16 ? xb->stride(0) : 0;
  if (has_res) TORCH_CHECK(srr % 8 == 0 && r->stride(1) == 1);
  if (want_bf16) TORCH_CHECK(sxb % 8 == 0 && xb->stride(1) == 1);
#define NA_LAUNCH(NRV, HR, WB)                                                                   \
  norm_a4_kernel<NRV, HR, WB><<<m_pad, 256, 0, st>>>(                                           \
      (const __nv_bfloat16*)x.data_ptr(), x.stride(0), rp, srr, (const __nv_bfloat16*)w.data_ptr(), \
      gs.data_ptr<float>(), xbp, sxb, (__nv_bfloat16*)ro.data_ptr(), ro.stride(0),              \
      (uint32_t*)apk.data_ptr(), as_.data_ptr<float>(), M, N, (float)eps, m_pad, tmb,            \
      dbg.has_value() ? dbg->data_ptr<float>() : nullptr)
#define NA_RES(NRV)                                       \
  if (has_res && want_bf16) NA_LAUNCH(NRV, true, true);   \
  else if (has_res) NA_LAUNCH(NRV, true, false);          \
  else if (want_bf16) NA_LAUNCH(NRV, false, true);        \
  else NA_LAUNCH(NRV, false, false);
  if (nr == 1) { NA_RES(1) } else if (nr == 2) { NA_RES(2) } else { NA_RES(4) }
#undef NA_RES
#undef NA_LAUNCH
}

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
#define AA_G(MD)                                                                                \
  if (gpc == 2) AA_LAUNCH(MD, 2); else if (gpc == 4) AA_LAUNCH(MD, 4);                          \
  else if (gpc == 8) AA_LAUNCH(MD, 8); else if (gpc == 16) AA_LAUNCH(MD, 16);                   \
  else TORCH_CHECK(false, "act_a4: gpc must be 2/4/8/16");
  if (mode == 0) { AA_G(0) } else { AA_G(1) }
#undef AA_G
#undef AA_LAUNCH
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("norm_a4", &norm_a4);
  m.def("act_a4", &act_a4);
}
