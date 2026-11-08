// ATDEV_C1_BA_STREAM side-stream in_proj_ba accumulator without shared memory (c1_norm.ba_tot_cuda).
//
// tot[m, n] = sum_{s=0..7} ( mma.m16n8k16 chain over the 40 k16 steps of K-split s, from 0 ) -- the accumulation
// of c1_norm._ba_gemv_r_kernel (Triton tl.dot, BK 128, SPLIT 8: one sequential mma chain per output element and
// split, then tot = 0 + p0 + p1 + ... + p7 in fp32), stored unscaled (rstd is applied by ba_scale / GT_BAP).
// Grid (N / 8 n8-tiles, 8 splits) of ONE warp per CTA, no shared memory, ~60 registers: unlike the Triton kernel
// (smem-staged tl.dot) these CTAs fit next to the qkvz Marlin consumer's 2 CTAs / SM (whose dynamic smem fills
// the SM), so the side stream really runs concurrently instead of delaying Marlin CTAs. Rows >= M read as 0.
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int kSplit = 8;

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t ldg32(const __nv_bfloat16* p) {
  return __ldg(reinterpret_cast<const unsigned int*>(p));
}

template <int U>
__global__ void __launch_bounds__(32) ba_tot_kernel(const __nv_bfloat16* __restrict__ x, int64_t sx,
                                                    const __nv_bfloat16* __restrict__ w, int64_t sw,
                                                    float* __restrict__ out, float* __restrict__ ws,
                                                    int* __restrict__ tick, int M, int N, int K) {
  const int nt = blockIdx.x, sp = blockIdx.y, lane = threadIdx.x;
  const int g = lane >> 2, t = lane & 3;
  const int ks = K / kSplit, k0 = sp * ks, steps = ks / 16;
  const bool r0 = g < M, r1 = g + 8 < M;
  const __nv_bfloat16* xa = x + (int64_t)g * sx + k0 + 2 * t;
  const __nv_bfloat16* xb = x + (int64_t)(g + 8) * sx + k0 + 2 * t;
  const __nv_bfloat16* wb = w + (int64_t)(nt * 8 + g) * sw + k0 + 2 * t;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int s0 = 0; s0 < steps; s0 += U) {
    uint32_t a[U][4], b[U][2];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int c = 16 * (s0 + u);
      a[u][0] = r0 ? ldg32(xa + c) : 0u;
      a[u][1] = r1 ? ldg32(xb + c) : 0u;
      a[u][2] = r0 ? ldg32(xa + c + 8) : 0u;
      a[u][3] = r1 ? ldg32(xb + c + 8) : 0u;
      b[u][0] = ldg32(wb + c);
      b[u][1] = ldg32(wb + c + 8);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) mma16816(acc, a[u], b[u][0], b[u][1]);
  }
  // partial of split sp: ws[sp][16][N]
  float* dst = ws + (int64_t)sp * 16 * N + nt * 8 + 2 * t;
  *reinterpret_cast<float2*>(dst + (int64_t)g * N) = make_float2(acc[0], acc[1]);
  *reinterpret_cast<float2*>(dst + (int64_t)(g + 8) * N) = make_float2(acc[2], acc[3]);
  __threadfence();
  __syncwarp();
  int last = 0;
  if (lane == 0) last = atomicAdd(tick + nt, 1) == kSplit - 1;
  last = __shfl_sync(0xffffffffu, last, 0);
  if (!last) return;
  __threadfence();
  // 16 rows x 8 cols, 4 per lane: tot = 0 + p0 + ... + p7 (in order)
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const int idx = lane * 4 + e, r = idx >> 3, c = nt * 8 + (idx & 7);
    float tot = 0.f;
#pragma unroll
    for (int s = 0; s < kSplit; ++s) tot += __ldcg(ws + ((int64_t)s * 16 + r) * N + c);
    if (r < M) out[(int64_t)r * N + c] = tot;
  }
  __syncwarp();
  if (lane == 0) atomicExch(tick + nt, 0);
}

}  // namespace

void ba_tot(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws, torch::Tensor tick) {
  const int M = (int)x.size(0), K = (int)x.size(1), N = (int)w.size(0);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kFloat);
  TORCH_CHECK(M <= 16 && N % 8 == 0 && K % (16 * kSplit) == 0 && w.size(1) == K);
  TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && x.stride(0) % 2 == 0 && w.stride(0) % 2 == 0);
  TORCH_CHECK(out.is_contiguous() && out.size(0) == M && out.size(1) == N);
  TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.numel() >= kSplit * 16 * N);
  TORCH_CHECK(tick.scalar_type() == at::kInt && tick.numel() >= N / 8);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 4 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 4 == 0);
  const at::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  dim3 grid(N / 8, kSplit);
  const int steps = K / kSplit / 16;
  auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  if (steps % 8 == 0)
    ba_tot_kernel<8><<<grid, 32, 0, st>>>(xp, x.stride(0), wp, w.stride(0), out.data_ptr<float>(),
                                         ws.data_ptr<float>(), tick.data_ptr<int>(), M, N, K);
  else
    ba_tot_kernel<1><<<grid, 32, 0, st>>>(xp, x.stride(0), wp, w.stride(0), out.data_ptr<float>(),
                                         ws.data_ptr<float>(), tick.data_ptr<int>(), M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("ba_tot", &ba_tot); }
