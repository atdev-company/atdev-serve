// C32n2-ba: GDN in_proj_ba (x [M, K] bf16 @ W[N, K]^T, N = 96, K = 5120, M <= 128) as ONE low-CTA launch that
// reproduces cuBLAS's result bit for bit (cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8, split-K S,
// + cublasLt::splitKreduce_kernel): per output element and k-split one sequential mma.m16n8k16 chain over the
// split's k16 steps from 0 (fp32), then the S split partials summed in fp32 in split order, bf16 RN.
// The split count per M and the reduce form are calibrated against torch.mm at load time (c32n_ba.py); every
// M without an exact match goes to torch.mm.
//
// Grid (N / 8 / NPC n-groups, S splits), NPC warps per CTA, warp = one n8 tile x all MT m16 tiles. The CTA's x / W
// k-chunk goes to smem by cp.async (k64 groups), fragments by ldmatrix. Partials to ws[s][row][n]; then a per-n-group
// barrier of the S CTAs (all co-resident: <= 48 CTAs), each CTA reduces 1/S of its group's elements in split order;
// the last CTA to leave resets the counters (graph-safe).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

struct BaArgs {
  const __nv_bfloat16* x; long long sx;
  const __nv_bfloat16* w; long long sw;
  __nv_bfloat16* out; long long so;
  float* ws; int* cnt;
  int M, N, K, S, ksz;  // ksz = k range per split (multiple of 16), last split takes the rest
  int mode;             // bit0: partials only (probe), bit1: reduce starts from 0.0f (else from p0)
};

__device__ __forceinline__ void cp_async16(uint32_t saddr, const void* g, bool pred) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(saddr), "l"(g), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x2(uint32_t& r0, uint32_t& r1, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
template <int N_>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_)); }
__device__ __forceinline__ void cp_wait_rt(int n) {
  switch (n) {
    case 0: cp_wait<0>(); break; case 1: cp_wait<1>(); break; case 2: cp_wait<2>(); break;
    case 3: cp_wait<3>(); break; case 4: cp_wait<4>(); break; case 5: cp_wait<5>(); break;
    case 6: cp_wait<6>(); break; default: cp_wait<7>(); break;
  }
}

// one CTA: n-group gi (NPC n8 tiles), split sp. smem: x rows [MT*16][KP] then W rows [NPC*8][KP], KP = ksz + 8
// (bf16, 16 B pad: ldmatrix conflict-free). cp.async in k64 groups, compute group by group.
template <int MT, int NPC>
__device__ __forceinline__ void ba_cta(const BaArgs& p, int gi, int sp, __nv_bfloat16* sm) {
  constexpr int NT = NPC * 32;
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int k0 = sp * p.ksz, kc = min(p.K, k0 + p.ksz) - k0;
  const int KP = p.ksz + 8;
  const int XR = MT * 16, WR = NPC * 8, n0 = gi * WR;
  const uint32_t sb = static_cast<uint32_t>(__cvta_generic_to_shared(sm));
  const int ngrp = (kc + 63) / 64;
  for (int gq = 0; gq < ngrp; ++gq) {
    const int c0 = gq * 8, c1 = min(kc / 8, c0 + 8), cw = c1 - c0;  // 16 B chunks of this k64 group
    for (int e = threadIdx.x; e < (XR + WR) * cw; e += NT) {
      const int r = e / cw, c = c0 + e % cw;
      const __nv_bfloat16* src;
      bool ok;
      if (r < XR) { ok = r < p.M; src = p.x + (long long)(ok ? r : 0) * p.sx + k0 + 8 * c; }
      else { ok = true; src = p.w + (long long)(n0 + r - XR) * p.sw + k0 + 8 * c; }
      cp_async16(sb + (uint32_t)((r * KP + 8 * c) * 2), src, ok);
    }
    asm volatile("cp.async.commit_group;\n" ::);
  }
  float acc[MT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.f;
  const int l15 = lane & 15;
  const uint32_t a_base = sb + (uint32_t)((((lane & 7) + ((lane >> 3) & 1) * 8) * KP + (lane >> 4) * 8) * 2);
  const uint32_t b_base = sb + (uint32_t)(((XR + wid * 8 + (l15 & 7)) * KP + (l15 >> 3) * 8) * 2);
  for (int gq = 0; gq < ngrp; ++gq) {
    cp_wait_rt(ngrp - 1 - gq);
    __syncthreads();
    const int s1 = min(kc, gq * 64 + 64) / 16;
    for (int st = gq * 4; st < s1; ++st) {
      uint32_t b0, b1;
      ldsm_x2(b0, b1, b_base + st * 32);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        uint32_t a[4];
        ldsm_x4(a, a_base + (uint32_t)(mt * 16 * KP * 2) + st * 32);
        mma16816(acc[mt], a, b0, b1);
      }
    }
  }
  const int g = lane >> 2, t = lane & 3;
  const int nt = gi * NPC + wid, col = nt * 8 + 2 * t;
  if (p.S == 1) {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = mt * 16 + h * 8 + g;
        if (r < p.M)
          *reinterpret_cast<__nv_bfloat162*>(p.out + (long long)r * p.so + col) =
              __floats2bfloat162_rn(acc[mt][2 * h], acc[mt][2 * h + 1]);
      }
    return;
  }
  const int MP = MT * 16;
  float* dst = p.ws + (long long)sp * MP * p.N + col;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int h = 0; h < 2; ++h)
      __stcg(reinterpret_cast<float2*>(dst + (long long)(mt * 16 + h * 8 + g) * p.N),
             make_float2(acc[mt][2 * h], acc[mt][2 * h + 1]));
  if (p.mode & 1) return;
  __threadfence();
  __syncthreads();
  int* arrive = p.cnt + 2 * gi;
  if (threadIdx.x == 0) {
    atomicAdd(arrive, 1);
    while (ld_acquire(arrive) < p.S) __nanosleep(20);
  }
  __syncthreads();
  const int GC = NPC * 8, E = p.M * GC;
  const int per = (E + p.S - 1) / p.S, e0 = sp * per, e1 = min(E, e0 + per);
  for (int e = e0 + (int)threadIdx.x; e < e1; e += NT) {
    const int r = e / GC, c = gi * GC + e % GC;
    const float* src = p.ws + (long long)r * p.N + c;
    float s = (p.mode & 2) ? __fadd_rn(0.f, __ldcg(src)) : __ldcg(src);
    for (int q = 1; q < p.S; ++q) s = __fadd_rn(s, __ldcg(src + (long long)q * MP * p.N));
    p.out[(long long)r * p.so + c] = __float2bfloat16_rn(s);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int* leave = p.cnt + 2 * gi + 1;
    if (atomicAdd(leave, 1) == p.S - 1) {
      atomicExch(arrive, 0);
      atomicExch(leave, 0);
    }
  }
}

template <int MT, int NPC>
__global__ void __launch_bounds__(NPC * 32) ba_kernel(BaArgs p) {
  extern __shared__ int4 smem_[];
  ba_cta<MT, NPC>(p, blockIdx.x, blockIdx.y, reinterpret_cast<__nv_bfloat16*>(smem_));
}

BaArgs make_ba(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int S,
               int ksz, int mode) {
  const int M = (int)x.size(0), K = (int)x.size(1), N = (int)w.size(0);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(M >= 1 && M <= 128 && w.size(1) == K && out.size(0) == M && out.size(1) == N);
  TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && out.stride(1) == 1 && x.stride(0) % 2 == 0 &&
              w.stride(0) % 2 == 0 && out.stride(0) % 2 == 0);
  TORCH_CHECK(ksz % 16 == 0 && ksz > 0 && (long long)ksz * S >= K && (long long)ksz * (S - 1) < K && K % 16 == 0);
  TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.numel() >= (long long)S * 128 * N);
  TORCH_CHECK(cnt.scalar_type() == at::kInt && cnt.numel() >= 2 * (N / 8));
  BaArgs p;
  p.x = (const __nv_bfloat16*)x.data_ptr(); p.sx = x.stride(0);
  p.w = (const __nv_bfloat16*)w.data_ptr(); p.sw = w.stride(0);
  p.out = (__nv_bfloat16*)out.data_ptr(); p.so = out.stride(0);
  p.ws = ws.data_ptr<float>(); p.cnt = cnt.data_ptr<int>();
  p.M = M; p.N = N; p.K = K; p.S = S; p.ksz = ksz; p.mode = mode;
  return p;
}

#define MT_SWITCH(MT_, ...)                       \
  switch (MT_) {                                  \
    case 1: { constexpr int MT = 1; __VA_ARGS__; break; } \
    case 2: { constexpr int MT = 2; __VA_ARGS__; break; } \
    case 3: { constexpr int MT = 3; __VA_ARGS__; break; } \
    case 4: { constexpr int MT = 4; __VA_ARGS__; break; } \
    case 5: { constexpr int MT = 5; __VA_ARGS__; break; } \
    case 6: { constexpr int MT = 6; __VA_ARGS__; break; } \
    case 7: { constexpr int MT = 7; __VA_ARGS__; break; } \
    default: { constexpr int MT = 8; __VA_ARGS__; break; } \
  }

}  // namespace

// npc = warps (n8 tiles) per CTA: 4 / 6 / 12 (N % (8 npc) == 0)
void ba_gemm(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t S,
             int64_t ksz, int64_t mode, int64_t npc) {
  BaArgs p = make_ba(x, w, out, ws, cnt, (int)S, (int)ksz, (int)mode);
  TORCH_CHECK(p.N % (8 * npc) == 0);
  const at::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t st = at::cuda::getCurrentCUDAStream(x.get_device());
  const int mt = (p.M + 15) / 16;
  dim3 grid(p.N / 8 / (int)npc, (int)S);
  const size_t sm = (size_t)(16 * mt + 8 * npc) * (ksz + 8) * 2;
  TORCH_CHECK(sm <= 160 * 1024, "c32n_ba: smem ", sm);
#define BA_L(NPCV)                                                                                     \
  MT_SWITCH(mt, ({ auto kf = ba_kernel<MT, NPCV>;                                                      \
                   cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, 160 * 1024); \
                   kf<<<grid, NPCV * 32, sm, st>>>(p); }))
  if (npc == 4) { BA_L(4); }
  else if (npc == 6) { BA_L(6); }
  else if (npc == 12) { BA_L(12); }
  else TORCH_CHECK(false, "npc 4 / 6 / 12");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ba_gemm", &ba_gemm, "in_proj_ba, cuBLAS-exact split-K order, one launch");
}
