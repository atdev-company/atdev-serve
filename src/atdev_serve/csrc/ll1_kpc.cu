// ll1_fuse kpc: DFlash2 drafter DFlashGroupedConv.prepare at the c=1 block (M = 16 rows) as ONE kernel:
//   kernel_projection (bf16 [16, 5120] x [1280, 5120]^T) + split-K reduce + side-0 grouped conv (taps 2, group 16).
// Stock: cuBLAS ampere_s16816gemm_bf16_64x64_ldg8_stages_64x6_tn grid [20, 1, 5] (split-K S = 5, 1024 k per split)
// + cublasLt splitKreduce_kernel + Triton _dflash2_grouped_conv_kernel (+ ctxfold's copy into the fold buffer).
// Bit-identity rests on:
//   * per output element and split: acc = 0; for every k16 chunk of the split in k order:
//     acc = mma.m16n8k16.row.col.f32.bf16 (canonical fragment layout, as ldmatrix of CUTLASS / cuBLAS)
//   * reduce: v = ((((0 + p0) + p1) + p2) + p3) + p4, out = bf16_rn(v)
//   * conv (Triton's PTX with fp fusion): c0 = base[0][c] + d[r][0][g], c1 = base[1][c] + d[r][1][g] (tap masked to
//     (0, 0) at position r % block == 0), acc = fma(c0, x[r][c], c1 * x[r-1][c]), out = bf16_rn(acc).
// Work split (no cross-CTA reduction): CTA = 8 groups of one side = 2 n8 tiles (tap-0 and tap-1 columns of the
// groups) x 5 splits = 10 warps; warp (tile, split) runs the split's sequential mma chain. The 5 split partials meet
// in shared memory and are summed in split order; side-0 CTAs then run the conv of their 128 channels. 80 CTAs.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

namespace {

constexpr int M = 16, K = 5120, N = 1280, S = 5, KS = K / S;  // 1024 k per split
constexpr int G = 320, CH = 5120;                             // groups, channels
constexpr int KT = 64, NKT = KS / KT;                         // k tile, tiles per split (16)
constexpr int STAGES = 4;
constexpr int GPC = 8;                       // groups per CTA
constexpr int WARPS = 2 * S;                 // (tile, split)
constexpr int THREADS = 32 * WARPS;          // 320
constexpr int NCTA = 2 * G / GPC;            // 80
constexpr int A_ST = S * M * KT;             // A elements per stage (5 splits x 16 rows x 64 k)
constexpr int B_ST = S * 2 * 8 * KT;         // B elements per stage (5 splits x 16 cols x 64 k)

__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void cp16(void* smem, const void* g) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int n>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(n)); }
__device__ __forceinline__ void ldm_x4(uint32_t* r, const void* smem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
// smem tile rows of 64 bf16 (128 B = 8 x 16 B chunks), chunk index swizzled by (row & 7)
__device__ __forceinline__ int swz(int row, int chunk) { return row * 64 + ((chunk ^ (row & 7)) << 3); }

struct alignas(16) V8 {
  __nv_bfloat16 v[8];
};

__global__ void __launch_bounds__(THREADS) kpc_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t sx, const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ base, int64_t sbase, __nv_bfloat16* __restrict__ kp,
    __nv_bfloat16* __restrict__ out, int64_t sout, int block_size) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smem_raw);  // [STAGES][S][16][64]
  __nv_bfloat16* sB = sA + STAGES * A_ST;                          // [STAGES][S][16 cols][64]
  __nv_bfloat16* sX = sB + STAGES * B_ST;                          // conv inputs: x [16][128], base [2][128]
  const int side = blockIdx.x / (NCTA / 2), gb = (blockIdx.x % (NCTA / 2)) * GPC;  // first group
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wsplit = warp >> 1, wtile = warp & 1;
  // projection column of local column lc (0..15): tile 0 = tap 0 groups, tile 1 = tap 1 groups
  auto pcol = [&](int lc) { return side * 640 + (lc >> 3) * G + gb + (lc & 7); };
  const bool conv_cta = side == 0 && out != nullptr;
  if (conv_cta) {  // x rows [0, 16) x 128 channels + base taps 0/1 (part of the first cp.async group)
    const int c0 = gb * 16;
    for (int i = tid; i < M * 16 + 32; i += THREADS) {
      if (i < M * 16) cp16(sX + i * 8, x + (i >> 4) * sx + c0 + (i & 15) * 8);
      else cp16(sX + M * 128 + (i - M * 16) * 8, base + ((i - M * 16) >> 4) * sbase + c0 + ((i - M * 16) & 15) * 8);
    }
  }
  auto load_stage = [&](int st, int kt) {
    // A: S x 16 rows x 8 chunks = 640; B: S x 16 cols x 8 chunks = 640 -> 2 + 2 per thread
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int c = tid + i * THREADS;  // 0..639
      const int s = c / 128, rem = c % 128, row = rem >> 3, ch = rem & 7;
      const int k0 = s * KS + kt * KT;
      cp16(sA + st * A_ST + s * M * KT + swz(row, ch), x + row * sx + k0 + ch * 8);
      cp16(sB + st * B_ST + s * 16 * KT + swz(row, ch), w + (int64_t)pcol(row) * K + k0 + ch * 8);
    }
  };
#pragma unroll
  for (int st = 0; st < STAGES - 1; ++st) {
    load_stage(st, st);
    cp_commit();
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int kt = 0; kt < NKT; ++kt) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    {
      const int nk = kt + STAGES - 1;
      if (nk < NKT) load_stage(nk % STAGES, nk);
      cp_commit();
    }
    const __nv_bfloat16* a_s = sA + (kt % STAGES) * A_ST + wsplit * M * KT;
    const __nv_bfloat16* b_s = sB + (kt % STAGES) * B_ST + wsplit * 16 * KT;
#pragma unroll
    for (int c2 = 0; c2 < 2; ++c2) {
      uint32_t b4[4];
      {
        const int r = wtile * 8 + (lane & 7), ch = c2 * 4 + (lane >> 3);
        ldm_x4(b4, b_s + swz(r, ch));
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int c = c2 * 2 + h;
        uint32_t a[4];
        {
          const int mi = lane >> 3, r = (lane & 7) + ((mi & 1) ? 8 : 0), ch = 2 * c + (mi >> 1);
          ldm_x4(a, a_s + swz(r, ch));
        }
        const uint32_t b[2] = {b4[2 * h], b4[2 * h + 1]};
        mma16816(acc, a, b);
      }
    }
  }
  cp_wait<0>();
  __syncthreads();
  // ---- split partials -> smem [S][16 rows][16 cols] fp32 (reuses the pipeline buffers)
  float* sP = reinterpret_cast<float*>(smem_raw);
  {
    const int grp = lane >> 2, tig = lane & 3, lc = wtile * 8 + tig * 2;
    float* o = sP + wsplit * M * 16;
    o[grp * 16 + lc] = acc[0];
    o[grp * 16 + lc + 1] = acc[1];
    o[(grp + 8) * 16 + lc] = acc[2];
    o[(grp + 8) * 16 + lc + 1] = acc[3];
  }
  __syncthreads();
  __shared__ __align__(16) __nv_bfloat16 sd[M][16];
  if (tid < M * 16) {
    const int r = tid >> 4, lc = tid & 15;
    float v = 0.f;
#pragma unroll
    for (int s = 0; s < S; ++s) v += sP[s * M * 16 + r * 16 + lc];
    const __nv_bfloat16 b = __float2bfloat16(v);
    kp[r * N + pcol(lc)] = b;
    sd[r][lc] = b;
  }
  if (!conv_cta) return;
  __syncthreads();
  // ---- side-0 conv of channels [16 gb, 16 gb + 128) for all rows: 16 rows x 16 vectors of 8
  if (tid < M * 16) {
    const int r = tid >> 4, cv = tid & 15, cl = 8 * cv, lg = cv >> 1;
    const bool tm = (r % block_size) >= 1;
    const V8 xv = *reinterpret_cast<const V8*>(sX + r * 128 + cl);
    const V8 b0 = *reinterpret_cast<const V8*>(sX + M * 128 + cl);
    V8 xpv, b1;
    if (tm) {
      xpv = *reinterpret_cast<const V8*>(sX + (r - 1) * 128 + cl);
      b1 = *reinterpret_cast<const V8*>(sX + M * 128 + 128 + cl);
    }
    const float d0 = __bfloat162float(sd[r][lg]);
    const float d1 = tm ? __bfloat162float(sd[r][8 + lg]) : 0.f;
    V8 o;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      // Triton's PTX (fp fusion): acc = fma(c0, x, c1 * x_prev) -- the tap-1 product is rounded, tap 0 fused
      const float c0 = __bfloat162float(b0.v[e]) + d0;
      const float coef = (tm ? __bfloat162float(b1.v[e]) : 0.f) + d1;
      const float xp = tm ? __bfloat162float(xpv.v[e]) : 0.f;
      o.v[e] = __float2bfloat16(__fmaf_rn(c0, __bfloat162float(xv.v[e]), __fmul_rn(coef, xp)));
    }
    *reinterpret_cast<V8*>(out + r * sout + gb * 16 + cl) = o;
  }
}

}  // namespace

void kpc(torch::Tensor x, torch::Tensor w, torch::Tensor base, torch::Tensor ws, torch::Tensor cnt, torch::Tensor kp,
         c10::optional<torch::Tensor> out, int64_t block_size) {
  TORCH_CHECK(x.size(0) == M && x.size(1) == K && x.stride(1) == 1 && x.stride(0) % 8 == 0);
  TORCH_CHECK(w.size(0) == N && w.size(1) == K && w.is_contiguous());
  TORCH_CHECK(base.size(0) == 2 && base.size(1) == CH && base.stride(1) == 1 && base.stride(0) % 8 == 0);
  TORCH_CHECK(kp.is_contiguous() && kp.size(0) == M && kp.size(1) == N);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int smem = STAGES * (A_ST + B_ST) * 2 + (M + 2) * 128 * 2;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(kpc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr = true;
  }
  __nv_bfloat16* op = nullptr;
  int64_t so = 0;
  if (out.has_value()) {
    TORCH_CHECK(out->stride(1) == 1 && out->stride(0) % 8 == 0);
    op = reinterpret_cast<__nv_bfloat16*>(out->data_ptr());
    so = out->stride(0);
  }
  kpc_kernel<<<NCTA, THREADS, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(base.data_ptr()),
      base.stride(0), reinterpret_cast<__nv_bfloat16*>(kp.data_ptr()), op, so, (int)block_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("kpc", &kpc); }
