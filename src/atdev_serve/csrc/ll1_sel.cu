// ll1_fuse sel: the DFlash2 CandidateSelector at c=1 (1 request, D = 15 steps, top-16, rank 256) as ONE
// kernel. Stock (Inductor graph of CandidateSelector.forward, 6 launches): hidden_projection = cuBLAS/CUTLASS
// s16816gemm_bf16_64x64_64x6 grid [4, 1, 16] (split-K S = 16, 320 k per split) + splitKreduce; successor gather;
// predecessor gather * hidden (bf16); einsum blpr,blcr->blpc (CUTLASS wmma bf16 16x16, k in order); + unary (f32).
// Bit-identity:
//   * hidden_projection: per split acc = mma.m16n8k16 over the split's k16 chunks in order; v = 0 + p0 + ... + p15;
//     lin = bf16_rn(v)
//   * A[p][r] = bf16_rn(pred_cb[pid(l, p)][r] * lin[l][r]), pid(0, p) = anchor, pid(l, p) = cand[l - 1][p]
//   * es[p][c] = bf16_rn(mma chain over r k16 chunks in order of A[p] . succ_cb[cand[l][c]])
//   * scores[l][p][c] = unary[l][c] + float(es[p][c])
// Grid: 128 CTAs x 4 warps. Phase 1: CTA = (column group cg of 32 columns, split s); the last split CTA of a column
// group reduces it into lin. Phase 2: CTAs 0..14 (l) wait for all 8 column groups (all 128 CTAs are co-resident:
// checked by occupancy at the first call), then compute row block l of the scores.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

namespace {

constexpr int L = 15, P = 16, R = 256, K = 5120, S = 16, KS = K / S;  // 320 k per split
constexpr int CG = R / 32;                                             // 8 column groups
constexpr int NCTA = CG * S;                                           // 128
constexpr int THREADS = 128;
constexpr int ROWP = KS + 8;  // smem row pitch (elements): 656 B -> conflict-free ldmatrix
constexpr int RP = R + 8;     // phase-2 row pitch (528 B)

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
__device__ __forceinline__ void ldm_x4(uint32_t* r, const void* smem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldm_x2(uint32_t* r, const void* smem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}

struct alignas(16) V8 {
  __nv_bfloat16 v[8];
};

__global__ void __launch_bounds__(THREADS) sel_kernel(
    const __nv_bfloat16* __restrict__ h, int64_t sh, const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ pcb, const __nv_bfloat16* __restrict__ scb, const int64_t* __restrict__ cand,
    const float* __restrict__ unary, const int* __restrict__ anchor, float* __restrict__ ws,
    __nv_bfloat16* __restrict__ lin, int* __restrict__ cnt, float* __restrict__ out) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smem_raw);  // [16][ROWP]
  __nv_bfloat16* sB = sA + 16 * ROWP;                              // [32][ROWP]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cg = blockIdx.x % CG, split = blockIdx.x / CG;
  const int k0 = split * KS;
  __nv_bfloat16* sP = sB + 32 * ROWP;  // [16][RP] pred rows (scaled in place by lin later)
  __nv_bfloat16* sS = sP + 16 * RP;    // [16][RP] succ rows
  // ---- phase-2 operand rows (independent of the projection) prefetched at the start by CTAs 0..L-1
  if (blockIdx.x < L) {
    const int l = blockIdx.x;
    for (int i = tid; i < 2 * 16 * (R / 8); i += THREADS) {
      const int which = i / (16 * (R / 8)), j = i % (16 * (R / 8)), p = j / (R / 8), v = j % (R / 8);
      if (which == 0) {
        const int64_t pid = (l == 0) ? (int64_t)anchor[0] : cand[(l - 1) * P + p];
        cp16(sP + p * RP + v * 8, pcb + pid * R + v * 8);
      } else {
        cp16(sS + p * RP + v * 8, scb + cand[l * P + p] * R + v * 8);
      }
    }
  }
  // ---- phase 1: load the split's A (15 rows + 1 zero row) and B (32 rows) slices
  for (int c = tid; c < 16 * (KS / 8); c += THREADS) {
    const int row = c / (KS / 8), ch = c % (KS / 8);
    if (row < L) cp16(sA + row * ROWP + ch * 8, h + row * sh + k0 + ch * 8);
    else *reinterpret_cast<uint4*>(sA + row * ROWP + ch * 8) = make_uint4(0, 0, 0, 0);
  }
  for (int c = tid; c < 32 * (KS / 8); c += THREADS) {
    const int row = c / (KS / 8), ch = c % (KS / 8);
    cp16(sB + row * ROWP + ch * 8, w + (int64_t)(cg * 32 + row) * K + k0 + ch * 8);
  }
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::);
  __syncthreads();
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
  for (int c = 0; c < KS / 16; ++c) {
    uint32_t a[4], b[2];
    {
      const int mi = lane >> 3, r = (lane & 7) + ((mi & 1) ? 8 : 0);
      ldm_x4(a, sA + r * ROWP + c * 16 + (mi >> 1) * 8);
    }
    {
      const int mi = (lane >> 3) & 1, r = warp * 8 + (lane & 7);
      ldm_x2(b, sB + r * ROWP + c * 16 + mi * 8);
    }
    mma16816(acc, a, b);
  }
  const int grp = lane >> 2, tig = lane & 3;
  {
    const int col = cg * 32 + warp * 8 + tig * 2;
    float* o = ws + (int64_t)split * L * R;
    if (grp < L) { o[grp * R + col] = acc[0]; o[grp * R + col + 1] = acc[1]; }
    if (grp + 8 < L) { o[(grp + 8) * R + col] = acc[2]; o[(grp + 8) * R + col + 1] = acc[3]; }
  }
  __threadfence();
  __syncthreads();
  __shared__ int s_last;
  if (tid == 0) {
    const int old = atomicAdd(cnt + cg, 1);
    s_last = (old == S - 1);
    if (s_last) cnt[cg] = 0;
  }
  __syncthreads();
  if (s_last) {
    __threadfence();
    for (int i = tid; i < L * 32; i += THREADS) {
      const int r = i >> 5, col = cg * 32 + (i & 31);
      float v = 0.f;
#pragma unroll
      for (int s = 0; s < S; ++s) v += __ldcg(ws + (int64_t)s * L * R + r * R + col);
      lin[r * R + col] = __float2bfloat16(v);
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) atomicAdd(cnt + CG, 1);  // column groups done
  }
  if (blockIdx.x >= L) return;
  // ---- phase 2: row block l
  const int l = blockIdx.x;
  if (tid == 0) {
    while (atomicAdd(cnt + CG, 0) < CG) __nanosleep(64);
  }
  __syncthreads();
  __threadfence();
  {
    // A = bf16(pred * lin) in place (pred rows landed with phase 1's cp.async group)
    for (int i = tid; i < 16 * (R / 8); i += THREADS) {
      const int p = i / (R / 8), v = i % (R / 8);
      uint4 lraw = __ldcg(reinterpret_cast<const uint4*>(lin + l * R + v * 8));  // written by other CTAs: L2
      const V8 lv = *reinterpret_cast<const V8*>(&lraw);
      V8 pv = *reinterpret_cast<const V8*>(sP + p * RP + v * 8);
#pragma unroll
      for (int e = 0; e < 8; ++e) pv.v[e] = __float2bfloat16(__bfloat162float(pv.v[e]) * __bfloat162float(lv.v[e]));
      *reinterpret_cast<V8*>(sP + p * RP + v * 8) = pv;
    }
  }
  __syncthreads();
  if (warp >= 2) return;
  float e4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
  for (int c = 0; c < R / 16; ++c) {
    uint32_t a[4], b[2];
    {
      const int mi = lane >> 3, r = (lane & 7) + ((mi & 1) ? 8 : 0);
      ldm_x4(a, sP + r * RP + c * 16 + (mi >> 1) * 8);
    }
    {
      const int mi = (lane >> 3) & 1, r = warp * 8 + (lane & 7);
      ldm_x2(b, sS + r * RP + c * 16 + mi * 8);
    }
    mma16816(e4, a, b);
  }
  const int col = warp * 8 + tig * 2;
  float* O = out + l * P * P;
  const float u0 = unary[l * P + col], u1 = unary[l * P + col + 1];
  O[grp * P + col] = u0 + __bfloat162float(__float2bfloat16(e4[0]));
  O[grp * P + col + 1] = u1 + __bfloat162float(__float2bfloat16(e4[1]));
  O[(grp + 8) * P + col] = u0 + __bfloat162float(__float2bfloat16(e4[2]));
  O[(grp + 8) * P + col + 1] = u1 + __bfloat162float(__float2bfloat16(e4[3]));
  if (tid == 0) {
    const int old = atomicAdd(cnt + CG + 1, 1);
    if (old == L - 1) {  // last row block: re-arm for the next launch
      cnt[CG + 1] = 0;
      atomicExch(cnt + CG, 0);
    }
  }
}

}  // namespace

static int g_ok = -1;

bool sel_ok() {
  if (g_ok < 0) {
    const int smem = (16 + 32) * ROWP * 2 + 2 * 16 * RP * 2;
    cudaFuncSetAttribute(sel_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int per_sm = 0, sms = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, sel_kernel, THREADS, smem);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    g_ok = (per_sm * sms >= NCTA) ? 1 : 0;
  }
  return g_ok == 1;
}

void sel(torch::Tensor h, torch::Tensor w, torch::Tensor pcb, torch::Tensor scb, torch::Tensor cand,
         torch::Tensor unary, torch::Tensor anchor, torch::Tensor ws, torch::Tensor lin, torch::Tensor cnt,
         torch::Tensor out) {
  TORCH_CHECK(sel_ok(), "ll1 sel: 128 CTAs not co-resident");
  TORCH_CHECK(h.size(0) == L && h.size(1) == K && h.stride(1) == 1 && h.stride(0) % 8 == 0);
  TORCH_CHECK(w.size(0) == R && w.size(1) == K && w.is_contiguous());
  TORCH_CHECK(pcb.size(1) == R && pcb.is_contiguous() && scb.size(1) == R && scb.is_contiguous());
  TORCH_CHECK(cand.is_contiguous() && cand.numel() == L * P && cand.scalar_type() == at::kLong);
  TORCH_CHECK(unary.is_contiguous() && unary.numel() == L * P && unary.scalar_type() == at::kFloat);
  TORCH_CHECK(anchor.scalar_type() == at::kInt && out.is_contiguous() && out.numel() == L * P * P);
  const int smem = (16 + 32) * ROWP * 2 + 2 * 16 * RP * 2;
  sel_kernel<<<NCTA, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()), h.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(pcb.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(scb.data_ptr()), cand.data_ptr<int64_t>(), unary.data_ptr<float>(),
      anchor.data_ptr<int>(), ws.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(lin.data_ptr()),
      cnt.data_ptr<int>(), out.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sel", &sel);
  m.def("sel_ok", &sel_ok);
}
