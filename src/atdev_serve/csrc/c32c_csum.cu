// Consumer-sum: the residual add + Gemma RMSNorm -> s4 producer (one CTA per padded row) reading its
// x input as the split-K slice partials of the producer W4A4 GEMM instead of the GEMM's bf16 output.
// The MarlinA4 csum build (A4_PAR_REDUCE=7) stores every slice's fp32 partial to P[slice][row][col]
// (slot stride 128 rows) and skips its lock chain and bf16 write; here x[row][c] = bf16(P[0] + P[1] +
// ... + P[cnt-1]) (slots added in slice order = the serial chain's fp32 order, rounded like
// write_result) -> the GEMM-output part is bit-identical to the chain. The norm is rounding-level vs
// fused_quant's Triton chunk kernel (rare fp32 ties). The s4 quotient is the branch-free form of
// c32_prod.cu (bit-identical to div.rn).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int CS_ROWS = 128;  // P slot stride (rows)
constexpr int CS_TN = 64;     // GEMM column tile (slice count granularity)
constexpr int CS_SMAX = 4;

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
__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }

// split-K slice count of GEMM column tile t (MarlinA4 stripe schedule, init_part2_slice / a4_host.cu)
struct Sched {
  int n_tiles, k_tiles, part2, iters;
};
__device__ __forceinline__ int slice_count(const Sched& sc, int t) {
  if (t < sc.n_tiles - sc.part2) return 1;  // part-1 column: whole K in one CTA
  const int cp = t - (sc.n_tiles - sc.part2);
  const int col_first = sc.iters * ((sc.k_tiles * cp + sc.iters - 1) / sc.iters);
  if (col_first > sc.k_tiles * (cp + 1)) return 1;
  const int off = col_first - sc.k_tiles * cp;
  return (sc.k_tiles - off + sc.iters - 1) / sc.iters + (off > 0 ? 1 : 0);
}

__device__ __forceinline__ void a4_pack8(const float (&y)[8], int s, int g, int row, int m_pad, int tmb, float gs,
                                         uint32_t* __restrict__ apk, float* __restrict__ as_) {
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

template <int NR, bool WANT_BF16>
__global__ void __launch_bounds__(256) csum_norm_a4_kernel(
    const float* __restrict__ P, Sched sc, const __nv_bfloat16* __restrict__ r, int sr,
    const __nv_bfloat16* __restrict__ w, const float* __restrict__ gs_ptr, __nv_bfloat16* __restrict__ xb,
    int sxb, __nv_bfloat16* __restrict__ ro, int sro, uint32_t* __restrict__ apk, float* __restrict__ as_, int M,
    int N, float eps, int m_pad, int tmb) {
  __shared__ float red[8];
  const int row = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const bool valid = row < M;
  const size_t slot = (size_t)CS_ROWS * N;
  // all loads of the row first (NR x cnt x 2 float4 + residual), then the slot sums in slice order
  constexpr int NRV = NR > 3 ? 3 : NR;  // N <= 6144 (2048-column repetitions with data)
  float4 q[NRV][CS_SMAX][2];
  uint4 ra[NRV];
  int nk[NRV];
#pragma unroll
  for (int k = 0; k < NRV; k++) {
    const int c = k * 2048 + 8 * t;
    const bool m = valid && c < N;
    nk[k] = m ? slice_count(sc, c / CS_TN) : 0;
    ra[k] = make_uint4(0, 0, 0, 0);
    const float4* p = reinterpret_cast<const float4*>(P + (size_t)row * N + (m ? c : 0));
#pragma unroll
    for (int i = 0; i < CS_SMAX; i++) {
      q[k][i][0] = make_float4(0.f, 0.f, 0.f, 0.f);
      q[k][i][1] = q[k][i][0];
      if (i < nk[k]) {
        q[k][i][0] = __ldg(p + i * (slot / 4));
        q[k][i][1] = __ldg(p + i * (slot / 4) + 1);
      }
    }
    if (m) ra[k] = *reinterpret_cast<const uint4*>(r + (size_t)row * sr + c);
  }
  float v[NR][8];
#pragma unroll
  for (int k = 0; k < NR; k++) {
    float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    uint4 rk = make_uint4(0, 0, 0, 0);
    if (k < NRV) {
      s[0] = q[k][0][0].x; s[1] = q[k][0][0].y; s[2] = q[k][0][0].z; s[3] = q[k][0][0].w;
      s[4] = q[k][0][1].x; s[5] = q[k][0][1].y; s[6] = q[k][0][1].z; s[7] = q[k][0][1].w;
#pragma unroll
      for (int i = 1; i < CS_SMAX; i++)
        if (i < nk[k]) {
          s[0] += q[k][i][0].x; s[1] += q[k][i][0].y; s[2] += q[k][i][0].z; s[3] += q[k][i][0].w;
          s[4] += q[k][i][1].x; s[5] += q[k][i][1].y; s[6] += q[k][i][1].z; s[7] += q[k][i][1].w;
        }
      rk = ra[k];
    }
    const __nv_bfloat16* rh = reinterpret_cast<const __nv_bfloat16*>(&rk);
#pragma unroll
    for (int j = 0; j < 8; j++) v[k][j] = __fadd_rn(bf(__float2bfloat16_rn(s[j])), bf(rh[j]));
  }
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
  const float gs = *gs_ptr;
#pragma unroll
  for (int k = 0; k < NR; k++) {
    const int c = k * 2048 + 8 * t;
    if (k * 2048 >= N) break;
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
      if (WANT_BF16) *reinterpret_cast<uint4*>(xb + (size_t)row * sxb + c) = *reinterpret_cast<uint4*>(xbb);
    }
    if (c < N) a4_pack8(y, t & 15, c / 128, row, m_pad, tmb, gs, apk, as_);
  }
}

// x (bf16, the GEMM output) by summing the P slots only (test / fallback consumers): out[M, N] bf16
__global__ void csum_to_bf16_kernel(const float* __restrict__ P, Sched sc,
                                    __nv_bfloat16* __restrict__ out, int so, int M, int N) {
  const int row = blockIdx.y, c = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (row >= M || c >= N) return;
  const size_t slot = (size_t)CS_ROWS * N;
  const int n = slice_count(sc, c / CS_TN);
  const float4* p = reinterpret_cast<const float4*>(P + (size_t)row * N + c);
  float4 a = __ldcg(p), b = __ldcg(p + 1);
  for (int i = 1; i < n; i++) {
    const float4 qa = __ldcg(p + i * (slot / 4)), qb = __ldcg(p + i * (slot / 4) + 1);
    a.x += qa.x; a.y += qa.y; a.z += qa.z; a.w += qa.w;
    b.x += qb.x; b.y += qb.y; b.z += qb.z; b.w += qb.w;
  }
  __nv_bfloat16 o[8] = {__float2bfloat16_rn(a.x), __float2bfloat16_rn(a.y), __float2bfloat16_rn(a.z),
                        __float2bfloat16_rn(a.w), __float2bfloat16_rn(b.x), __float2bfloat16_rn(b.y),
                        __float2bfloat16_rn(b.z), __float2bfloat16_rn(b.w)};
  *reinterpret_cast<uint4*>(out + (size_t)row * so + c) = *reinterpret_cast<uint4*>(o);
}

Sched make_sched(int N, int K, int64_t grid) {
  Sched sc;
  sc.n_tiles = N / CS_TN;
  sc.k_tiles = K / 128;
  sc.part2 = sc.n_tiles;
  if (sc.n_tiles > grid) {
    sc.part2 = sc.n_tiles % grid;
    if (sc.part2 * 3 <= grid) sc.part2 += grid;
  }
  sc.iters = (sc.k_tiles * sc.part2 + grid - 1) / grid;
  return sc;
}

}  // namespace

// P: [4][128][N] fp32 slice partials of a MarlinA4 csum GEMM with N, K (gemm K), tn = 64 and `grid` CTAs
void csum_norm_a4(torch::Tensor P, int64_t K, int64_t grid, torch::Tensor r, torch::Tensor w, torch::Tensor gs,
                  c10::optional<torch::Tensor> xb, torch::Tensor ro, torch::Tensor apk, torch::Tensor as_, int64_t M,
                  double eps, int64_t m_pad, int64_t tmb) {
  const int N = r.size(1);
  TORCH_CHECK(P.scalar_type() == at::kFloat && P.is_contiguous() && P.numel() >= (int64_t)CS_SMAX * CS_ROWS * N);
  TORCH_CHECK(K % 128 == 0 && N % CS_TN == 0);
  const Sched sc = make_sched(N, K, grid);
  TORCH_CHECK(N % 256 == 0 && N <= 6144 && r.stride(1) == 1 && r.stride(0) % 8 == 0 && M <= CS_ROWS);
  TORCH_CHECK(w.is_contiguous() && ro.stride(0) % 8 == 0 && m_pad <= 16 * tmb);
  int block = 2048;
  while (block < N) block <<= 1;
  const int nr = block / 2048;
  const bool want_bf16 = xb.has_value();
  __nv_bfloat16* xbp = want_bf16 ? (__nv_bfloat16*)xb->data_ptr() : nullptr;
  const int sxb = want_bf16 ? xb->stride(0) : 0;
  const at::cuda::OptionalCUDAGuard guard(r.device());
  auto st = at::cuda::getCurrentCUDAStream();
#define CS_LAUNCH(NRV, WB)                                                                                    \
  csum_norm_a4_kernel<NRV, WB><<<m_pad, 256, 0, st>>>(                                                     \
      P.data_ptr<float>(), sc, (const __nv_bfloat16*)r.data_ptr(), r.stride(0),          \
      (const __nv_bfloat16*)w.data_ptr(), gs.data_ptr<float>(), xbp, sxb, (__nv_bfloat16*)ro.data_ptr(), \
      ro.stride(0), (uint32_t*)apk.data_ptr(), as_.data_ptr<float>(), (int)M, N, (float)eps, (int)m_pad, (int)tmb)
  if (nr == 1) {
    if (want_bf16) CS_LAUNCH(1, true); else CS_LAUNCH(1, false);
  } else if (nr == 2) {
    if (want_bf16) CS_LAUNCH(2, true); else CS_LAUNCH(2, false);
  } else {
    if (want_bf16) CS_LAUNCH(4, true); else CS_LAUNCH(4, false);
  }
#undef CS_LAUNCH
}

void csum_to_bf16(torch::Tensor P, int64_t K, int64_t grid, torch::Tensor out, int64_t M) {
  const int N = out.size(1);
  const Sched sc = make_sched(N, K, grid);
  TORCH_CHECK(N % 8 == 0 && out.stride(1) == 1 && out.stride(0) % 8 == 0 && M <= CS_ROWS);
  const at::cuda::OptionalCUDAGuard guard(out.device());
  auto st = at::cuda::getCurrentCUDAStream();
  if (M == 0) return;
  csum_to_bf16_kernel<<<dim3((N / 8 + 127) / 128, M), 128, 0, st>>>(P.data_ptr<float>(), sc,
                                                                    (__nv_bfloat16*)out.data_ptr(), out.stride(0),
                                                                    (int)M, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("csum_norm_a4", &csum_norm_a4);
  m.def("csum_to_bf16", &csum_to_bf16);
}
