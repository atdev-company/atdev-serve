// LL5-tree (ATDEV_LL5_TREE=1): gdn_tree._tree_build_kernel (best-first k-best-paths tree over DFlash2's CRF lattice,
// CHAIN=False, MIX=False, N = NP = 16 nodes, K = 16, one warp per request) as a CUDA kernel with the Triton kernel's
// exact arithmetic (sm_80 PTX of the Triton build, inspected):
//   S = S0 * inv_temp (mul.f32); m = max_k S (butterfly 8,4,2,1 over the 16 k-lanes)
//   e = ex2.approx.f32((fma(S0, inv, -m)) * log2e); sum = butterfly add 8,4,2,1 (k-lanes)
//   lse = m + logf(sum) (libdevice __nv_logf); E = S - lse (sub.f32); cv = val + E (masked -inf)
//   argmax with ties to the smallest index (Triton tie_break_left), max / argmax are exact.
// Difference in schedule only: each node's lattice row (fixed once the node exists) is processed once, when the node
// is created, instead of being reloaded and renormalised by every later iteration (same values every time); the
// request's [D, 16, 16] scores are staged in shared memory by 4 warps first (one DRAM / L2 round trip).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace {

constexpr int NN = 16, K = 16, HALF_ROWS = NN / 2, MAXD = 16;

__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// a half-warp (16 lanes, k = lane & 15) holds one lattice row; butterflies stay inside the half
__device__ __forceinline__ float row_E(float s0, float inv) {
  const float s = __fmul_rn(s0, inv);
  float m = s;
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  const float x = __fmaf_rn(s0, inv, -m);
  float e = ex2_approx(__fmul_rn(x, 1.4426950408889634f));
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) e = __fadd_rn(e, __shfl_xor_sync(0xffffffffu, e, o));
  const float lse = __fadd_rn(m, logf(e));
  return __fsub_rn(s, lse);
}

__device__ __forceinline__ float hmax(float v) {
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// lane = h * 16 + k: half h holds rows (nodes) j = 2 rep + h, rep = 0..7 (the Triton [2, 16] x 8 layout).
// Rolled iteration loop with the node state in shared memory (an unrolled 15-iteration body thrashes the i-cache).
__global__ void __launch_bounds__(128) tree_build_kernel(
    const float* __restrict__ scores, const int64_t* __restrict__ cand, const int* __restrict__ rmap, int rmap_stride,
    int64_t* __restrict__ tok_out, int64_t stride_tok_out, int* __restrict__ mpar, int* __restrict__ mdep,
    int64_t* __restrict__ manc, int* __restrict__ mconv, float inv, int D) {
  __shared__ __align__(16) float ssc[MAXD * K * K];
  __shared__ float sE[NN][K];
  __shared__ float sval[NN];
  __shared__ int sdep[NN], sused[NN], spar[NN];
  __shared__ int64_t sanc[NN], stok[NN];
  const int r = blockIdx.x, tid = threadIdx.x;
  const float* sbase = scores + (int64_t)r * D * K * K;
  {
    const int n4 = D * K * K / 4;
    const float4* src = reinterpret_cast<const float4*>(sbase);
    for (int i = tid; i < n4; i += blockDim.x) reinterpret_cast<float4*>(ssc)[i] = src[i];
  }
  if (tid < NN) {
    sval[tid] = tid == 0 ? 0.f : -__int_as_float(0x7f800000);
    sdep[tid] = 0; sused[tid] = 0; spar[tid] = tid == 0 ? -1 : 0;
    sanc[tid] = tid == 0 ? 1 : 0; stok[tid] = 0;
  }
  __syncthreads();
  if (tid >= 32) return;
  const int lane = tid, k = lane & 15, h = lane >> 4;
  const int rs = rmap[(int64_t)r * rmap_stride];
  const int64_t* cbase = cand + (int64_t)r * D * K;
  const float NEG = -__int_as_float(0x7f800000);
  {
    const float e0 = row_E(ssc[k], inv);  // root row (dep 0, cid 0)
    if (h == 0) sE[0][k] = e0;
  }
  __syncwarp();
#pragma unroll 1
  for (int it = 1; it < NN; ++it) {
    float cvR[HALF_ROWS];
    float bv = NEG;
    int bj = 0;
#pragma unroll
    for (int q = 0; q < HALF_ROWS; ++q) {
      const int j = 2 * q + h;
      const bool alive = j < it && sdep[j] < D;
      const bool ub = (sused[j] >> k) & 1;
      cvR[q] = (alive && !ub) ? __fadd_rn(sval[j], sE[j][k]) : NEG;
    }
#pragma unroll
    for (int q = 0; q < HALF_ROWS; ++q) {
      const float rb = hmax(cvR[q]);
      if (rb > bv) { bv = rb; bj = 2 * q + h; }  // ascending j within the half: ties keep the smaller j
    }
    {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, 16);
      const int oj = __shfl_xor_sync(0xffffffffu, bj, 16);
      if (ov > bv || (ov == bv && oj < bj)) { bv = ov; bj = oj; }
    }
    const float best = bv;
    const int jstar = bj;  // 0 when every candidate is -inf (Triton argmax of all -inf)
    float cj = NEG;
#pragma unroll
    for (int q = 0; q < HALF_ROWS; ++q)
      if (q == (jstar >> 1)) cj = cvR[q];
    const float cm = hmax(cj);
    const unsigned bal = __ballot_sync(0xffffffffu, cj == cm);
    const unsigned hit = (jstar & 1) ? (bal >> 16) : (bal & 0xffffu);
    const int cstar = hit ? __ffs(hit) - 1 : 0;
    const bool ok = best > NEG;
    const int pdep = sdep[jstar];
    const int64_t panc = sanc[jstar];
    const int ndep = ok ? pdep + 1 : 1, ncid = ok ? cstar : 0;
    float en = 0.f;
    if (it + 1 < NN && ndep < D) en = row_E(ssc[(ndep * K + ncid) * K + k], inv);
    __syncwarp();
    if (lane == 0) {
      const int64_t ctok = cbase[min(pdep, D - 1) * K + cstar];
      spar[it] = ok ? jstar : 0;
      sdep[it] = ndep;
      sval[it] = ok ? best : NEG;
      stok[it] = ok ? ctok : 0;
      sanc[it] = (ok ? panc : 1) | ((int64_t)1 << it);
      if (ok) sused[jstar] |= (1 << cstar);
    }
    if (h == 0) sE[it][k] = en;
    __syncwarp();
  }
  if (rs < 0) return;
  const int64_t mb = (int64_t)rs * NN;
  if (lane < NN) {
    const int j = lane;
    const int pj = spar[j];
    if (j >= 1) tok_out[(int64_t)r * stride_tok_out + (j - 1)] = stok[j];
    mpar[mb + j] = pj;
    mdep[mb + j] = sdep[j];
    manc[mb + j] = sanc[j];
    const int c1 = pj;
    const int pg1 = (c1 >= 0 && c1 < NN) ? spar[c1] : 0;
    const int c2 = c1 > 0 ? pg1 : (c1 == 0 ? -1 : c1 - 1);
    const int pg2 = (c2 >= 0 && c2 < NN) ? spar[c2] : 0;
    const int c3 = c2 > 0 ? pg2 : (c2 == 0 ? -1 : c2 - 1);
    mconv[(mb + j) * 3 + 0] = c1;
    mconv[(mb + j) * 3 + 1] = c2;
    mconv[(mb + j) * 3 + 2] = c3;
  }
}

}  // namespace

void tree_build(torch::Tensor scores, torch::Tensor cand, torch::Tensor rmap, int64_t rmap_stride, torch::Tensor tok_out,
                torch::Tensor mpar, torch::Tensor mdep, torch::Tensor manc, torch::Tensor mconv, double inv_temp) {
  TORCH_CHECK(scores.is_contiguous() && scores.scalar_type() == at::kFloat && scores.dim() == 4 &&
              scores.size(2) == K && scores.size(3) == K);
  TORCH_CHECK(cand.is_contiguous() && cand.scalar_type() == at::kLong && rmap.scalar_type() == at::kInt);
  TORCH_CHECK(tok_out.scalar_type() == at::kLong && tok_out.stride(1) == 1);
  TORCH_CHECK(mpar.scalar_type() == at::kInt && mdep.scalar_type() == at::kInt && manc.scalar_type() == at::kLong &&
              mconv.scalar_type() == at::kInt && mpar.is_contiguous() && mdep.is_contiguous() && manc.is_contiguous() &&
              mconv.is_contiguous());
  const int r = scores.size(0), d = scores.size(1);
  TORCH_CHECK(d <= MAXD && (d * K * K) % 4 == 0 && reinterpret_cast<uintptr_t>(scores.data_ptr()) % 16 == 0);
  if (r == 0) return;
  tree_build_kernel<<<r, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      scores.data_ptr<float>(), cand.data_ptr<int64_t>(), rmap.data_ptr<int>(), (int)rmap_stride,
      tok_out.data_ptr<int64_t>(), tok_out.stride(0), mpar.data_ptr<int>(), mdep.data_ptr<int>(),
      manc.data_ptr<int64_t>(), mconv.data_ptr<int>(), (float)inv_temp, d);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("tree_build", &tree_build); }
