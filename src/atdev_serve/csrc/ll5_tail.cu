// Draft-vocab top-k tail of the DFlash2 drafter (ll1_fuse "tail"): after the int4 screen and the per-chunk top-16
// (draft_vocab._topkx_chunk_kernel), this replaces the stock chain (k = screen = 16): _topkx_merge_kernel (global
// top-16 of the per-chunk candidates + torch.topk(sorted=False) output order + ids[]), weight.index_select (16
// full-head rows), torch.bmm (cuBLAS gemvx: exact bf16 logits), torch.topk(exact, 16, sorted=True) (sbtopk
// gatherTopK + bitonicSortKVInPlace), full.gather, values.float() -> 6 launches. One warp per (row, candidate):
//   merge: top-16 of the chunk composites (composites are unique) -> torch's sorted=False order: key > kth first,
//     then key == kth, each in index order (== _topkx_merge_kernel)
//   rescore: dot(weight[full[j]], hidden[row]) with cuBLAS gemvx's order (lane t accumulates k = t, t + 32, ...
//     with fma, then a shfl_down 16..1 tree)
//   sort: exact emulation of torch.topk(16, sorted=True) on the 16 bf16 values (sbtopk gather order + the
//     bitonicSortKVInPlace<2,-1,16,16> network on shuffles) -> ids, float(values)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

namespace {

constexpr int S = 16;

__device__ __forceinline__ bool isnan_bf(__nv_bfloat16 x) { return __hisnan(x); }
__device__ __forceinline__ bool gt_op(__nv_bfloat16 a, __nv_bfloat16 b) {  // GTOp<bf16, handleNaN=true>
  return (isnan_bf(a) && !isnan_bf(b)) || (__bfloat162float(a) > __bfloat162float(b));
}
__device__ __forceinline__ unsigned radix_key(__nv_bfloat16 v) {
  const unsigned x = __bfloat16_as_ushort(v);
  const unsigned mask = (x & 0x8000u) ? 0xffffu : 0x8000u;
  return isnan_bf(v) ? 0xffffu : (x ^ mask);
}
// one warp per (row, candidate j) -> 16 M warps spread over the GPU (the 16 full-head rows of a row are 160 KB:
// one CTA per row was DRAM-per-SM bound). Every warp redoes the cheap merge (the 64 chunk lists are sorted
// descending: 16 pops of a 64-way tournament, lists in smem), computes candidate j's exact logit, and the last
// arriving warp of the row (arrival counter, re-armed) runs the torch.topk(sorted) emulation.
__global__ void __launch_bounds__(32) tail2_kernel(
    const int* __restrict__ cc, int nl, const int64_t* __restrict__ ids, const __nv_bfloat16* __restrict__ w,
    int64_t sw, const __nv_bfloat16* __restrict__ h, int64_t sh, int64_t* __restrict__ oid,
    float* __restrict__ oval, __nv_bfloat16* __restrict__ gex, int* __restrict__ cnt) {
  const int r = blockIdx.x >> 4, j = blockIdx.x & 15, lane = threadIdx.x;
  const int NEG = -2147483647 - 1;
  __shared__ int sl[64 * S];  // nl (<= 64) sorted lists of 16
  __shared__ int64_t s_full[S];
  __shared__ __align__(16) __nv_bfloat16 sw_[256];
  for (int i = lane; i < nl * S; i += 32) sl[i] = cc[(int64_t)r * nl * S + i];
  __syncwarp();
  // tournament: lane owns lists 2 lane, 2 lane + 1
  int p0 = 0, p1 = 0;
  const int l0 = 2 * lane, l1 = 2 * lane + 1;
  int mysel = NEG;
  for (int s = 0; s < S; ++s) {
    const int h0 = (l0 < nl && p0 < S) ? sl[l0 * S + p0] : NEG;
    const int h1 = (l1 < nl && p1 < S) ? sl[l1 * S + p1] : NEG;
    int m = max(h0, h1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == s) mysel = m;
    if (h0 == m) ++p0;  // composites are unique: exactly one head matches
    else if (h1 == m) ++p1;
  }
  {
    const int key = mysel >> 16, idx = 65535 - (mysel & 0xFFFF);
    int kth = lane < S ? key : 0x7fffffff;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) kth = min(kth, __shfl_xor_sync(0xffffffffu, kth, o));
    const int ordk = (key > kth ? 0 : 1) * 65536 + idx;
    int pos = 0;
#pragma unroll
    for (int q = 0; q < S; ++q) pos += __shfl_sync(0xffffffffu, ordk, q) < ordk;
    if (lane < S) s_full[pos] = ids[idx];
  }
  __syncwarp();
  // exact logit of candidate j (cuBLAS gemvx order)
  {
    const __nv_bfloat16* a = w + s_full[j] * sw;
    const __nv_bfloat16* x = h + (int64_t)r * sh;
    constexpr int NV = 5120 / 256;
    uint4 rv[NV], xv[NV];
#pragma unroll
    for (int c = 0; c < NV; ++c) {
      rv[c] = *reinterpret_cast<const uint4*>(a + c * 256 + lane * 8);
      xv[c] = *reinterpret_cast<const uint4*>(x + c * 256 + lane * 8);
    }
    __shared__ __align__(16) __nv_bfloat16 sx_[256];
    float acc = 0.f;
#pragma unroll
    for (int c = 0; c < NV; ++c) {
      __syncwarp();
      *reinterpret_cast<uint4*>(sw_ + lane * 8) = rv[c];
      *reinterpret_cast<uint4*>(sx_ + lane * 8) = xv[c];
      __syncwarp();
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int kk = e * 32 + lane;
        acc = __fmaf_rn(__bfloat162float(sw_[kk]), __bfloat162float(sx_[kk]), acc);
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) gex[r * S + j] = __float2bfloat16(acc);
  }
  __threadfence();
  __syncwarp();
  int last = 0;
  if (lane == 0) {
    last = atomicAdd(cnt + r, 1) == S - 1;
    if (last) cnt[r] = 0;
  }
  last = __shfl_sync(0xffffffffu, last, 0);
  if (!last) return;
  __threadfence();
  // torch.topk(ex, 16, sorted=True) emulation in registers: slot = lane, 32 slots.
  // (a) sbtopk gather order: key > kmin first, then key == kmin, each in index order (ballot ranks);
  // (b) the same bitonicSortKVInPlace<2,-1,16,16> network (pair (p, p + stride), dir = (p & size) != 0 in the
  //     merge stages, false in the final pass) with partner values exchanged by shuffles.
  unsigned short ex = 0;
  if (lane < S) ex = __ldcg(reinterpret_cast<const unsigned short*>(gex + r * S + lane));
  const __nv_bfloat16 exb = __ushort_as_bfloat16(ex);
  unsigned rk = lane < S ? radix_key(exb) : 0xffffffffu;
  unsigned kmin = rk;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) kmin = min(kmin, __shfl_xor_sync(0xffffffffu, kmin, o));
  const bool c0 = lane < S && rk > kmin;
  const unsigned b0 = __ballot_sync(0xffffffffu, c0);
  const unsigned lt = (1u << lane) - 1u;
  const int pos = c0 ? __popc(b0 & lt) : (__popc(b0) + __popc((~b0) & 0xffffu & lt));
  // slot p <- element i with pos_i == p (p < 16); slots 16..31 invalid (key 0, val 0)
  int src = 0;
#pragma unroll
  for (int q = 0; q < S; ++q) {
    const int pq = __shfl_sync(0xffffffffu, pos, q);
    if (pq == lane) src = q;
  }
  unsigned short kraw = __shfl_sync(0xffffffffu, ex, src);
  int v = src;
  bool valid = lane < S;
  if (!valid) { kraw = 0; v = 0; }
#pragma unroll
  for (unsigned size = 2; size <= 32; size *= 2) {
#pragma unroll
    for (unsigned stride = size / 2; stride > 0; stride /= 2) {
      const bool dir = size < 32 ? ((lane & size) != 0) : false;
      const unsigned short pk = __shfl_xor_sync(0xffffffffu, kraw, stride);
      const int pv = __shfl_xor_sync(0xffffffffu, v, stride);
      const bool pvalid = __shfl_xor_sync(0xffffffffu, (int)valid, stride) != 0;
      const bool lower = (lane & stride) == 0;
      const __nv_bfloat16 kA = __ushort_as_bfloat16(lower ? kraw : pk), kB = __ushort_as_bfloat16(lower ? pk : kraw);
      const bool vA = lower ? valid : pvalid, vB = lower ? pvalid : valid;
      const bool swap = (gt_op(kA, kB) && vA) || !vB;
      if (swap == dir) { kraw = pk; v = pv; valid = pvalid; }
    }
  }
  if (lane < S) {
    oval[r * S + lane] = __bfloat162float(__ushort_as_bfloat16(kraw));
    oid[r * S + lane] = s_full[v];
  }
}

}  // namespace

void tail2(torch::Tensor cc, torch::Tensor ids, torch::Tensor w, torch::Tensor h, torch::Tensor oid,
           torch::Tensor oval, torch::Tensor gex, torch::Tensor cnt) {
  TORCH_CHECK(cc.scalar_type() == at::kInt && cc.is_contiguous() && cc.size(1) % S == 0 && cc.size(1) <= 64 * S);
  TORCH_CHECK(ids.scalar_type() == at::kLong && ids.is_contiguous());
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.stride(1) == 1 && h.scalar_type() == at::kBFloat16 &&
              h.stride(1) == 1 && w.size(1) == 5120 && h.size(1) == 5120 && w.stride(0) % 8 == 0 &&
              h.stride(0) % 8 == 0);
  TORCH_CHECK(oid.is_contiguous() && oval.is_contiguous() && oval.scalar_type() == at::kFloat);
  const int m = cc.size(0);
  TORCH_CHECK(gex.numel() >= m * S && cnt.numel() >= m);
  if (m == 0) return;
  tail2_kernel<<<m * S, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      cc.data_ptr<int>(), (int)(cc.size(1) / S), ids.data_ptr<int64_t>(),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), w.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()), h.stride(0), oid.data_ptr<int64_t>(),
      oval.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(gex.data_ptr()), cnt.data_ptr<int>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tail2", &tail2);
}
