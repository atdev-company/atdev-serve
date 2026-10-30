// ATDEV_DRAFT_VOCAB_TOPKX=2: bit-exact emulation of
//   vals, idx = torch.topk(exact, 16, dim=-1, sorted=True); ids = full.gather(-1, idx)
// for exact [M, 16] bf16 (the draft-vocab rescoring output), as torch 2.13 runs it on CUDA:
//   sbtopk::gatherTopK (radix order: key = bits ^ (sign ? 0xffff : 0x8000), NaN -> 0xffff; elements with key > k-th
//   key in index order, then those == k-th key in index order; with k == slice size that is all 16), then
//   bitonicSortKVInPlace<2, -1, 16, 16, bf16, int64, GTOp<bf16, true>> (Power2SortSize 32, slots 16..31 invalid;
//   the network and bitonicSwap are copied from ATen/native/cuda/SortUtils.cuh).
// One CTA (16 threads) per row; the per-row computation is the same as torch's (its CTA holds up to 16 rows).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

namespace {

__device__ __forceinline__ bool isnan_bf(__nv_bfloat16 x) { return __hisnan(x); }
__device__ __forceinline__ bool gt_op(__nv_bfloat16 a, __nv_bfloat16 b) {  // GTOp<bf16, handleNaN=true>
  return (isnan_bf(a) && !isnan_bf(b)) || (__bfloat162float(a) > __bfloat162float(b));
}
__device__ __forceinline__ unsigned radix_key(__nv_bfloat16 v) {
  const unsigned x = __bfloat16_as_ushort(v);
  const unsigned mask = (x & 0x8000u) ? 0xffffu : 0x8000u;
  return isnan_bf(v) ? 0xffffu : (x ^ mask);
}

template <typename K, typename V>
__device__ __forceinline__ void swap_vars(K& a, K& b) { K t = a; a = b; b = t; }

__device__ __forceinline__ void bitonic_swap(__nv_bfloat16& kA, long& vA, bool& validA, __nv_bfloat16& kB, long& vB,
                                             bool& validB, bool dir) {
  bool swap = (gt_op(kA, kB) && validA) || !validB;
  if (swap == dir) {
    swap_vars<__nv_bfloat16, long>(kA, kB);
    swap_vars<long, long>(vA, vB);
    swap_vars<bool, bool>(validA, validB);
  }
}

__global__ void topk16_sorted_kernel(const __nv_bfloat16* __restrict__ ex, long sex, const long* __restrict__ full,
                                     long sfull, long* __restrict__ oid, long soid, __nv_bfloat16* __restrict__ oval,
                                     long soval) {
  constexpr int S = 16, P2 = 32;
  __shared__ __nv_bfloat16 keys[P2];
  __shared__ long vals[P2];
  __shared__ bool valid[P2];
  __shared__ __nv_bfloat16 gk[S];
  __shared__ long gv[S];
  const int r = blockIdx.x, tx = threadIdx.x;
  // ---- sbtopk gather (k == slice size): > kth first, then == kth, each in index order
  if (tx == 0) {
    __nv_bfloat16 v[S];
    unsigned kmin = 0xffffffffu;
#pragma unroll
    for (int i = 0; i < S; ++i) {
      v[i] = ex[r * sex + i];
      const unsigned k = radix_key(v[i]);
      kmin = k < kmin ? k : kmin;
    }
    int w = 0;
#pragma unroll
    for (int i = 0; i < S; ++i)
      if (radix_key(v[i]) > kmin) { gk[w] = v[i]; gv[w] = i; ++w; }
#pragma unroll
    for (int i = 0; i < S; ++i)
      if (radix_key(v[i]) == kmin) { gk[w] = v[i]; gv[w] = i; ++w; }
  }
  __syncthreads();
  // ---- bitonicSortKVInPlace load (items_per_thread 2, blockDim.x 16)
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int idx = tx + k * 16;
    const bool ok = idx < S;
    keys[idx] = ok ? gk[idx] : __nv_bfloat16();
    vals[idx] = ok ? gv[idx] : 0L;
    valid[idx] = ok;
  }
  // ---- bitonicSort<32> (SortUtils.cuh)
#pragma unroll
  for (unsigned size = 2; size < P2; size *= 2) {
    const bool flag = ((tx & (size / 2)) != 0);
#pragma unroll
    for (unsigned stride = size / 2; stride > 0; stride /= 2) {
      __syncthreads();
      const unsigned pos = 2 * tx - (tx & (stride - 1));
      bitonic_swap(keys[pos], vals[pos], valid[pos], keys[pos + stride], vals[pos + stride], valid[pos + stride],
                   flag);
    }
  }
#pragma unroll
  for (unsigned stride = P2 / 2; stride > 0; stride /= 2) {
    __syncthreads();
    const unsigned pos = 2 * tx - (tx & (stride - 1));
    bitonic_swap(keys[pos], vals[pos], valid[pos], keys[pos + stride], vals[pos + stride], valid[pos + stride],
                 false);
  }
  __syncthreads();
  // ---- outputs: vals, full.gather(-1, idx)
  oval[r * soval + tx] = keys[tx];
  oid[r * soid + tx] = full[r * sfull + vals[tx]];
}

}  // namespace

void topk16_sorted(torch::Tensor ex, torch::Tensor full, torch::Tensor oid, torch::Tensor oval) {
  TORCH_CHECK(ex.scalar_type() == at::kBFloat16 && ex.size(1) == 16 && ex.stride(1) == 1);
  TORCH_CHECK(full.scalar_type() == at::kLong && full.size(1) == 16 && full.stride(1) == 1);
  TORCH_CHECK(oid.scalar_type() == at::kLong && oval.scalar_type() == at::kBFloat16 && oid.stride(1) == 1 &&
              oval.stride(1) == 1);
  const int m = (int)ex.size(0);
  if (m == 0) return;
  topk16_sorted_kernel<<<m, 16, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(ex.data_ptr()), ex.stride(0), full.data_ptr<long>(), full.stride(0),
      oid.data_ptr<long>(), oid.stride(0), reinterpret_cast<__nv_bfloat16*>(oval.data_ptr()), oval.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("topk16_sorted", &topk16_sorted); }
