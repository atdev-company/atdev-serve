// LL1-tail (ATDEV_LL1_TAIL=1): the DFlash2 draft-vocab top-k tail after the int4 screen as ONE kernel per row.
// Stock (draft_vocab.DraftVocabHead.top_k, TOPKX=1, k = screen = 16): _topkx_merge_kernel (global top-16 of the
// per-chunk candidates + torch.topk(sorted=False) output order + ids[]), weight.index_select (16 full-head rows),
// torch.bmm (cuBLAS gemvx: exact bf16 logits), torch.topk(exact, 16, sorted=True) (sbtopk gatherTopK +
// bitonicSortKVInPlace), full.gather, values.float() -> 6 launches. Here (one CTA of 16 warps per row):
//   A merge: top-16 of the chunk composites by 16 rounds of block max (composites are unique) -> torch's
//     sorted=False order: key > kth first, then key == kth, each in index order (== _topkx_merge_kernel)
//   B rescore: warp j computes dot(weight[full[j]], hidden[row]) with cuBLAS gemvx's order (GEMV_* below,
//     found by scratchpad probe: lane t accumulates k = t, t + 32, ... with fma, then a shfl_down 16..1 tree)
//   C sort: exact emulation of torch.topk(16, sorted=True) on the 16 bf16 values (copied from csrc/topk16.cu:
//     sbtopk gather order + bitonicSortKVInPlace<2,-1,16,16> network) -> ids, float(values)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

#ifndef TAIL_STOP
#define TAIL_STOP 0
#endif
namespace {

constexpr int S = 16;
constexpr int THREADS = 512;

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

// stage 1 (== draft_vocab._topkx_chunk_kernel's per-chunk top-16 SET; the merge does not depend on the order):
// one warp per 1024-element chunk, lane holds 32 composites, 16 rounds of warp max. Composite of a 16-bit float:
// radix key = bits ^ (sign ? 0xffff : 0x8000), NaN -> 0xffff; comp = ((key - 32768) << 16) | (65535 - index).
__device__ __forceinline__ int comp_of(unsigned short u, int idx) {
  const bool nan = ((u & 0x7f80u) == 0x7f80u) && ((u & 0x007fu) != 0);
  int key = (u & 0x8000u) ? (u ^ 0xffffu) : (u ^ 0x8000u);
  key = nan ? 0xffff : key;
  return ((key - 32768) << 16) | (65535 - idx);
}

__global__ void __launch_bounds__(128) chunk_kernel(const unsigned short* __restrict__ x, int64_t sx, int n,
                                                     int nc, int* __restrict__ cc) {
  const int r = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int c = blockIdx.x * 4 + warp;
  if (c >= nc) return;
  const int NEG = -2147483647 - 1;
  int v[32];
  const unsigned short* xr = x + (int64_t)r * sx + c * 1024;
#pragma unroll
  for (int q = 0; q < 4; ++q) {  // 4 x 16 B per lane: elements c*1024 + q*256 + lane*8 + e
    const int base = c * 1024 + q * 256 + lane * 8;
    uint4 raw = make_uint4(0, 0, 0, 0);
    if (base + 8 <= n) raw = *reinterpret_cast<const uint4*>(xr + q * 256 + lane * 8);
    const unsigned short* u = reinterpret_cast<const unsigned short*>(&raw);
#pragma unroll
    for (int e = 0; e < 8; ++e) v[q * 8 + e] = (base + e < n) ? comp_of(u[e], base + e) : NEG;
  }
  int mysel = NEG;
#pragma unroll 1
  for (int s = 0; s < 16; ++s) {
    int m = v[0];
#pragma unroll
    for (int i = 1; i < 32; ++i) m = max(m, v[i]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == s) mysel = m;
#pragma unroll
    for (int i = 0; i < 32; ++i) v[i] = (v[i] == m) ? NEG : v[i];
  }
  if (lane < 16) cc[((int64_t)r * nc + c) * 16 + lane] = mysel;
}

__global__ void __launch_bounds__(THREADS) tail_kernel(
    const int* __restrict__ cc, int ncs, const int64_t* __restrict__ ids, const __nv_bfloat16* __restrict__ w,
    int64_t sw, const __nv_bfloat16* __restrict__ h, int64_t sh, int K, int64_t* __restrict__ oid,
    float* __restrict__ oval) {
  const int r = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  __shared__ int s_sel[S];
  __shared__ int64_t s_full[S];
  __shared__ __nv_bfloat16 s_ex[S];
  __shared__ __align__(16) __nv_bfloat16 sX[5120];       // hidden row
  __shared__ __align__(16) __nv_bfloat16 sW[THREADS / 32][256];  // per-warp transposition chunk
  const int NEG = -2147483647 - 1;
  // hidden row -> smem (all warps), overlapped with warp 0's merge
  for (int i = tid; i < K / 8; i += THREADS)
    *reinterpret_cast<uint4*>(sX + i * 8) = *reinterpret_cast<const uint4*>(h + (int64_t)r * sh + i * 8);
  // ---- A: top-16 composites of the ncs (<= 2048) chunk candidates, 16 rounds of block max
  __shared__ int s_wmax[THREADS / 32];
  {
    constexpr int PER = 4;
    int v[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int j = tid + i * THREADS;
      v[i] = j < ncs ? cc[(int64_t)r * ncs + j] : NEG;
    }
    for (int s = 0; s < S; ++s) {
      int m = v[0];
#pragma unroll
      for (int i = 1; i < PER; ++i) m = max(m, v[i]);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
      if (lane == 0) s_wmax[warp] = m;
      __syncthreads();
      if (warp == 0) {
        int x = lane < THREADS / 32 ? s_wmax[lane] : NEG;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) x = max(x, __shfl_xor_sync(0xffffffffu, x, o));
        if (lane == 0) s_sel[s] = x;
      }
      __syncthreads();
      const int mx = s_sel[s];
#pragma unroll
      for (int i = 0; i < PER; ++i) v[i] = (v[i] == mx) ? NEG : v[i];
    }
  }
  if (warp == 0) {  // torch.topk(sorted=False) order: (cat, idx) rank among the 16 (lanes 0..15)
    const int mysel = lane < S ? s_sel[lane] : NEG;
    const int key = mysel >> 16, idx = 65535 - (mysel & 0xFFFF);
    int kth = lane < S ? key : 0x7fffffff;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) kth = min(kth, __shfl_xor_sync(0xffffffffu, kth, o));
    const int ordk = (key > kth ? 0 : 1) * 65536 + idx;
    int pos = 0;
#pragma unroll
    for (int j = 0; j < S; ++j) pos += __shfl_sync(0xffffffffu, ordk, j) < ordk;
    if (lane < S) s_full[pos] = ids[idx];
  }
  __syncthreads();
  // ---- B: exact bf16 logits, warp j = candidate j (cuBLAS gemvx order: lane t sums k = t, t + 32, ... by fma,
  //      then shfl_down 16..1); row loaded with coalesced 16 B vectors, transposed through per-warp smem
  {
    const __nv_bfloat16* a = w + s_full[warp] * sw;
    constexpr int NV = 5120 / 256;  // 20 chunks of 256 elements (one 16 B vector per lane)
    uint4 rv[NV];
#pragma unroll
    for (int c = 0; c < NV; ++c) rv[c] = *reinterpret_cast<const uint4*>(a + c * 256 + lane * 8);
    float acc = 0.f;
    __nv_bfloat16* cw = sW[warp];
#pragma unroll
    for (int c = 0; c < NV; ++c) {
      __syncwarp();
      *reinterpret_cast<uint4*>(cw + lane * 8) = rv[c];
      __syncwarp();
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int kk = j * 32 + lane;
        acc = __fmaf_rn(__bfloat162float(cw[kk]), __bfloat162float(sX[c * 256 + kk]), acc);
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) s_ex[warp] = __float2bfloat16(acc);
  }
  __syncthreads();
  // ---- C: torch.topk(ex, 16, sorted=True) emulation (topk16.cu) on warp 0's first 16 lanes
  __shared__ __nv_bfloat16 keys[32];
  __shared__ long vals[32];
  __shared__ bool valid[32];
  __shared__ __nv_bfloat16 gk[S];
  __shared__ long gv[S];
  if (tid == 0) {
    unsigned kmin = 0xffffffffu;
    for (int i = 0; i < S; ++i) { const unsigned k = radix_key(s_ex[i]); kmin = k < kmin ? k : kmin; }
    int c = 0;
    for (int i = 0; i < S; ++i)
      if (radix_key(s_ex[i]) > kmin) { gk[c] = s_ex[i]; gv[c] = i; ++c; }
    for (int i = 0; i < S; ++i)
      if (radix_key(s_ex[i]) == kmin) { gk[c] = s_ex[i]; gv[c] = i; ++c; }
  }
  __syncthreads();
  if (tid >= 16) return;
  const int tx = tid;
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int idx = tx + k * 16;
    const bool ok = idx < S;
    keys[idx] = ok ? gk[idx] : __nv_bfloat16();
    vals[idx] = ok ? gv[idx] : 0L;
    valid[idx] = ok;
  }
  constexpr unsigned P2 = 32;
#pragma unroll
  for (unsigned size = 2; size < P2; size *= 2) {
    const bool flag = ((tx & (size / 2)) != 0);
#pragma unroll
    for (unsigned stride = size / 2; stride > 0; stride /= 2) {
      __syncwarp(0xffffu);
      const unsigned pos = 2 * tx - (tx & (stride - 1));
      bitonic_swap(keys[pos], vals[pos], valid[pos], keys[pos + stride], vals[pos + stride], valid[pos + stride],
                   flag);
    }
  }
#pragma unroll
  for (unsigned stride = P2 / 2; stride > 0; stride /= 2) {
    __syncwarp(0xffffu);
    const unsigned pos = 2 * tx - (tx & (stride - 1));
    bitonic_swap(keys[pos], vals[pos], valid[pos], keys[pos + stride], vals[pos + stride], valid[pos + stride],
                 false);
  }
  __syncwarp(0xffffu);
  oval[r * S + tx] = __bfloat162float(keys[tx]);
  oid[r * S + tx] = s_full[vals[tx]];
}

// v2: one warp per (row, candidate j) -> 16 M warps spread over the GPU (the 16 full-head rows of a row are 160 KB:
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
#if TAIL_STOP == 1
  if (lane == 0) gex[r * S + j] = __float2bfloat16((float)s_full[j]);
  return;
#endif
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
#if TAIL_STOP == 2
  return;
#endif
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
  // torch.topk(ex, 16, sorted=True) emulation (topk16.cu) on lanes 0..15
  __shared__ __nv_bfloat16 s_ex[S];
  __shared__ __nv_bfloat16 keys[32];
  __shared__ long vals[32];
  __shared__ bool valid[32];
  __shared__ __nv_bfloat16 gk[S];
  __shared__ long gv[S];
  if (lane < S) s_ex[lane] = __ushort_as_bfloat16(__ldcg(reinterpret_cast<const unsigned short*>(gex + r * S + lane)));
  __syncwarp();
  if (lane == 0) {
    unsigned kmin = 0xffffffffu;
    for (int i = 0; i < S; ++i) { const unsigned k = radix_key(s_ex[i]); kmin = k < kmin ? k : kmin; }
    int c = 0;
    for (int i = 0; i < S; ++i)
      if (radix_key(s_ex[i]) > kmin) { gk[c] = s_ex[i]; gv[c] = i; ++c; }
    for (int i = 0; i < S; ++i)
      if (radix_key(s_ex[i]) == kmin) { gk[c] = s_ex[i]; gv[c] = i; ++c; }
  }
  __syncwarp();
  if (lane >= 16) return;
  const int tx = lane;
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int idx = tx + k * 16;
    const bool ok = idx < S;
    keys[idx] = ok ? gk[idx] : __nv_bfloat16();
    vals[idx] = ok ? gv[idx] : 0L;
    valid[idx] = ok;
  }
  constexpr unsigned P2 = 32;
#pragma unroll
  for (unsigned size = 2; size < P2; size *= 2) {
    const bool flag = ((tx & (size / 2)) != 0);
#pragma unroll
    for (unsigned stride = size / 2; stride > 0; stride /= 2) {
      __syncwarp(0xffffu);
      const unsigned pos = 2 * tx - (tx & (stride - 1));
      bitonic_swap(keys[pos], vals[pos], valid[pos], keys[pos + stride], vals[pos + stride], valid[pos + stride],
                   flag);
    }
  }
#pragma unroll
  for (unsigned stride = P2 / 2; stride > 0; stride /= 2) {
    __syncwarp(0xffffu);
    const unsigned pos = 2 * tx - (tx & (stride - 1));
    bitonic_swap(keys[pos], vals[pos], valid[pos], keys[pos + stride], vals[pos + stride], valid[pos + stride],
                 false);
  }
  __syncwarp(0xffffu);
  oval[r * S + tx] = __bfloat162float(keys[tx]);
  oid[r * S + tx] = s_full[vals[tx]];
}

}  // namespace

void chunk(torch::Tensor x, torch::Tensor cc) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 &&
              x.stride(0) % 8 == 0 && cc.is_contiguous());
  const int m = x.size(0), n = x.size(1), nc = (n + 1023) / 1024;
  TORCH_CHECK(n % 8 == 0 && cc.size(1) == nc * 16);
  if (m == 0) return;
  chunk_kernel<<<dim3((nc + 3) / 4, m), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const unsigned short*>(x.data_ptr()), x.stride(0), n, nc, cc.data_ptr<int>());
}

void tail(torch::Tensor cc, torch::Tensor ids, torch::Tensor w, torch::Tensor h, torch::Tensor oid,
          torch::Tensor oval) {
  TORCH_CHECK(cc.scalar_type() == at::kInt && cc.is_contiguous() && cc.size(1) <= 4 * THREADS);
  TORCH_CHECK(ids.scalar_type() == at::kLong && ids.is_contiguous());
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.stride(1) == 1 && h.scalar_type() == at::kBFloat16 &&
              h.stride(1) == 1 && w.size(1) == h.size(1) && h.size(1) == 5120 && w.stride(0) % 8 == 0 &&
              h.stride(0) % 8 == 0);
  TORCH_CHECK(oid.is_contiguous() && oval.is_contiguous() && oval.scalar_type() == at::kFloat);
  const int m = cc.size(0);
  if (m == 0) return;
  tail_kernel<<<m, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      cc.data_ptr<int>(), (int)cc.size(1), ids.data_ptr<int64_t>(), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
      w.stride(0), reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()), h.stride(0), (int)h.size(1),
      oid.data_ptr<int64_t>(), oval.data_ptr<float>());
}

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
  m.def("tail", &tail);
  m.def("chunk", &chunk);
}
