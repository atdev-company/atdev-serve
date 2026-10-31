// ll1_fuse ctx: the DFlash2 drafter_ctxfold context-K/V chain of one layer as ONE kernel.
// Stock chain (drafter_ctxfold._ctx_kv_store, 4 launches): k = qkv[:, q:q+kv].contiguous(); vLLM rms_norm_kernel
// (K-norm per (row, kv head), hidden 128); vLLM rotary_embedding_kernel (neox, rot_dim 128); vLLM
// reshape_and_cache_flash_kernel (bf16 cache, kAuto). This kernel reproduces each step's arithmetic exactly:
//   * rms_norm: hidden 128 -> VEC 8, blockDim 16: thread l sums x*x of elements [8l, 8l+8) in order (variance += x*x),
//     cub::BlockReduce<float,1024>::Reduce(.., 16) = WarpReduceShfl partial tile: shfl_down offsets 1,2,4,8 (16 valid
//     lanes), lane 0 holds the sum; s = rsqrtf(var / 128 + eps); out = bf16(x * s * w).
//   * rotary (neox): x_f, y_f = float(kn[i]), float(kn[i + 64]); cos/sin from the bf16 cache row of the position;
//     kx = bf16(x_f * cos_f - y_f * sin_f); ky = bf16(y_f * cos_f + x_f * sin_f) (same expressions -> same FMA
//     contraction as the vLLM build).
//   * cache insert: slot < 0 -> skip the row; k / v rows copied to key_cache / value_cache at
//     block * block_stride + offset * page_stride + head * head_stride.
// One CTA per context row, 16 threads per kv head (blockDim = 16 * nkv).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

namespace {

struct alignas(16) BF8 {
  __nv_bfloat16 v[8];
};

template <bool FMA_X_FIRST>
__global__ void ctx_kv_kernel(const __nv_bfloat16* __restrict__ qkv, int64_t sq, int k_off, int v_off,
                              const __nv_bfloat16* __restrict__ knw, float eps, const int64_t* __restrict__ pos,
                              const __nv_bfloat16* __restrict__ cs, int rot_dim, const int64_t* __restrict__ slots,
                              __nv_bfloat16* __restrict__ kc, __nv_bfloat16* __restrict__ vc, int64_t block_stride,
                              int64_t page_stride, int64_t head_stride, int block_size) {
  const int row = blockIdx.x;
  const int64_t slot = slots[row];
  if (slot < 0) return;
  const int g = threadIdx.x >> 4;  // kv head
  const int l = threadIdx.x & 15;  // 8-element vector of the head
  const __nv_bfloat16* krow = qkv + row * sq + k_off + g * 128;
  const __nv_bfloat16* vrow = qkv + row * sq + v_off + g * 128;
  const BF8 kv = *reinterpret_cast<const BF8*>(krow + 8 * l);
  const BF8 vv = *reinterpret_cast<const BF8*>(vrow + 8 * l);
  // ---- rms_norm (vLLM rms_norm_kernel<bf16, 8, 2, true>, blockDim 16)
  float variance = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float x = __bfloat162float(kv.v[i]);
    variance += x * x;
  }
  // cub WarpReduceShfl partial tile (16 valid lanes): lane + offset <= 15 within the 16-lane group
#pragma unroll
  for (int off = 1; off < 16; off <<= 1) {
    float o = __shfl_down_sync(0xffffffffu, variance, off);
    if (l + off <= 15) variance = o + variance;
  }
  const float sv = rsqrtf(__shfl_sync(0xffffffffu, variance, threadIdx.x & ~15) / 128 + eps);
  const BF8 wv = *reinterpret_cast<const BF8*>(knw + 8 * l);
  BF8 kn;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float x = __bfloat162float(kv.v[i]);
    float w = __bfloat162float(wv.v[i]);
    kn.v[i] = __float2bfloat16(x * sv * w);
  }
  // ---- rotary (neox, rot_dim 128 -> embed 64): partner vector is lane l ^ 8
  BF8 pn;
  {
    const uint4 mine = *reinterpret_cast<const uint4*>(&kn);
    uint4 o;
    o.x = __shfl_xor_sync(0xffffffffu, mine.x, 8);
    o.y = __shfl_xor_sync(0xffffffffu, mine.y, 8);
    o.z = __shfl_xor_sync(0xffffffffu, mine.z, 8);
    o.w = __shfl_xor_sync(0xffffffffu, mine.w, 8);
    *reinterpret_cast<uint4*>(&pn) = o;
  }
  const int embed = rot_dim / 2;  // 64
  const int ri = 8 * (l & 7);     // rot offset of this vector's first element
  const __nv_bfloat16* crow = cs + pos[row] * rot_dim;
  const BF8 cv = *reinterpret_cast<const BF8*>(crow + ri);
  const BF8 sn = *reinterpret_cast<const BF8*>(crow + embed + ri);
  BF8 ko;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float cos_f = __bfloat162float(cv.v[i]);
    const float sin_f = __bfloat162float(sn.v[i]);
    if (l < 8) {  // x half: x = own, y = partner
      const float x_f = __bfloat162float(kn.v[i]);
      const float y_f = __bfloat162float(pn.v[i]);
      ko.v[i] = __float2bfloat16(x_f * cos_f - y_f * sin_f);
    } else {  // y half: y = own, x = partner
      const float x_f = __bfloat162float(pn.v[i]);
      const float y_f = __bfloat162float(kn.v[i]);
      ko.v[i] = __float2bfloat16(y_f * cos_f + x_f * sin_f);
    }
  }
  // ---- cache insert
  const int64_t blk = slot / block_size, off = slot % block_size;
  const int64_t dst = blk * block_stride + off * page_stride + g * head_stride + 8 * l;
  *reinterpret_cast<BF8*>(kc + dst) = ko;
  *reinterpret_cast<BF8*>(vc + dst) = vv;
}

}  // namespace

void ctx_kv(torch::Tensor qkv, int64_t k_off, int64_t v_off, int64_t nkv, torch::Tensor knw, double eps,
            torch::Tensor pos, torch::Tensor cs, torch::Tensor slots, torch::Tensor kc, torch::Tensor vc) {
  TORCH_CHECK(qkv.scalar_type() == at::kBFloat16 && qkv.stride(1) == 1);
  TORCH_CHECK(cs.scalar_type() == at::kBFloat16 && cs.size(1) == 128 && cs.is_contiguous());
  TORCH_CHECK(kc.scalar_type() == at::kBFloat16 && kc.stride(-1) == 1 && kc.size(-1) == 128);
  TORCH_CHECK(kc.stride(0) == vc.stride(0) && kc.stride(1) == vc.stride(1) && kc.stride(2) == vc.stride(2));
  TORCH_CHECK(pos.scalar_type() == at::kLong && slots.scalar_type() == at::kLong);
  TORCH_CHECK(qkv.stride(0) % 8 == 0 && k_off % 8 == 0 && v_off % 8 == 0);
  const int rows = slots.size(0);
  if (rows == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  ctx_kv_kernel<true><<<rows, 16 * nkv, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), qkv.stride(0), (int)k_off, (int)v_off,
      reinterpret_cast<const __nv_bfloat16*>(knw.data_ptr()), (float)eps, pos.data_ptr<int64_t>(),
      reinterpret_cast<const __nv_bfloat16*>(cs.data_ptr()), (int)cs.size(1), slots.data_ptr<int64_t>(),
      reinterpret_cast<__nv_bfloat16*>(kc.data_ptr()), reinterpret_cast<__nv_bfloat16*>(vc.data_ptr()),
      kc.stride(0), kc.stride(1), kc.stride(2), (int)kc.size(1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("ctx_kv", &ctx_kv); }
