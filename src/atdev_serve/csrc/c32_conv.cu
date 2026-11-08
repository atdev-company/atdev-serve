// C32a-conv-pdl: vLLM causal_conv1d_update (width 4, SiLU, no bias, spec decoding, varlen, null block),
// bit-identical to the stock Triton kernel's op sequence (products rounded to bf16 by fma.rn.bf16 with -0,
// fp32 adds in tap order from 0, SiLU = acc / (1 + ex2.approx(-acc * log2e)) with div.full, bf16 RN out),
// optionally launched as a programmatic dependent (PDL) of the producer GEMM: the request metadata, the
// conv-state window and the weights are loaded BEFORE griddepcontrol.wait (they do not depend on the
// GEMM), only the GEMM output x after it.
// grid (dim / (2 * 128), num_reqs), 128 threads, one thread = 2 channels (bf16x2).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {
__device__ __forceinline__ float cv_prod(__nv_bfloat16 x, __nv_bfloat16 w) {
  unsigned short r;
  const unsigned short xb = __bfloat16_as_ushort(x), wb = __bfloat16_as_ushort(w), z = 0x8000;
  asm("fma.rn.bf16 %0, %1, %2, %3;" : "=h"(r) : "h"(xb), "h"(wb), "h"(z));
  return __bfloat162float(__ushort_as_bfloat16(r));
}
__device__ __forceinline__ float cv_silu(float a) {
  float t = __fsub_rn(0.0f, a), e, r;
  t = __fmul_rn(t, 1.44269502162933349609375f);
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
  e = __fadd_rn(e, 1.0f);
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(e));
  return r;
}

template <int TM, bool PDL>
__global__ void __launch_bounds__(128) conv_upd(__nv_bfloat16* __restrict__ x, long long ldx,
                                                __nv_bfloat16* __restrict__ cs, long long s_seq, long long s_tok,
                                                const __nv_bfloat16* __restrict__ w, long long w_dim,
                                                const int* __restrict__ csi, long long csi_stride,
                                                const int* __restrict__ qsl, const int* __restrict__ nacc,
                                                int num_lines, int null_id) {
  const int r = blockIdx.y;
  const long long c = 2 * ((long long)blockIdx.x * 128 + threadIdx.x);
  const int coord = csi[(size_t)r * csi_stride];
  const int qs = qsl[r], qe = qsl[r + 1], L = qe - qs;
  const bool act = coord != null_id && L > 0;
  __nv_bfloat162 s[3], xv[TM];
  __nv_bfloat16 wv[2][4];
  __nv_bfloat16* base = cs + (long long)coord * s_seq + c;
  if (act) {
    const long long off = (long long)nacc[r] - 1;
#pragma unroll
    for (int q = 0; q < 3; q++) s[q] = *reinterpret_cast<const __nv_bfloat162*>(base + (off + q) * s_tok);
    const uint4 wq = *reinterpret_cast<const uint4*>(w + c * w_dim);  // w_dim == 4: 2 channels x 4 taps
    const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wq);
#pragma unroll
    for (int e = 0; e < 2; e++)
#pragma unroll
      for (int j = 0; j < 4; j++) wv[e][j] = wh[e * 4 + j];
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (PDL) asm volatile("griddepcontrol.wait;" ::: "memory");  // PDL: Hopper+ only
#endif
  if (!act) return;
#pragma unroll
  for (int t = 0; t < TM; t++)
    if (t < L) xv[t] = *reinterpret_cast<const __nv_bfloat162*>(x + (size_t)(qs + t) * ldx + c);
  const bool okl = coord < num_lines;
  const __nv_bfloat16 zb = __ushort_as_bfloat16(0);
  const __nv_bfloat162 z2 = __halves2bfloat162(zb, zb);
  *reinterpret_cast<__nv_bfloat162*>(base) = okl ? s[1] : z2;
  *reinterpret_cast<__nv_bfloat162*>(base + s_tok) = okl ? s[2] : z2;
#pragma unroll
  for (int t = 0; t < TM; t++)
    if (t < L) *reinterpret_cast<__nv_bfloat162*>(base + (2 + t) * s_tok) = xv[t];
#pragma unroll
  for (int t = 0; t < TM; t++) {
    if (t >= L) break;
    const __nv_bfloat162 c0 = t == 0 ? s[0] : (t == 1 ? s[1] : (t == 2 ? s[2] : xv[t - 3]));
    const __nv_bfloat162 c1 = t == 0 ? s[1] : (t == 1 ? s[2] : xv[t - 2]);
    const __nv_bfloat162 c2 = t == 0 ? s[2] : xv[t - 1];
    const __nv_bfloat162 xt = xv[t];
    const __nv_bfloat16 a0[2] = {c0.x, c0.y}, a1[2] = {c1.x, c1.y}, a2[2] = {c2.x, c2.y}, a3[2] = {xt.x, xt.y};
    float o[2];
#pragma unroll
    for (int e = 0; e < 2; e++) {
      float acc = 0.0f;
      acc = __fadd_rn(acc, cv_prod(a0[e], wv[e][0]));
      acc = __fadd_rn(acc, cv_prod(a1[e], wv[e][1]));
      acc = __fadd_rn(acc, cv_prod(a2[e], wv[e][2]));
      acc = __fadd_rn(acc, cv_prod(a3[e], wv[e][3]));
      o[e] = cv_silu(acc);
    }
    *reinterpret_cast<__nv_bfloat162*>(x + (size_t)(qs + t) * ldx + c) =
        __halves2bfloat162(__float2bfloat16_rn(o[0]), __float2bfloat16_rn(o[1]));
  }
}
}  // namespace

// x: [tokens, >= dim] view (unit column stride), conv_state (lines, dim, state_len) view with unit dim
// stride, weight (dim, 4) contiguous. In place on x. pdl: launch as a programmatic dependent of the
// previous kernel on the stream.
void conv_update(torch::Tensor x, torch::Tensor conv_state, torch::Tensor weight, torch::Tensor csi,
                 torch::Tensor qsl, torch::Tensor nacc, int64_t num_reqs, int64_t null_id, int64_t max_q,
                 bool pdl) {
  const int dim = (int)conv_state.size(1);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 2 == 0 && x.size(1) >= dim);
  TORCH_CHECK(conv_state.scalar_type() == at::kBFloat16 && conv_state.stride(1) == 1 &&
              conv_state.stride(0) % 2 == 0 && conv_state.stride(2) % 2 == 0);
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16 && weight.is_contiguous() && weight.size(1) == 4 &&
              weight.size(0) == dim);
  TORCH_CHECK(dim % 256 == 0, "conv_update: dim % 256");
  TORCH_CHECK(max_q >= 1 && max_q <= 4, "conv_update: max_query_len <= 4");
  TORCH_CHECK(csi.scalar_type() == at::kInt && qsl.scalar_type() == at::kInt && nacc.scalar_type() == at::kInt &&
              qsl.stride(0) == 1 && nacc.stride(0) == 1);
  if (num_reqs <= 0) return;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(dim / 256, (unsigned)num_reqs);
  cfg.blockDim = dim3(128);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  auto* xp = (__nv_bfloat16*)x.data_ptr();
  auto* cp = (__nv_bfloat16*)conv_state.data_ptr();
  const auto* wp = (const __nv_bfloat16*)weight.data_ptr();
  cudaError_t e;
  if (pdl)
    e = cudaLaunchKernelEx(&cfg, conv_upd<4, true>, xp, (long long)x.stride(0), cp, (long long)conv_state.stride(0),
                           (long long)conv_state.stride(2), wp, (long long)weight.stride(0), csi.data_ptr<int>(),
                           (long long)csi.stride(0), qsl.data_ptr<int>(), nacc.data_ptr<int>(),
                           (int)conv_state.size(0), (int)null_id);
  else
    e = cudaLaunchKernelEx(&cfg, conv_upd<4, false>, xp, (long long)x.stride(0), cp, (long long)conv_state.stride(0),
                           (long long)conv_state.stride(2), wp, (long long)weight.stride(0), csi.data_ptr<int>(),
                           (long long)csi.stride(0), qsl.data_ptr<int>(), nacc.data_ptr<int>(),
                           (int)conv_state.size(0), (int)null_id);
  TORCH_CHECK(e == cudaSuccess, "conv_update launch: ", cudaGetErrorString(e));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_update", &conv_update, "causal_conv1d_update (width 4, SiLU, spec, varlen), bit-identical, PDL opt");
}
