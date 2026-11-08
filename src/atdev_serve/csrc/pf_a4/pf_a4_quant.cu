// PFX-s9: int8 -> s4 activation re-quantisation for the large-M (prefill) W4A4 GEMM.
// Same output layout as marlin_plus_a4 a4_quant_kernel (A_pk [chunk][K/64][tmb][32][4] words, A_s [chunk][K/128][16 tmb]):
// one warp per (row, 128-group); lane holds 4 consecutive int8 codes. The group scale is
//   s = max(amax_q8, 1e-8) / 7   (amax over the group's int8 codes),   A_s = s * rs[row]
// (rs = the int8 path's per-token scale incl. input_global_scale), q4 = clamp(rint(q8 / s), -8, 7). Rows >= M are zero.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

__global__ void pf_a4_quant_i8_kernel(const int8_t* __restrict__ x, int ldx, int M, int K, const float* __restrict__ rs,
                                      uint32_t* __restrict__ A_pk, float* __restrict__ A_s, int m_pad, int tmb) {
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int ngroups = K >> 7;
  const int row = gw / ngroups, g = gw - row * ngroups;
  if (row >= m_pad) return;
  const int chunk_rows = 16 * tmb, ch = row / chunk_rows, rr = row - ch * chunk_rows;
  uint32_t* A_pk_c = A_pk + (size_t)ch * (K / 64) * tmb * 32 * 4;
  float* A_s_c = A_s + (size_t)ch * (K / 128) * chunk_rows;
  float v[4] = {0.f, 0.f, 0.f, 0.f};
  if (row < M) {
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(x + (size_t)row * ldx + g * 128 + lane * 4);
#pragma unroll
    for (int e = 0; e < 4; e++) v[e] = (float)(int8_t)((raw >> (8 * e)) & 0xFF);
  }
  float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float s = fmaxf(amax, 1e-8f) * (1.0f / 7.0f);
  const int par = (lane >> 2) & 1;
  uint32_t w = 0;
#pragma unroll
  for (int e = 0; e < 4; e++) {
    float q = rintf(__fdiv_rn(v[e], s));
    q = fminf(fmaxf(q, -8.f), 7.f);
    const uint32_t nib = (uint32_t)((int)q) & 0xFu;
    w |= nib << (4 * (2 * e + par));
  }
  w |= __shfl_xor_sync(0xffffffffu, w, 4);
  if (!par) {
    const int kb = 2 * g + (lane >> 4);
    const int kr = (lane & 15) >> 3;
    const int l = (rr & 7) * 4 + (lane & 3);
    const int r = ((rr & 15) >> 3) + 2 * kr;
    const int mb = rr >> 4;
    A_pk_c[((size_t)(kb * tmb + mb) * 32 + l) * 4 + r] = w;
  }
  if (lane == 0) A_s_c[(size_t)g * chunk_rows + rr] = (row < M) ? s * rs[row] : s;
}


__device__ __forceinline__ float qdiv(float v, float s, float inv) {
  float q = v * inv;
  float r = fmaf(-q, s, v);
  return fmaf(r, inv, q);
}
__global__ void pf_a4_quant_i8_kernel2(const int8_t* __restrict__ x, int ldx, int M, int K, const float* __restrict__ rs,
                                      uint32_t* __restrict__ A_pk, float* __restrict__ A_s, int m_pad, int tmb) {
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int ngroups = K >> 7;
  const int row = gw / ngroups, g = gw - row * ngroups;
  if (row >= m_pad) return;
  const int chunk_rows = 16 * tmb, ch = row / chunk_rows, rr = row - ch * chunk_rows;
  uint32_t* A_pk_c = A_pk + (size_t)ch * (K / 64) * tmb * 32 * 4;
  float* A_s_c = A_s + (size_t)ch * (K / 128) * chunk_rows;
  uint32_t raw = 0;
  if (row < M) raw = *reinterpret_cast<const uint32_t*>(x + (size_t)row * ldx + g * 128 + lane * 4);
  // |int8| per byte via SIMD abs (|-128| = 128 does not fit a byte -> use the signed values via cvt below)
  int a0 = (int)(int8_t)(raw & 0xFF), a1 = (int)(int8_t)((raw >> 8) & 0xFF), a2 = (int)(int8_t)((raw >> 16) & 0xFF), a3 = (int)(int8_t)(raw >> 24);
  int am = max(max(abs(a0), abs(a1)), max(abs(a2), abs(a3)));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) am = max(am, __shfl_xor_sync(0xffffffffu, am, o));
  const float s = fmaxf((float)am, 1e-8f) * (1.0f / 7.0f);
  const float inv = 1.0f / s;  // IEEE division, once per warp
  const int par = (lane >> 2) & 1;
  uint32_t w = 0;
  const int av[4] = {a0, a1, a2, a3};
#pragma unroll
  for (int e = 0; e < 4; e++) {
    float q = rintf(qdiv((float)av[e], s, inv));
    q = fminf(fmaxf(q, -8.f), 7.f);
    const uint32_t nib = (uint32_t)((int)q) & 0xFu;
    w |= nib << (4 * (2 * e + par));
  }
  w |= __shfl_xor_sync(0xffffffffu, w, 4);
  if (!par) {
    const int kb = 2 * g + (lane >> 4);
    const int kr = (lane & 15) >> 3;
    const int l = (rr & 7) * 4 + (lane & 3);
    const int r = ((rr & 15) >> 3) + 2 * kr;
    const int mb = rr >> 4;
    A_pk_c[((size_t)(kb * tmb + mb) * 32 + l) * 4 + r] = w;
  }
  if (lane == 0) A_s_c[(size_t)g * chunk_rows + rr] = (row < M) ? s * rs[row] : s;
}

void pf_a4_quant_i8(torch::Tensor x, torch::Tensor rs, torch::Tensor a_pk, torch::Tensor a_s, int64_t tmb, int64_t variant) {
  const int M = x.size(0), K = x.size(1);
  TORCH_CHECK(x.scalar_type() == at::kChar && x.stride(1) == 1 && x.stride(0) % 4 == 0 && K % 128 == 0);
  TORCH_CHECK(rs.scalar_type() == at::kFloat && rs.numel() >= M);
  const int nch = (M + 16 * tmb - 1) / (16 * tmb);
  TORCH_CHECK(a_pk.numel() * 4 >= (int64_t)nch * (K / 64) * tmb * 32 * 4 * 4);
  TORCH_CHECK(a_s.numel() >= (int64_t)nch * (K / 128) * 16 * tmb);
  const at::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  const int m_pad = nch * 16 * tmb;
  const int64_t warps = (int64_t)m_pad * (K / 128);
  const int threads = 256;
  if (variant == 0) pf_a4_quant_i8_kernel<<<(unsigned)((warps * 32 + threads - 1) / threads), threads, 0, stream>>>(
      reinterpret_cast<const int8_t*>(x.data_ptr()), (int)x.stride(0), M, K, rs.data_ptr<float>(),
      reinterpret_cast<uint32_t*>(a_pk.data_ptr()), a_s.data_ptr<float>(), m_pad, (int)tmb);
  if (variant == 1) pf_a4_quant_i8_kernel2<<<(unsigned)((warps * 32 + threads - 1) / threads), threads, 0, stream>>>(
      reinterpret_cast<const int8_t*>(x.data_ptr()), (int)x.stride(0), M, K, rs.data_ptr<float>(),
      reinterpret_cast<uint32_t*>(a_pk.data_ptr()), a_s.data_ptr<float>(), m_pad, (int)tmb);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_a4_i8", &pf_a4_quant_i8, "int8 codes -> s4 fragments (PFX-s9)", py::arg("x"), py::arg("rs"), py::arg("a_pk"),
        py::arg("a_s"), py::arg("tmb"), py::arg("variant") = 0);
}
