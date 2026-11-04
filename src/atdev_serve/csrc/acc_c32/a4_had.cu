// ACC-c32: rotated W4A4 activation quant = 128-block Sylvester Hadamard (normalised) of every (row, 128-group)
// followed by marlin_plus_a4's quant_a4 (same scale formula and s4 fragment layout as a4_host.cu
// a4_quant_kernel; only the values are rotated). One warp per (row, group): lane L holds k = g*128 + 4L + e.
// Butterflies on all 7 index bits (2 in registers, 5 across lanes), fp32, then * 2^-3.5.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cstdint>

__device__ __forceinline__ void fwht128_lane4(float (&v)[4], int lane) {
#pragma unroll
  for (int b = 1; b <= 2; b <<= 1) {
    float t[4];
#pragma unroll
    for (int e = 0; e < 4; e++) t[e] = (e & b) ? v[e ^ b] - v[e] : v[e] + v[e ^ b];
#pragma unroll
    for (int e = 0; e < 4; e++) v[e] = t[e];
  }
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const float p = __shfl_xor_sync(0xffffffffu, v[e], o);
      v[e] = (lane & o) ? p - v[e] : v[e] + p;
    }
  }
#pragma unroll
  for (int e = 0; e < 4; e++) v[e] *= 0.08838834764831845f;
}

__global__ void a4_had_quant_kernel(const __nv_bfloat16* __restrict__ x, int ldx, int M, int K,
                                    const float* __restrict__ gs, uint32_t* __restrict__ A_pk,
                                    float* __restrict__ A_s, int m_pad, int tmb) {
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
    const uint2 raw = *reinterpret_cast<const uint2*>(x + (size_t)row * ldx + g * 128 + lane * 4);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
    float2 f0 = __bfloat1622float2(h[0]), f1 = __bfloat1622float2(h[1]);
    v[0] = f0.x; v[1] = f0.y; v[2] = f1.x; v[3] = f1.y;
  }
  fwht128_lane4(v, lane);
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
  if (lane == 0) A_s_c[(size_t)g * chunk_rows + rr] = s * gs[0];
}

// bf16 [M, K] -> bf16 [M, K] rotated (prefill / non-a4 paths); one warp per (row, group)
__global__ void had_rot_kernel(const __nv_bfloat16* __restrict__ x, int ldx, __nv_bfloat16* __restrict__ y,
                               int M, int K) {
  const long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int ngroups = K >> 7;
  const long row = gw / ngroups;
  const int g = gw - row * ngroups;
  if (row >= M) return;
  const uint2 raw = *reinterpret_cast<const uint2*>(x + row * ldx + g * 128 + lane * 4);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
  float2 f0 = __bfloat1622float2(h[0]), f1 = __bfloat1622float2(h[1]);
  float v[4] = {f0.x, f0.y, f1.x, f1.y};
  fwht128_lane4(v, lane);
  __nv_bfloat162 o0 = __floats2bfloat162_rn(v[0], v[1]), o1 = __floats2bfloat162_rn(v[2], v[3]);
  uint2 out;
  out.x = *reinterpret_cast<uint32_t*>(&o0); out.y = *reinterpret_cast<uint32_t*>(&o1);
  *reinterpret_cast<uint2*>(y + row * (long)K + g * 128 + lane * 4) = out;
}

void quant_a4_had(torch::Tensor x, torch::Tensor gs, torch::Tensor a_pk, torch::Tensor a_s, int64_t tmb) {
  const int M = x.size(0), K = x.size(1);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 4 == 0 && K % 128 == 0);
  const int nch = (M + 16 * tmb - 1) / (16 * tmb);
  TORCH_CHECK(a_pk.numel() * 4 >= (int64_t)nch * (K / 64) * tmb * 32 * 4 * 4);
  TORCH_CHECK(a_s.numel() >= (int64_t)nch * (K / 128) * 16 * tmb);
  const at::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  const int m_pad = nch * 16 * tmb;
  const int warps = m_pad * (K / 128);
  a4_had_quant_kernel<<<(warps * 32 + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int)x.stride(0), M, K, gs.data_ptr<float>(),
      reinterpret_cast<uint32_t*>(a_pk.data_ptr()), a_s.data_ptr<float>(), m_pad, (int)tmb);
}

torch::Tensor had_rot(torch::Tensor x) {
  const int M = x.size(0), K = x.size(1);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 4 == 0 && K % 128 == 0);
  auto y = torch::empty({M, K}, x.options());
  if (M == 0) return y;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  const long warps = (long)M * (K / 128);
  had_rot_kernel<<<(unsigned)((warps * 32 + 255) / 256), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int)x.stride(0),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), M, K);
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_a4_had", &quant_a4_had, "128-block Hadamard + per-(token,128-group) s4 quant + fragment pack");
  m.def("had_rot", &had_rot, "128-block Hadamard rotation of bf16 rows (fp32 butterflies, bf16 out)");
}
