// A4ACC lever 1: fused producer -> (optional silu*u / a*sigmoid(g)) -> bf16 -> H128 block rotation -> bf16 ->
// per-token int8 (vLLM formula: s = amax/127, q = sign(v)*floor(|v|*(127/amax)+0.5)); out scale = s * gs.
// One CTA (16 warps) per row; each warp owns 128-groups g = warp, warp+16, ...; rotated values stay in registers.
// Replaces: fp32 torch silu/gate chain + had_rot launch + per_token_quant_int8 launch of the had3 large-M path.
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

__device__ __forceinline__ void ld4(const __nv_bfloat16* p, float (&f)[4]) {
  const uint2 raw = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
  float2 a = __bfloat1622float2(h[0]), b = __bfloat1622float2(h[1]);
  f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

constexpr int NW = 16, GMAX = 9;  // K <= 16*9*128 = 18432

// S4: additionally emit the s4 fragments + per-(group,row) scales straight from the rotated bf16 values (no int8 -> s4
// double rounding), in the chunked tmb=8 layout of pf_a4_quant (A_pk [chunk][K/64][8][32][4] words, A_s [chunk][K/128][128]).
// Rows >= M (up to m_pad) get zero codes.
template <int MODE, bool S4>
__global__ void __launch_bounds__(NW * 32)
rot_i8_kernel(const __nv_bfloat16* __restrict__ x, long ldx, const __nv_bfloat16* __restrict__ x2, long ldx2,
              const float* __restrict__ gs, int8_t* __restrict__ q, float* __restrict__ s, int K, int M,
              uint32_t* __restrict__ A_pk, float* __restrict__ A_s, float clip) {
  const int row = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (row >= M) {  // S4 padding rows
    if (S4) {
      const int NGp = K >> 7;
      const int ch = row >> 7, rr = row & 127;
      for (int g = warp; g < NGp; g += NW) {
        if (!(lane & 4)) {
          const int kb = 2 * g + (lane >> 4), kr = (lane & 15) >> 3, l = (rr & 7) * 4 + (lane & 3);
          const int r = ((rr & 15) >> 3) + 2 * kr, mb = rr >> 4;
          (A_pk + (size_t)ch * (K / 64) * 8 * 32 * 4)[((size_t)(kb * 8 + mb) * 32 + l) * 4 + r] = 0u;
        }
        if (lane == 0) (A_s + (size_t)ch * (K / 128) * 128)[(size_t)g * 128 + rr] = 1.0f;
      }
    }
    return;
  }
  const int NG = K >> 7;
  float v[GMAX][4];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < GMAX; i++) {
    const int g = warp + i * NW;
    if (g < NG) {
      float a[4];
      const long off = g * 128 + lane * 4;
      ld4(x + row * ldx + off, a);
      if (MODE == 1) {  // x = [gate | up] row of width 2K: silu(gate) * up
        float u[4];
        ld4(x + row * ldx + K + off, u);
#pragma unroll
        for (int e = 0; e < 4; e++) a[e] = rbf(a[e] / (1.f + __expf(-a[e])) * u[e]);
      } else if (MODE == 2) {  // a * sigmoid(g)
        float gg[4];
        ld4(x2 + row * ldx2 + off, gg);
#pragma unroll
        for (int e = 0; e < 4; e++) a[e] = rbf(a[e] / (1.f + __expf(-gg[e])));
      }
      fwht128_lane4(a, lane);
#pragma unroll
      for (int e = 0; e < 4; e++) { v[i][e] = rbf(a[e]); amax = fmaxf(amax, fabsf(v[i][e])); }
      if (S4) {
        float ga = fmaxf(fmaxf(fabsf(a[0]), fabsf(a[1])), fmaxf(fabsf(a[2]), fabsf(a[3])));  // fp32 rotated values (as decode quant_a4_had)
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ga = fmaxf(ga, __shfl_xor_sync(0xffffffffu, ga, o));
        const float sc = fmaxf(ga * clip, 1e-8f) * (1.0f / 7.0f);
        const int par = (lane >> 2) & 1;
        uint32_t w4 = 0;
#pragma unroll
        for (int e = 0; e < 4; e++) {
          float qq = rintf(__fdiv_rn(a[e], sc));
          qq = fminf(fmaxf(qq, -8.f), 7.f);
          w4 |= ((uint32_t)((int)qq) & 0xFu) << (4 * (2 * e + par));
        }
        w4 |= __shfl_xor_sync(0xffffffffu, w4, 4);
        const int ch = row >> 7, rr = row & 127;
        if (!par) {
          const int kb = 2 * g + (lane >> 4), kr = (lane & 15) >> 3, l = (rr & 7) * 4 + (lane & 3);
          const int r = ((rr & 15) >> 3) + 2 * kr, mb = rr >> 4;
          (A_pk + (size_t)ch * (K / 64) * 8 * 32 * 4)[((size_t)(kb * 8 + mb) * 32 + l) * 4 + r] = w4;
        }
        if (lane == 0) (A_s + (size_t)ch * (K / 128) * 128)[(size_t)g * 128 + rr] = sc * gs[0];
      }
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  __shared__ float sm[NW];
  if (lane == 0) sm[warp] = amax;
  __syncthreads();
  amax = sm[0];
#pragma unroll
  for (int w = 1; w < NW; w++) amax = fmaxf(amax, sm[w]);
  amax = fmaxf(amax, 1e-10f);
  const float mul = 127.0f / amax;
#pragma unroll
  for (int i = 0; i < GMAX; i++) {
    const int g = warp + i * NW;
    if (g < NG) {
      uint32_t w = 0;
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const float t = v[i][e] * mul;
        const float r = copysignf(floorf(fabsf(t) + 0.5f), t);
        w |= ((uint32_t)(int)r & 0xFFu) << (8 * e);
      }
      *reinterpret_cast<uint32_t*>(q + (long)row * K + g * 128 + lane * 4) = w;
    }
  }
  if (threadIdx.x == 0) s[row] = (amax / 127.0f) * gs[0];
}

static std::vector<torch::Tensor> run(int mode, torch::Tensor x, c10::optional<torch::Tensor> x2, torch::Tensor gs, int64_t K,
                                      c10::optional<torch::Tensor> a_pk = c10::nullopt, c10::optional<torch::Tensor> a_s = c10::nullopt,
                                      double clip = 1.0) {
  const int M = x.size(0);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 4 == 0 && K % 128 == 0 && K <= NW * GMAX * 128);
  auto q = torch::empty({M, K}, x.options().dtype(at::kChar));
  auto s = torch::empty({M}, x.options().dtype(at::kFloat));
  if (M == 0) return {q, s};
  const at::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t st = at::cuda::getCurrentCUDAStream(x.get_device());
  const __nv_bfloat16* p2 = nullptr; long l2 = 0;
  if (x2.has_value()) { TORCH_CHECK(x2->stride(1) == 1 && x2->stride(0) % 4 == 0); p2 = reinterpret_cast<const __nv_bfloat16*>(x2->data_ptr()); l2 = x2->stride(0); }
  const auto* px = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const bool s4 = a_pk.has_value();
  uint32_t* pk = s4 ? reinterpret_cast<uint32_t*>(a_pk->data_ptr()) : nullptr;
  float* sa = s4 ? a_s->data_ptr<float>() : nullptr;
  const int m_pad = s4 ? ((M + 127) / 128) * 128 : M;
  if (s4) {
    const int nch = m_pad / 128;
    TORCH_CHECK(a_pk->numel() * 4 >= (int64_t)nch * (K / 64) * 8 * 32 * 4 * 4 && a_s->numel() >= (int64_t)nch * (K / 128) * 128);
  }
#define L(MD, S) rot_i8_kernel<MD, S><<<m_pad, NW * 32, 0, st>>>(px, x.stride(0), p2, l2, gs.data_ptr<float>(), reinterpret_cast<int8_t*>(q.data_ptr()), s.data_ptr<float>(), (int)K, M, pk, sa, (float)clip)
  if (s4) { if (mode == 0) L(0, true); else if (mode == 1) L(1, true); else L(2, true); }
  else { if (mode == 0) L(0, false); else if (mode == 1) L(1, false); else L(2, false); }
  return {q, s};
}
std::vector<torch::Tensor> rot_quant(torch::Tensor x, torch::Tensor gs, c10::optional<torch::Tensor> a_pk, c10::optional<torch::Tensor> a_s, double clip) { return run(0, x, c10::nullopt, gs, x.size(1), a_pk, a_s, clip); }
std::vector<torch::Tensor> silu_rot_quant(torch::Tensor gu, torch::Tensor gs, c10::optional<torch::Tensor> a_pk, c10::optional<torch::Tensor> a_s, double clip) { return run(1, gu, c10::nullopt, gs, gu.size(1) / 2, a_pk, a_s, clip); }
std::vector<torch::Tensor> gate_rot_quant(torch::Tensor a, torch::Tensor g, torch::Tensor gs, c10::optional<torch::Tensor> a_pk, c10::optional<torch::Tensor> a_s, double clip) { return run(2, a, g, gs, a.size(1), a_pk, a_s, clip); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rot_quant", &rot_quant, py::arg("x"), py::arg("gs"), py::arg("a_pk") = py::none(), py::arg("a_s") = py::none(), py::arg("clip") = 1.0);
  m.def("silu_rot_quant", &silu_rot_quant, py::arg("gu"), py::arg("gs"), py::arg("a_pk") = py::none(), py::arg("a_s") = py::none(), py::arg("clip") = 1.0);
  m.def("gate_rot_quant", &gate_rot_quant, py::arg("a"), py::arg("g"), py::arg("gs"), py::arg("a_pk") = py::none(), py::arg("a_s") = py::none(), py::arg("clip") = 1.0);
}
