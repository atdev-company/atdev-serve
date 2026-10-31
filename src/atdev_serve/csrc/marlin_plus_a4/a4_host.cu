// marlin_plus_a4 host side: W4A4 kernel (mp_a4.cuh) + activation quant/pack.
// cfg = {thread_n, threads, stages, blocks_per_sm (= MINB), thread_m_blocks, MW, DEFER}.
// A_pk layout: [K/64][tmb][32 lanes][4 words] (s4 mma A fragments, int8-path nibble
// order), A_s: [K/128][16*tmb] fp32 (x input_global_scale). prob_m <= 16*tmb.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include "kernel.h"
#include "mp_a4_decl.h"

namespace MARLIN_NAMESPACE_NAME {
__global__ void MarlinA4Default(MARLIN_KERNEL_PARAMS) {};
using A4FuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

static A4FuncPtr get_a4_kernel(int tmb, int tnb, int threads, int stages, int MW,
                               int DEFER, int MINB) {
  auto kernel = MarlinA4Default;
#include "a4_selector.h"
  return kernel;
}
static int a4_smem(int tmb, int tn, int stages) {
  int tnb = tn / 16;
  int red = (2 * tnb + 1) * 16 * tmb;
  int b = stages * (tn * 4);
  int s = stages * 2 * tnb;
  int as = stages * (16 * tmb / 4);
  int a = stages * 2 * tmb * 32;
  return 16 * (std::max(red, b) + s + as + a);
}

// one warp per (row, 128-group):
//   s = max(amax, 1e-8) / 7;  q = clamp(rint(x / s), -8, 7)
// then packed into the s4 mma A-fragment layout with the int8-path nibble order:
//   lane L of the warp holds k = g*128 + 4L + e (e < 4); word (kb, mb, l, r) with
//   kb = 2g + L/16, mb = row/16, l = (row%8)*4 + L%4, r = (row%16)/8 + 2*((L%16)/8),
//   nibble i = 2e + ((L/4)&1).
__global__ void a4_quant_kernel(const __nv_bfloat16* __restrict__ x, int ldx, int M,
                                int K, const float* __restrict__ gs,
                                uint32_t* __restrict__ A_pk, float* __restrict__ A_s,
                                int m_pad, int tmb) {
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int ngroups = K >> 7;
  const int row = gw / ngroups, g = gw - row * ngroups;
  if (row >= m_pad) return;
  // row chunks of 16*tmb rows (the kernel's `parallel` m-blocks): [chunk][K/64][tmb][32][4] words and
  // [chunk][K/128][16*tmb] scales; a single chunk is the original layout
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
  float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  // torch `tensor / 7.0` (CUDA, scalar divisor) multiplies by the fp32 reciprocal
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
}  // namespace MARLIN_NAMESPACE_NAME

namespace mp = MARLIN_NAMESPACE_NAME;

void mp_quant_a4(torch::Tensor x, torch::Tensor gs, torch::Tensor a_pk, torch::Tensor a_s,
                 int64_t tmb) {
  const int M = x.size(0), K = x.size(1);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 4 == 0);
  TORCH_CHECK(K % 128 == 0);
  const int nch = (M + 16 * tmb - 1) / (16 * tmb);
  TORCH_CHECK(a_pk.numel() * 4 >= (int64_t)nch * (K / 64) * tmb * 32 * 4 * 4);
  TORCH_CHECK(a_s.numel() >= (int64_t)nch * (K / 128) * 16 * tmb);
  const at::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  const int m_pad = nch * 16 * tmb;
  const int warps = m_pad * (K / 128);
  const int threads = 256;
  mp::a4_quant_kernel<<<(warps * 32 + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int)x.stride(0), M, K,
      gs.data_ptr<float>(), reinterpret_cast<uint32_t*>(a_pk.data_ptr()), a_s.data_ptr<float>(),
      m_pad, (int)tmb);
}

static torch::Tensor a4_gemm_impl(torch::Tensor a_pk, torch::Tensor a_s, torch::Tensor b_q,
                                  torch::Tensor b_s, torch::Tensor out, torch::Tensor locks,
                                  torch::Tensor c_tmp, int64_t prob_m, int64_t size_n, int64_t size_k,
                                  std::vector<int64_t> cfg) {
  const int prob_n = size_n, prob_k = size_k;
  TORCH_CHECK(cfg.size() == 7 || cfg.size() == 8, "a4 cfg = {tn, threads, stages, bps, tmb, MW, DEFER[, grid]}");
  const int grid_ovr = cfg.size() == 8 ? (int)cfg[7] : 0;
  const int tn = cfg[0], threads = cfg[1], stages = cfg[2], bps = cfg[3];
  const int tmb = cfg[4], MW = cfg[5], DEFER = cfg[6];
  TORCH_CHECK(out.size(0) == prob_m && out.size(1) == prob_n && out.is_contiguous());
  TORCH_CHECK(prob_m <= 16 * tmb || prob_m % (16 * tmb) == 0,
              "marlin_plus a4: M <= 16*tmb or a multiple of it");
  TORCH_CHECK(b_s.size(0) * 128 == prob_k, "marlin_plus a4: group 128 only");
  if (prob_m == 0) return out;
  if (prob_k % 128 || prob_n % tn) TORCH_CHECK(false, "marlin_plus a4: invalid config (tile)");
  const int dev = a_pk.get_device();
  const at::cuda::OptionalCUDAGuard guard(a_pk.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);
  int sms = 0, max_smem = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  auto kern = mp::get_a4_kernel(tmb, tn / 16, threads, stages, MW, DEFER, bps);
  TORCH_CHECK(kern != mp::MarlinA4Default, "marlin_plus a4: kernel not instantiated tmb=", tmb,
              " tn=", tn, " thr=", threads, " st=", stages, " MW=", MW, " DEFER=", DEFER,
              " bps=", bps);
  int smem = mp::a4_smem(tmb, tn, stages);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, kern);
  int dyn_max = max_smem / bps - (int)fa.sharedSizeBytes - (bps > 1 ? 1024 : 0);
  if (smem > dyn_max)
    TORCH_CHECK(false, "marlin_plus a4: invalid config (smem ", smem, " > ", dyn_max, ")");
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, threads, smem);
  if (occ < bps)
    TORCH_CHECK(false, "marlin_plus a4: invalid config (occupancy ", occ, " < ", bps, ")");
  // parallel split-K reduction: slots per column, passed as `lda`
  TORCH_CHECK(grid_ovr <= sms * bps, "marlin_plus a4: grid override > resident CTAs");
  const int grid = grid_ovr > 0 ? grid_ovr : sms * bps, n_tiles = prob_n / tn, k_tiles = prob_k / 128;
  // replicate the kernel's stripe schedule: slices per column <= ceil(k_tiles / iters) + 1
  int part2 = n_tiles;
  if (n_tiles > grid) {
    part2 = n_tiles % grid;
    if (part2 * 3 <= grid) part2 += grid;
  }
  const int iters = (k_tiles * part2 + grid - 1) / grid;
  int par_ms = (k_tiles + iters - 1) / iters + 1;
  const int64_t c_size = (int64_t)16 * tmb * tn * 4;  // bytes per partial
  const int64_t lock_bound = std::max(grid + 8, n_tiles + 2);
#if defined(A4_PAR_REDUCE) && A4_PAR_REDUCE == 7
  // consumer-sum: c_tmp = P [4][128][N] fp32 (slice partials), single m-chunk only
  TORCH_CHECK(prob_m <= 16 * tmb && prob_m <= 128 && par_ms <= 4,
              "marlin_plus a4 csum: M <= min(16 tmb, 128) and <= 4 slices per column");
  TORCH_CHECK(c_tmp.numel() * (int64_t)c_tmp.element_size() >= (int64_t)4 * 128 * prob_n * 4,
              "marlin_plus a4 csum: P too small");
#else
  TORCH_CHECK(c_tmp.numel() * (int64_t)c_tmp.element_size() >= lock_bound * par_ms * c_size,
              "marlin_plus a4: c_tmp too small for the parallel reduction (need ",
              lock_bound * par_ms * c_size / (1 << 20), " MB)");
  TORCH_CHECK(locks.numel() >= lock_bound, "marlin_plus a4: locks too small");
#endif
  kern<<<grid, threads, smem, stream>>>(
      (const int4*)a_pk.data_ptr(), (const int4*)b_q.data_ptr(), (int4*)out.data_ptr(),
      (int4*)c_tmp.data_ptr(), nullptr, a_s.data_ptr<float>(), (const int4*)b_s.data_ptr(),
      nullptr, nullptr, (int)prob_m, prob_n, prob_k, par_ms, locks.data_ptr<int>(), false,
      false, true, smem, nullptr, 0, nullptr, nullptr);
  return out;
}

torch::Tensor mp_gemm_a4(torch::Tensor a_pk, torch::Tensor a_s, torch::Tensor b_q,
                         torch::Tensor b_s, torch::Tensor out, torch::Tensor locks,
                         torch::Tensor c_tmp, int64_t prob_m, int64_t size_n, int64_t size_k,
                         std::vector<int64_t> cfg) {
  return a4_gemm_impl(a_pk, a_s, b_q, b_s, out, locks, c_tmp, prob_m, size_n, size_k, cfg);
}



std::vector<std::vector<int64_t>> mp_a4_instantiated();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_a4", &mp_gemm_a4, "marlin_plus W4A4 gemm", py::arg("a_pk"), py::arg("a_s"),
        py::arg("b_q"), py::arg("b_s"), py::arg("out"), py::arg("locks"), py::arg("c_tmp"),
        py::arg("prob_m"), py::arg("size_n"), py::arg("size_k"), py::arg("cfg"));
  m.def("quant_a4", &mp_quant_a4, "per-(token,128-group) s4 quant + fragment pack",
        py::arg("x"), py::arg("gs"), py::arg("a_pk"), py::arg("a_s"), py::arg("tmb"));
  m.def("instantiated", &mp_a4_instantiated, "instantiated W4A4 configs");
}
