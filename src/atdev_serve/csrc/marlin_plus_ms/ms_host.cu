// marlin_plus REFRAME A2 host side: M-split W4A8 kernel (mp_a8ms.cuh).
// Separate extension from marlin_plus (keeps that module's build cache valid).
// cfg = {thread_k, thread_n, threads, stages, blocks_per_sm (= MINB),
//        thread_m_blocks, MW, DEFER}. thread_m_blocks is fixed for every chunk;
// a remainder chunk runs the same kernel with predicated rows.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "kernel.h"
#include "mp_a8ms_decl.h"

namespace MARLIN_NAMESPACE_NAME {
__global__ void MarlinMSDefault(MARLIN_KERNEL_PARAMS) {};
using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

static MarlinFuncPtr get_ms_kernel(int tmb, int tnb, int tkb, int threads, int stages,
                                   int MW, int DEFER, int MINB, int U8) {
  auto kernel = MarlinMSDefault;
#include "ms_selector.h"
  return kernel;
}
static int ms_smem(int tmb, int tn, int tk, int stages, int u8 = 0) {
  int tnb = tn / 16;
  int red = (2 * tnb + 1) * 16 * tmb;
  int b = stages * (tn * (tk / 32));
  int s = stages * 2 * tnb;
  int a = stages * (tk / 16) * 16 * tmb;
  return 16 * (4 * tmb + std::max(red, b) + s + a) + (u8 ? stages * 16 * tmb * 4 : 0);
}
}  // namespace MARLIN_NAMESPACE_NAME

namespace mp = MARLIN_NAMESPACE_NAME;

torch::Tensor mp_gemm_ms(torch::Tensor a, torch::Tensor b_q, torch::Tensor b_s,
                         c10::optional<torch::Tensor> a_s_opt, torch::Tensor out,
                         torch::Tensor locks, torch::Tensor c_tmp, int64_t size_n,
                         int64_t size_k, std::vector<int64_t> cfg,
                         c10::optional<torch::Tensor> fq_gscale,
                         c10::optional<torch::Tensor> fq_aq,
                         c10::optional<torch::Tensor> fq_as,
                         c10::optional<torch::Tensor> fq_bar,
                         c10::optional<torch::Tensor> a_sa) {
  const int prob_m = a.size(0);
  const int prob_n = size_n, prob_k = size_k;
  TORCH_CHECK(a.size(1) == size_k && a.stride(1) == 1);
  TORCH_CHECK(out.size(0) == prob_m && out.size(1) == prob_n && out.is_contiguous());
  TORCH_CHECK(cfg.size() == 8, "ms cfg = {tk, tn, threads, stages, bps, tmb, MW, DEFER}");
  if (prob_m == 0) return out;
  const bool fq = fq_gscale.has_value();
  if (fq) {
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && fq_aq.has_value() &&
                fq_as.has_value() && fq_bar.has_value());
    TORCH_CHECK(fq_aq.value().numel() >= (int64_t)prob_m * prob_k);
    TORCH_CHECK(fq_as.value().numel() >= prob_m);
    TORCH_CHECK(a.stride(0) % 8 == 0 && prob_k % 8 == 0);
  } else {
    TORCH_CHECK(a.scalar_type() == at::kChar && a_s_opt.has_value());
    TORCH_CHECK(a.stride(0) % 16 == 0);
  }
  TORCH_CHECK(b_s.size(0) * 128 == prob_k, "marlin_plus ms: group 128 only");
  const int tk = cfg[0], tn = cfg[1], threads = cfg[2], stages = cfg[3];
  const int bps = cfg[4], tmb = cfg[5], MW = cfg[6], DEFER = cfg[7];
  if (prob_k % tk || prob_n % tn) TORCH_CHECK(false, "marlin_plus ms: invalid config (tile)");
  const int dev = a.get_device();
  const at::cuda::OptionalCUDAGuard guard(a.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);
  int sms = 0, max_smem = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  // P3 U8 (bit-exact, unsigned-nibble B + C = -8*Sa): enabled when the caller hands an Sa buffer
  // (int32 [M][K/128]); in fq mode the prologue fills it, otherwise the activation producer did
  const bool u8 = a_sa.has_value();
  if (u8) TORCH_CHECK(a_sa.value().scalar_type() == at::kInt && a_sa.value().numel() >= (int64_t)256 * (prob_k / 128) &&
                      prob_m <= 256, "marlin_plus ms: a_sa must be int32 [K/128][256] (group-major), M <= 256");
  auto kern = mp::get_ms_kernel(tmb, tn / 16, tk / 16, threads, stages, MW, DEFER, bps, u8 ? 1 : 0);
  TORCH_CHECK(kern != mp::MarlinMSDefault, "marlin_plus ms: kernel not instantiated tmb=", tmb,
              " tk=", tk, " tn=", tn, " thr=", threads, " st=", stages, " MW=", MW,
              " DEFER=", DEFER, " bps=", bps);
  int smem = mp::ms_smem(tmb, tn, tk, stages, u8 ? 1 : 0);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, kern);
  int dyn_max = max_smem / bps - (int)fa.sharedSizeBytes - (bps > 1 ? 1024 : 0);
  if (smem > dyn_max)
    TORCH_CHECK(false, "marlin_plus ms: invalid config (smem ", smem, " > ", dyn_max, ")");
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, threads, smem);
  if (occ < bps)
    TORCH_CHECK(false, "marlin_plus ms: invalid config (occupancy ", occ, " < ", bps, ")");
  const int lda = fq ? prob_k : a.stride(0);
  const int4* A_ptr = fq ? (const int4*)fq_aq.value().data_ptr() : (const int4*)a.data_ptr();
  const int4* A_src = fq ? (const int4*)a.data_ptr() : nullptr;
  const int lda_src = fq ? a.stride(0) : 0;
  int* bar_ptr = fq ? fq_bar.value().data_ptr<int>() : nullptr;
  const float* gs_ptr = fq ? fq_gscale.value().data_ptr<float>() : nullptr;
  const float* a_s_ptr = fq ? fq_as.value().data_ptr<float>() : a_s_opt.value().data_ptr<float>();
  const int4* B_ptr = (const int4*)b_q.data_ptr();
  int4* C_ptr = (int4*)out.data_ptr();
  int4* C_tmp_ptr = (int4*)c_tmp.data_ptr();
  const int4* s_ptr = (const int4*)b_s.data_ptr();
  int* locks_ptr = locks.data_ptr<int>();
  const int* sa_ptr = u8 ? a_sa.value().data_ptr<int>() : nullptr;
  int max_par = 16;
  if (prob_n <= 4096) max_par = 16 * 8;
  int rest_m = prob_m;
  while (rest_m) {
    int par_count = rest_m / (tmb * 16);
    if (par_count > max_par) par_count = max_par;
    int m_split = par_count > 0 ? par_count * tmb * 16 : rest_m;
    kern<<<sms * bps, threads, smem, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, nullptr, a_s_ptr, s_ptr, nullptr,
        reinterpret_cast<const int4*>(sa_ptr), m_split, prob_n, prob_k, lda, locks_ptr, false, false,
        true, smem, A_src, lda_src, bar_ptr, gs_ptr);
    A_ptr += m_split * (lda / 16);
    if (sa_ptr) sa_ptr += m_split;  // group-major [K/128][256]: next rows
    a_s_ptr += m_split;
    if (fq) A_src += m_split * (lda_src / 8);
    C_ptr += m_split * (prob_n / 8);
    rest_m -= m_split;
  }
  return out;
}

std::vector<std::vector<int64_t>> mp_ms_instantiated();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_ms", &mp_gemm_ms, "marlin_plus M-split W4A8 gemm", py::arg("a"), py::arg("b_q"),
        py::arg("b_s"), py::arg("a_s"), py::arg("out"), py::arg("locks"), py::arg("c_tmp"),
        py::arg("size_n"), py::arg("size_k"), py::arg("cfg"),
        py::arg("fq_gscale") = py::none(), py::arg("fq_aq") = py::none(),
        py::arg("fq_as") = py::none(), py::arg("fq_bar") = py::none(), py::arg("a_sa") = py::none());
  m.def("instantiated", &mp_ms_instantiated, "instantiated M-split configs");
}
