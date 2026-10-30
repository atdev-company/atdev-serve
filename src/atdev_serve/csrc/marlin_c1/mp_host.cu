// marlin_plus host side: vLLM 0.31 Marlin (copied, Apache-2.0) with the
// launch configuration exposed, so every (thread_k, thread_n, threads,
// stages, blocks_per_sm, max_m_blocks) can be chosen per shape at runtime.
//
// The kernel instantiations live in generated mp_kernels_*.cu files and the
// selector in mp_selector.h (see atdev_serve/marlin_plus.py::generate()).
// Weight/scale layouts are exactly vLLM's (gptq_marlin_repack +
// marlin_permute_scales [+ marlin_act_int8_process_scales]), so this module
// reuses the tensors vLLM already holds: no extra weight memory.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "kernel.h"
#ifndef E6_MODE
  #define E6_MODE 0
#endif

namespace MARLIN_NAMESPACE_NAME {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS) {};
using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

static MarlinFuncPtr get_kernel(const vllm::ScalarType a_type,
                                const vllm::ScalarType b_type,
                                const vllm::ScalarType c_type,
                                const vllm::ScalarType s_type,
                                int thread_m_blocks, int thread_n_blocks,
                                int thread_k_blocks, bool m_block_size_8,
                                bool has_zp, int group_blocks, int threads,
                                bool is_zp_float, int stages) {
  auto kernel = MarlinDefault;
#include "mp_selector.h"
  return kernel;
}

struct TCfg {
  int thread_k, thread_n, threads;
};

static int cache_size(TCfg c, int thread_m_blocks, int group_size,
                      bool is_a_8bit, int stages) {
  int tb_k = c.thread_k, tb_n = c.thread_n, tb_m = thread_m_blocks * 16;
  int sh_a = stages * (tb_m * tb_k) * (is_a_8bit ? 1 : 2);
  int sh_b = stages * (tb_k * tb_n / 8) * 4;
  int sh_red = tb_m * (tb_n + 8) * 2;
  int sh_bias = tb_n * 2;
  int tmp = (sh_b > sh_red ? sh_red : sh_b) + sh_bias;
  tmp = std::max(std::max(sh_b, sh_red), tmp);
  int tb_groups = group_size == -1 ? 1 : div_ceil(tb_k, group_size);
  int sh_s = tb_groups * tb_n * 2 * stages;
  return tmp + sh_a + sh_s;
}

static bool valid(TCfg c, int tmb, int k, int n, int group_size, bool a8,
                  int stages, int max_smem) {
  if (k % c.thread_k || n % c.thread_n) return false;
  if (c.thread_n < min_thread_n || c.thread_k < min_thread_k) return false;
  if (c.threads < 128) return false;
  return cache_size(c, tmb, group_size, a8, stages) <= max_smem;
}

// Stock vLLM 0.31 heuristic (marlin.cu determine_exec_config + the
// small-grid {128,64,128} override), used when thread_k == -1.
static TCfg stock_cfg(int tmb, int m_split, int n, int k, int group_size,
                      bool a8, int stages, int max_smem, int sms,
                      const vllm::ScalarType& at, const vllm::ScalarType& bt,
                      const vllm::ScalarType& ct, const vllm::ScalarType& st,
                      bool m8) {
  static const TCfg small[] = {{128, 128, 256}, {64, 128, 128}, {128, 64, 128}};
  static const TCfg large[] = {{64, 256, 256}, {64, 128, 128}, {128, 64, 128}};
  const TCfg* cfgs = tmb > 1 ? large : small;
  TCfg r{-1, -1, -1};
  for (int i = 0; i < 3; i++) {
    TCfg c = cfgs[i];
    if (!valid(c, tmb, k, n, group_size, a8, stages, max_smem - 512)) continue;
    auto kern = get_kernel(at, bt, ct, st, tmb, c.thread_n / 16, c.thread_k / 16,
                           m8, false, group_size / 16, c.threads, false, stages);
    if (kern == MarlinDefault) continue;
    r = c;
    break;
  }
  if (r.thread_n != -1) {
    if (n / r.thread_n * div_ceil(m_split, tmb * 16) * 4 <= sms) {
      TCfg alt{128, 64, 128};
      if (valid(alt, tmb, k, n, group_size, a8, stages, max_smem)) r = alt;
    }
  }
  return r;
}

}  // namespace MARLIN_NAMESPACE_NAME

namespace mp = MARLIN_NAMESPACE_NAME;

// a: [M, K] bf16 or int8 (row stride lda); b_q: Marlin-packed; b_s: permuted
// scales [K/128, N] (int16-in-bf16 for the int8 path); a_s: [M] fp32 (int8
// only, already multiplied by input_global_scale); out: [M, N] bf16.
// cfg = {thread_k, thread_n, threads, stages, blocks_per_sm, max_m_blocks}
// (thread_k = -1: stock heuristic, stages 4, 1 block/SM, max_m_blocks 4).
static torch::Tensor mp_gemm_impl(const MpEpi& epi, torch::Tensor a, torch::Tensor b_q, torch::Tensor b_s,
                      c10::optional<torch::Tensor> a_s_opt, torch::Tensor out,
                      torch::Tensor locks, torch::Tensor c_tmp, int64_t size_n,
                      int64_t size_k, std::vector<int64_t> cfg,
                      bool use_fp32_reduce, bool use_atomic_add,
                      c10::optional<torch::Tensor> fq_gscale,
                      c10::optional<torch::Tensor> fq_aq,
                      c10::optional<torch::Tensor> fq_as,
                      c10::optional<torch::Tensor> fq_bar) {
  const int prob_m = a.size(0);
  const int prob_n = size_n, prob_k = size_k;
  TORCH_CHECK(a.size(1) == size_k && a.stride(1) == 1);
  TORCH_CHECK(out.size(0) == prob_m && out.size(1) == (epi.mode == 3 ? prob_n / 2 : prob_n) && out.is_contiguous());
  TORCH_CHECK(cfg.size() == 6);
  if (prob_m == 0) return out;
  // fused quant: `a` is bf16, the kernel quantises it into fq_aq / fq_as
  const bool fq = fq_gscale.has_value();
  const bool a8 = fq || a.scalar_type() == at::kChar;
  if (fq) {
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && fq_aq.has_value() &&
                fq_as.has_value() && fq_bar.has_value());
    TORCH_CHECK(fq_aq.value().numel() >= (int64_t)prob_m * prob_k);
    TORCH_CHECK(fq_as.value().numel() >= prob_m);
    TORCH_CHECK(a.stride(0) % 8 == 0 && prob_k % 8 == 0);
  }
  TORCH_CHECK(a8 || a.scalar_type() == at::kBFloat16);
  TORCH_CHECK(out.scalar_type() == at::kBFloat16);
  vllm::ScalarType at = a8 ? vllm::kS8 : vllm::kBFloat16;
  vllm::ScalarType bt = vllm::kU4B8, ct = vllm::kBFloat16, st = vllm::kBFloat16;
  const int num_groups = b_s.size(0);
  const int group_size = prob_k / num_groups;
  TORCH_CHECK(group_size == 128, "marlin_plus: group 128 only");
  const int group_blocks = group_size / 16;
  const float* a_s_ptr = nullptr;
  if (a8 && !fq) {
    TORCH_CHECK(a_s_opt.has_value());
    a_s_ptr = a_s_opt.value().data_ptr<float>();
  }
  const int dev = a.get_device();
  const at::cuda::OptionalCUDAGuard guard(a.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);
  int sms = 0, max_smem = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

  int thread_k = cfg[0], thread_n = cfg[1], threads = cfg[2];
  int stages = cfg[3] > 0 ? cfg[3] : 4;
  int bps = cfg[4] > 0 ? cfg[4] : 1;
  int max_tmb = cfg[5] > 0 ? cfg[5] : 4;
  const bool auto_cfg = thread_k < 0;

  const int lda = fq ? prob_k : a.stride(0);
  const int4* A_ptr = fq ? (const int4*)fq_aq.value().data_ptr()
                         : (const int4*)a.data_ptr();
  const int4* A_src = fq ? (const int4*)a.data_ptr() : nullptr;
  const int lda_src = fq ? a.stride(0) : 0;
  int* bar_ptr = fq ? fq_bar.value().data_ptr<int>() : nullptr;
  const float* gs_ptr = fq ? fq_gscale.value().data_ptr<float>() : nullptr;
  if (fq) a_s_ptr = fq_as.value().data_ptr<float>();
  const int4* B_ptr = (const int4*)b_q.data_ptr();
  int4* C_ptr = (int4*)out.data_ptr();
  int4* C_tmp_ptr = (int4*)c_tmp.data_ptr();
  const int4* s_ptr = (const int4*)b_s.data_ptr();
  int* locks_ptr = locks.data_ptr<int>();

  int max_par = 16;
  if (prob_n <= 4096) max_par = 16 * 8;
  int rest_m = prob_m;
  while (rest_m) {
    int par_count = rest_m / (max_tmb * 16);
    if (par_count > max_par) par_count = max_par;
    int m_split = par_count > 0 ? par_count * max_tmb * 16 : rest_m;
    int tmb = std::min(mp::div_ceil(m_split, 16), max_tmb);
    bool m8 = m_split <= 8 && !a8;
    mp::TCfg c{thread_k, thread_n, threads};
    int ch_stages = stages, ch_bps = bps;
    bool ch_auto = auto_cfg;
    if (!ch_auto) {
      // remainder chunk (fewer m-blocks than tuned) or uninstantiated combo:
      // fall back to the stock heuristic for this chunk
      auto kk = mp::get_kernel(at, bt, ct, st, tmb, c.thread_n / 16, c.thread_k / 16,
                               m8, false, group_blocks, c.threads, false, stages);
      if (kk == mp::MarlinDefault) { ch_auto = true; ch_stages = 4; ch_bps = 1; }
      TORCH_CHECK(epi.mode == 0 || !ch_auto, "mp_gemm_epi: config not instantiated");
    }
    int smem = max_smem / ch_bps - (ch_bps > 1 ? 1024 : 0);
    if (ch_auto) {
      c = mp::stock_cfg(tmb, m_split, prob_n, prob_k, group_size, a8, ch_stages,
                        max_smem, sms, at, bt, ct, st, m8);
      TORCH_CHECK(c.thread_k != -1, "marlin_plus: no stock config");
    }
    TORCH_CHECK(mp::valid(c, tmb, prob_k, prob_n, group_size, a8, ch_stages, smem),
                "marlin_plus: invalid config k=", c.thread_k, " n=", c.thread_n,
                " thr=", c.threads, " stages=", ch_stages, " tmb=", tmb);
    auto kern = mp::get_kernel(at, bt, ct, st, tmb, c.thread_n / 16,
                               c.thread_k / 16, m8, false, group_blocks,
                               c.threads, false, ch_stages);
    TORCH_CHECK(kern != mp::MarlinDefault,
                "marlin_plus: kernel not instantiated: tmb=", tmb, " m8=", m8,
                " k=", c.thread_k, " n=", c.thread_n, " thr=", c.threads,
                " stages=", ch_stages, " a8=", a8);
    {  // static __shared__ (fused-quant prologue) counts against the opt-in max
      cudaFuncAttributes fa;
      cudaFuncGetAttributes(&fa, kern);
      int dyn_max = max_smem / ch_bps - (int)fa.sharedSizeBytes - (ch_bps > 1 ? 1024 : 0);
      if (smem > dyn_max) smem = dyn_max;
      TORCH_CHECK(mp::valid(c, tmb, prob_k, prob_n, group_size, a8, ch_stages, smem),
                  "marlin_plus: invalid config (smem) k=", c.thread_k, " n=", c.thread_n);
    }
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (ch_bps > 1 || fq) {
      // persistent grid + inter-CTA waits: every CTA must be co-resident
      int occ = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, c.threads, smem);
      TORCH_CHECK(occ >= ch_bps, "marlin_plus: invalid config (occupancy ", occ,
                  " < blocks_per_sm ", ch_bps, ")");
    }
    // vLLM rule: atomic-add reduction only for small outputs. ATDEV_MP_FORCE_ATOMIC=1
    // (A2 experiment) lifts the size limit to test whether the serial split-K
    // reduction chain is the fixed per-call overhead.
    static const bool force_atomic = getenv("ATDEV_MP_FORCE_ATOMIC") != nullptr;
    bool part_atomic = use_atomic_add && (force_atomic || mp::div_ceil(m_split, 64) * prob_n <= 2048);
    // P3 microbench: ATDEV_MP_GRID=<n> launches n (<= sms*bps) persistent CTAs instead of sms*bps
    // (fewer stripes per column -> shorter serial split-K chain). Not used in serving.
    int grid = sms * ch_bps;
    if (const char* ge = getenv("ATDEV_MP_GRID")) {
      int gv = atoi(ge);
      if (gv > 0 && gv < grid) grid = gv;
    }
    kern<<<grid, c.threads, smem, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, nullptr, a_s_ptr, s_ptr, nullptr,
        nullptr, m_split, prob_n, prob_k, lda, locks_ptr, false, part_atomic,
        use_fp32_reduce, smem, A_src, lda_src, bar_ptr, gs_ptr, epi);
    A_ptr += m_split * (lda / (a8 ? 16 : 8));
    if (a8) a_s_ptr += m_split;
    if (fq) A_src += m_split * (lda_src / 8);
    C_ptr += m_split * (prob_n / 8);
    rest_m -= m_split;
  }
  return out;
}

torch::Tensor mp_gemm(torch::Tensor a, torch::Tensor b_q, torch::Tensor b_s,
                      c10::optional<torch::Tensor> a_s_opt, torch::Tensor out,
                      torch::Tensor locks, torch::Tensor c_tmp, int64_t size_n,
                      int64_t size_k, std::vector<int64_t> cfg,
                      bool use_fp32_reduce, bool use_atomic_add,
                      c10::optional<torch::Tensor> fq_gscale,
                      c10::optional<torch::Tensor> fq_aq,
                      c10::optional<torch::Tensor> fq_as,
                      c10::optional<torch::Tensor> fq_bar) {
  MpEpi epi{};
  return mp_gemm_impl(epi, a, b_q, b_s, a_s_opt, out, locks, c_tmp, size_n, size_k, cfg, use_fp32_reduce,
                      use_atomic_add, fq_gscale, fq_aq, fq_as, fq_bar);
}

// E6 fused epilogues (bf16 W4A16, M <= 16, one m-split). mode 1: R [M, N] bf16 residual in, W [N] bf16 Gemma
// weight, C2 [M, N] bf16 residual out, P [16 * T] fp32 partials out (T = N / thread_n); out = bf16(x * (1 + W)).
// mode 2/3: Pin [16 * T_in] partials of the producer, n_norm = producer N (the norm width); mode 3 out [M, N / 2].
torch::Tensor mp_gemm_epi(torch::Tensor a, torch::Tensor b_q, torch::Tensor b_s, torch::Tensor out,
                          torch::Tensor locks, torch::Tensor c_tmp, int64_t size_n, int64_t size_k,
                          std::vector<int64_t> cfg, int64_t mode, c10::optional<torch::Tensor> R,
                          c10::optional<torch::Tensor> W, c10::optional<torch::Tensor> C2,
                          c10::optional<torch::Tensor> P, c10::optional<torch::Tensor> Pin, int64_t T_in,
                          int64_t n_norm, double eps, c10::optional<torch::Tensor> Rout) {
  TORCH_CHECK(a.size(0) <= 16 && a.scalar_type() == at::kBFloat16, "mp_gemm_epi: bf16, M <= 16 only");
  TORCH_CHECK(mode == 0 || a.size(0) > 8, "mp_gemm_epi: fused epilogues need 8 < M <= 16 (m16 layout)");
  TORCH_CHECK(cfg.size() == 6 && cfg[0] > 0, "mp_gemm_epi: explicit config required");
  const int thread_n = cfg[1], threads = cfg[2];
  TORCH_CHECK(mode == E6_MODE, "mp_gemm_epi: this build has E6_MODE=", E6_MODE, ", called with mode ", mode);
  MpEpi epi{};
  epi.mode = (int)mode;
  if (mode == 1) {
    TORCH_CHECK(R && W && C2 && P);
    TORCH_CHECK(R->is_contiguous() && C2->is_contiguous() && W->is_contiguous());
    TORCH_CHECK(R->size(0) == a.size(0) && R->size(1) == size_n && C2->sizes() == R->sizes() && W->numel() == size_n);
    TORCH_CHECK(P->numel() >= a.size(0) * (size_n / thread_n) && P->scalar_type() == at::kFloat);
    epi.R = (const int4*)R->data_ptr();
    epi.W = (const int4*)W->data_ptr();
    epi.C2 = (int4*)C2->data_ptr();
    epi.P = P->data_ptr<float>();
  } else if (mode == 2 || mode == 3) {
    TORCH_CHECK(Pin && Pin->scalar_type() == at::kFloat && Pin->numel() >= a.size(0) * T_in);
    TORCH_CHECK(T_in <= (threads / 16) * 10, "mp_gemm_epi: too many partials ", T_in);
    if (mode == 3) TORCH_CHECK(thread_n == 128, "mp_gemm_epi: SiLU pair needs thread_n = 128");
    epi.Pin = Pin->data_ptr<float>();
    epi.T_in = (int)T_in;
    epi.n_norm = (int)n_norm;
    epi.eps = (float)eps;
    if (Rout) {
      TORCH_CHECK(Rout->scalar_type() == at::kFloat && Rout->numel() >= a.size(0));
      epi.Rout = Rout->data_ptr<float>();
    }
  } else {
    TORCH_CHECK(mode == 0);
  }
  return mp_gemm_impl(epi, a, b_q, b_s, c10::nullopt, out, locks, c_tmp, size_n, size_k, cfg, true, false,
                      c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
}

std::vector<std::vector<int64_t>> mp_instantiated();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_epi", &mp_gemm_epi, "marlin_c1 gemm with E6 epilogue", py::arg("a"), py::arg("b_q"),
        py::arg("b_s"), py::arg("out"), py::arg("locks"), py::arg("c_tmp"), py::arg("size_n"),
        py::arg("size_k"), py::arg("cfg"), py::arg("mode"), py::arg("R") = py::none(), py::arg("W") = py::none(),
        py::arg("C2") = py::none(), py::arg("P") = py::none(), py::arg("Pin") = py::none(),
        py::arg("T_in") = 0, py::arg("n_norm") = 0, py::arg("eps") = 1e-6, py::arg("Rout") = py::none());
  m.def("gemm", &mp_gemm, "marlin_plus gemm", py::arg("a"), py::arg("b_q"),
        py::arg("b_s"), py::arg("a_s"), py::arg("out"), py::arg("locks"),
        py::arg("c_tmp"), py::arg("size_n"), py::arg("size_k"), py::arg("cfg"),
        py::arg("use_fp32_reduce") = true, py::arg("use_atomic_add") = false,
        py::arg("fq_gscale") = py::none(), py::arg("fq_aq") = py::none(),
        py::arg("fq_as") = py::none(), py::arg("fq_bar") = py::none());
  m.def("instantiated", &mp_instantiated, "instantiated kernel configs");
}
