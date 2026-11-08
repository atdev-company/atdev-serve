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
torch::Tensor mp_gemm(torch::Tensor a, torch::Tensor b_q, torch::Tensor b_s,
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
  TORCH_CHECK(out.size(0) == prob_m && out.size(1) == prob_n && out.is_contiguous());
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
        use_fp32_reduce, smem, A_src, lda_src, bar_ptr, gs_ptr);
    A_ptr += m_split * (lda / (a8 ? 16 : 8));
    if (a8) a_s_ptr += m_split;
    if (fq) A_src += m_split * (lda_src / 8);
    C_ptr += m_split * (prob_n / 8);
    rest_m -= m_split;
  }
  return out;
}

namespace MARLIN_NAMESPACE_NAME {
using ChainFuncPtr = void (*)(MpJobs, int, int4*, int*, int*, bool, int);
__global__ void MarlinChainDefault(MpJobs, int, int4*, int*, int*, bool, int) {}

static ChainFuncPtr get_chain_kernel(const vllm::ScalarType a_type, int thread_m_blocks,
                                     int thread_n_blocks, int thread_k_blocks,
                                     bool m_block_size_8, int threads, int stages) {
  auto kernel = MarlinChainDefault;
#include "mp_chain_selector.h"
  return kernel;
}
}  // namespace MARLIN_NAMESPACE_NAME

// P1 stage 2: njobs (<= MP_MAX_CHAIN) GEMMs in one persistent launch with a grid
// barrier between them. All jobs: same M (<= 16*max_tmb, one m-chunk), same
// dtype path (bf16 or int8 with a_s given), group 128. cfg = {thread_k,
// thread_n, threads, stages}.
// types[j]: 0 GEMM, 1 norm (a = h, b_q = residual in or 0-size, b_s = weight bf16 [k],
// out = x, aux[j] = residual out), 2 silu (a = gate_up [m, 2n], out [m, n]).
void mp_gemm_chain(std::vector<torch::Tensor> a, std::vector<torch::Tensor> b_q,
                   std::vector<torch::Tensor> b_s, std::vector<torch::Tensor> out,
                   std::vector<int64_t> size_n, std::vector<int64_t> size_k,
                   c10::optional<std::vector<torch::Tensor>> a_s_opt, torch::Tensor locks,
                   torch::Tensor c_tmp, torch::Tensor bar, std::vector<int64_t> cfg,
                   bool use_fp32_reduce, std::vector<int64_t> types,
                   c10::optional<std::vector<torch::Tensor>> aux_opt, double eps) {
  const int nj = a.size();
  TORCH_CHECK(nj >= 1 && nj <= MP_MAX_CHAIN && cfg.size() == 4);
  TORCH_CHECK(b_q.size() == nj && b_s.size() == nj && out.size() == nj &&
              size_n.size() == nj && size_k.size() == nj);
  if (types.empty()) types.assign(nj, 0);
  TORCH_CHECK(types.size() == nj);
  int g0 = -1;
  for (int j = 0; j < nj; j++) if (types[j] == 0 && g0 < 0) g0 = j;
  const int prob_m = a[0].size(0);
  const bool a8 = g0 >= 0 && a[g0].scalar_type() == at::kChar;
  if (a8) TORCH_CHECK(a_s_opt.has_value() && a_s_opt.value().size() == nj);
  const int dev = a[0].get_device();
  const at::cuda::OptionalCUDAGuard guard(a[0].device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);
  int sms = 0, max_smem = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  int thread_k = cfg[0], thread_n = cfg[1], threads = cfg[2], stages = cfg[3];
  int tmb = std::min(mp::div_ceil(prob_m, 16), 4);
  TORCH_CHECK(prob_m <= tmb * 16, "mp chain: M must fit one m-chunk");
  bool m8 = prob_m <= 8 && !a8;
  vllm::ScalarType at = a8 ? vllm::kS8 : vllm::kBFloat16;
  mp::MpJobs jobs{};
  for (int j = 0; j < nj; j++) {
    mp::MpJob& jb = jobs.j[j];
    jb.type = types[j];
    TORCH_CHECK(a[j].size(0) == prob_m && a[j].stride(1) == 1);
    TORCH_CHECK(out[j].size(0) == prob_m && out[j].size(1) == size_n[j] && out[j].is_contiguous());
    jb.A = (const int4*)a[j].data_ptr();
    jb.C = (int4*)out[j].data_ptr();
    jb.m = prob_m; jb.n = size_n[j]; jb.k = size_k[j];
    jb.lda = a[j].stride(0);
    jb.B = nullptr; jb.s = nullptr; jb.a_s = nullptr; jb.aux = nullptr; jb.w = nullptr;
    jb.eps = (float)eps;
    if (jb.type == 0) {
      TORCH_CHECK(a[j].size(1) == size_k[j] && a[j].scalar_type() == (a8 ? at::kChar : at::kBFloat16));
      TORCH_CHECK(b_s[j].size(0) * 128 == size_k[j], "mp chain: group 128 only");
      TORCH_CHECK(size_k[j] % thread_k == 0 && size_n[j] % thread_n == 0);
      TORCH_CHECK(a[j].stride(0) % (a8 ? 16 : 8) == 0);
      jb.B = (const int4*)b_q[j].data_ptr();
      jb.s = (const int4*)b_s[j].data_ptr();
      jb.a_s = a8 ? a_s_opt.value()[j].data_ptr<float>() : nullptr;
    } else if (jb.type == 1) {
      TORCH_CHECK(a[j].scalar_type() == at::kBFloat16 && a[j].size(1) == size_k[j] &&
                  size_n[j] == size_k[j] && size_k[j] % 8 == 0 && a[j].stride(0) % 8 == 0);
      TORCH_CHECK(b_s[j].numel() == size_k[j] && b_s[j].scalar_type() == at::kBFloat16);
      if (b_q[j].numel() > 0) {
        TORCH_CHECK(b_q[j].is_contiguous() && b_q[j].sizes() == out[j].sizes());
        jb.B = (const int4*)b_q[j].data_ptr();
        TORCH_CHECK(aux_opt.has_value() && aux_opt.value()[j].is_contiguous() &&
                    aux_opt.value()[j].sizes() == out[j].sizes());
        jb.aux = (int4*)aux_opt.value()[j].data_ptr();
      }
      jb.w = b_s[j].data_ptr();
    } else if (jb.type == 2) {
      TORCH_CHECK(a[j].scalar_type() == at::kBFloat16 && a[j].size(1) >= 2 * size_n[j] &&
                  size_n[j] % 8 == 0 && a[j].stride(0) % 8 == 0);
    } else {
      TORCH_CHECK(false, "mp chain: unknown job type");
    }
  }
  auto kern = mp::get_chain_kernel(at, tmb, thread_n / 16, thread_k / 16, m8, threads, stages);
  TORCH_CHECK(kern != mp::MarlinChainDefault, "mp chain: kernel not instantiated: tmb=", tmb,
              " m8=", m8, " k=", thread_k, " n=", thread_n, " thr=", threads, " st=", stages,
              " a8=", a8);
  // shared memory: the largest job's requirement (cache_size depends only on the tile)
  mp::TCfg c{thread_k, thread_n, threads};
  // same dynamic smem as mp_gemm (stock marlin_plus launch) so the kernel sees
  // the same L1/smem split and sh_* layout
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, kern);
  int smem = max_smem - (int)fa.sharedSizeBytes - 512;
  if (g0 >= 0)
    TORCH_CHECK(mp::valid(c, tmb, size_k[g0], size_n[g0], 128, a8, stages, smem),
                "mp chain: invalid config");
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, threads, smem);
  TORCH_CHECK(occ >= 1, "mp chain: kernel does not fit one CTA/SM");
  kern<<<sms, threads, smem, stream>>>(jobs, nj, (int4*)c_tmp.data_ptr(), locks.data_ptr<int>(),
                                       bar.data_ptr<int>(), use_fp32_reduce, smem);
}

std::vector<std::vector<int64_t>> mp_instantiated();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_chain", &mp_gemm_chain, "marlin_plus chained gemms (one launch)", py::arg("a"),
        py::arg("b_q"), py::arg("b_s"), py::arg("out"), py::arg("size_n"), py::arg("size_k"),
        py::arg("a_s") = py::none(), py::arg("locks"), py::arg("c_tmp"), py::arg("bar"),
        py::arg("cfg"), py::arg("use_fp32_reduce") = true,
        py::arg("types") = std::vector<int64_t>(), py::arg("aux") = py::none(),
        py::arg("eps") = 1e-6);
  m.def("gemm", &mp_gemm, "marlin_plus gemm", py::arg("a"), py::arg("b_q"),
        py::arg("b_s"), py::arg("a_s"), py::arg("out"), py::arg("locks"),
        py::arg("c_tmp"), py::arg("size_n"), py::arg("size_k"), py::arg("cfg"),
        py::arg("use_fp32_reduce") = true, py::arg("use_atomic_add") = false,
        py::arg("fq_gscale") = py::none(), py::arg("fq_aq") = py::none(),
        py::arg("fq_as") = py::none(), py::arg("fq_bar") = py::none());
  m.def("instantiated", &mp_instantiated, "instantiated kernel configs");
}
