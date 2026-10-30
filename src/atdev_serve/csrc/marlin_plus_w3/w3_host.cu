// marlin_plus W3 host side (bet #2): vLLM Marlin W4A16 launch logic (copied from mp_host.cu,
// bf16 activations only) for the 3-bit planar weight stream (gen/marlin_template_w3.h).
// B: int32 [K/16, 3, N/2] (planes), produced by atdev_serve.w3_mixed.pack_w3 from the Marlin
// int4 repack; scales: the unchanged Marlin-permuted bf16 [K/128, N].
// cfg = {thread_k, thread_n, threads, stages, blocks_per_sm, max_m_blocks}; thread_k = -1 ->
// stock heuristic (stages 4, 1 CTA/SM) -> output bit-identical to stock Marlin W4A16 on the
// same (3-bit-valued) weights.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "kernel.h"
#include "w3_decl.h"

namespace MARLIN_NAMESPACE_NAME {

__global__ void MarlinW3Default(MARLIN_KERNEL_PARAMS) {};
using W3FuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

static W3FuncPtr get_kernel(int thread_m_blocks, int thread_n_blocks, int thread_k_blocks,
                            bool m_block_size_8, int threads, int stages) {
  auto kernel = MarlinW3Default;
#include "w3_selector.h"
  return kernel;
}

struct TCfg {
  int thread_k, thread_n, threads;
};

static int cache_size(TCfg c, int thread_m_blocks, bool m8, int stages) {
  int tb_k = c.thread_k, tb_n = c.thread_n, tb_m = m8 ? 8 : thread_m_blocks * 16;
  int sh_a = stages * (tb_m * tb_k) * 2;
  int sh_b = stages * (tb_k * tb_n * 3 / 32) * 4;  // 3-bit planar
  int sh_red = tb_m * (tb_n + 8) * 2;
  int sh_bias = tb_n * 2;
  int tmp = (sh_b > sh_red ? sh_red : sh_b) + sh_bias;
  tmp = std::max(std::max(sh_b, sh_red), tmp);
  int tb_groups = div_ceil(tb_k, 128);
  int sh_s = tb_groups * tb_n * 2 * stages;
  return tmp + sh_a + sh_s;
}

static bool valid(TCfg c, int tmb, bool m8, int k, int n, int stages, int max_smem) {
  if (k % c.thread_k || n % c.thread_n) return false;
  if (c.thread_n < min_thread_n || c.thread_k < min_thread_k) return false;
  if (c.threads < 128) return false;
  return cache_size(c, tmb, m8, stages) <= max_smem;
}

static TCfg stock_cfg(int tmb, int m_split, int n, int k, bool m8, int stages, int max_smem,
                      int sms) {
  static const TCfg small[] = {{128, 128, 256}, {64, 128, 128}, {128, 64, 128}};
  static const TCfg large[] = {{64, 256, 256}, {64, 128, 128}, {128, 64, 128}};
  const TCfg* cfgs = tmb > 1 ? large : small;
  TCfg r{-1, -1, -1};
  for (int i = 0; i < 3; i++) {
    TCfg c = cfgs[i];
    if (!valid(c, tmb, m8, k, n, stages, max_smem - 512)) continue;
    auto kern = get_kernel(tmb, c.thread_n / 16, c.thread_k / 16, m8, c.threads, stages);
    if (kern == MarlinW3Default) continue;
    r = c;
    break;
  }
  if (r.thread_n != -1) {
    if (n / r.thread_n * div_ceil(m_split, tmb * 16) * 4 <= sms) {
      TCfg alt{128, 64, 128};
      if (valid(alt, tmb, m8, k, n, stages, max_smem) &&
          get_kernel(tmb, 4, 8, m8, 128, stages) != MarlinW3Default)
        r = alt;
    }
  }
  return r;
}

}  // namespace MARLIN_NAMESPACE_NAME

namespace mp = MARLIN_NAMESPACE_NAME;

torch::Tensor w3_gemm(torch::Tensor a, torch::Tensor b_q3, torch::Tensor b_s, torch::Tensor out,
                      torch::Tensor locks, torch::Tensor c_tmp, int64_t size_n, int64_t size_k,
                      std::vector<int64_t> cfg) {
  const int prob_m = a.size(0);
  const int prob_n = size_n, prob_k = size_k;
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && a.size(1) == size_k && a.stride(1) == 1);
  TORCH_CHECK(a.stride(0) % 8 == 0, "w3: A row stride must be a multiple of 8");
  TORCH_CHECK(out.size(0) == prob_m && out.size(1) == prob_n && out.is_contiguous() &&
              out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(cfg.size() == 6);
  TORCH_CHECK(b_q3.scalar_type() == at::kInt && b_q3.is_contiguous() &&
              b_q3.numel() == (int64_t)prob_k / 16 * 3 * (prob_n / 2), "w3: B shape");
  TORCH_CHECK(b_s.size(0) * 128 == prob_k, "w3: group 128 only");
  TORCH_CHECK(prob_k % 128 == 0 && prob_n % 64 == 0);
  if (prob_m == 0) return out;
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

  const int lda = a.stride(0);
  const int4* A_ptr = (const int4*)a.data_ptr();
  const int4* B_ptr = (const int4*)b_q3.data_ptr();
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
    bool m8 = m_split <= 8;
    mp::TCfg c{thread_k, thread_n, threads};
    int ch_stages = stages, ch_bps = bps;
    bool ch_auto = auto_cfg;
    if (!ch_auto) {
      auto kk = mp::get_kernel(tmb, c.thread_n / 16, c.thread_k / 16, m8, c.threads, stages);
      if (kk == mp::MarlinW3Default) { ch_auto = true; ch_stages = 4; ch_bps = 1; }
    }
    int smem = max_smem / ch_bps - (ch_bps > 1 ? 1024 : 0);
    if (ch_auto) {
      c = mp::stock_cfg(tmb, m_split, prob_n, prob_k, m8, ch_stages, max_smem, sms);
      TORCH_CHECK(c.thread_k != -1, "w3: no stock config for tmb=", tmb);
    }
    TORCH_CHECK(mp::valid(c, tmb, m8, prob_k, prob_n, ch_stages, smem),
                "w3: invalid config k=", c.thread_k, " n=", c.thread_n, " thr=", c.threads,
                " stages=", ch_stages, " tmb=", tmb);
    auto kern = mp::get_kernel(tmb, c.thread_n / 16, c.thread_k / 16, m8, c.threads, ch_stages);
    TORCH_CHECK(kern != mp::MarlinW3Default, "w3: kernel not instantiated: tmb=", tmb, " m8=", m8,
                " k=", c.thread_k, " n=", c.thread_n, " thr=", c.threads, " stages=", ch_stages);
    {
      cudaFuncAttributes fa;
      cudaFuncGetAttributes(&fa, kern);
      int dyn_max = max_smem / ch_bps - (int)fa.sharedSizeBytes - (ch_bps > 1 ? 1024 : 0);
      if (smem > dyn_max) smem = dyn_max;
      TORCH_CHECK(mp::valid(c, tmb, m8, prob_k, prob_n, ch_stages, smem),
                  "w3: invalid config (smem) k=", c.thread_k, " n=", c.thread_n);
    }
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (ch_bps > 1) {
      int occ = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, c.threads, smem);
      TORCH_CHECK(occ >= ch_bps, "w3: invalid config (occupancy ", occ, " < ", ch_bps, ")");
    }
    kern<<<sms * ch_bps, c.threads, smem, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, nullptr, nullptr, s_ptr, nullptr, nullptr, m_split,
        prob_n, prob_k, lda, locks_ptr, false, false, true, smem, nullptr, 0, nullptr, nullptr);
    A_ptr += m_split * (lda / 8);
    C_ptr += m_split * (prob_n / 8);
    rest_m -= m_split;
  }
  return out;
}

std::vector<std::vector<int64_t>> w3_instantiated();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &w3_gemm, "marlin_plus W3 (3-bit planar) W3A16 gemm", py::arg("a"), py::arg("b_q3"),
        py::arg("b_s"), py::arg("out"), py::arg("locks"), py::arg("c_tmp"), py::arg("size_n"),
        py::arg("size_k"), py::arg("cfg"));
  m.def("instantiated", &w3_instantiated, "instantiated kernel configs");
}
