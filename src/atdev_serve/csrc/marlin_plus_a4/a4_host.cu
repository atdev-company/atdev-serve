// marlin_plus P3 host side: W4A4 kernel (mp_a4.cuh) + activation quant/pack.
// cfg = {thread_n, threads, stages, blocks_per_sm (= MINB), thread_m_blocks, MW, DEFER}.
// A_pk layout: [K/64][tmb][32 lanes][4 words] (s4 mma A fragments, int8-path nibble
// order), A_s: [K/128][16*tmb] fp32 (x input_global_scale). prob_m <= 16*tmb.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <map>
#include <tuple>

#include "kernel.h"
#include "mp_a4_decl.h"

namespace MARLIN_NAMESPACE_NAME {
__global__ void MarlinA4Default(MARLIN_KERNEL_PARAMS) {};
using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);
__global__ void MarlinA4DefaultX(MARLIN_KERNEL_PARAMS A4_XPARAMS) {};
using A4FuncPtr = void (*)(MARLIN_KERNEL_PARAMS A4_XPARAMS);

static A4FuncPtr get_a4_kernel(int tmb, int tnb, int threads, int stages, int MW,
                               int DEFER, int MINB) {
  auto kernel = MarlinA4DefaultX;
#include "a4_selector.h"
  return kernel;
}
using A4ReduceFn = void (*)(const int4*, int4*, int, int, const int*, int);
static A4ReduceFn get_a4_reduce(int tmb, int tnb, int threads, int MW) {
  A4ReduceFn r = nullptr;
#include "a4_reduce_selector.h"
  return r;
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

// one warp per (row, 128-group): bit-equal to a4_sim.fake_a4's quantisation
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
                                  std::vector<int64_t> cfg, const mp::A4Ne* nep,
                                  const mp::A4Silu* slp = nullptr) {
  const int prob_n = size_n, prob_k = size_k;
#if A4_NE
  mp::A4Ne ne{};
  if (nep != nullptr) ne = *nep;
#else
  TORCH_CHECK(nep == nullptr, "marlin_plus a4: C32-NE arguments need an A4_NE=1 build");
#endif
#if A4_SILU
  TORCH_CHECK(slp != nullptr, "marlin_plus a4: A4_SILU builds run gemm_a4_silu only");
  const mp::A4Silu sl = *slp;
#else
  TORCH_CHECK(slp == nullptr, "marlin_plus a4: C32h arguments need an A4_SILU=1 build");
#endif
  TORCH_CHECK(cfg.size() == 7 || cfg.size() == 8, "a4 cfg = {tn, threads, stages, bps, tmb, MW, DEFER[, grid]}");
  const int grid_ovr = cfg.size() == 8 ? (int)cfg[7] : 0;
  const int tn = cfg[0], threads = cfg[1], stages = cfg[2], bps = cfg[3];
  const int tmb = cfg[4], MW = cfg[5], DEFER = cfg[6];
  if (slp == nullptr) TORCH_CHECK(out.size(0) == prob_m && out.size(1) == prob_n && out.is_contiguous());
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
  TORCH_CHECK(kern != mp::MarlinA4DefaultX, "marlin_plus a4: kernel not instantiated tmb=", tmb,
              " tn=", tn, " thr=", threads, " st=", stages, " MW=", MW, " DEFER=", DEFER,
              " bps=", bps);
  int smem = mp::a4_smem(tmb, tn, stages);
#if A4_SILU && A4_SILU_DEFER == 2
  smem += 16 * tmb * 152;  // + [8][16 tmb] u64 PA prefetch
#elif A4_SILU && A4_SILU_DEFER
  smem += 16 * tmb * 88;  // C32h deferred quant: bf16 y tile (80 B rows) + group scale / reciprocal
#endif
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
  // parallel split-K reduction (A4_PAR_REDUCE builds): slots per column MS, passed as `lda`
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
#if !defined(A4_PAR_REDUCE) && !defined(A4_OVLCHAIN)
  // serial fp32 split-K chain: one 16*tmb x tn partial per lock slot (slot < max(grid, part2) + 2);
  // tmb 6/8 tiles are 1.5-2x the tmb=4 ones, so check the shared C_tmp instead of overrunning it
  TORCH_CHECK(c_tmp.numel() * (int64_t)c_tmp.element_size() >=
                  (int64_t)(std::max(grid, part2) + 2) * c_size,
              "marlin_plus a4: c_tmp too small for tmb=", tmb, " tn=", tn, " (need ",
              (int64_t)(std::max(grid, part2) + 2) * c_size, " B)");
#endif
#if defined(A4_PAR_REDUCE) && A4_PAR_REDUCE == 7
  // C32c consumer-sum: c_tmp = P [4][128][N] fp32 (slice partials), single m-chunk only
  TORCH_CHECK(prob_m <= 16 * tmb && prob_m <= 128 && par_ms <= 4,
              "marlin_plus a4 csum: M <= min(16 tmb, 128) and <= 4 slices per column");
  TORCH_CHECK(c_tmp.numel() * (int64_t)c_tmp.element_size() >= (int64_t)4 * 128 * prob_n * 4,
              "marlin_plus a4 csum: P too small");
#elif defined(A4_PAR_REDUCE)
  TORCH_CHECK(c_tmp.numel() * (int64_t)c_tmp.element_size() >= lock_bound * par_ms * c_size,
              "marlin_plus a4: c_tmp too small for the parallel reduction (need ",
              lock_bound * par_ms * c_size / (1 << 20), " MB)");
  TORCH_CHECK(locks.numel() >= lock_bound, "marlin_plus a4: locks too small");
#endif
#if defined(A4_PAR_REDUCE) && A4_PAR_REDUCE == 2
  // split columns of the striped part: (lock slot, column, slice_count), replicating the kernel's schedule
  std::vector<int> cols;
  {
    const int gmn = n_tiles;
    for (int col_par = 0; col_par < part2; col_par++) {
      int col_first = iters * ((k_tiles * col_par + iters - 1) / iters);
      int slice_count = 1;
      if (col_first <= k_tiles * (col_par + 1)) {
        int col_off = col_first - k_tiles * col_par;
        slice_count = (k_tiles - col_off + iters - 1) / iters;
        if (col_off > 0) slice_count++;
      }
      if (slice_count <= 1) continue;
      int slice_col = (col_par + gmn - part2) % n_tiles;
      // lock slot: part2 >= grid -> blockIdx of the CTA holding the column's head slice + 1;
      // else the column index (locks_off starts at (iters*b)/k_tiles - 1 and ++ per slice)
      int lock = (part2 >= grid) ? (col_first - 1) / iters + 1 : col_par;
      cols.push_back(lock); cols.push_back(slice_col); cols.push_back(slice_count);
    }
  }
  // cached per (shape, tile, grid): built on the first (eager) call so CUDA-graph capture never copies
  static std::map<std::tuple<int, int, int, int, int>, torch::Tensor> cols_cache;
  torch::Tensor cols_dev;
  const int ncols = (int)cols.size() / 3;
  if (ncols) {
    auto key = std::make_tuple(prob_n, prob_k, tn, grid, dev);
    auto it = cols_cache.find(key);
    if (it == cols_cache.end()) {
      auto cols_cpu = torch::from_blob(cols.data(), {(int64_t)cols.size()}, torch::kInt32).clone();
      it = cols_cache.emplace(key, cols_cpu.to(a_pk.device())).first;
    }
    cols_dev = it->second;
  }
#endif
  kern<<<grid, threads, smem, stream>>>(
      (const int4*)a_pk.data_ptr(), (const int4*)b_q.data_ptr(), slp ? nullptr : (int4*)out.data_ptr(),
      (int4*)c_tmp.data_ptr(), nullptr, a_s.data_ptr<float>(), (const int4*)b_s.data_ptr(),
      nullptr, nullptr, (int)prob_m, prob_n, prob_k, par_ms, locks.data_ptr<int>(), false,
      false, true, smem, nullptr, 0, nullptr, nullptr A4_XARGS);
#if defined(A4_PAR_REDUCE) && A4_PAR_REDUCE == 2
  if (ncols) {
    auto red = mp::get_a4_reduce(tmb, tn / 16, threads, MW);
    TORCH_CHECK(red != nullptr, "marlin_plus a4: reducer not instantiated");
    red<<<ncols, threads, 0, stream>>>((const int4*)c_tmp.data_ptr(), (int4*)out.data_ptr(), (int)prob_m,
                                       prob_n, cols_dev.data_ptr<int>(), par_ms);
  }
#endif
  return out;
}

torch::Tensor mp_gemm_a4(torch::Tensor a_pk, torch::Tensor a_s, torch::Tensor b_q,
                         torch::Tensor b_s, torch::Tensor out, torch::Tensor locks,
                         torch::Tensor c_tmp, int64_t prob_m, int64_t size_n, int64_t size_k,
                         std::vector<int64_t> cfg) {
  return a4_gemm_impl(a_pk, a_s, b_q, b_s, out, locks, c_tmp, prob_m, size_n, size_k, cfg, nullptr);
}

#if A4_NE
static int a4_tmb(const std::vector<int64_t>& cfg) { return (int)cfg[4]; }

#endif
#if A4_NE == 2
// consumer: out rows scaled by rs[row] (rs: fp32, >= 16 tmb entries)
torch::Tensor mp_gemm_a4_rs(torch::Tensor a_pk, torch::Tensor a_s, torch::Tensor b_q, torch::Tensor b_s,
                            torch::Tensor out, torch::Tensor locks, torch::Tensor c_tmp, int64_t prob_m,
                            int64_t size_n, int64_t size_k, std::vector<int64_t> cfg, torch::Tensor rs) {
  TORCH_CHECK(prob_m <= 16 * a4_tmb(cfg), "C32-NE consumer: M <= 16 tmb only");
  TORCH_CHECK(rs.scalar_type() == at::kFloat && rs.is_contiguous() && rs.numel() >= 16 * a4_tmb(cfg));
  mp::A4Ne ne{};
  ne.rs = rs.data_ptr<float>();
  return a4_gemm_impl(a_pk, a_s, b_q, b_s, out, locks, c_tmp, prob_m, size_n, size_k, cfg, &ne);
}
#endif
#if A4_NE == 1 || A4_NE == 3
// producer: out is not written; ro = bf16(bf16(A W) + r), next linear's s4 input of x (1 + w) into
// (apk, as) for a consumer with tmb_c, rstd[m_pad] (0 past M), optional xb = bf16(x (1 + w))
void mp_gemm_a4_ne(torch::Tensor a_pk, torch::Tensor a_s, torch::Tensor b_q, torch::Tensor b_s,
                   torch::Tensor locks, torch::Tensor c_tmp, int64_t prob_m, int64_t size_n, int64_t size_k,
                   std::vector<int64_t> cfg, torch::Tensor r, torch::Tensor ro, torch::Tensor w,
                   c10::optional<torch::Tensor> xb, torch::Tensor S, torch::Tensor P, torch::Tensor cnt,
                   torch::Tensor apk, torch::Tensor as_, torch::Tensor gs, torch::Tensor rstd, double eps,
                   int64_t tmb_c, torch::Tensor hbuf) {
  const int tmb = a4_tmb(cfg), tn = (int)cfg[0], n_tiles = (int)size_n / tn;
#if A4_NE == 3
  {
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, r.get_device());
    const int grid = cfg.size() == 8 && cfg[7] > 0 ? (int)cfg[7] : sms * (int)cfg[3];
    TORCH_CHECK(grid >= 16 * tmb_c, "C32-NE GB: grid < m_pad");
    TORCH_CHECK(cfg[1] == 128 && size_n == 5120, "C32-NE GB: 128 threads, N = 5120");
  }
#endif
  TORCH_CHECK(prob_m >= 1 && prob_m <= 16 * tmb, "C32-NE producer: 1 <= M <= 16 tmb");
  TORCH_CHECK(128 % tn == 0, "C32-NE producer: tn must divide 128");
  TORCH_CHECK(prob_m <= 16 * tmb_c && tmb_c >= 1);
  auto bf = [](const torch::Tensor& t) { return t.scalar_type() == at::kBFloat16 && t.stride(1) == 1 &&
                                                t.stride(0) % 8 == 0; };
  TORCH_CHECK(bf(r) && bf(ro) && r.size(1) == size_n && ro.size(1) == size_n);
  TORCH_CHECK(r.size(0) >= prob_m && ro.size(0) >= prob_m);
  TORCH_CHECK(w.is_contiguous() && w.scalar_type() == at::kBFloat16 && w.numel() == size_n);
  TORCH_CHECK(S.numel() >= (int64_t)n_tiles * 16 * tmb * tn && P.numel() >= (int64_t)16 * tmb * n_tiles);
  TORCH_CHECK(cnt.numel() >= n_tiles / (128 / tn) + 1 && cnt.scalar_type() == at::kInt);
  TORCH_CHECK(rstd.numel() >= 16 * tmb_c && rstd.scalar_type() == at::kFloat);
  TORCH_CHECK(apk.numel() * 4 >= (int64_t)(size_n / 64) * tmb_c * 32 * 4 * 4);
  TORCH_CHECK(as_.numel() >= (int64_t)(size_n / 128) * 16 * tmb_c);
  mp::A4Ne ne{};
  ne.S = S.data_ptr<float>();
  ne.P = P.data_ptr<float>();
  ne.cnt = cnt.data_ptr<int>();
  ne.r = (const __nv_bfloat16*)r.data_ptr();
  ne.ro = (__nv_bfloat16*)ro.data_ptr();
  ne.w = (const __nv_bfloat16*)w.data_ptr();
  ne.apk = (uint32_t*)apk.data_ptr();
  ne.as = as_.data_ptr<float>();
  ne.gs = gs.data_ptr<float>();
  ne.rstd = rstd.data_ptr<float>();
  ne.ldr = r.stride(0);
  ne.ldro = ro.stride(0);
  if (xb.has_value()) {
    TORCH_CHECK(bf(*xb) && xb->size(1) == size_n);
    ne.xb = (__nv_bfloat16*)xb->data_ptr();
    ne.ldxb = xb->stride(0);
  }
  ne.m_pad = 16 * tmb_c;
  ne.tmb_c = tmb_c;
  ne.eps = (float)eps;
  // hbuf: bf16 scratch >= M x N for the plain GEMM tile (read back by the end-of-kernel norm epilogue)
  TORCH_CHECK(hbuf.scalar_type() == at::kBFloat16 && hbuf.is_contiguous() && hbuf.numel() >= prob_m * size_n);
  auto out = hbuf.view(-1).narrow(0, 0, prob_m * size_n).view({prob_m, size_n});
  a4_gemm_impl(a_pk, a_s, b_q, b_s, out, locks, c_tmp, prob_m, size_n, size_k, cfg, &ne);
}
#endif

#if A4_SILU
// C32h: gate_up GEMM (n8-tile interleaved weights, size_n = 2 N_c) whose epilogue writes the down GEMM's s4 input
// y = s4(bf16(silu(g) u)) into (apk, as) for a down GEMM with the same tmb; no bf16 output
void mp_gemm_a4_silu(torch::Tensor a_pk, torch::Tensor a_s, torch::Tensor b_q, torch::Tensor b_s,
                     torch::Tensor locks, torch::Tensor c_tmp, int64_t prob_m, int64_t size_n, int64_t size_k,
                     std::vector<int64_t> cfg, torch::Tensor apk, torch::Tensor as_, torch::Tensor gs,
                     torch::Tensor PA, torch::Tensor cnt, c10::optional<torch::Tensor> dbg) {
  const int tmb = (int)cfg[4], tn = (int)cfg[0], n_tiles = (int)size_n / tn;
  TORCH_CHECK(tn == 64 && size_n % 256 == 0, "C32h: tn = 64, N % 256 == 0");
  TORCH_CHECK(prob_m >= 1 && prob_m <= 16 * tmb, "C32h: 1 <= M <= 16 tmb");
  const int64_t m_pad = 16 * tmb, kc = size_n / 2;
  TORCH_CHECK(apk.numel() * apk.element_size() >= (kc / 64) * tmb * 32 * 4 * 4, "C32h: apk too small");
  TORCH_CHECK(as_.numel() >= (kc / 128) * m_pad && as_.scalar_type() == at::kFloat, "C32h: as too small");
  TORCH_CHECK(PA.numel() >= (int64_t)n_tiles * 2 * m_pad * (A4_SILU_TAG ? 2 : 1) && PA.scalar_type() == at::kFloat,
              "C32h: PA too small");
  TORCH_CHECK(cnt.numel() >= 2 * (n_tiles / 4) && cnt.scalar_type() == at::kInt, "C32h: cnt too small");
#if A4_SILU_TAG
  {
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, a_pk.get_device());
    TORCH_CHECK(cnt.numel() >= sms * cfg[3] && PA.data_ptr<float>() != nullptr && (uintptr_t)PA.data_ptr() % 8 == 0,
                "C32h TAG: cnt >= grid epochs, PA 8-byte aligned");
  }
#endif
  TORCH_CHECK(gs.scalar_type() == at::kFloat);
  mp::A4Silu sl{};
  sl.apk = (uint32_t*)apk.data_ptr();
  sl.as = as_.data_ptr<float>();
  sl.gs = gs.data_ptr<float>();
  sl.PA = PA.data_ptr<float>();
  sl.cnt = cnt.data_ptr<int>();
  sl.m_pad = (int)m_pad;
  sl.dbg = dbg.has_value() ? (unsigned long long*)dbg->data_ptr() : nullptr;
  a4_gemm_impl(a_pk, a_s, b_q, b_s, a_pk, locks, c_tmp, prob_m, size_n, size_k, cfg, nullptr, &sl);
}
#endif

// ---- warp-specialised variant (mp_a4ws.cuh): cfg = {tn, threads_c, stages, bps, tmb, MW}
namespace MARLIN_NAMESPACE_NAME {
static MarlinFuncPtr get_a4ws_kernel(int tmb, int tnb, int threads_c, int stages, int MW, int MINB) {
  auto kernel = MarlinA4Default;
#include "a4ws_selector.h"
  return kernel;
}
static int a4ws_smem(int tmb, int tn, int stages) {
  int tnb = tn / 16;
  int stage = 2 * tmb * 32 + tmb * 4 + tn * 4 + 2 * tnb;
  int red = (2 * tnb + 1) * 16 * tmb;
  return 16 * (stages * stage + red);
}
}  // namespace MARLIN_NAMESPACE_NAME

torch::Tensor mp_gemm_a4ws(torch::Tensor a_pk, torch::Tensor a_s, torch::Tensor b_q, torch::Tensor b_s,
                           torch::Tensor out, torch::Tensor locks, torch::Tensor c_tmp, int64_t prob_m,
                           int64_t size_n, int64_t size_k, std::vector<int64_t> cfg) {
  TORCH_CHECK(cfg.size() == 6, "a4ws cfg = {tn, threads_c, stages, bps, tmb, MW}");
  const int tn = cfg[0], threads_c = cfg[1], stages = cfg[2], bps = cfg[3], tmb = cfg[4], MW = cfg[5];
  const int prob_n = size_n, prob_k = size_k;
  TORCH_CHECK(out.size(0) == prob_m && out.size(1) == prob_n && out.is_contiguous());
  TORCH_CHECK(prob_m <= 16 * tmb || prob_m % (16 * tmb) == 0);
  TORCH_CHECK(b_s.size(0) * 128 == prob_k);
  if (prob_m == 0) return out;
  if (prob_k % 128 || prob_n % tn) TORCH_CHECK(false, "marlin_plus a4ws: invalid config (tile)");
  const int dev = a_pk.get_device();
  const at::cuda::OptionalCUDAGuard guard(a_pk.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);
  int sms = 0, max_smem = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  auto kern = mp::get_a4ws_kernel(tmb, tn / 16, threads_c, stages, MW, bps);
  TORCH_CHECK(kern != mp::MarlinA4Default, "marlin_plus a4ws: kernel not instantiated");
  int smem = mp::a4ws_smem(tmb, tn, stages);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, kern);
  int dyn_max = max_smem / bps - (int)fa.sharedSizeBytes - (bps > 1 ? 1024 : 0);
  if (smem > dyn_max) TORCH_CHECK(false, "marlin_plus a4ws: invalid config (smem)");
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, threads_c + 32, smem);
  if (occ < bps) TORCH_CHECK(false, "marlin_plus a4ws: invalid config (occupancy ", occ, ")");
  kern<<<sms * bps, threads_c + 32, smem, stream>>>(
      (const int4*)a_pk.data_ptr(), (const int4*)b_q.data_ptr(), (int4*)out.data_ptr(),
      (int4*)c_tmp.data_ptr(), nullptr, a_s.data_ptr<float>(), (const int4*)b_s.data_ptr(),
      nullptr, nullptr, (int)prob_m, prob_n, prob_k, prob_k, locks.data_ptr<int>(), false,
      false, true, smem, nullptr, 0, nullptr, nullptr);
  return out;
}

std::vector<std::vector<int64_t>> mp_a4_instantiated();
std::vector<std::vector<int64_t>> mp_a4ws_instantiated();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_a4", &mp_gemm_a4, "marlin_plus W4A4 gemm", py::arg("a_pk"), py::arg("a_s"),
        py::arg("b_q"), py::arg("b_s"), py::arg("out"), py::arg("locks"), py::arg("c_tmp"),
        py::arg("prob_m"), py::arg("size_n"), py::arg("size_k"), py::arg("cfg"));
  m.def("quant_a4", &mp_quant_a4, "per-(token,128-group) s4 quant + fragment pack",
        py::arg("x"), py::arg("gs"), py::arg("a_pk"), py::arg("a_s"), py::arg("tmb"));
  m.def("instantiated", &mp_a4_instantiated, "instantiated W4A4 configs");
  m.def("gemm_a4ws", &mp_gemm_a4ws, "marlin_plus W4A4 gemm, warp-specialised producer", py::arg("a_pk"),
        py::arg("a_s"), py::arg("b_q"), py::arg("b_s"), py::arg("out"), py::arg("locks"), py::arg("c_tmp"),
        py::arg("prob_m"), py::arg("size_n"), py::arg("size_k"), py::arg("cfg"));
  m.def("instantiated_ws", &mp_a4ws_instantiated, "instantiated warp-specialised configs");
#if A4_NE == 2
  m.def("gemm_a4_rs", &mp_gemm_a4_rs, "W4A4 gemm, output rows scaled by rs (C32-NE consumer build)");
#endif
#if A4_NE == 1 || A4_NE == 3
  m.def("gemm_a4_ne", &mp_gemm_a4_ne, "W4A4 gemm + residual/RMSNorm/s4 epilogue (C32-NE producer build)");
#endif
#if A4_SILU
  m.def("gemm_a4_silu", &mp_gemm_a4_silu, "W4A4 gate_up gemm + SiLU.mul -> s4 epilogue (C32h build)",
        py::arg("a_pk"), py::arg("a_s"), py::arg("b_q"), py::arg("b_s"), py::arg("locks"), py::arg("c_tmp"),
        py::arg("prob_m"), py::arg("size_n"), py::arg("size_k"), py::arg("cfg"), py::arg("apk"), py::arg("as_"),
        py::arg("gs"), py::arg("PA"), py::arg("cnt"), py::arg("dbg") = py::none());
#endif
  m.attr("ne") = (int)A4_NE;
  m.attr("silu") = (int)A4_SILU;
}
