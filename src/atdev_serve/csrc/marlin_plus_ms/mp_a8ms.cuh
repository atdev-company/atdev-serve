// marlin_plus REFRAME A2: W4A8 (int8 x uint4b8, group 128) Marlin variant with
// the M dimension split across warps ("MS" = M-split).
//
// Stock Marlin (and marlin_plus v1) gives every warp ALL rows of the CTA tile
// (64 at M=64) x 32 columns: 64 int32 accumulators + 64 group partials per
// thread, 224-255 registers, 8 warps/SM, latency-bound at ~1.0 TB/s.
// Here the CTA's warps form an MW x (thread_n/32) grid: each warp owns
// thread_m_blocks/MW m-blocks x 32 columns. The B fragment is dequantised by
// each of the MW warps that share a column strip (cost: +dequant), but the
// per-thread accumulator state shrinks by MW, which allows
//   - 2 CTAs/SM or 512-thread CTAs (16 warps/SM instead of 8), and
//   - DEFER: double-buffered group partials, folded one k-step after the
//     group ends (no wait on the group's last mma), and
//   - a single weight pass for M=128 (thread_m_blocks = 8).
// Memory layouts (A int8 smem XOR swizzle, Marlin-repacked B, int16 group
// scales, a_scales, fp32 C_tmp striped reduction) are exactly stock Marlin's,
// so vLLM's tensors are reused unchanged. Integer accumulation and the fold
// are exact; only the fp32 cross-CTA reduction order can differ from stock.
// Derived from vLLM v0.31.0 csrc marlin_template.h (Apache-2.0).
#pragma once
#include "marlin_template.h"

// P3 timing ablations (numerically wrong; ATDEV_MS_DEFINES): MS_ABL bits 1 = no int4->int8 dequant (raw words
// fed to the mma), 2 = no mma, 4 = no group fold, 8 = no split-K reduction chain, 32 = fold = plain add, 64 = no global loads
#ifndef MS_ABL
  #define MS_ABL 0
#endif
// P3 (b) bit-exact issue-slot trims: MS_AADDR = precomputed shared addresses for the A ldmatrix (no per-step
// XOR-swizzle rematerialisation); MS_CINIT = U8 first mma takes C = -8*Sa as a separate operand (no MOVs)
#ifndef MS_AADDR
  #define MS_AADDR 0
#endif
#ifndef MS_CINIT
  #define MS_CINIT 0
#endif
// MS_FF (DEFER=0 only, bit-exact): "fold-first" -- a group's fold is not done after its last mma (stall on the
// mma latency + 48 IMADs at the stage barrier for every warp at once) but per accumulator right before the
// first mma of the next group overwrites it (same registers, no double buffer); pending fold flushed at slice end
// Default on, applied only to 3-m-block-per-warp kernels (tmb=6 MW=2: P3 k=2 M=96 bucket; -0.38 ms/step there,
// neutral at M=64, so the M<=64 kernels keep their code unchanged).
#ifndef MS_FF
  #define MS_FF 1
#endif
namespace MARLIN_NAMESPACE_NAME {

__device__ __forceinline__ void ms_mma_s8(const int* a, const int* b, int* c) {
  MP_ASM_PURE(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}
__device__ __forceinline__ void ms_mma_s8u8(const int* a, const int* b, int* c) {
  MP_ASM_PURE(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32.satfinite "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}
__device__ __forceinline__ void ms_mma_s8u8_ci(const int* a, const int* b, int c0, int c1, int* d) {
  MP_ASM_PURE(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32.satfinite "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%11,%11};\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c0), "r"(c1));
}
__device__ __forceinline__ void ms_mma_s8u8_zc(const int* a, const int* b, int* c) {
  const int z = 0;
  MP_ASM_PURE(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32.satfinite "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
      : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(z));
}
__device__ __forceinline__ void ms_mma_s8_zc(const int* a, const int* b, int* c) {
  const int z = 0;
  MP_ASM_PURE(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
      : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "r"(z));
}

#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800)

template <int threads, int thread_m_blocks, int thread_n_blocks,
          int thread_k_blocks, int stages, int MW, int DEFER, int MINB, int U8>
__global__ void __launch_bounds__(threads, MINB) MarlinA8MS(MARLIN_KERNEL_PARAMS) {
  // U8 (P3, bit-exact): B as unsigned nibbles u = q + 8 into mma s8.u8; each group's first mma starts
  // from C = -8 * sum_k a[row, k] over the group (Sa, int32 [rows][K/128], passed in zp_ptr), so the
  // group partial is exactly sum a*q as with the signed dequant. Sa layout: group-major
  // [K/128][MS_SA_LD] (row stride MS_SA_LD = 256 = FQ_MAX_M), so a stage's rows are one contiguous load.
  constexpr int group_blocks = 8;  // group 128
  constexpr int tmb = thread_m_blocks;
  constexpr int TMW = tmb / MW;              // m-blocks per warp
  constexpr int tb_n_warps = thread_n_blocks / 2;  // 32 columns per warp
  constexpr int KS = thread_k_blocks / 2;    // k32 sub-steps per stage
  constexpr int G = group_blocks / thread_k_blocks;  // stages per group (1|2)
  constexpr int GS = group_blocks / 2;       // k32 sub-steps per group (4)
  static_assert(tmb % MW == 0, "MW must divide thread_m_blocks");
  static_assert(threads == 32 * MW * tb_n_warps, "threads = 32*MW*(tn/32)");
  static_assert(KS >= 2, "thread_k >= 64");
  static_assert(group_blocks % thread_k_blocks == 0, "thread_k <= 128");
  static_assert(stages % (2 * G) == 0, "stages must hold an even number of groups");
  constexpr int m_block_size = 16 * tmb;

  if (A_src != nullptr) {
    mp_fused_quant_prologue<threads>(A_src, lda_src, prob_m, prob_k,
                                     const_cast<int4*>(A), lda,
                                     const_cast<float*>(a_scales_ptr),
                                     a_gscale_ptr, grid_bar,
                                     U8 ? reinterpret_cast<int*>(const_cast<int4*>(zp_ptr)) : nullptr);
  }

  const int4* A0 = A;
  int4* C0 = C;
  constexpr int MS_SA_LD = 256;
  const int* Sa0 = reinterpret_cast<const int*>(zp_ptr);
  const int* Sa = Sa0;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int nw = warp % tb_n_warps;  // column strip of this warp
  const int mh = warp / tb_n_warps;  // row group of this warp

  extern __shared__ int4 sh[];
  float* sh_a_s = reinterpret_cast<float*>(sh);
  int4* sh_new = sh + 4 * tmb;

  int parallel = 1;
  if (prob_m > m_block_size) {
    parallel = prob_m / m_block_size;
    prob_m = m_block_size;
  }
  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int global_mn_tiles = parallel * n_tiles;
  int part2_mn_tiles = global_mn_tiles;
  int part1_mn_iters = 0;
  bool in_part2 = false;
  if (global_mn_tiles > gridDim.x) {
    part2_mn_tiles = global_mn_tiles % gridDim.x;
    if (part2_mn_tiles * 3 <= gridDim.x) part2_mn_tiles += gridDim.x;
    part1_mn_iters = (global_mn_tiles - part2_mn_tiles) / gridDim.x;
  }
  int iters = div_ceil(k_tiles * part2_mn_tiles, gridDim.x);
  iters = G * div_ceil(iters, G);  // stripes start on a group boundary

  int slice_row = 0;
  int slice_col_par = blockIdx.x;
  int slice_col;
  int slice_iters = k_tiles;
  int slice_count = 1;
  int slice_idx = 0;
  int par_id = 0;
  int locks_off = 0;
  if (part2_mn_tiles >= gridDim.x) {
    locks_off = blockIdx.x;
  } else {
    locks_off = (iters * blockIdx.x) / k_tiles - 1;
  }

  auto load_a_scales = [&]() {
    __syncthreads();
  #pragma unroll
    for (int r = threadIdx.x; r < m_block_size; r += threads) {
      int a_s_gl_rd = par_id * m_block_size + r;
      cp_async1_ca_pred(&sh_a_s[r], &a_scales_ptr[a_s_gl_rd], r < prob_m);
    }
  };

  bool first_init = true;
  auto init_part2_slice = [&]() {
    slice_iters =
        iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= part2_mn_tiles) slice_iters = 0;
    if (slice_iters == 0) return;
    if (slice_row + slice_iters > k_tiles) slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * div_ceil(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = div_ceil(k_tiles - col_off, iters);
      if (col_off > 0) slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0) slice_idx--;
      }
    }
    if (part2_mn_tiles >= gridDim.x) {
      if (slice_count > 1 && slice_idx == slice_count - 1) {
        locks_off++;
      }
    } else {
      locks_off++;
    }
    if (slice_col == n_tiles) {
      A += m_block_size * lda / 16;
      if (U8) Sa += m_block_size;
      C += m_block_size * prob_n / 8;
      slice_col = 0;
      par_id++;
    }
    if (first_init || slice_col == 0) load_a_scales();
  };
  auto init_part1_slice = [&]() {
    if (part1_mn_iters) {
      part1_mn_iters--;
      par_id = slice_col_par / n_tiles;
      slice_col = slice_col_par % n_tiles;
      slice_iters = k_tiles;
      A = A0 + m_block_size / 16 * par_id * lda;
      if (U8) Sa = Sa0 + m_block_size * par_id;
      C = C0 + m_block_size / 8 * par_id * prob_n;
      load_a_scales();
    }
  };
  auto init_slice = [&]() {
    if (!in_part2 && !part1_mn_iters) {
      in_part2 = true;
      slice_col_par = (iters * blockIdx.x) / k_tiles;
      slice_row = (iters * blockIdx.x) % k_tiles;
      slice_col = (slice_col_par + global_mn_tiles - part2_mn_tiles) % n_tiles;
      par_id = (slice_col_par + global_mn_tiles - part2_mn_tiles) / n_tiles;
      A = A0 + m_block_size / 16 * par_id * lda;
      if (U8) Sa = Sa0 + m_block_size * par_id;
      C = C0 + m_block_size / 8 * par_id * prob_n;
    }
    if (!in_part2) {
      init_part1_slice();
    } else {
      init_part2_slice();
      first_init = false;
    }
  };
  init_slice();

  // ---- A (int8, 16 per int4) ----
  int a_gl_stride = lda / 16;
  constexpr int a_sh_stride = thread_k_blocks;
  constexpr int a_gl_rd_delta_o = thread_k_blocks;
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
  constexpr int a_sh_stage = a_sh_stride * m_block_size;
  constexpr int a_sh_wr_iters = div_ceil(a_sh_stage, a_sh_wr_delta);

  // ---- B (uint4b8, Marlin int8-path layout: one k32 row = prob_n int4) ----
  int b_gl_stride = prob_n;
  constexpr int b_sh_stride = thread_n_blocks * 16;  // int4 per k32 row
  int b_gl_rd_delta_o = b_gl_stride * KS;
  constexpr int b_sh_stage = b_sh_stride * KS;
  constexpr int b_ld_iters = div_ceil(b_sh_stage, threads);

  // ---- group scales (int16 in bf16 slots) ----
  int s_gl_stride = prob_n / 8;
  constexpr int s_sh_stride = 2 * thread_n_blocks;
  constexpr int s_sh_stage = s_sh_stride;

  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  int a_sh_rd = a_sh_stride * (lane % 16) + lane / 16;

  int b_gl_rd = b_sh_stride * slice_col + b_gl_rd_delta_o * slice_row;
  int b_gl_off[b_ld_iters];
  #pragma unroll
  for (int i = 0; i < b_ld_iters; i++) {
    int idx = threads * i + threadIdx.x;
    b_gl_off[i] = b_gl_stride * (idx / b_sh_stride) + (idx % b_sh_stride);
  }
  const int b_sh_rd = 32 * nw + lane;

  int sa_g = (thread_k_blocks * slice_row) / group_blocks;  // U8: group index of the next Sa stage load
  int s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                s_sh_stride * slice_col + threadIdx.x;
  const int s_sh_wr = threadIdx.x;
  const bool s_sh_wr_pred = threadIdx.x < s_sh_stage;
  const int s_sh_rd = 4 * nw + (lane % 4);

  bool a_sh_wr_pred[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ (row % 8);
  };
  int a_sh_wr_trans[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[KS][TMW];
  #pragma unroll
  for (int i = 0; i < KS; i++) {
  #pragma unroll
    for (int j = 0; j < TMW; j++)
      a_sh_rd_trans[i][j] =
          transform_a(2 * i + a_sh_rd_delta_i * (mh * TMW + j) + a_sh_rd);
  }

  constexpr int sh_red_size = (2 * thread_n_blocks + 1) * 16 * tmb;
  constexpr int sh_b_size = stages * b_sh_stage;
  constexpr int sh_br_size = sh_red_size > sh_b_size ? sh_red_size : sh_b_size;
  int4* sh_b = sh_new;
  int4* sh_red = sh_new;
  int4* sh_s = sh_new + sh_br_size;
  int4* sh_a = sh_s + stages * s_sh_stage;
  int* sh_sa = reinterpret_cast<int*>(sh_a + stages * a_sh_stage);  // U8: [stages][m_block_size]
  uint32_t a_rd_addr[KS][TMW];  // MS_AADDR: shared address of the stage-0 ldmatrix row per (k32, m-block)
  if constexpr (MS_AADDR) {
  #pragma unroll
    for (int i = 0; i < KS; i++)
  #pragma unroll
      for (int j = 0; j < TMW; j++)
        a_rd_addr[i][j] = static_cast<uint32_t>(__cvta_generic_to_shared(&sh_a[a_sh_rd_trans[i][j]]));
  }

  using FragS = typename MarlinScalarType<vllm::kBFloat16.id()>::FragS;
  int frag_a[2][TMW][4];
  int frag_bq[2][4];
  FragS frag_s[2][4];
  int frag_sa[2][TMW][2];  // U8: Sa of this thread's rows (lane/4, +8) per m-block
  int acc[TMW][2][2][4];                  // int32 (then fp32) accumulators
  int part[DEFER ? 2 : 1][TMW][2][2][4];  // int32 group partials
  int4 sv_saved = make_int4(0, 0, 0, 0);  // scales of the pending group
  bool pending = false;

  auto zero_accums = [&]() {
  #pragma unroll
    for (int i = 0; i < TMW * 16; i++) reinterpret_cast<int*>(acc)[i] = 0;
    if constexpr (MS_FF && !DEFER && TMW >= 3 && !(MS_ABL & 36)) {
  #pragma unroll
      for (int i = 0; i < TMW * 16; i++) reinterpret_cast<int*>(part[0])[i] = 0;
    }
  };

  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    if (pred && !(MS_ABL & 64)) {  // MS_ABL & 64: no global loads (compute-only timing)
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        cp_async4_pred(
            &sh_a_stage[a_sh_wr_trans[i]],
            &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
            a_sh_wr_pred[i]);
      }
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < b_ld_iters; i++) {
        if (b_sh_stage % threads == 0 || threads * i + threadIdx.x < b_sh_stage)
          cp_async4(&sh_b_stage[threads * i + threadIdx.x], &B[b_gl_rd + b_gl_off[i]]);
      }
      b_gl_rd += b_gl_rd_delta_o;
      if (pipe % G == 0) {
        int4* sh_s_stage = sh_s + s_sh_stage * pipe;
        if (s_sh_wr_pred) cp_async4(&sh_s_stage[s_sh_wr], &scales_ptr[s_gl_rd]);
        s_gl_rd += s_gl_stride;
        if constexpr (U8) {  // m_block_size contiguous ints of group sa_g (rows >= prob_m: unused)
          if (threadIdx.x < m_block_size / 4)
            cp_async4(reinterpret_cast<int4*>(&sh_sa[pipe * m_block_size]) + threadIdx.x,
                      reinterpret_cast<const int4*>(&Sa[sa_g * MS_SA_LD]) + threadIdx.x);
          sa_g++;
        }
      }
    }
    cp_async_fence();
  };

  auto wait_for_stage = [&]() {
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  auto fetch_to_registers = [&](int k, int pipe) {
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
  #pragma unroll
    for (int i = 0; i < TMW; i++) {
      uint32_t smem = MS_AADDR ? a_rd_addr[k % KS][i] + static_cast<uint32_t>(pipe * a_sh_stage * 16)
                               : static_cast<uint32_t>(__cvta_generic_to_shared(&sh_a_stage[a_sh_rd_trans[k % KS][i]]));
      asm volatile(
          "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
          : "=r"(frag_a[k % 2][i][0]), "=r"(frag_a[k % 2][i][1]),
            "=r"(frag_a[k % 2][i][2]), "=r"(frag_a[k % 2][i][3])
          : "r"(smem));
    }
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;
    *reinterpret_cast<int4*>(frag_bq[k % 2]) =
        sh_b_stage[b_sh_stride * (k % KS) + b_sh_rd];
  };

  auto fetch_scales_to_registers = [&](int k, int full_pipe) {
    int pipe = full_pipe % stages;
    if (pipe % G == 0) {
      if (k % KS == 0) {
        int4* sh_s_stage = sh_s + s_sh_stage * (G * (pipe / G));
        reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
        if constexpr (U8) {
          const int* sa_st = sh_sa + m_block_size * (G * (pipe / G));
  #pragma unroll
          for (int i = 0; i < TMW; i++) {
            frag_sa[k % 2][i][0] = sa_st[(mh * TMW + i) * 16 + lane / 4];
            frag_sa[k % 2][i][1] = sa_st[(mh * TMW + i) * 16 + 8 + lane / 4];
          }
        }
      } else {
        reinterpret_cast<int4*>(&frag_s[1])[0] =
            reinterpret_cast<int4*>(&frag_s[0])[0];
        if constexpr (U8) {
  #pragma unroll
          for (int i = 0; i < TMW; i++) { frag_sa[1][i][0] = frag_sa[0][i][0]; frag_sa[1][i][1] = frag_sa[0][i][1]; }
        }
      }
    }
  };

  // acc += part[b] * group scale (int32, exact)
  auto fold = [&](int (&p)[TMW][2][2][4], int4 svr) {
    if constexpr (MS_ABL & 4) { acc[0][0][0][0] += p[0][0][0][0]; return; }
    if constexpr (MS_ABL & 32) {  // fold without the scale multiply (keeps every mma live)
  #pragma unroll
      for (int i = 0; i < TMW * 16; i++) reinterpret_cast<int*>(acc)[i] += reinterpret_cast<int*>(p)[i];
      return;
    }
    const uint16_t* s16 = reinterpret_cast<const uint16_t*>(&svr);
  #pragma unroll
    for (int j = 0; j < 2; j++) {
  #pragma unroll
      for (int h = 0; h < 2; h++) {
        // frag_s[.][j*2+h] holds the bf16x2 (= 2 x int16) scales of this
        // thread's two columns in n8-tile (j, h)
        const int s0 = (int)s16[(j * 2 + h) * 2 + 0];
        const int s1 = (int)s16[(j * 2 + h) * 2 + 1];
  #pragma unroll
        for (int i = 0; i < TMW; i++) {
  #pragma unroll
          for (int g = 0; g < 4; g++)
            acc[i][j][h][g] += p[i][j][h][g] * (g % 2 ? s1 : s0);
        }
      }
    }
  };

  auto fold1 = [&](int i, int j, int h, int4 svr) {
    const uint16_t* s16 = reinterpret_cast<const uint16_t*>(&svr);
    const int s0 = (int)s16[(j * 2 + h) * 2 + 0];
    const int s1 = (int)s16[(j * 2 + h) * 2 + 1];
  #pragma unroll
    for (int g = 0; g < 4; g++) acc[i][j][h][g] += part[0][i][j][h][g] * (g % 2 ? s1 : s0);
  };
  constexpr bool FF = MS_FF && !DEFER && TMW >= 3 && !(MS_ABL & 36);
  auto mma_step = [&](int (&P)[TMW][2][2][4], int k2, bool first) {
  #pragma unroll
    for (int j = 0; j < 2; j++) {
      int fb[2][2];
      if constexpr (MS_ABL & 1) {
        fb[0][0] = frag_bq[k2][j * 2]; fb[0][1] = frag_bq[k2][j * 2] >> 4;
        fb[1][0] = frag_bq[k2][j * 2 + 1]; fb[1][1] = frag_bq[k2][j * 2 + 1] >> 4;
      } else if constexpr ((MS_ABL & 16) || U8) {  // u8 nibble expansion (u = q + 8), mma s8.u8 below
        fb[0][0] = frag_bq[k2][j * 2] & 0x0F0F0F0F; fb[0][1] = (frag_bq[k2][j * 2] >> 4) & 0x0F0F0F0F;
        fb[1][0] = frag_bq[k2][j * 2 + 1] & 0x0F0F0F0F; fb[1][1] = (frag_bq[k2][j * 2 + 1] >> 4) & 0x0F0F0F0F;
      } else {
        dequant<int32_t, vllm::kU4B8.id(), true>(frag_bq[k2][j * 2], fb[0]);
        dequant<int32_t, vllm::kU4B8.id(), true>(frag_bq[k2][j * 2 + 1], fb[1]);
      }
      if constexpr (MS_ABL & 2) {
  #pragma unroll
        for (int i = 0; i < TMW; i++) { P[i][j][0][0] ^= fb[0][0] ^ frag_a[k2][i][0]; P[i][j][1][0] ^= fb[1][1]; }
        continue;
      }
  #pragma unroll
      for (int i = 0; i < TMW; i++) {
        if constexpr (FF) {
          if (first) { fold1(i, j, 0, sv_saved); fold1(i, j, 1, sv_saved); }  // slice start: part = 0 -> no-op
        }
        if constexpr (U8) {
          if (MS_CINIT && first) {
            const int c0 = -8 * frag_sa[k2][i][0], c1 = -8 * frag_sa[k2][i][1];
            ms_mma_s8u8_ci(frag_a[k2][i], fb[0], c0, c1, P[i][j][0]);
            ms_mma_s8u8_ci(frag_a[k2][i], fb[1], c0, c1, P[i][j][1]);
            continue;
          }
          if (first) {  // C = -8 * Sa(row): exact correction of the +8 offset of the unsigned nibbles
            const int c0 = -8 * frag_sa[k2][i][0], c1 = -8 * frag_sa[k2][i][1];
  #pragma unroll
            for (int h = 0; h < 2; h++) {
              P[i][j][h][0] = c0; P[i][j][h][1] = c0; P[i][j][h][2] = c1; P[i][j][h][3] = c1;
            }
          }
          ms_mma_s8u8(frag_a[k2][i], fb[0], P[i][j][0]);
          ms_mma_s8u8(frag_a[k2][i], fb[1], P[i][j][1]);
        } else if constexpr (MS_ABL & 16) {
          if (first) {
            ms_mma_s8u8_zc(frag_a[k2][i], fb[0], P[i][j][0]);
            ms_mma_s8u8_zc(frag_a[k2][i], fb[1], P[i][j][1]);
          } else {
            ms_mma_s8u8(frag_a[k2][i], fb[0], P[i][j][0]);
            ms_mma_s8u8(frag_a[k2][i], fb[1], P[i][j][1]);
          }
        } else if (first) {
          ms_mma_s8_zc(frag_a[k2][i], fb[0], P[i][j][0]);
          ms_mma_s8_zc(frag_a[k2][i], fb[1], P[i][j][1]);
        } else {
          ms_mma_s8(frag_a[k2][i], fb[0], P[i][j][0]);
          ms_mma_s8(frag_a[k2][i], fb[1], P[i][j][1]);
        }
      }
    }
  };
  auto matmul_a8 = [&](int k, int tile) {
    const int k2 = k % 2;
    const int q = (tile % G) * KS + k;  // k32 sub-step within the group
    const bool first = q == 0;
    const bool last = q == GS - 1;
    if constexpr (DEFER) {
      const int gp = (tile / G) % 2;  // compile-time after unrolling
      if (gp == 0) mma_step(part[0], k2, first);
      else mma_step(part[DEFER ? 1 : 0], k2, first);
      // fold the previous group one sub-step into the next one
      if (q == 1 && pending) {
        if (gp == 0) fold(part[DEFER ? 1 : 0], sv_saved);
        else fold(part[0], sv_saved);
        pending = false;
      }
      if (last) {
        sv_saved = reinterpret_cast<int4*>(&frag_s[k2])[0];
        pending = true;
      }
    } else if constexpr (FF) {
      mma_step(part[0], k2, first);  // folds the previous group into acc before overwriting its partials
      if (last) sv_saved = reinterpret_cast<int4*>(&frag_s[k2])[0];
    } else {
      mma_step(part[0], k2, first);
      if (last) fold(part[0], reinterpret_cast<int4*>(&frag_s[k2])[0]);
    }
  };
  // slice end: fold a still-pending group (DEFER); its buffer parity is the
  // parity of the last group of the slice, `last_gp`
  int last_gp = 0;
  auto flush_fold = [&]() {
    if constexpr (FF) fold(part[0], sv_saved);  // slices end on a group boundary: always one group pending
    if constexpr (DEFER) {
      if (pending) {
        if (last_gp == 0) fold(part[0], sv_saved);
        else fold(part[DEFER ? 1 : 0], sv_saved);
        pending = false;
      }
    }
  };

  auto global_reduce_fp32 = [&](bool first = false, bool last = false) {
    constexpr int tb_m = tmb * 16;
    constexpr int tb_n = thread_n_blocks * 16;
    constexpr int c_size = tb_m * tb_n * sizeof(float) / 16;
    constexpr int th_size = TMW * 16 * sizeof(float) / 16;
    int c_cur_offset = locks_off * c_size;
    if (!first) {
      float* fc = reinterpret_cast<float*>(&acc);
  #pragma unroll
      for (int k = 0; k < th_size; k++) {
        sh_red[threadIdx.x] = C_tmp[c_cur_offset + threads * k + threadIdx.x];
        float* sc = reinterpret_cast<float*>(&sh_red[threadIdx.x]);
  #pragma unroll
        for (int f = 0; f < 4; f++) fc[k * 4 + f] += sc[f];
      }
    }
    if (!last) {
      int4* fc = reinterpret_cast<int4*>(&acc);
  #pragma unroll
      for (int k = 0; k < th_size; k++)
        C_tmp[c_cur_offset + threads * k + threadIdx.x] = fc[k];
    }
  };

  auto write_result = [&]() {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta = c_sh_stride * (threads / (2 * thread_n_blocks));
    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr = (4 * c_sh_stride) * (lane / 4) + lane % 4 + 16 * nw +
                  (mh * TMW) * 16 * (4 * c_sh_stride);
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));
    int c_gl_wr_end = c_gl_stride * prob_m;
    using c2 = nv_bfloat162;
    auto write = [&](int idx, float c0, float c1) {
      reinterpret_cast<c2*>(sh_red)[idx] =
          __halves2bfloat162(__float2bfloat16(c0), __float2bfloat16(c1));
    };
    __syncthreads();
    const float* fc = reinterpret_cast<const float*>(&acc);
  #pragma unroll
    for (int i = 0; i < TMW; i++) {
  #pragma unroll
      for (int j = 0; j < 2; j++) {
        const float* c0 = fc + ((i * 2 + j) * 2 + 0) * 4;
        const float* c1 = fc + ((i * 2 + j) * 2 + 1) * 4;
        int wr = c_sh_wr + 8 * j;
        write(wr + (4 * c_sh_stride) * 0 + 0, c0[0], c0[1]);
        write(wr + (4 * c_sh_stride) * 8 + 0, c0[2], c0[3]);
        write(wr + (4 * c_sh_stride) * 0 + 4, c1[0], c1[1]);
        write(wr + (4 * c_sh_stride) * 8 + 4, c1[2], c1[3]);
      }
      c_sh_wr += 16 * (4 * c_sh_stride);
    }
    __syncthreads();
  #pragma unroll
    for (int i = 0; i < div_ceil(16 * tmb, threads / (2 * thread_n_blocks)); i++) {
      if (c_gl_wr < c_gl_wr_end) {
        C[c_gl_wr] = sh_red[c_sh_rd];
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
    __syncthreads();
  };

  auto start_pipes = [&]() {
  #pragma unroll
    for (int i = 0; i < stages - 1; i++) fetch_to_shared(i, i, i < slice_iters);
    zero_accums();
    wait_for_stage();
    fetch_to_registers(0, 0);
    fetch_scales_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
  };
  if (slice_iters) start_pipes();

  while (slice_iters) {
  #pragma unroll
    for (int pipe = 0; pipe < stages;) {
  #pragma unroll
      for (int k = 0; k < KS; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        fetch_scales_to_registers(k + 1, pipe);
        if (k == KS - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);
          pipe++;
          wait_for_stage();
        }
        const int tile = pipe - (k >= KS - 2 ? 1 : 0);
        matmul_a8(k, tile);
        if constexpr (DEFER) last_gp = (tile / G) % 2;
      }
      slice_iters--;
      if (slice_iters == 0) break;
    }
    a_gl_rd += a_gl_rd_delta_o * stages;

    if (slice_iters == 0) {
      flush_fold();
      {  // int32 -> fp32 * per-row activation scale
        float* fc = reinterpret_cast<float*>(&acc);
  #pragma unroll
        for (int i = 0; i < TMW; i++) {
          const float s_lo = sh_a_s[(mh * TMW + i) * 16 + lane / 4];
          const float s_hi = sh_a_s[(mh * TMW + i) * 16 + 8 + lane / 4];
  #pragma unroll
          for (int jh = 0; jh < 4; jh++) {
  #pragma unroll
            for (int g = 0; g < 4; g++) {
              int v = reinterpret_cast<int*>(acc)[(i * 4 + jh) * 4 + g];
              fc[(i * 4 + jh) * 4 + g] = __int2float_rn(v) * (g / 2 ? s_hi : s_lo);
            }
          }
        }
      }
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      if (slice_count > 1 && !(MS_ABL & 8)) {
        barrier_acquire(&locks[locks_off], slice_idx);
        global_reduce_fp32(slice_idx == 0, last);
        barrier_release(&locks[locks_off], last);
      }
      if (last) write_result();
      slice_row = 0;
      if (!in_part2) {
        slice_col_par += gridDim.x;
      } else {
        slice_col_par++;
        slice_col++;
      }
      init_slice();
      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                  (threadIdx.x % a_gl_rd_delta_o);
        a_gl_rd += a_gl_rd_delta_o * slice_row;
        b_gl_rd = b_sh_stride * slice_col + b_gl_rd_delta_o * slice_row;
        s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                  s_sh_stride * slice_col + threadIdx.x;
        sa_g = (thread_k_blocks * slice_row) / group_blocks;
        start_pipes();
      }
    }
  }
}

#else
template <int threads, int thread_m_blocks, int thread_n_blocks,
          int thread_k_blocks, int stages, int MW, int DEFER, int MINB, int U8>
__global__ void MarlinA8MS(MARLIN_KERNEL_PARAMS) {}
#endif

}  // namespace MARLIN_NAMESPACE_NAME
