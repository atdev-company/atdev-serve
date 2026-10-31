// marlin_plus_a4: W4A4 GEMM (s4 activations x uint4b8 weights, group 128 on both sides) inside the
// marlin_plus_ms (M-split W4A8) pipeline.
//
// Keeps the M-split W4A8 kernel (mp_a8ms.cuh) unchanged in everything that
// streams the weights: persistent striped split-K tile schedule, cp.async
// multi-stage ring for B / group scales, fp32 C_tmp split-K reduction,
// bf16 epilogue through smem.  What changes:
//   * mma: m16n8k64 s4.s4 (2x the int8 rate on sm80).  The int8-path Marlin
//     weight word (8 nibbles) is used directly as a B fragment: uint4b8 -> s4
//     is one `xor 0x88888888` (u - 8 mod 16 flips the top bit), no lop3
//     dequant.  The nibble order of the int8 path (nibble i <-> k32-local
//     (i%2)*16 + (lane%4)*4 + i/2) is a k permutation; the activation packer
//     (a4_quant_kernel) applies the same permutation so the dot product is
//     unchanged.
//   * A: s4, pre-packed in mma fragment order [K/64][tmb][32 lanes][4 words]
//     by the quant kernel; one int4 smem load per (k64, m-block) per thread,
//     no ldmatrix.  Half the A bytes of the int8 path.
//   * scales: per-(row, 128-group) activation scales (x input_global_scale)
//     and the int8 path's int16 group weight scales.  The group fold is in
//     fp32: the group's mma chain starts from C = 0x4B400000 so the int32
//     accumulator *is* the float 12582912 + P (|P| <= 8*8*128 < 2^22), and
//     acc += (float(acc_i) - 12582912) * s16 * a_s, folded into 2 FP32 ops per
//     element (fma with -magic*s).
// thread_k = 128 = one group per pipeline stage.
//
// Split-K reduction (A4_PAR_REDUCE, set by the build):
//   6  gather: non-last slices store a bf16 partial to their own C_tmp slot and release; the last
//      slice waits for all of them and adds the slots in slice order to its registers (one hop
//      instead of the serial lock chain; rounding-level vs the fp32 chain, deterministic);
//   7  consumer-sum: every slice stores its fp32 partial to P[slice][row][col] and exits; the next
//      norm kernel sums them (c32c_csum.cu).
// Derived from vLLM v0.31.0 csrc marlin_template.h (Apache-2.0).
#pragma once
#include "marlin_template.h"

namespace MARLIN_NAMESPACE_NAME {

__device__ __forceinline__ void a4_mma(const uint32_t* a, uint32_t b0,
                                       uint32_t b1, int* c) {
  MP_ASM_PURE(
      "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void a4_mma_magic(const uint32_t* a, uint32_t b0,
                                             uint32_t b1, int* c) {
  const int m = 0x4B400000;
  MP_ASM_PURE(
      "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
      : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(m));
}

__device__ __forceinline__ void a4_mma_q(const uint32_t* a, uint32_t b0, uint32_t b1, int* c,
                                         const int* q) {
  MP_ASM_PURE(
      "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(q[0]), "r"(q[1]),
        "r"(q[2]), "r"(q[3]));
}

#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800)

template <int threads, int thread_m_blocks, int thread_n_blocks, int stages,
          int MW, int DEFER, int MINB>
__global__ void __launch_bounds__(threads, MINB) MarlinA4(MARLIN_KERNEL_PARAMS) {
  constexpr int thread_k_blocks = 8;        // tk = 128 = one group
  constexpr int tmb = thread_m_blocks;
  constexpr int TMW = tmb / MW;             // m-blocks per warp
  constexpr int tb_n_warps = thread_n_blocks / 2;  // 32 columns per warp
  constexpr int KS = 2;                     // k64 steps per stage
  static_assert(tmb % MW == 0, "MW must divide thread_m_blocks");
  static_assert(threads == 32 * MW * tb_n_warps, "threads = 32*MW*(tn/32)");
  static_assert(stages % 2 == 0, "stages must be even");
  constexpr int m_block_size = 16 * tmb;
  constexpr float kMagic = 12582912.0f;

  const int4* A0 = A;
  int4* C0 = C;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int nw = warp % tb_n_warps;  // column strip of this warp
  const int mh = warp / tb_n_warps;  // row group of this warp

  extern __shared__ int4 sh[];

  int parallel = 1;
  if (prob_m > m_block_size) {
    parallel = prob_m / m_block_size;
    prob_m = m_block_size;
  }
  const int k_tiles = prob_k / 16 / thread_k_blocks;
  const int n_tiles = prob_n / 16 / thread_n_blocks;
  const int global_mn_tiles = parallel * n_tiles;
  int part2_mn_tiles = global_mn_tiles;
  int part1_mn_iters = 0;
  bool in_part2 = false;
  if (global_mn_tiles > gridDim.x) {
    part2_mn_tiles = global_mn_tiles % gridDim.x;
    if (part2_mn_tiles * 3 <= gridDim.x) part2_mn_tiles += gridDim.x;
    part1_mn_iters = (global_mn_tiles - part2_mn_tiles) / gridDim.x;
  }
  const int iters = div_ceil(k_tiles * part2_mn_tiles, gridDim.x);

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

  // ---- A (s4 fragments): per k-tile 2 k64 blocks x tmb m-blocks x 32 int4 ----
  constexpr int a_sh_stage = KS * tmb * 32;  // int4
  constexpr int a_sh_wr_iters = div_ceil(a_sh_stage, threads);
  const int a_chunk = k_tiles * a_sh_stage;  // int4 per 64-row chunk
  // ---- activation scales: per k-tile m_block_size floats ----
  constexpr int as_sh_stage = m_block_size / 4;  // int4
  const int as_chunk = k_tiles * as_sh_stage;
  const int4* AS = reinterpret_cast<const int4*>(a_scales_ptr);
  // ---- B (uint4b8, Marlin int8-path layout: one k32 row = prob_n int4) ----
  const int b_gl_stride = prob_n;
  constexpr int b_sh_stride = thread_n_blocks * 16;  // int4 per k32 row
  const int b_gl_rd_delta_o = b_gl_stride * 4;       // 4 k32 rows per stage
  constexpr int b_sh_stage = b_sh_stride * 4;
  constexpr int b_ld_iters = div_ceil(b_sh_stage, threads);
  // ---- group scales (int16 in bf16 slots) ----
  const int s_gl_stride = prob_n / 8;
  constexpr int s_sh_stride = 2 * thread_n_blocks;
  constexpr int s_sh_stage = s_sh_stride;

  int a_gl_rd, as_gl_rd, b_gl_rd, s_gl_rd;  // set after init_slice()
  int b_gl_off[b_ld_iters];
  #pragma unroll
  for (int i = 0; i < b_ld_iters; i++) {
    int idx = threads * i + threadIdx.x;
    b_gl_off[i] = b_gl_stride * (idx / b_sh_stride) + (idx % b_sh_stride);
  }
  const int b_sh_rd = 32 * nw + lane;
  const int s_sh_wr = threadIdx.x;
  const bool s_sh_wr_pred = threadIdx.x < s_sh_stage;
  const int s_sh_rd = 4 * nw + (lane % 4);

  constexpr int sh_red_size = (2 * thread_n_blocks + 1) * 16 * tmb;
  constexpr int sh_b_size = stages * b_sh_stage;
  constexpr int sh_br_size = sh_red_size > sh_b_size ? sh_red_size : sh_b_size;
  int4* sh_b = sh;
  int4* sh_red = sh;
  int4* sh_s = sh + sh_br_size;
  int4* sh_as = sh_s + stages * s_sh_stage;
  int4* sh_a = sh_as + stages * as_sh_stage;

  bool first_init = true;
  auto init_part2_slice = [&]() {
    slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
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
      if (slice_count > 1 && slice_idx == slice_count - 1) locks_off++;
    } else {
      locks_off++;
    }
    if (slice_col == n_tiles) {
      A += a_chunk;
      AS += as_chunk;
      C += m_block_size * prob_n / 8;
      slice_col = 0;
      par_id++;
    }
    first_init = false;
  };
  auto init_part1_slice = [&]() {
    if (part1_mn_iters) {
      part1_mn_iters--;
      par_id = slice_col_par / n_tiles;
      slice_col = slice_col_par % n_tiles;
      slice_iters = k_tiles;
      A = A0 + a_chunk * par_id;
      AS = reinterpret_cast<const int4*>(a_scales_ptr) + as_chunk * par_id;
      C = C0 + m_block_size / 8 * par_id * prob_n;
    }
  };
  auto init_slice = [&]() {
    if (!in_part2 && !part1_mn_iters) {
      in_part2 = true;
      slice_col_par = (iters * blockIdx.x) / k_tiles;
      slice_row = (iters * blockIdx.x) % k_tiles;
      slice_col = (slice_col_par + global_mn_tiles - part2_mn_tiles) % n_tiles;
      par_id = (slice_col_par + global_mn_tiles - part2_mn_tiles) / n_tiles;
      A = A0 + a_chunk * par_id;
      AS = reinterpret_cast<const int4*>(a_scales_ptr) + as_chunk * par_id;
      C = C0 + m_block_size / 8 * par_id * prob_n;
    }
    if (!in_part2) init_part1_slice();
    else init_part2_slice();
  };
  init_slice();
  (void)first_init;
  a_gl_rd = a_sh_stage * slice_row;
  as_gl_rd = as_sh_stage * slice_row;
  b_gl_rd = b_sh_stride * slice_col + b_gl_rd_delta_o * slice_row;
  s_gl_rd = s_gl_stride * slice_row + s_sh_stride * slice_col + threadIdx.x;
  // 0: source pointers (a_p / as_p include threadIdx.x; b_p[i] include b_gl_off[i])
  const int4* a_p = nullptr;
  const int4* as_p = nullptr;
  const int4* b_p[b_ld_iters];
  const int4* s_p = nullptr;
  auto set_ptrs = [&]() {
    
  };
  set_ptrs();

  uint32_t frag_a[2][TMW][4];
  uint32_t frag_b[2][2][4];   // [k64 parity][k32 row][n8 tile]
  int4 frag_s[2];             // 8 x int16 group scales of this thread's 8 columns
  float frag_as[2][TMW][2];   // activation scales of this thread's rows (lane/4, +8)
  float acc[TMW][2][2][4];
  int part[DEFER ? 2 : 1][TMW][2][2][4];
  int4 sv_saved;
  float as_saved[TMW][2];
  bool pending = false;
  int mq[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) mq[i] = 0x4B400000 + ((prob_k >> 31) & (i + 1));
  (void)mq;

  auto zero_accums = [&]() {
  #pragma unroll
    for (int i = 0; i < TMW * 16; i++) reinterpret_cast<float*>(acc)[i] = 0.f;
  };

  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
      {
  #pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        int idx = threads * i + threadIdx.x;
        if (a_sh_stage % threads == 0 || idx < a_sh_stage)
          cp_async4(&sh_a_stage[idx], &A[a_gl_rd + a_sh_stage * a_off + idx]);
      }
      int4* sh_as_stage = sh_as + as_sh_stage * pipe;
      if (threadIdx.x < as_sh_stage)
        cp_async4(&sh_as_stage[threadIdx.x],
                  &AS[as_gl_rd + as_sh_stage * a_off + threadIdx.x]);
      }
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < b_ld_iters; i++) {
        if (4 == 4 ? (b_sh_stage % threads == 0 || threads * i + threadIdx.x < b_sh_stage)
                              : (threads * i + threadIdx.x < b_sh_stage * 4 / 4))
          cp_async4(&sh_b_stage[threads * i + threadIdx.x], &B[b_gl_rd + b_gl_off[i]]);
      }
      b_gl_rd += b_gl_rd_delta_o;
      int4* sh_s_stage = sh_s + s_sh_stage * pipe;
      if (s_sh_wr_pred) cp_async4(&sh_s_stage[s_sh_wr], &scales_ptr[s_gl_rd]);
      s_gl_rd += s_gl_stride;
    }
    cp_async_fence();
  };

  auto wait_for_stage = [&]() {
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  auto fetch_to_registers = [&](int k, int pipe) {
    const int kk = k % KS;
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
  #pragma unroll
    for (int i = 0; i < TMW; i++)
      *reinterpret_cast<int4*>(frag_a[k % 2][i]) =
          sh_a_stage[(kk * tmb + mh * TMW + i) * 32 + lane];
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;
    *reinterpret_cast<int4*>(frag_b[k % 2][0]) = sh_b_stage[b_sh_stride * (2 * kk) + b_sh_rd];
    *reinterpret_cast<int4*>(frag_b[k % 2][1]) = sh_b_stage[b_sh_stride * (2 * kk + 1) + b_sh_rd];
  };

  auto fetch_scales_to_registers = [&](int k, int full_pipe) {
    const int pipe = full_pipe % stages;
    if (k % KS == 0) {
      frag_s[k % 2] = sh_s[s_sh_stage * pipe + s_sh_rd];
      const float* as_st = reinterpret_cast<const float*>(sh_as + as_sh_stage * pipe);
  #pragma unroll
      for (int i = 0; i < TMW; i++) {
        frag_as[k % 2][i][0] = as_st[(mh * TMW + i) * 16 + lane / 4];
        frag_as[k % 2][i][1] = as_st[(mh * TMW + i) * 16 + 8 + lane / 4];
      }
    } else {
      frag_s[1] = frag_s[0];
  #pragma unroll
      for (int i = 0; i < TMW; i++) {
        frag_as[1][i][0] = frag_as[0][i][0];
        frag_as[1][i][1] = frag_as[0][i][1];
      }
    }
  };

  // acc += (P as float) * s16[col] * a_s[row, group]
  auto fold = [&](int (&p)[TMW][2][2][4], int4 svr, float (&as)[TMW][2], int jsel = -1) {
    const int16_t* s16 = reinterpret_cast<const int16_t*>(&svr);
  #pragma unroll
    for (int j = 0; j < 2; j++) {
      if (jsel >= 0 && j != jsel) continue;
  #pragma unroll
      for (int h = 0; h < 2; h++) {
        const float s0 = (float)s16[(j * 2 + h) * 2 + 0];
        const float s1 = (float)s16[(j * 2 + h) * 2 + 1];
        // (bits(magic + P) * s) - magic * s in ONE fma: the exact value is P * s, rounded once,
        // i.e. the same float as FADD(-magic) (exact, |P| < 2^22) then FMUL(s) (rounded once).
        // -magic * s is exact (1.5 * |s16| < 2^17 fits the 24-bit mantissa). 2 FP32 ops/element
        // instead of 3 (+ 2 FMUL per (j,h), amortised over 4*TMW elements).
        const float n0 = -kMagic * s0, n1 = -kMagic * s1;
  #pragma unroll
        for (int i = 0; i < TMW; i++) {
  #pragma unroll
          for (int g = 0; g < 4; g++) {
            const float ps = fmaf(__int_as_float(p[i][j][h][g]), g % 2 ? s1 : s0, g % 2 ? n1 : n0);
            acc[i][j][h][g] = fmaf(ps, as[i][g / 2], acc[i][j][h][g]);
          }
        }
      }
    }
  };

  auto mma_step = [&](int (&P)[TMW][2][2][4], int k2, bool first, int jsel = -1) {
    
    
  #pragma unroll
    for (int j = 0; j < 2; j++) {
      if (jsel >= 0 && j != jsel) continue;
  #pragma unroll
      for (int h = 0; h < 2; h++) {
        const uint32_t b0 = frag_b[k2][0][j * 2 + h] ^ 0x88888888u;
        const uint32_t b1 = frag_b[k2][1][j * 2 + h] ^ 0x88888888u;
  #pragma unroll
        for (int i = 0; i < TMW; i++) {
          if (first) {
            a4_mma_q(frag_a[k2][i], b0, b1, P[i][j][h], mq);
          }
          else a4_mma(frag_a[k2][i], b0, b1, P[i][j][h]);
        }
      }
    }
  };
  bool abl_run = false;
  auto matmul_a4 = [&](int k, int tile) {
    const int k2 = k % 2;
    const bool first = k == 0;
    const bool last = k == KS - 1;
    if constexpr (DEFER) {
      const int gp = tile % 2;
      if (gp == 0) mma_step(part[0], k2, first);
      else mma_step(part[DEFER ? 1 : 0], k2, first);
      if (k == 1 && pending) {
        if (gp == 0) fold(part[DEFER ? 1 : 0], sv_saved, as_saved);
        else fold(part[0], sv_saved, as_saved);
        pending = false;
      }
      if (last) {
        sv_saved = frag_s[k2];
  #pragma unroll
        for (int i = 0; i < TMW; i++) {
          as_saved[i][0] = frag_as[k2][i][0];
          as_saved[i][1] = frag_as[k2][i][1];
        }
        pending = true;
      }
    } else {
      mma_step(part[0], k2, first);
      if (last) fold(part[0], frag_s[k2], frag_as[k2]);
    }
  };
  int last_gp = 0;
  auto flush_fold = [&]() {
    
    
    if constexpr (DEFER) {
      if (pending) {
        if (last_gp == 0) fold(part[0], sv_saved, as_saved);
        else fold(part[DEFER ? 1 : 0], sv_saved, as_saved);
        pending = false;
      }
    }
  };

  // parallel reduction: partial slot of (column locks_off, slice s)
  const int par_ms = lda;  // slots per column (host)

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
      const float* fs = reinterpret_cast<const float*>(&acc);
  #pragma unroll
      for (int k = 0; k < th_size; k++) {
        float* dst = reinterpret_cast<float*>(&C_tmp[c_cur_offset + threads * k + threadIdx.x]);
  #pragma unroll
        for (int f = 0; f < 4; f++) dst[f] = fs[k * 4 + f];
      }
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
      constexpr float rs0 = 1.0f, rs1 = 1.0f;
  #pragma unroll
      for (int j = 0; j < 2; j++) {
        const float* c0 = fc + ((i * 2 + j) * 2 + 0) * 4;
        const float* c1 = fc + ((i * 2 + j) * 2 + 1) * 4;
        int wr = c_sh_wr + 8 * j;
        {
          write(wr + (4 * c_sh_stride) * 0 + 0, c0[0], c0[1]);
          write(wr + (4 * c_sh_stride) * 8 + 0, c0[2], c0[3]);
          write(wr + (4 * c_sh_stride) * 0 + 4, c1[0], c1[1]);
          write(wr + (4 * c_sh_stride) * 8 + 4, c1[2], c1[3]);
        }
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
    a_gl_rd += a_sh_stage * (stages - 1);
    as_gl_rd += as_sh_stage * (stages - 1);
    
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
        matmul_a4(k, tile);
        
        if constexpr (DEFER) last_gp = tile % 2;
      }
      slice_iters--;
      if (slice_iters == 0) break;
    }
    a_gl_rd += a_sh_stage * stages;
    as_gl_rd += as_sh_stage * stages;
    

    if (slice_iters == 0) {
      flush_fold();
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      if constexpr (A4_PAR_REDUCE == 7) {
        // consumer-sum (GEMM side): every slice stores its fp32 partial to P[slice_idx][row][col]
        // (P = C_tmp, slot stride 16 tmb x prob_n); no lock chain, no bf16 write. The consumer sums the
        // column tile's slots 0..cnt-1 in slice order (= the serial chain's fp32 order: bit-identical).
        float* P = reinterpret_cast<float*>(C_tmp) + (size_t)slice_idx * 128 * prob_n;  // slot stride 128 rows
        const float* fc = reinterpret_cast<const float*>(&acc);
  #pragma unroll
        for (int i = 0; i < TMW; i++)
  #pragma unroll
          for (int j = 0; j < 2; j++)
  #pragma unroll
            for (int h = 0; h < 2; h++)
  #pragma unroll
              for (int g = 0; g < 4; g += 2) {
                const int row = (mh * TMW + i) * 16 + lane / 4 + (g >= 2 ? 8 : 0);
                const int col = slice_col * thread_n_blocks * 16 + 32 * nw + 8 * (2 * j + h) + 2 * (lane % 4);
                const int e = ((i * 2 + j) * 2 + h) * 4 + g;
                if (row < prob_m) __stcg(reinterpret_cast<float2*>(P + (size_t)row * prob_n + col), make_float2(fc[e], fc[e + 1]));
              }
        last = false;
      } else
      if (slice_count > 1) {
        {
          // gather (rounding-level, deterministic): non-last slices store their partial to their own
          // slot (locks_off, slice_idx) and release; the last slice waits for all of them and adds the
          // slots in slice order to its own registers. One hop instead of the serial chain.
          constexpr int c_size = tmb * 16 * thread_n_blocks * 16 * sizeof(float) / 16;
          constexpr int th_size = TMW * 16 * sizeof(float) / 16;
          int4* base = C_tmp + (size_t)(locks_off * par_ms) * c_size;
          if (!last) {
            const float* fs = reinterpret_cast<const float*>(&acc);
            {
              // half-width (bf16) partial: 8 values per int4
              int4* dst = base + (size_t)slice_idx * c_size;
              float sc = 1.f;
              
  #pragma unroll
              for (int k2 = 0; k2 < th_size / 2; k2++) {
                uint32_t w[4];
  #pragma unroll
                for (int j = 0; j < 4; j++) {
                  const float a = fs[8 * k2 + 2 * j] * sc, b = fs[8 * k2 + 2 * j + 1] * sc;
                  {
                    __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
                    w[j] = *reinterpret_cast<uint32_t*>(&h);
                  }
                }
                __stcg(&dst[threads * k2 + threadIdx.x], make_int4(w[0], w[1], w[2], w[3]));
              }
            }
            barrier_release(&locks[locks_off], false);
          } else {
            barrier_acquire(&locks[locks_off], slice_count - 1);
            float* fc = reinterpret_cast<float*>(&acc);
            {
              constexpr int H = th_size / 2;
              constexpr int NB = 1;
              auto addh = [&](const int4& v, int k2, float d) {
                const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
  #pragma unroll
                for (int j = 0; j < 4; j++) {
                  float a, b;
                  {
                    a = __uint_as_float(w[j] << 16);
                    b = __uint_as_float(w[j] & 0xffff0000u);
                  }
                  fc[8 * k2 + 2 * j] += a;
                  fc[8 * k2 + 2 * j + 1] += b;
                }
              };
  #pragma unroll 1
              for (int s = 0; s < slice_count - 1; s += NB) {
                const int nb = min(NB, slice_count - 1 - s);
                int4 v[NB][H];
                float d[NB];
  #pragma unroll
                for (int q = 0; q < NB; q++) {
                  if (q < nb) {
                    const int4* src = base + (size_t)(s + q) * c_size;
  #pragma unroll
                    for (int k2 = 0; k2 < H; k2++) v[q][k2] = __ldcg(&src[threads * k2 + threadIdx.x]);
                    d[q] = 1.f;
                  }
                }
  #pragma unroll
                for (int q = 0; q < NB; q++)
                  if (q < nb) {
  #pragma unroll
                    for (int k2 = 0; k2 < H; k2++) addh(v[q][k2], k2, d[q]);
                  }
              }
            }
            barrier_release(&locks[locks_off], true);
          }
        }
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
        a_gl_rd = a_sh_stage * slice_row;
        as_gl_rd = as_sh_stage * slice_row;
        b_gl_rd = b_sh_stride * slice_col + b_gl_rd_delta_o * slice_row;
        s_gl_rd = s_gl_stride * slice_row + s_sh_stride * slice_col + threadIdx.x;
        set_ptrs();
        start_pipes();
      }
    }
  }
}

#else
template <int threads, int thread_m_blocks, int thread_n_blocks, int stages,
          int MW, int DEFER, int MINB>
__global__ void MarlinA4(MARLIN_KERNEL_PARAMS) {}
#endif

}  // namespace MARLIN_NAMESPACE_NAME
