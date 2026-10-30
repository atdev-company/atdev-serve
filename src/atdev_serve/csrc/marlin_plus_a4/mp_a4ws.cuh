// marlin_plus P3 (alt 1): warp-specialised W4A4 kernel. Same math, data layouts and tile schedule as
// MarlinA4 (mp_a4.cuh, DEFER=0), so the output is bit-identical; only the pipeline organisation changes:
//   * one producer warp (the last warp) walks the CTA's tile schedule and streams every stage
//     (s4 A fragments, per-(row,group) activation scales, uint4b8 B, int16 group scales) with cp.async,
//     signalling stage s through `full[s]` (cp.async.mbarrier.arrive.noinc: the arrive fires when the
//     producer lane's copies have landed) after waiting for `empty[s]`;
//   * the consumer warps wait on `full[s]`, load both k64 steps' fragments + scales of the stage into
//     registers, release the stage (`empty[s]`, one arrive per warp) and only then run the mma + fold,
//     so the next loads overlap the math instead of a CTA-wide __syncthreads per stage;
//   * the split-K reduction / bf16 epilogue use their own smem (no aliasing with the ring), so the
//     producer keeps prefetching the next slice while the consumers reduce, and there is no per-slice
//     pipeline refill (start_pipes) any more. Consumer-only syncs use named barrier 1.
#pragma once
#include "mp_a4.cuh"  // includes marlin_template.h (no include guard there)

namespace MARLIN_NAMESPACE_NAME {

__device__ __forceinline__ void ws_mbar_init(uint64_t* bar, int count) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(a), "r"(count));
}
__device__ __forceinline__ void ws_mbar_arrive(uint64_t* bar) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" ::"r"(a) : "memory");
}
__device__ __forceinline__ void ws_cp_async_arrive(uint64_t* bar) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(a) : "memory");
}
__device__ __forceinline__ void ws_mbar_wait(uint64_t* bar, int parity) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t done;
  do {
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(done)
        : "r"(a), "r"(parity)
        : "memory");
  } while (!done);
}
template <int NT>
__device__ __forceinline__ void ws_bar_sync() {  // consumer-only named barrier
  asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory");
}

#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800)

template <int threads_c, int thread_m_blocks, int thread_n_blocks, int stages, int MW, int MINB>
__global__ void __launch_bounds__(threads_c + 32, MINB) MarlinA4WS(MARLIN_KERNEL_PARAMS) {
  constexpr int thread_k_blocks = 8;  // tk = 128 = one group per stage
  constexpr int tmb = thread_m_blocks;
  constexpr int TMW = tmb / MW;
  constexpr int tb_n_warps = thread_n_blocks / 2;
  constexpr int NCW = threads_c / 32;  // consumer warps
  static_assert(tmb % MW == 0 && threads_c == 32 * MW * tb_n_warps, "bad layout");
  constexpr int m_block_size = 16 * tmb;
  constexpr float kMagic = 12582912.0f;

  const int4* A0 = A;
  int4* C0 = C;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const bool producer = warp == NCW;

  extern __shared__ int4 sh[];
  __shared__ __align__(8) uint64_t full_bar[stages], empty_bar[stages];

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

  int slice_row = 0, slice_col_par = blockIdx.x, slice_col = 0, slice_iters = k_tiles;
  int slice_count = 1, slice_idx = 0, par_id = 0, locks_off = 0;
  if (part2_mn_tiles >= gridDim.x) locks_off = blockIdx.x;
  else locks_off = (iters * blockIdx.x) / k_tiles - 1;

  constexpr int a_sh_stage = 2 * tmb * 32;
  const int a_chunk = k_tiles * a_sh_stage;
  constexpr int as_sh_stage = m_block_size / 4;
  const int as_chunk = k_tiles * as_sh_stage;
  const int4* AS = reinterpret_cast<const int4*>(a_scales_ptr);
  const int b_gl_stride = prob_n;
  constexpr int b_sh_stride = thread_n_blocks * 16;
  const int b_gl_rd_delta_o = b_gl_stride * 4;
  constexpr int b_sh_stage = b_sh_stride * 4;
  const int s_gl_stride = prob_n / 8;
  constexpr int s_sh_stage = 2 * thread_n_blocks;

  // smem: ring of stages x [A | AS | B | S], then the reduction / epilogue area
  constexpr int st_a = 0, st_as = a_sh_stage, st_b = st_as + as_sh_stage, st_s = st_b + b_sh_stage;
  constexpr int stage_size = st_s + s_sh_stage;
  int4* sh_red = sh + stages * stage_size;

  // ---- tile schedule (identical in producer and consumers) ----
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
    if (!in_part2) {
      if (part1_mn_iters) {
        part1_mn_iters--;
        par_id = slice_col_par / n_tiles;
        slice_col = slice_col_par % n_tiles;
        slice_iters = k_tiles;
        A = A0 + a_chunk * par_id;
        AS = reinterpret_cast<const int4*>(a_scales_ptr) + as_chunk * par_id;
        C = C0 + m_block_size / 8 * par_id * prob_n;
      }
    } else {
      init_part2_slice();
    }
  };
  auto next_slice = [&]() {
    slice_row = 0;
    if (!in_part2) slice_col_par += gridDim.x;
    else {
      slice_col_par++;
      slice_col++;
    }
    init_slice();
  };

  if (threadIdx.x == 0) {
  #pragma unroll
    for (int s = 0; s < stages; s++) {
      ws_mbar_init(&full_bar[s], 32);    // 32 producer lanes, async arrive each
      ws_mbar_init(&empty_bar[s], NCW);  // one arrive per consumer warp
    }
  }
  __syncthreads();
  init_slice();

  if (producer) {
    // ------------------------------------------------------------------ producer warp
    int t = 0;  // global tile counter of this CTA
    while (slice_iters) {
      const int b_base = b_sh_stride * slice_col;
      for (int it = 0; it < slice_iters; it++) {
        const int row = slice_row + it;
        const int s = t % stages;
        ws_mbar_wait(&empty_bar[s], ((t / stages) & 1) ^ 1);
        int4* st = sh + s * stage_size;
  #pragma unroll
        for (int i = lane; i < a_sh_stage; i += 32) cp_async4(&st[st_a + i], &A[a_sh_stage * row + i]);
        if (lane < as_sh_stage) cp_async4(&st[st_as + lane], &AS[as_sh_stage * row + lane]);
  #pragma unroll
        for (int i = lane; i < b_sh_stage; i += 32)
          cp_async4(&st[st_b + i], &B[b_base + b_gl_rd_delta_o * row + b_gl_stride * (i / b_sh_stride) +
                                      (i % b_sh_stride)]);
        if (lane < s_sh_stage)
          cp_async4(&st[st_s + lane], &scales_ptr[s_gl_stride * row + 2 * thread_n_blocks * slice_col + lane]);
        ws_cp_async_arrive(&full_bar[s]);
        t++;
      }
      next_slice();
    }
    cp_async_wait<0>();
    return;
  }

  // -------------------------------------------------------------------- consumer warps
  const int nw = warp % tb_n_warps;
  const int mh = warp / tb_n_warps;
  const int b_sh_rd = 32 * nw + lane;
  const int s_sh_rd = 4 * nw + (lane % 4);
  uint32_t frag_a[2][TMW][4];
  uint32_t frag_b[2][2][4];
  int4 frag_s;
  float frag_as[TMW][2];
  float acc[TMW][2][2][4];
  int part[TMW][2][2][4];

  auto global_reduce_fp32 = [&](bool first, bool last) {
    constexpr int c_size = tmb * 16 * thread_n_blocks * 16 * sizeof(float) / 16;
    constexpr int th_size = TMW * 16 * sizeof(float) / 16;
    const int c_cur_offset = locks_off * c_size;
    if (!first) {
      float* fc = reinterpret_cast<float*>(&acc);
  #pragma unroll
      for (int k = 0; k < th_size; k++) {
        int4 v = __ldcg(&C_tmp[c_cur_offset + threads_c * k + threadIdx.x]);
        const float* sc = reinterpret_cast<const float*>(&v);
  #pragma unroll
        for (int f = 0; f < 4; f++) fc[k * 4 + f] += sc[f];
      }
    }
    if (!last) {
      int4* fc = reinterpret_cast<int4*>(&acc);
  #pragma unroll
      for (int k = 0; k < th_size; k++) C_tmp[c_cur_offset + threads_c * k + threadIdx.x] = fc[k];
    }
  };

  auto write_result = [&]() {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads_c / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta = c_sh_stride * (threads_c / (2 * thread_n_blocks));
    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr = (4 * c_sh_stride) * (lane / 4) + lane % 4 + 16 * nw + (mh * TMW) * 16 * (4 * c_sh_stride);
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));
    int c_gl_wr_end = c_gl_stride * prob_m;
    auto write = [&](int idx, float c0, float c1) {
      reinterpret_cast<nv_bfloat162*>(sh_red)[idx] =
          __halves2bfloat162(__float2bfloat16(c0), __float2bfloat16(c1));
    };
    ws_bar_sync<threads_c>();
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
    ws_bar_sync<threads_c>();
  #pragma unroll
    for (int i = 0; i < div_ceil(16 * tmb, threads_c / (2 * thread_n_blocks)); i++) {
      if (c_gl_wr < c_gl_wr_end) {
        C[c_gl_wr] = sh_red[c_sh_rd];
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
  };

  int t = 0;
  while (slice_iters) {
  #pragma unroll
    for (int i = 0; i < TMW * 16; i++) reinterpret_cast<float*>(acc)[i] = 0.f;
    for (int it = 0; it < slice_iters; it++) {
      const int s = t % stages;
      ws_mbar_wait(&full_bar[s], (t / stages) & 1);
      const int4* st = sh + s * stage_size;
  #pragma unroll
      for (int kk = 0; kk < 2; kk++) {
  #pragma unroll
        for (int i = 0; i < TMW; i++)
          *reinterpret_cast<int4*>(frag_a[kk][i]) = st[st_a + (kk * tmb + mh * TMW + i) * 32 + lane];
        *reinterpret_cast<int4*>(frag_b[kk][0]) = st[st_b + b_sh_stride * (2 * kk) + b_sh_rd];
        *reinterpret_cast<int4*>(frag_b[kk][1]) = st[st_b + b_sh_stride * (2 * kk + 1) + b_sh_rd];
      }
      frag_s = st[st_s + s_sh_rd];
      {
        const float* as_st = reinterpret_cast<const float*>(st + st_as);
  #pragma unroll
        for (int i = 0; i < TMW; i++) {
          frag_as[i][0] = as_st[(mh * TMW + i) * 16 + lane / 4];
          frag_as[i][1] = as_st[(mh * TMW + i) * 16 + 8 + lane / 4];
        }
      }
      __syncwarp();
      if (lane == 0) ws_mbar_arrive(&empty_bar[s]);  // stage is in registers: release it
      t++;
      // mma: k64 step 0 starts from the magic constant, step 1 accumulates (same order as MarlinA4)
  #pragma unroll
      for (int kk = 0; kk < 2; kk++) {
  #pragma unroll
        for (int j = 0; j < 2; j++) {
  #pragma unroll
          for (int h = 0; h < 2; h++) {
            const uint32_t b0 = frag_b[kk][0][j * 2 + h] ^ 0x88888888u;
            const uint32_t b1 = frag_b[kk][1][j * 2 + h] ^ 0x88888888u;
  #pragma unroll
            for (int i = 0; i < TMW; i++) {
              if (kk == 0) a4_mma_magic(frag_a[kk][i], b0, b1, part[i][j][h]);
              else a4_mma(frag_a[kk][i], b0, b1, part[i][j][h]);
            }
          }
        }
      }
      // fold (identical to MarlinA4::fold)
      {
        const int16_t* s16 = reinterpret_cast<const int16_t*>(&frag_s);
  #pragma unroll
        for (int j = 0; j < 2; j++) {
  #pragma unroll
          for (int h = 0; h < 2; h++) {
            const float s0 = (float)s16[(j * 2 + h) * 2 + 0];
            const float s1 = (float)s16[(j * 2 + h) * 2 + 1];
  #pragma unroll
            for (int i = 0; i < TMW; i++) {
  #pragma unroll
              for (int g = 0; g < 4; g++) {
                const float tt = __int_as_float(part[i][j][h][g]) - kMagic;
                acc[i][j][h][g] = fmaf(tt * (g % 2 ? s1 : s0), frag_as[i][g / 2], acc[i][j][h][g]);
              }
            }
          }
        }
      }
    }
    // slice end: serial lock-ordered fp32 reduction (same order as MarlinA4), consumer-only syncs
    const bool last = slice_idx == slice_count - 1;
    if (slice_count > 1) {
      if (threadIdx.x == 0) {
        int state = -1;
        do
          asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(&locks[locks_off]));
        while (state != slice_idx);
      }
      ws_bar_sync<threads_c>();
      global_reduce_fp32(slice_idx == 0, last);
      ws_bar_sync<threads_c>();
      if (threadIdx.x == 0) {
        if (last) locks[locks_off] = 0;
        else {
          asm volatile("fence.acq_rel.gpu;\n");
          asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(&locks[locks_off]), "r"(1));
        }
      }
    }
    if (last) write_result();
    next_slice();
  }
}

#else
template <int threads_c, int thread_m_blocks, int thread_n_blocks, int stages, int MW, int MINB>
__global__ void MarlinA4WS(MARLIN_KERNEL_PARAMS) {}
#endif

}  // namespace MARLIN_NAMESPACE_NAME
