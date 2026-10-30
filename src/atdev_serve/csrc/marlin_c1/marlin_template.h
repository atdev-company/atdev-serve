/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

#include "marlin.cuh"
#include "marlin_dtypes.cuh"
#include "dequant.h"
#include "marlin_mma.h"
#include "core/scalar_type.hpp"

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

namespace MARLIN_NAMESPACE_NAME {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750

template <typename scalar_t,  // compute dtype, half or nv_float16
          const vllm::ScalarTypeId b_type_id,  // weight MarlinScalarType id
          const vllm::ScalarTypeId s_type_id,  // weight scale ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const bool m_block_size_8,  // whether m_block_size == 8
                                      // only works when thread_m_blocks == 1
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const int group_blocks,  // number of consecutive 16x16 blocks
                                   // with a separate quantization scale
          const bool is_zp_float   // is zero point of float16 type?
          >
__global__ void Marlin(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    int4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
    const int4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                          // (k/groupsize)xn
    int prob_m,                           // batch dimension m
    int prob_n,                           // output dimension n
    int prob_k,                           // reduction dimension k
    int* locks,           // extra global storage for barrier synchronization
    bool use_fp32_reduce  // whether to use fp32 global reduce
) {}

}  // namespace marlin

#else

// Instruction for loading a full 16x16 matrix fragment of operand A from shared
// memory, directly in tensor core layout.
template <int count, vllm::ScalarTypeId type_id>
__device__ inline void ldsm(typename MarlinScalarType<type_id>::FragA& frag_a,
                            const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (count == 4) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(smem));
  } else if constexpr (count == 2) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(a[0]), "=r"(a[1])
                 : "r"(smem));
  } else if constexpr (count == 1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                 : "=r"(a[0])
                 : "r"(smem));
  } else {
    static_assert(count == 1 || count == 2 || count == 4, "invalid count");
  }
}

// Multiply dequantized values by the corresponding quantization scale; used
// only for grouped quantization.
template <vllm::ScalarTypeId type_id>
__device__ inline void scale(typename MarlinScalarType<type_id>::FragB& frag_b,
                             typename MarlinScalarType<type_id>::FragS& frag_s,
                             int i) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<type_id>::scalar_t2;
  scalar_t2 s = MarlinScalarType<type_id>::num2num2(
      reinterpret_cast<scalar_t*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

template <vllm::ScalarTypeId type_id>
__device__ inline void scale_and_sub(
    typename MarlinScalarType<type_id>::FragB& frag_b,
    typename MarlinScalarType<type_id>::scalar_t s,
    typename MarlinScalarType<type_id>::scalar_t zp) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<type_id>::scalar_t2;
  scalar_t2 s2 = MarlinScalarType<type_id>::num2num2(s);
  scalar_t2 zp2 = MarlinScalarType<type_id>::num2num2(zp);
  frag_b[0] = __hfma2(frag_b[0], s2, __hneg2(zp2));
  frag_b[1] = __hfma2(frag_b[1], s2, __hneg2(zp2));
}

template <vllm::ScalarTypeId type_id>
__device__ inline void sub_zp(
    typename MarlinScalarType<type_id>::FragB& frag_b,
    typename MarlinScalarType<type_id>::scalar_t2& frag_zp, int i) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<type_id>::scalar_t2;
  scalar_t2 zp = MarlinScalarType<type_id>::num2num2(
      reinterpret_cast<scalar_t*>(&frag_zp)[i]);
  frag_b[0] = __hsub2(frag_b[0], zp);
  frag_b[1] = __hsub2(frag_b[1], zp);
}

// Given 2 floats multiply by 2 scales (halves)
template <vllm::ScalarTypeId type_id>
__device__ inline void scale_float(
    float* c, typename MarlinScalarType<type_id>::FragS& s) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  scalar_t* s_ptr = reinterpret_cast<scalar_t*>(&s);
  c[0] = __fmul_rn(c[0], MarlinScalarType<type_id>::num2float(s_ptr[0]));
  c[1] = __fmul_rn(c[1], MarlinScalarType<type_id>::num2float(s_ptr[1]));
}

// Wait until barrier reaches `count`, then lock for current threadblock.
__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      // Guarantee that subsequent writes by this threadblock will be visible
      // globally.
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

// Release barrier and increment visitation count.
__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    // Make sure that all writes since acquiring this barrier are visible
    // globally, while releasing the barrier.
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                 :
                 : "l"(lock), "r"(val));
  }
}

// Wait until value of lock to be negative, and then add 1
__device__ inline void wait_negative_and_add(int* lock) {
  if (threadIdx.x == 0) {
    int state = 0;
    do
      // Guarantee that subsequent writes by this threadblock will be visible
      // globally.
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    while (state >= 0);
    atomicAdd(lock, 1);
  }
  __syncthreads();
}

#ifndef MP_OVL_MAXS
  #define MP_OVL_MAXS 6
#endif
#include "mp_prologue.inc"
// E6: the epilogue mode is a compile-time constant of the build (one extension per mode, marlin_c1_build);
// the runtime epi.mode must match (checked on the host). E6_RSTD_AT: where a consumer computes rstd
// (0: loads before the pipeline fill, sum after; 1: all before the fill; 2: in write_result).
#ifndef E6_MODE
  #define E6_MODE 0
#endif
#ifndef E6_RSTD_AT
  #define E6_RSTD_AT 0
#endif

template <const vllm::ScalarTypeId a_type_id,  // A ScalarType id
          const vllm::ScalarTypeId b_type_id,  // B ScalarType id
          const vllm::ScalarTypeId c_type_id,  // C ScalarType id
          const vllm::ScalarTypeId s_type_id,  // B_SCALE ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const bool m_block_size_8,  // whether m_block_size == 8
                                      // only works when thread_m_blocks == 1
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const int group_blocks,  // number of consecutive 16x16 blocks
                                   // with a separate quantization scale
          const bool is_zp_float   // is zero point of float16 type?
          >
__device__ __forceinline__ void MarlinBody(
    const int4* __restrict__ A0,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,   // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C0,        // fp16 output buffer of shape mxn
    int4* __restrict__ C_tmp,     // fp32 tmp output buffer (for reduce)
    const int4* __restrict__ b_bias_ptr,
    // float scales of input matrix, only used when is_a_8bit == true.
    // shape (m,)
    const float* __restrict__ a_scales_ptr,
    // fp16 quantization scales. shape (k/groupsize, n)
    const int4* __restrict__ scales_ptr,
    // float global scale (for nvfp4// only)
    const float* __restrict__ global_scale_ptr,
    // 4bit packed zero-points of shape
    // (k/groupsize, n/pack_factor)
    const int4* __restrict__ zp_ptr,
    int prob_m,  // batch dimension m
    int prob_n,  // output dimension n
    int prob_k,  // reduction dimension k
    int lda,     // A.stride(0), equal to prob_k is A is contiguous
    int* locks,  // extra global storage for barrier synchronization
    bool has_bias,
    bool use_atomic_add,   // whether to use atomic add to reduce
    bool use_fp32_reduce,  // whether to use fp32 global reduce
    int max_shared_mem,
    // marlin_plus: fused per-token int8 activation quantisation (int8 path).
    // A_src != nullptr: bf16 activations [prob_m, lda_src]; every CTA
    // quantises rows blockIdx.x, blockIdx.x + gridDim.x, ... into A0 (int8,
    // stride lda) and a_scales_ptr (= absmax/127 * a_gscale), then a grid
    // barrier (grid_bar[0] = count, grid_bar[1] = generation).
    const int4* __restrict__ A_src, int lda_src, int* grid_bar,
    const float* __restrict__ a_gscale_ptr, const MpEpi epi) {
  // E6 consumer row rstd. 128 B so the dynamic smem base keeps its bank alignment (a 64-B static array
  // shifts Marlin's swizzled layouts by 16 banks: +2.4 us on gate_up)
  __shared__ __align__(128) float sh_r[32];
  if constexpr (a_type_id == vllm::kS8.id()) {
    if (A_src != nullptr) {
      mp_fused_quant_prologue<threads>(A_src, lda_src, prob_m, prob_k,
                                       const_cast<int4*>(A0), lda,
                                       const_cast<float*>(a_scales_ptr),
                                       a_gscale_ptr, grid_bar);
    }
  }
  // Each threadblock processes one "stripe" of the B matrix with (roughly) the
  // same size, which might involve multiple column "slices" (of width 16 *
  // `thread_n_blocks`). Stripes are defined as shown in the 3x3 matrix 5 SM
  // example:
  //   0 1 3
  //   0 2 3
  //   1 2 4
  // While this kind of partitioning makes things somewhat more complicated, it
  // ensures good utilization of all SMs for many kinds of shape and GPU
  // configurations, while requiring as few slow global cross-threadblock
  // reductions as possible.

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
  // FP8 computation is only supported for Ada Lovelace or newer architectures.
  if constexpr (a_type_id == vllm::kFE4M3fn.id()) return;
  #endif

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
  // Turing TensorCore only supports fp16 and int8
  if constexpr (a_type_id != vllm::kFloat16.id() && a_type_id != vllm::kS8.id())
    return;
  #endif

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
  constexpr auto num_bits = vllm::ScalarType::from_id(b_type_id).size_bits();
  // Disable use_fp16_accum for NVFP4 and cases when group_size == -1 &&
  // num_bits == 4
  constexpr bool use_fp16_accum =
      a_type_id == vllm::kFloat16.id() &&
      (!(b_type_id == vllm::kFE2M1f.id() && s_type_id == vllm::kFE4M3fn.id()) &&
       !(group_blocks == -1 && num_bits == 4));
  #else
  constexpr bool use_fp16_accum = false;
  #endif
  using Adtype = MarlinScalarType<a_type_id>;
  using Cdtype = MarlinScalarType<c_type_id>;
  const int4* A = A0;
  int4* C = C0;

  using scalar_t = typename MarlinScalarType<a_type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<a_type_id>::scalar_t2;
  using scalar_32bit_t = typename MarlinScalarType<a_type_id>::scalar_32bit_t;

  using c_scalar_t = typename MarlinScalarType<c_type_id>::scalar_t;
  using c_scalar_t2 = typename MarlinScalarType<c_type_id>::scalar_t2;

  using FragA = typename MarlinScalarType<a_type_id>::FragA;
  using FragB = typename MarlinScalarType<a_type_id>::FragB;
  using FragC = typename MarlinScalarType<a_type_id>::FragC;
  using FragS = typename MarlinScalarType<c_type_id>::FragS;
  using FragZP = typename MarlinScalarType<c_type_id>::FragZP;

  static constexpr auto a_type = vllm::ScalarType::from_id(a_type_id);
  static constexpr auto b_type = vllm::ScalarType::from_id(b_type_id);
  static constexpr auto c_type = vllm::ScalarType::from_id(c_type_id);
  static constexpr auto s_type = vllm::ScalarType::from_id(s_type_id);
  if constexpr (b_type == vllm::kFE2M1f) {
    static_assert(s_type == vllm::kFE4M3fn && group_blocks == 1 ||
                  s_type == vllm::kFE8M0fnu && group_blocks == 2);
  } else if constexpr (s_type == vllm::kFE8M0fnu) {
    // MXFP8: FP8 weights with e8m0 microscaling block scales
    static_assert(b_type == vllm::kFE4M3fn && group_blocks == 2);
  } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    static_assert(s_type == vllm::kBFloat16);
  } else if constexpr (std::is_same<scalar_t, half>::value) {
    static_assert(s_type == vllm::kFloat16);
  }

  constexpr bool is_a_8bit = a_type.size_bits() == 8;
  constexpr bool is_8bit_scale = s_type.size_bits() == 8;
  if constexpr (!is_a_8bit) {
    static_assert(std::is_same<scalar_t, c_scalar_t>::value);
  }
  constexpr bool has_zp = b_type == vllm::kU4 || b_type == vllm::kU8;
  constexpr bool is_int_type = b_type == vllm::kU4 || b_type == vllm::kU8 ||
                               b_type == vllm::kS4 || b_type == vllm::kS8 ||
                               b_type == vllm::kU4B8 || b_type == vllm::kU8B128;
  // see comments of dequant.h for more details
  constexpr bool dequant_skip_flop =
      is_a_8bit || (b_type == vllm::kFE4M3fn && !(s_type == vllm::kFE8M0fnu)) ||
      b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn ||
      has_zp && !is_zp_float && !std::is_same<scalar_t, nv_bfloat16>::value ||
      has_zp && !is_zp_float && !(b_type == vllm::kU8);

  float global_scale_f32 = 1.0f;

  if constexpr (b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) {
    global_scale_f32 = global_scale_ptr[0];
  }

  static_assert(group_blocks != 0,
                "runtime activation ordering is not supported");
  constexpr int m_block_size = m_block_size_8 ? 8 : (16 * thread_m_blocks);

  extern __shared__ int4 sh[];
  float* sh_a_s = reinterpret_cast<float*>(sh);
  int4* sh_new = sh + (is_a_8bit ? (4 * thread_m_blocks) : 0);
  constexpr int pack_factor = 32 / b_type.size_bits();
  static_assert(thread_m_blocks == 1 || !m_block_size_8);

  // For larger GEMMs we run multiple batchsize 64 versions in parallel for a
  // better partitioning with less reductions
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

  if constexpr (group_blocks != -1) {
    if (group_blocks >= thread_k_blocks) {
      // Ensure that the number of tiles in each stripe is a multiple of the
      // groupsize; this avoids an annoying special case where a stripe starts
      // in the middle of group.
      iters = (group_blocks / thread_k_blocks) *
              div_ceil(iters, (group_blocks / thread_k_blocks));
    }
  }

  int slice_row = 0;
  int slice_col_par = blockIdx.x;
  int slice_col;
  int slice_iters =
      k_tiles;  // number of threadblock tiles in the current slice
  // total number of active threadblocks in the current slice
  int slice_count = 1;
  // index of threadblock in current slice; numbered bottom to top
  int slice_idx = 0;

  int par_id = 0;
  int locks_off = 0;

  if (part2_mn_tiles >= gridDim.x) {
    // when part2_mn_tiles >= sms
    // then there are at most $sms$ conflict tile blocks
    locks_off = blockIdx.x;
  } else {
    locks_off = (iters * blockIdx.x) / k_tiles - 1;
  }

  // Compute all information about the current slice which is required for
  // synchronization.
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

    if (first_init && use_atomic_add && slice_count > 1 && slice_idx == 0) {
      constexpr int threads_per_m = 16 * thread_n_blocks / 8;
      int m_per_thread =
          div_ceil(thread_m_blocks * 16, threads / threads_per_m);
      if (m_block_size_8) m_per_thread = div_ceil(8, threads / threads_per_m);
      for (int i = 0; i < m_per_thread; i++) {
        int row = threads / threads_per_m * i + threadIdx.x / threads_per_m;
        if (row < prob_m) {
          int col = slice_col * 16 * thread_n_blocks / 8 +
                    threadIdx.x % threads_per_m;
          C[row * prob_n / 8 + col] = {0, 0, 0, 0};
        }
      }
      // After write zero to output, write a negative value to lock.
      // Every SM that processes the same slice would wait for
      // the negative value, and then atomicAdd 1 to it.
      // After all SMs are processed, the lock value would back to 0 again.
      __syncthreads();
      if (threadIdx.x == 0) locks[locks_off] = 1 - slice_count;
    }

    if (slice_col == n_tiles) {
      A += 16 * thread_m_blocks * lda / (is_a_8bit ? 16 : 8);
      C += 16 * thread_m_blocks * prob_n / 8;
      slice_col = 0;
      par_id++;
    }
    if (is_a_8bit && (first_init || slice_col == 0)) {
      __syncthreads();
      int a_s_gl_rd = par_id * 16 * thread_m_blocks + threadIdx.x;
      cp_async1_ca_pred(&sh_a_s[threadIdx.x], &a_scales_ptr[a_s_gl_rd],
                        threadIdx.x < prob_m);
    }
  };

  auto init_part1_slice = [&]() {
    if (part1_mn_iters) {
      part1_mn_iters--;
      par_id = slice_col_par / n_tiles;
      slice_col = slice_col_par % n_tiles;
      slice_iters = k_tiles;
      A = A0 + 16 * thread_m_blocks / (is_a_8bit ? 16 : 8) * par_id * lda;
      C = C0 + 16 * thread_m_blocks / 8 * par_id * prob_n;
      if (is_a_8bit) {
        __syncthreads();
        int a_s_gl_rd = par_id * 16 * thread_m_blocks + threadIdx.x;
        cp_async1_ca_pred(&sh_a_s[threadIdx.x], &a_scales_ptr[a_s_gl_rd],
                          threadIdx.x < prob_m);
      }
    }
  };

  auto init_slice = [&]() {
    if (!in_part2 && !part1_mn_iters) {
      in_part2 = true;
      slice_col_par = (iters * blockIdx.x) / k_tiles;
      slice_row = (iters * blockIdx.x) % k_tiles;
      slice_col = (slice_col_par + global_mn_tiles - part2_mn_tiles) % n_tiles;
      par_id = (slice_col_par + global_mn_tiles - part2_mn_tiles) / n_tiles;
      A = A0 + 16 * thread_m_blocks / (is_a_8bit ? 16 : 8) * par_id * lda;
      C = C0 + 16 * thread_m_blocks / 8 * par_id * prob_n;
    }
    if (!in_part2) {
      init_part1_slice();
    } else {
      init_part2_slice();
      first_init = false;
    }
  };

  init_slice();

  // A sizes/strides

  // stride of the A matrix in global memory
  int a_gl_stride = lda / (is_a_8bit ? 16 : 8);
  // stride of an A matrix tile in shared memory
  constexpr int a_sh_stride = 16 * thread_k_blocks / (is_a_8bit ? 16 : 8);
  // delta between subsequent A tiles in global memory
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / (is_a_8bit ? 16 : 8);
  // between subsequent accesses within a tile
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
  // between shared memory writes
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
  // within a shared memory tile
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
  // overall size of a tile
  constexpr int a_sh_stage = a_sh_stride * m_block_size;
  // number of shared write iterations for a tile
  constexpr int a_sh_wr_iters = div_ceil(a_sh_stage, a_sh_wr_delta);

  // B sizes/strides
  int b_gl_stride = 16 * prob_n / (pack_factor * (is_a_8bit ? 2 : 4));
  constexpr int b_sh_stride =
      ((thread_n_blocks * 16) * 16 / pack_factor) / (is_a_8bit ? 2 : 4);
  constexpr int b_thread_vecs = b_type.size_bits() == 4 ? 1 : 2;
  constexpr int b_sh_stride_threads = b_sh_stride / b_thread_vecs;

  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks / (is_a_8bit ? 2 : 1);
  constexpr int b_sh_wr_delta = threads * b_thread_vecs;
  constexpr int b_sh_stage =
      b_sh_stride * thread_k_blocks / (is_a_8bit ? 2 : 1);
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  // Scale sizes/strides
  int s_gl_stride = prob_n / (is_8bit_scale ? 16 : 8);
  constexpr int s_sh_stride = 16 * thread_n_blocks / (is_8bit_scale ? 16 : 8);
  constexpr int s_tb_groups =
      group_blocks != -1 && group_blocks < thread_k_blocks
          ? thread_k_blocks / group_blocks
          : 1;
  constexpr int s_sh_stage = s_tb_groups * s_sh_stride;
  int s_gl_rd_delta = s_gl_stride;

  constexpr int tb_n_warps = thread_n_blocks / (is_a_8bit ? 2 : 4);

  // Zero-points sizes/strides
  int zp_gl_stride = is_zp_float ? prob_n / 8 : (prob_n / pack_factor) / 4;
  constexpr int zp_sh_stride = is_zp_float
                                   ? 16 * thread_n_blocks / 8
                                   : ((16 * thread_n_blocks) / pack_factor) / 4;
  constexpr int zp_tb_groups = s_tb_groups;
  constexpr int zp_sh_stage = has_zp ? zp_tb_groups * zp_sh_stride : 0;
  int zp_gl_rd_delta = zp_gl_stride;

  // Global A read index of current thread.
  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;
  // Shared write index of current thread.
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  // Shared read index.
  int a_sh_rd =
      a_sh_stride * ((threadIdx.x % 32) % (16 / (m_block_size_8 ? 2 : 1))) +
      (threadIdx.x % 32) / (16 / (m_block_size_8 ? 2 : 1));
  a_sh_rd += 2 * ((threadIdx.x / 32) / tb_n_warps) * b_sh_wr_iters;

  int b_gl_rd;
  if (threads <= b_sh_stride) {
    b_gl_rd = threadIdx.x;
  } else {
    b_gl_rd =
        b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
  }

  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  auto b_sh_rd = threadIdx.x * b_thread_vecs;
  b_sh_rd += b_sh_rd / b_sh_stride * (b_sh_stride * (b_sh_wr_iters - 1));

  int s_gl_rd;
  if constexpr (group_blocks == -1) {
    s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
  } else if constexpr (group_blocks >= thread_k_blocks) {
    s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
              s_sh_stride * slice_col + threadIdx.x;
  } else {
    s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                             threadIdx.x / s_sh_stride) +
              s_sh_stride * slice_col + threadIdx.x % s_sh_stride;
  }
  auto s_sh_wr = threadIdx.x;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stage;

  // Zero-points
  int zp_gl_rd;
  if constexpr (has_zp) {
    if constexpr (group_blocks == -1) {
      zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
    } else if constexpr (group_blocks >= thread_k_blocks) {
      zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                 zp_sh_stride * slice_col + threadIdx.x;
    } else {
      zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                                 threadIdx.x / zp_sh_stride) +
                 zp_sh_stride * slice_col + threadIdx.x % zp_sh_stride;
    }
  }
  auto zp_sh_wr = threadIdx.x;
  bool zp_sh_wr_pred = zp_sh_stage > 0 && threadIdx.x < zp_sh_stage;

  // We use a different scale layout for grouped and column-wise quantization as
  // we scale a `half2` tile in column-major layout in the former and in
  // row-major in the latter case.
  int s_sh_rd;
  if constexpr (is_a_8bit) {
    s_sh_rd = 4 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 4);
  } else if constexpr (group_blocks != -1)
    s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 4;
  else if constexpr (group_blocks == -1 &&
                     (m_block_size_8 || (has_zp && !dequant_skip_flop)))
    s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 8;
  else
    s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) % 4;

  int bias_sh_rd;
  if constexpr (m_block_size_8) {
    bias_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 8;
  } else {
    bias_sh_rd = (is_a_8bit ? 4 : 8) * ((threadIdx.x / 32) % tb_n_warps) +
                 (threadIdx.x % 32) % 4;
  }

  int bias_sh_wr = threadIdx.x;
  int bias_gl_rd = (thread_n_blocks * 16 / 8) * slice_col + threadIdx.x;

  // Zero-points have the same read layout as the scales
  // (without column-wise case)
  constexpr int num_col_threads = 8;
  constexpr int num_row_threads = 4;
  constexpr int num_ints_per_thread = 8 / pack_factor;
  int zp_sh_rd;
  if constexpr (has_zp) {
    if constexpr (is_zp_float) {
      if constexpr (group_blocks != -1) {
        zp_sh_rd =
            8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 4;
      }
    } else if (is_a_8bit) {
      zp_sh_rd = num_ints_per_thread * num_col_threads *
                     ((threadIdx.x / 32) % tb_n_warps / 2) +
                 num_ints_per_thread * ((threadIdx.x % 32) / num_row_threads);
    } else {
      zp_sh_rd = num_ints_per_thread * num_col_threads *
                     ((threadIdx.x / 32) % tb_n_warps) +
                 num_ints_per_thread * ((threadIdx.x % 32) / num_row_threads);
    }
  }

  // Precompute which thread should not read memory in which iterations; this is
  // needed if there are more threads than required for a certain tilesize or
  // when the batchsize is not a multiple of 16.
  bool a_sh_wr_pred[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;

  // To ensure that writing and reading A tiles to/from shared memory, the
  // latter in fragment format, is fully bank conflict free, we need to use a
  // rather fancy XOR-based layout. The key here is that neither reads nor
  // writes of the 16-byte `int4` blocks of 8 consecutive threads involve the
  // same shared memory banks. Further, it seems (based on NSight-Compute) that
  // each warp must also write a consecutive memory segment?
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ (row % 8);
  };
  // Since the computation of this remapping is non-trivial and, due to our main
  // loop unrolls, all shared memory accesses are static, we simply precompute
  // both transformed reads and writes.
  int a_sh_wr_trans[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
  #pragma unroll
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] = transform_a(2 * i + a_sh_rd_delta_i * j + a_sh_rd);
  }

  // Since B-accesses have non-constant stride they have to be computed at
  // runtime; we break dependencies between subsequent accesses with a tile by
  // maintining multiple pointers (we have enough registers), a tiny
  // optimization.

  // Shared memory storage for global fetch pipelines.
  constexpr int sh_red_size = (2 * thread_n_blocks + 1) * 16 * thread_m_blocks;
  constexpr int sh_b_size = stages * b_sh_stage;
  int4* sh_b = sh_new;
  int4* sh_red = sh_new;
  constexpr int sh_size_b_red_min =
      (sh_red_size < sh_b_size ? sh_red_size : sh_b_size);
  constexpr int sh_size_b_red_max =
      (sh_red_size > sh_b_size ? sh_red_size : sh_b_size);
  constexpr int sh_bias_size = (thread_n_blocks * 16 / 8);
  constexpr int sh_b_red_bias_size =
      sh_size_b_red_max > (sh_size_b_red_min + sh_bias_size)
          ? sh_size_b_red_max
          : (sh_size_b_red_min + sh_bias_size);

  int4* sh_bias = sh_new + sh_size_b_red_min;
  int4* sh_zp = sh_new + sh_b_red_bias_size;
  constexpr int sh_s_size = stages * s_sh_stage;
  int4* sh_s = sh_zp + (stages * zp_sh_stage);
  int4* sh_a = sh_s + sh_s_size;

  // Register storage for double buffer of shared memory reads.
  FragA frag_a[2][thread_m_blocks];
  I4 frag_b_quant[2][b_thread_vecs];
  FragC frag_c[thread_m_blocks][is_a_8bit ? 2 : 4][2];
  FragC frag_c_tmp[thread_m_blocks][is_a_8bit ? 2 : 4][2];
  FragS frag_s[2][4];
  FragS frag_bias[2][4];
  int frag_qzp[2][num_ints_per_thread];  // Zero-points
  FragZP frag_zp;                        // Zero-points in fp16
  FragZP frag_zpf[2];                    // Zero-points in fp16 in HQQ

  if constexpr (is_a_8bit) {
  #pragma unroll
    for (int j = 0; j < 2; j++) {
  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
        for (int g = 0; g < 4; g++) {
          frag_c_tmp[i][j][0][g] = 0.0f;
        }

  #pragma unroll
        for (int g = 0; g < 4; g++) {
          frag_c_tmp[i][j][1][g] = 0.0f;
        }
      }
    }
  }

  // Zero accumulators.
  auto zero_accums = [&]() {
  #pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  // Asynchronously fetch the next A, B and s tile from global to the next
  // shared memory pipeline location.
  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    if (pred) {
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
      for (int i = 0; i < (b_sh_wr_iters * b_thread_vecs); i++) {
        constexpr int count = div_ceil(b_sh_stride, threads);
        int b_gl_idx =
            b_gl_rd + (i % count) * threads +
            b_gl_stride * (i / count) * div_ceil(threads, b_sh_stride);

        cp_async4(&sh_b_stage[threads * i + threadIdx.x], &B[b_gl_idx]);
      }

      b_gl_rd += b_gl_rd_delta_o;

      if constexpr (group_blocks != -1) {
        int4* sh_s_stage = sh_s + s_sh_stage * pipe;

        // Only fetch scales if this tile starts a new group
        if (pipe % div_ceil(group_blocks, thread_k_blocks) == 0) {
          if (s_sh_wr_pred) {
            cp_async4(&sh_s_stage[s_sh_wr], &scales_ptr[s_gl_rd]);
          }
          s_gl_rd += s_gl_rd_delta * s_tb_groups;
        }
      }

      if constexpr (has_zp && group_blocks != -1) {
        int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

        // Only fetch zero points if this tile starts a new group
        if (pipe % div_ceil(group_blocks, thread_k_blocks) == 0) {
          if (zp_sh_wr_pred) {
            cp_async4(&sh_zp_stage[zp_sh_wr], &zp_ptr[zp_gl_rd]);
          }
          zp_gl_rd += zp_gl_rd_delta * zp_tb_groups;
        }
      }
    }
    // Insert a fence even when we are winding down the pipeline to ensure that
    // waiting is also correct at this point.
    cp_async_fence();
  };

  auto fetch_col_zp_to_shared = [&]() {
    if (zp_sh_wr_pred) {
      cp_async4(&sh_zp[zp_sh_wr], &zp_ptr[zp_gl_rd]);
    }
  };

  auto fetch_col_scale_to_shared = [&]() {
    if (s_sh_wr_pred) {
      cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
    }
  };

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  // Load the next sub-tile from the current location in the shared memory pipe
  // into the current register buffer.
  auto fetch_to_registers = [&](int k, int pipe) {
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
  #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      ldsm<m_block_size_8 ? 2 : 4, a_type_id>(
          frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;

  #pragma unroll
    for (int i = 0; i < b_thread_vecs; i++) {
      frag_b_quant[k % 2][i] = *reinterpret_cast<I4*>(
          &sh_b_stage[b_sh_stride * (k % b_sh_wr_iters) + b_sh_rd + i]);
    }
  };

  auto fetch_scales_to_registers = [&](int k, int full_pipe) {
    int pipe = full_pipe % stages;
    using IT1 = typename std::conditional_t<is_a_8bit, int2, int4>;
    using IT0 = typename std::conditional_t<is_a_8bit, int, int2>;
    constexpr int group_blocks2 = div_ceil(group_blocks, is_a_8bit ? 2 : 1);

    if constexpr (group_blocks == -1) {
      // Load only when starting a new slice.
      if (k == 0 && full_pipe == 0 && dequant_skip_flop) {
        reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd];
        reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
      }
    } else if constexpr (group_blocks >= thread_k_blocks) {
      constexpr int g = group_blocks / thread_k_blocks;
      if (pipe % g == 0) {
        if (k % b_sh_wr_iters == 0) {
          int4* sh_s_stage = sh_s + s_sh_stage * (g * (pipe / g));
          reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
        } else {
          reinterpret_cast<int4*>(&frag_s[1])[0] =
              reinterpret_cast<int4*>(&frag_s[0])[0];
        }
      }
    } else if (group_blocks2 < b_sh_wr_iters || k % b_sh_wr_iters == 0) {
      auto warp_id = threadIdx.x / 32;
      int warp_row = warp_id / tb_n_warps;

      int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;
      int cur_group_id = k_blocks / group_blocks2;

      int4* sh_s_stage = sh_s + s_sh_stage * pipe;

      if constexpr (!is_8bit_scale) {
        reinterpret_cast<int4*>(&frag_s[k % 2])[0] =
            sh_s_stage[s_sh_rd + cur_group_id * s_sh_stride];
      } else {
        reinterpret_cast<int2*>(&frag_s[k % 2])[0] = reinterpret_cast<int2*>(
            sh_s_stage)[s_sh_rd + cur_group_id * (2 * s_sh_stride)];
      }
    } else if (group_blocks >= b_sh_wr_iters) {
      if constexpr (!is_8bit_scale) {
        reinterpret_cast<int4*>(&frag_s[1])[0] =
            reinterpret_cast<int4*>(&frag_s[0])[0];
      } else {
        reinterpret_cast<int2*>(&frag_s[1])[0] =
            reinterpret_cast<int2*>(&frag_s[0])[0];
      }
    }
  };

  auto fetch_zp_to_registers = [&](int k, int full_pipe) {
    static_assert(!has_zp || group_blocks != 0);

    if constexpr (has_zp && !is_zp_float) {
      int pipe = full_pipe % stages;

      if constexpr (group_blocks == -1) {
        // load only when starting a new slice
        if (k == 0 && full_pipe == 0 || is_a_8bit) {
  #pragma unroll
          for (int i = 0; i < num_ints_per_thread; i++) {
            frag_qzp[k % 2][i] = (reinterpret_cast<int*>(sh_zp))[zp_sh_rd + i];
          }
        }
      } else if constexpr (group_blocks >= thread_k_blocks) {
        constexpr int g = group_blocks / thread_k_blocks;
        if (pipe % g == 0 && k % b_sh_wr_iters == 0 || is_a_8bit) {
          int4* sh_zp_stage = sh_zp + zp_sh_stage * (g * (pipe / g));
  #pragma unroll
          for (int i = 0; i < num_ints_per_thread; i++) {
            frag_qzp[k % 2][i] =
                (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
          }
        }
      } else {
        auto warp_id = threadIdx.x / 32;

        int warp_row = warp_id / tb_n_warps;

        int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;
        int cur_group_id = k_blocks / div_ceil(group_blocks, is_a_8bit ? 2 : 1);

        int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

        sh_zp_stage += cur_group_id * zp_sh_stride;

  #pragma unroll
        for (int i = 0; i < num_ints_per_thread; i++) {
          frag_qzp[k % 2][i] =
              (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
        }
      }
    }

    else if constexpr (has_zp && is_zp_float) {
      int pipe = full_pipe % stages;

      if constexpr (group_blocks != -1) {
        if constexpr (group_blocks >= thread_k_blocks) {
          constexpr int g = group_blocks / thread_k_blocks;
          if (pipe % g == 0 && k % b_sh_wr_iters == 0) {
            int4* sh_zp_stage = sh_zp + zp_sh_stage * (g * (pipe / g));
            reinterpret_cast<int4*>(&frag_zpf[k % 2])[0] =
                sh_zp_stage[zp_sh_rd];
          }
        } else if (group_blocks < b_sh_wr_iters || k % b_sh_wr_iters == 0) {
          auto warp_id = threadIdx.x / 32;

          int warp_row = warp_id / tb_n_warps;
          int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;
          int cur_group_id = k_blocks / group_blocks;

          int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

          reinterpret_cast<int4*>(&frag_zpf[k % 2])[0] =
              sh_zp_stage[zp_sh_rd + cur_group_id * zp_sh_stride];
        }
      }
    }
  };

  auto dequant_data = [&](int q, scalar_32bit_t* frag_b_ptr, int zp = 0) {
    if constexpr (a_type.size_bits() != b_type.size_bits()) {
      if constexpr (is_a_8bit && has_zp) {
        sub_zp_and_dequant<scalar_32bit_t, b_type_id, dequant_skip_flop>(
            q, frag_b_ptr, zp);
      } else {
        dequant<scalar_32bit_t, b_type_id, dequant_skip_flop>(q, frag_b_ptr);
      }
    }
  };

  // Execute the actual tensor core matmul of a sub-tile.
  bool is_first_matmul_in_slice = true;
  auto matmul = [&](int k, int pipe) {
    if (is_a_8bit) return;
    int k2 = k % 2;
    constexpr int g =
        group_blocks > 0 ? div_ceil(group_blocks, thread_k_blocks) : 1;
    const bool is_new_zp =
        (group_blocks > 0 && (group_blocks < b_sh_wr_iters || k == 0)) &&
            (pipe % g == 0) ||
        (group_blocks == -1 && is_first_matmul_in_slice);
    if constexpr (has_zp && !is_zp_float) {
      if (is_new_zp) {
        if constexpr (group_blocks == -1) is_first_matmul_in_slice = false;
        int zp_quant_0, zp_quant_1;

        if constexpr (b_type.size_bits() == 4) {
          zp_quant_0 = frag_qzp[k2][0];
          zp_quant_1 = zp_quant_0 >> 8;
        } else {
          static_assert(b_type.size_bits() == 8);
          zp_quant_0 = frag_qzp[k2][0];
          zp_quant_1 = frag_qzp[k2][1];
        }

        dequant_data(zp_quant_0, reinterpret_cast<scalar_32bit_t*>(&frag_zp));
        dequant_data(zp_quant_1,
                     reinterpret_cast<scalar_32bit_t*>(&frag_zp) + 2);
      }
    }
    if constexpr (!dequant_skip_flop && has_zp && is_zp_float) {
      if (is_new_zp) {
        reinterpret_cast<int4*>(&frag_zp)[0] =
            reinterpret_cast<int4*>(&frag_zpf[k2])[0];
      }
    }

    if constexpr (s_type == vllm::kFE4M3fn || s_type == vllm::kFE8M0fnu) {
      int s_quant_0 = reinterpret_cast<int*>(frag_s[k2])[0];
      int s_quant_1 = reinterpret_cast<int*>(frag_s[k2])[1];

      dequant_fp8_scales<c_scalar_t2, s_type_id>(
          s_quant_0, reinterpret_cast<c_scalar_t2*>(&frag_s[k2]));
      dequant_fp8_scales<c_scalar_t2, s_type_id>(
          s_quant_1, reinterpret_cast<c_scalar_t2*>(&frag_s[k2]) + 2);
    }

  // We have the m dimension as the inner loop in order to encourage overlapping
  // dequantization and matmul operations.
  #pragma unroll
    for (int j = 0; j < 4; j++) {
      FragB frag_b0;
      FragB frag_b1;
      int b_quant_0, b_quant_1;

      if constexpr (b_type_id == vllm::kFE2M1f.id()) {
        b_quant_1 = frag_b_quant[k2][0][j];
        b_quant_0 = b_quant_1 << 8;
      } else if constexpr (b_type.size_bits() == 4) {
        b_quant_0 = frag_b_quant[k2][0][j];
        b_quant_1 = b_quant_0 >> 8;
      } else {
        static_assert(b_type.size_bits() == 8);
        int* frag_b_quant_ptr = reinterpret_cast<int*>(frag_b_quant[k2]);
        b_quant_0 = frag_b_quant_ptr[j * 2 + 0];
        b_quant_1 = frag_b_quant_ptr[j * 2 + 1];
      }

      dequant_data(b_quant_0, reinterpret_cast<scalar_32bit_t*>(&frag_b0));
      dequant_data(b_quant_1, reinterpret_cast<scalar_32bit_t*>(&frag_b1));

      if constexpr (dequant_skip_flop && has_zp && !is_zp_float && !is_a_8bit) {
        sub_zp<a_type_id>(frag_b0, frag_zp[j], 0);
        sub_zp<a_type_id>(frag_b1, frag_zp[j], 1);
      }

      if constexpr (!dequant_skip_flop && has_zp && !is_zp_float &&
                    group_blocks == -1 && !is_a_8bit) {
        int idx = (threadIdx.x / 4) % 2;
        scalar_t2 s2 = Adtype::nums2num2(
            reinterpret_cast<scalar_t*>(&frag_s[j / 2][j % 2 * 2 + 0])[idx],
            reinterpret_cast<scalar_t*>(&frag_s[j / 2][j % 2 * 2 + 1])[idx]);
        if (is_new_zp) frag_zp[j] = __hmul2(frag_zp[j], s2);
        scale_and_sub<a_type_id>(frag_b0, s2.x, frag_zp[j].x);
        scale_and_sub<a_type_id>(frag_b1, s2.y, frag_zp[j].y);
      } else if constexpr (!dequant_skip_flop && has_zp && group_blocks != -1 &&
                           !is_a_8bit) {
        if (is_new_zp)
          frag_zp[j] = __hmul2(frag_zp[j],
                               *reinterpret_cast<scalar_t2*>(&frag_s[k2][j]));
        scale_and_sub<a_type_id>(frag_b0, frag_s[k2][j][0].x, frag_zp[j].x);
        scale_and_sub<a_type_id>(frag_b1, frag_s[k2][j][0].y, frag_zp[j].y);
      } else if constexpr (group_blocks != -1 && !is_a_8bit) {
        scale<a_type_id>(frag_b0, frag_s[k2][j], 0);
        scale<a_type_id>(frag_b1, frag_s[k2][j], 1);
      }

  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        if constexpr (m_block_size_8) {
          mma_trans<a_type_id, use_fp16_accum>(frag_a[k2][i], frag_b0, frag_b1,
                                               frag_c[i][j][0]);
        } else {
          mma<a_type_id, use_fp16_accum>(frag_a[k2][i], frag_b0,
                                         frag_c[i][j][0]);
          mma<a_type_id, use_fp16_accum>(frag_a[k2][i], frag_b1,
                                         frag_c[i][j][1]);
        }
      }
    }
  };

  // marlin_plus MP_FOLD: (a) the first mma of every group accumulator uses
  // C = 0 (no zeroing MOVs after each fold); (b) with thread_k = 64 a 128-group
  // spans two stages, so fold int32 partials into frag_c once per group instead
  // of once per stage. Integer arithmetic -> bit-identical to stock.
#if defined(MP_FOLD) && MP_FOLD
  constexpr bool mp_fold_on = is_a_8bit && group_blocks > 2 && b_sh_wr_iters == 2;
#else
  constexpr bool mp_fold_on = false;
#endif
  constexpr bool mp_fold2 = mp_fold_on && group_blocks == 2 * thread_k_blocks;
  // MP_FOLD >= 2: defer the int32 group fold to the next group's first k-step
  // (just before its C=0 mma reuses the accumulator), so the IMADs never wait
  // on the mma results issued in the same k-step. Pending folds are flushed at
  // slice end (mp_flush_fold). Integer arithmetic -> bit-identical.
#if defined(MP_FOLD) && MP_FOLD >= 2
  constexpr bool mp_defer = mp_fold_on && a_type == vllm::kS8;
#else
  constexpr bool mp_defer = false;
#endif
  int mp_sv[2][2][2];  // saved int group scales [j][half][g%2]
  bool mp_pending = false;
  auto mp_fold_j = [&](int j) {
  #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
      for (int g = 0; g < 4; g++) {
        *reinterpret_cast<int32_t*>(&frag_c[i][j][0][g]) +=
            *reinterpret_cast<int32_t*>(&frag_c_tmp[i][j][0][g]) * mp_sv[j][0][g % 2];
        *reinterpret_cast<int32_t*>(&frag_c[i][j][1][g]) +=
            *reinterpret_cast<int32_t*>(&frag_c_tmp[i][j][1][g]) * mp_sv[j][1][g % 2];
      }
    }
  };
  auto mp_flush_fold = [&]() {
    if constexpr (mp_defer) {
      if (mp_pending) {
        mp_fold_j(0);
        mp_fold_j(1);
        mp_pending = false;
      }
    }
  };
  auto matmul_a8 = [&](int k, int tile) {
    int k2 = k % 2;
    const bool mp_first = mp_fold_on && k == 0 && (!mp_fold2 || tile % 2 == 0);
    const bool mp_last = k == 1 && (!mp_fold2 || tile % 2 == 1);
  #pragma unroll
    for (int j = 0; j < 2; j++) {
      FragB frag_b[2];

      if (is_a_8bit && b_type.size_bits() == 4 && !has_zp) {
#if defined(MP_ABL) && MP_ABL == 2
        // ablation (numerically wrong): no int4->int8 dequant
        reinterpret_cast<int*>(&frag_b)[0] = frag_b_quant[k2][0][j * 2];
        reinterpret_cast<int*>(&frag_b)[1] = frag_b_quant[k2][0][j * 2] >> 4;
        reinterpret_cast<int*>(&frag_b)[2] = frag_b_quant[k2][0][j * 2 + 1];
        reinterpret_cast<int*>(&frag_b)[3] = frag_b_quant[k2][0][j * 2 + 1] >> 4;
#else
        dequant_data(frag_b_quant[k2][0][j * 2],
                     reinterpret_cast<scalar_32bit_t*>(&frag_b));
        dequant_data(frag_b_quant[k2][0][j * 2 + 1],
                     reinterpret_cast<scalar_32bit_t*>(&frag_b) + 2);
#endif
      } else if (is_a_8bit && b_type.size_bits() == 4 && has_zp) {
        int off = (threadIdx.x / 32) % 2 * 2 + j;
        int zp = (frag_qzp[k2][0] >> (off * 8)) & 0xF;
        dequant_data(frag_b_quant[k2][0][j * 2],
                     reinterpret_cast<scalar_32bit_t*>(&frag_b), zp);
        zp = (frag_qzp[k2][0] >> (off * 8 + 4)) & 0xF;
        dequant_data(frag_b_quant[k2][0][j * 2 + 1],
                     reinterpret_cast<scalar_32bit_t*>(&frag_b) + 2, zp);
      } else {
        reinterpret_cast<int2*>(&frag_b)[0] =
            reinterpret_cast<int2*>(&frag_b_quant[k2][j])[0];
        reinterpret_cast<int2*>(&frag_b)[1] =
            reinterpret_cast<int2*>(&frag_b_quant[k2][j])[1];
      }

  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        if constexpr (mp_fold_on && a_type == vllm::kS8) {
          if (mp_first) {
            if constexpr (mp_defer) {
              if (i == 0 && mp_pending) mp_fold_j(j);
            }
            mma_zc_s8<a_type_id>(frag_a[k2][i], frag_b[0], frag_c_tmp[i][j][0]);
            mma_zc_s8<a_type_id>(frag_a[k2][i], frag_b[1], frag_c_tmp[i][j][1]);
            continue;
          }
        }
        mma<a_type_id, false, 32>(
            frag_a[k2][i], frag_b[0],
            (group_blocks == -1 ? frag_c : frag_c_tmp)[i][j][0]);
        mma<a_type_id, false, 32>(
            frag_a[k2][i], frag_b[1],
            (group_blocks == -1 ? frag_c : frag_c_tmp)[i][j][1]);
      }

      if constexpr (group_blocks != -1) {
#if defined(MP_ABL) && MP_ABL == 1
        if (false) {  // ablation (numerically wrong): no group fold
#else
        if (group_blocks == 2 || (mp_fold_on ? mp_last : k == 1)) {
#endif
          if constexpr (a_type == vllm::kS8 && mp_defer) {
            mp_sv[j][0][0] = (int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2][0])[0];
            mp_sv[j][0][1] = (int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2][0])[1];
            mp_sv[j][1][0] = (int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2 + 1][0])[0];
            mp_sv[j][1][1] = (int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2 + 1][0])[1];
            if (j == 1) mp_pending = true;
          } else if constexpr (a_type == vllm::kS8) {
            int2 s_vals[2];
            s_vals[0] = {
                (int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2][0])[0],
                (int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2][0])[1]};
            s_vals[1] = {
                (int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2 + 1][0])[0],
                (int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2 + 1][0])[1]};

  #pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
              for (int g = 0; g < 4; g++) {
                int scale = reinterpret_cast<int*>(&s_vals[0])[g % 2];
                *reinterpret_cast<int32_t*>(&frag_c[i][j][0][g]) +=
                    *reinterpret_cast<int32_t*>(&frag_c_tmp[i][j][0][g]) *
                    scale;
                if constexpr (!mp_fold_on) frag_c_tmp[i][j][0][g] = 0.0f;
              }

  #pragma unroll
              for (int g = 0; g < 4; g++) {
                int scale = reinterpret_cast<int*>(&s_vals[1])[g % 2];
                *reinterpret_cast<int32_t*>(&frag_c[i][j][1][g]) +=
                    *reinterpret_cast<int32_t*>(&frag_c_tmp[i][j][1][g]) *
                    scale;
                if constexpr (!mp_fold_on) frag_c_tmp[i][j][1][g] = 0.0f;
              }
            }
          } else {
            float2 s_vals[2];
            if constexpr (s_type_id != vllm::kFE8M0fnu.id()) {
              static_assert(a_type.size_bits() == 16 ||
                            s_type.size_bits() == 16);
              s_vals[0] = Cdtype::num22float2(frag_s[k2][j * 2][0]);
              s_vals[1] = Cdtype::num22float2(frag_s[k2][j * 2 + 1][0]);
            } else {
              int32_t* s_vals_int = reinterpret_cast<int32_t*>(&s_vals[0]);
              int32_t s_vals_e8m0 =
                  *reinterpret_cast<int32_t*>(&frag_s[k2][j][0]);

              s_vals_int[0] = (s_vals_e8m0 & 0xFF) << 23;
              s_vals_int[1] = (s_vals_e8m0 & 0xFF00) << 15;
              s_vals_int[2] = (s_vals_e8m0 & 0xFF0000) << 7;
              s_vals_int[3] = (s_vals_e8m0 & 0xFF000000) >> 1;
            }

  #pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
              for (int g = 0; g < 4; g++) {
                float scale = reinterpret_cast<float*>(&s_vals[0])[g % 2];
                frag_c[i][j][0][g] += frag_c_tmp[i][j][0][g] * scale;
                frag_c_tmp[i][j][0][g] = 0.0f;
              }

  #pragma unroll
              for (int g = 0; g < 4; g++) {
                float scale = reinterpret_cast<float*>(&s_vals[1])[g % 2];
                frag_c[i][j][1][g] += frag_c_tmp[i][j][1][g] * scale;
                frag_c_tmp[i][j][1][g] = 0.0f;
              }
            }
          }
        }
      }
    }
  };

  // Since we slice across the k dimension of a tile in order to increase the
  // number of warps while keeping the n dimension of a tile reasonable, we have
  // multiple warps that accumulate their partial sums of the same output
  // location; which we have to reduce over in the end. We do in shared memory.
  auto thread_block_reduce = [&]() {
    constexpr int red_off = threads / b_sh_stride_threads / 2;
    if (red_off >= 1) {
      auto red_idx = threadIdx.x / b_sh_stride_threads;
      constexpr int red_sh_stride =
          b_sh_stride_threads * (is_a_8bit ? 2 : 4) * 2;
      constexpr int red_sh_delta = b_sh_stride_threads;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride_threads) +
                      (threadIdx.x % b_sh_stride_threads);

      // Parallel logarithmic shared memory reduction. We make sure to avoid any
      // unnecessary read or write iterations, e.g., for two warps we write only
      // once by warp 1 and read only once by warp 0.

  #pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
  #pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
  #pragma unroll
            for (int j = 0; j < (is_a_8bit ? 2 : 4) * 2;
                 j += (m_block_size_8 ? 2 : 1)) {
              int red_sh_wr =
                  red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd = reinterpret_cast<float*>(
                    &sh_red[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh_red[red_sh_wr]);
  #pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC*>(
                      frag_c)[(is_a_8bit ? 2 : 4) * 2 * m_block + j][k] +=
                      c_rd[k] + c_wr[k];
              }
              sh_red[red_sh_wr] = reinterpret_cast<int4*>(
                  &frag_c)[(is_a_8bit ? 2 : 4) * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
  #pragma unroll
          for (int i = 0; i < (is_a_8bit ? 2 : 4) * 2;
               i += (m_block_size_8 ? 2 : 1)) {
            float* c_rd =
                reinterpret_cast<float*>(&sh_red[red_sh_delta * i + red_sh_rd]);
  #pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(
                  frag_c)[(is_a_8bit ? 2 : 4) * 2 * m_block + i][j] += c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

  // Since multiple threadblocks may process parts of the same column slice, we
  // finally have to globally reduce over the results. As the striped
  // partitioning minimizes the number of such reductions and our outputs are
  // usually rather small, we perform this reduction serially in L2 cache.
  auto global_reduce_fp16 = [&](bool first = false, bool last = false) {
    // We are very careful here to reduce directly in the output buffer to
    // maximize L2 cache utilization in this step. To do this, we write out
    // results in FP16 (but still reduce with FP32 compute).
    constexpr int active_threads = 32 * tb_n_warps;
    if (threadIdx.x < active_threads) {
      int c_gl_stride = prob_n / 8;
      int c_gl_wr_delta_o = 8 * c_gl_stride * (is_a_8bit ? 2 : 1);
      int c_gl_wr_delta_i = 4 * (active_threads / 32);
      int c_gl_wr;
      if constexpr (m_block_size_8) {
        c_gl_wr = c_gl_stride * ((threadIdx.x % 4) * 2) +
                  4 * (threadIdx.x / 32) + (threadIdx.x % 32) / 8;
        c_gl_wr += (2 * thread_n_blocks) * slice_col;
      } else {
        c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) * (is_a_8bit ? 2 : 1) +
                  4 * (threadIdx.x / 32) + threadIdx.x % 4;
        c_gl_wr += (2 * thread_n_blocks) * slice_col * (is_a_8bit ? 2 : 1);
      }
      constexpr int c_sh_wr_delta = active_threads;
      auto c_sh_wr = threadIdx.x;

      int row = (threadIdx.x % 32) / 4;

      if (!first) {
  // Interestingly, doing direct global accesses here really seems to mess up
  // the compiler and lead to slowdowns, hence we also use async-copies even
  // though these fetches are not actually asynchronous.
  #pragma unroll
        for (int i = 0; i < (m_block_size_8 ? 2 : thread_m_blocks * 4); i++) {
          if constexpr (m_block_size_8) {
            cp_async4_pred(&sh_red[c_sh_wr + c_sh_wr_delta * i],
                           &C[c_gl_wr + i * c_gl_stride +
                              (threadIdx.x % 8) / 4 * c_gl_wr_delta_i],
                           (threadIdx.x % 4) * 2 + i < prob_m);
          } else if constexpr (is_a_8bit) {
            int2* sh_red_int2 = reinterpret_cast<int2*>(sh_red);
            int2* c_int2 = reinterpret_cast<int2*>(C);
            cp_async2_ca_pred(
                &sh_red_int2[c_sh_wr + c_sh_wr_delta * i],
                &c_int2[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                        c_gl_wr_delta_i * (i % 2)],
                i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m);
          } else {
            cp_async4_pred(
                &sh_red[c_sh_wr + c_sh_wr_delta * i],
                &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                   c_gl_wr_delta_i * (i % 2)],
                i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m);
          }
        }
        cp_async_fence();
        cp_async_wait<0>();
      }

  #pragma unroll
      for (int i = 0; i < (m_block_size_8 ? 2 : thread_m_blocks * 4); i++) {
        bool mask = (!m_block_size_8) && (i < (thread_m_blocks - 1) * 4 ||
                                          8 * (i / 2) + row < prob_m) ||
                    (m_block_size_8) && ((threadIdx.x % 4) * 2 + i < prob_m);
        if (mask) {
          if (!first) {
            c_scalar_t* c_red_f16;
            if constexpr (is_a_8bit) {
              int2 tmp =
                  reinterpret_cast<int2*>(sh_red)[c_sh_wr + i * c_sh_wr_delta];
              c_red_f16 = reinterpret_cast<c_scalar_t*>(&tmp);
            } else {
              int4 tmp = sh_red[c_sh_wr + i * c_sh_wr_delta];
              c_red_f16 = reinterpret_cast<c_scalar_t*>(&tmp);
            }
  #pragma unroll
            for (int j = 0; j < 2 * (is_a_8bit ? 2 : 4); j++) {
              int delta = 0;
              if constexpr (m_block_size_8) {
                delta = j % 2 == 1 ? -2 : 0;
              }
              reinterpret_cast<float*>(
                  &frag_c)[(is_a_8bit ? 2 : 4) * 2 * 4 * (i / 4) + 4 * j +
                           (i % 4) + delta] += Cdtype::num2float(c_red_f16[j]);
            }
          }
          if (!last) {
            c_scalar_t c_f16[is_a_8bit ? 4 : 8];
  #pragma unroll
            for (int j = 0; j < 2 * (is_a_8bit ? 2 : 4); j++) {
              int delta = 0;
              if constexpr (m_block_size_8) {
                delta = j % 2 == 1 ? -2 : 0;
              }
              c_f16[j] = Cdtype::float2num(reinterpret_cast<float*>(
                  &frag_c)[(is_a_8bit ? 2 : 4) * 2 * 4 * (i / 4) + 4 * j +
                           (i % 4) + delta]);
            }
            if constexpr (m_block_size_8) {
              C[c_gl_wr + i * c_gl_stride +
                (threadIdx.x % 8) / 4 * c_gl_wr_delta_i] =
                  *reinterpret_cast<int4*>(c_f16);
            } else if constexpr (is_a_8bit) {
              int2* c_int2 = reinterpret_cast<int2*>(C);
              c_int2[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                     c_gl_wr_delta_i * (i % 2)] =
                  *reinterpret_cast<int2*>(c_f16);
            } else {
              C[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                c_gl_wr_delta_i * (i % 2)] = *reinterpret_cast<int4*>(c_f16);
            }
          }
        }
      }
    }
  };

  // Globally reduce over threadblocks that compute the same column block.
  // We use a tmp C buffer to reduce in full fp32 precision.
  auto global_reduce_fp32 = [&](bool first = false, bool last = false) {
    constexpr int tb_m = thread_m_blocks * 16;
    constexpr int tb_n = thread_n_blocks * 16;

    constexpr int c_size = tb_m * tb_n * sizeof(float) / 16;

    constexpr int active_threads = 32 * tb_n_warps;
    bool is_th_active = threadIdx.x < active_threads;

    constexpr int num_floats = thread_m_blocks * (is_a_8bit ? 2 : 4) * 2 * 4;
    constexpr int th_size = num_floats * sizeof(float) / 16;

#if defined(MP_OVLCHAIN) && MP_OVLCHAIN
    int c_cur_offset = locks_off * MP_OVL_MAXS * c_size;  // disjoint from other locks' OVL slots
#else
    int c_cur_offset = locks_off * c_size;
#endif

    if (!is_th_active) {
      return;
    }

    if (!first) {
      float* frag_c_ptr = reinterpret_cast<float*>(&frag_c);
  #pragma unroll
      for (int k = 0; k < th_size; k += (m_block_size_8 ? 2 : 1)) {
        sh_red[threadIdx.x] =
            C_tmp[c_cur_offset + active_threads * k + threadIdx.x];

        float* sh_c_ptr = reinterpret_cast<float*>(&sh_red[threadIdx.x]);
  #pragma unroll
        for (int f = 0; f < 4; f++) {
          frag_c_ptr[k * 4 + f] += sh_c_ptr[f];
        }
      }
    }

    if (!last) {
      int4* frag_c_ptr = reinterpret_cast<int4*>(&frag_c);
  #pragma unroll
      for (int k = 0; k < th_size; k += (m_block_size_8 ? 2 : 1)) {
        C_tmp[c_cur_offset + active_threads * k + threadIdx.x] = frag_c_ptr[k];
      }
    }
  };

#if defined(MP_OVLCHAIN) && MP_OVLCHAIN
  // P3 overlapped split-K chain (bit-exact to the serial lock chain): non-last slices store their own
  // fp32 partial to slot (locks_off, slice_idx) and arrive without waiting; the last slice waits once
  // for slice_count-1 arrivals and sums t = p0; t += p1; ...; frag_c += t, which is exactly the serial
  // chain's p_{n-1} + (... + (p1 + p0)) (fp32 add is commutative). Slots: MP_OVL_MAXS per lock.
  auto ovl_reduce_fp32 = [&](bool last) {
    constexpr int tb_m = thread_m_blocks * 16;
    constexpr int tb_n = thread_n_blocks * 16;
    constexpr int c_size = tb_m * tb_n * sizeof(float) / 16;
    constexpr int active_threads = 32 * tb_n_warps;
    constexpr int num_floats = thread_m_blocks * (is_a_8bit ? 2 : 4) * 2 * 4;
    constexpr int th_size = num_floats * sizeof(float) / 16;
    const bool is_th_active = threadIdx.x < active_threads;
    int4* slots = C_tmp + (size_t)locks_off * MP_OVL_MAXS * c_size;
    if (!last) {
      if (is_th_active) {
        int4* frag_c_ptr = reinterpret_cast<int4*>(&frag_c);
        int4* dst = slots + (size_t)slice_idx * c_size;
  #pragma unroll
        for (int k = 0; k < th_size; k += (m_block_size_8 ? 2 : 1))
          __stcg(&dst[active_threads * k + threadIdx.x], frag_c_ptr[k]);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        asm volatile("fence.acq_rel.gpu;\n");
        asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(&locks[locks_off]), "r"(1));
      }
    } else {
      if (threadIdx.x == 0) {
        int state = -1;
        do
          asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(&locks[locks_off]));
        while (state != slice_count - 1);
      }
      __syncthreads();
      // stage the predecessors' partials in the (idle) pipeline smem with cp.async (no data registers,
      // one L2 round trip), then sum in slice order from smem
      const int nsl = slice_count - 1;
      int4* S = sh_new;
      if (is_th_active) {
  #pragma unroll 1
        for (int sl = 0; sl < nsl; sl++) {
  #pragma unroll
          for (int k = 0; k < th_size; k += (m_block_size_8 ? 2 : 1))
            cp_async4(&S[(sl * th_size + k) * active_threads + threadIdx.x],
                      &slots[(size_t)sl * c_size + active_threads * k + threadIdx.x]);
        }
      }
      cp_async_fence();
      cp_async_wait<0>();
      if (is_th_active) {
        float* frag_c_ptr = reinterpret_cast<float*>(&frag_c);
  #pragma unroll
        for (int k = 0; k < th_size; k += (m_block_size_8 ? 2 : 1)) {
          float4 t = *reinterpret_cast<float4*>(&S[k * active_threads + threadIdx.x]);
  #pragma unroll 1
          for (int sl = 1; sl < nsl; sl++) {
            float4 u = *reinterpret_cast<float4*>(&S[(sl * th_size + k) * active_threads + threadIdx.x]);
            t.x += u.x; t.y += u.y; t.z += u.z; t.w += u.w;
          }
          frag_c_ptr[k * 4 + 0] += t.x; frag_c_ptr[k * 4 + 1] += t.y;
          frag_c_ptr[k * 4 + 2] += t.z; frag_c_ptr[k * 4 + 3] += t.w;
        }
      }
      __syncthreads();  // S aliases sh_red (write_result)
      if (threadIdx.x == 0) locks[locks_off] = 0;
    }
  };
#endif

  // Write out the reduce final result in the correct layout. We only actually
  // reshuffle matrix fragments in this step, the reduction above is performed
  // in fragment layout.
  // E6 epilogue helpers (c_scalar_t = bf16 on the c=1 path). 2 * thread_n_blocks consecutive lanes share
  // one output row in the final store loop of write_result.
  auto e6_unpack = [&](const int4& v, float* f) {
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&v);
  #pragma unroll
    for (int e = 0; e < 4; e++) {
      float2 q = __bfloat1622float2(h2[e]);
      f[2 * e] = q.x;
      f[2 * e + 1] = q.y;
    }
  };
  auto e6_pack = [&](const float* f) {
    int4 v;
    __nv_bfloat162* h2 = reinterpret_cast<__nv_bfloat162*>(&v);
  #pragma unroll
    for (int e = 0; e < 4; e++) h2[e] = __floats2bfloat162_rn(f[2 * e], f[2 * e + 1]);
    return v;
  };
  auto e6_producer_store = [&](const int4 hv, int gidx, int gstride, int tile, const int4 rv, const int4 wv) {
    const int row = gidx / gstride;
    float h[8], r[8], w[8], x[8], y[8];
    e6_unpack(hv, h);
    e6_unpack(rv, r);
    e6_unpack(wv, w);
    float s = 0.f;
  #pragma unroll
    for (int e = 0; e < 8; e++) {
      x[e] = h[e] + r[e];
      s = fmaf(x[e], x[e], s);
      y[e] = x[e] * (w[e] + 1.f);
    }
#if !(defined(E6_ABL) && E6_ABL == 5)
    epi.C2[gidx] = e6_pack(x);
#endif
    C[gidx] = e6_pack(y);
#if !(defined(E6_ABL) && E6_ABL == 4)
    const unsigned msk = __activemask();
  #pragma unroll
    for (int o = thread_n_blocks; o > 0; o >>= 1) s += __shfl_xor_sync(msk, s, o);
    if (threadIdx.x % (2 * thread_n_blocks) == 0)
      epi.P[row * (prob_n / (16 * thread_n_blocks)) + tile] = s;
#endif
  };
  auto e6_silu_store = [&](const int4 hv, int gidx, int gstride, int tile) {
    const unsigned msk = __activemask();
    int4 pv;
    pv.x = __shfl_xor_sync(msk, hv.x, thread_n_blocks);
    pv.y = __shfl_xor_sync(msk, hv.y, thread_n_blocks);
    pv.z = __shfl_xor_sync(msk, hv.z, thread_n_blocks);
    pv.w = __shfl_xor_sync(msk, hv.w, thread_n_blocks);
    const int lane_c = threadIdx.x % (2 * thread_n_blocks);
    if (lane_c < thread_n_blocks) {
      const int row = gidx / gstride;
      float g[8], u[8], y[8];
      e6_unpack(hv, g);
      e6_unpack(pv, u);
  #pragma unroll
      for (int e = 0; e < 8; e++) {
#if defined(E6_SILU) && E6_SILU == 1
        y[e] = __fdividef(g[e], 1.f + __expf(-g[e])) * u[e];
#elif defined(E6_SILU) && E6_SILU == 3
        y[e] = g[e] * __frcp_rn(1.f + expf(-g[e])) * u[e];
#elif defined(E6_SILU) && E6_SILU == 2
        y[e] = g[e] * u[e];  // ablation: no transcendental
#else
        y[e] = (g[e] / (1.f + expf(-g[e]))) * u[e];
#endif
      }
      C[row * (gstride / 2) + tile * thread_n_blocks + lane_c] = e6_pack(y);
    }
  };

  // E6 producer: the residual / norm-weight chunks of the final store loop, loaded before the split-K
  // reduction wait so their latency is hidden (same index math as write_result's store loop).
  constexpr int E6_NI = div_ceil(16 * thread_m_blocks, threads / (2 * thread_n_blocks));
  int4 e6_rpre[E6_NI];
  int4 e6_wpre;
  auto e6_prefetch = [&]() {
    const int gstride = prob_n / 8;
    int g = gstride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks)) +
            (2 * thread_n_blocks) * slice_col;
    const int gend = gstride * prob_m;
    e6_wpre = epi.W[(threadIdx.x % (2 * thread_n_blocks)) + (2 * thread_n_blocks) * slice_col];
  #pragma unroll
    for (int i = 0; i < E6_NI; i++) {
      e6_rpre[i] = g < gend ? epi.R[g] : int4{0, 0, 0, 0};
      g += gstride * (threads / (2 * thread_n_blocks));
    }
  };

  auto write_result = [&](bool last) {
    if constexpr (E6_MODE >= 2 && E6_RSTD_AT == 2) {
      float pv[threads / 16 >= 1 ? 10 : 10];
      const int row = threadIdx.x / (threads / 16), sub = threadIdx.x % (threads / 16);
  #pragma unroll
      for (int q = 0; q < 10; q++) {
        const int t = sub + q * (threads / 16);
        pv[q] = (row < prob_m && t < epi.T_in) ? epi.Pin[row * epi.T_in + t] : 0.f;
      }
      float s = 0.f;
  #pragma unroll
      for (int q = 0; q < 10; q++) s += pv[q];
  #pragma unroll
      for (int o = threads / 32; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
      if (sub == 0) {
        const float rr = row < prob_m ? rsqrtf(s / (float)epi.n_norm + epi.eps) : 0.f;
        sh_r[row] = rr;
        if (epi.Rout != nullptr && blockIdx.x == 0 && row < prob_m) epi.Rout[row] = rr;
      }
      __syncthreads();
    }
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta =
        c_sh_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr;
    if constexpr (m_block_size_8) {
      c_sh_wr = (8 * c_sh_stride) * ((threadIdx.x % 32) % 4 * 2) +
                (threadIdx.x % 32) / 4;
      c_sh_wr += 64 * (threadIdx.x / 32);
    } else {
      c_sh_wr =
          (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
      c_sh_wr += (is_a_8bit ? 16 : 32) * (threadIdx.x / 32);
    }

    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));

    int c_gl_wr_end = c_gl_stride * prob_m;
    // We first reorder in shared memory to guarantee the most efficient final
    // global write patterns
    auto write = [&](int idx, float c0, float c1, FragS& s, FragS& b_bias) {
      if constexpr (b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) {
        c0 *= global_scale_f32;
        c1 *= global_scale_f32;
      }
      c_scalar_t2 res =
          Cdtype::nums2num2(Cdtype::float2num(c0), Cdtype::float2num(c1));

      // For per-column quantization we finally apply the scale here (only for
      // 4-bit)
      if constexpr (group_blocks == -1 && !is_a_8bit &&
                    b_type.size_bits() == 4 &&
                    (has_zp && dequant_skip_flop || !has_zp)) {
        c_scalar_t2 tmp_scale = s[0];
        if constexpr (m_block_size_8) {
          tmp_scale = Cdtype::num2num2(
              reinterpret_cast<scalar_t*>(&s[0])[(threadIdx.x % 8) / 4]);
        }
        res = __hmul2(res, tmp_scale);
      }
      if (has_bias && last) {
        c_scalar_t2 tmp_bias = b_bias[0];
        if constexpr (m_block_size_8) {
          tmp_bias = Cdtype::num2num2(
              reinterpret_cast<scalar_t*>(&b_bias[0])[(threadIdx.x % 8) / 4]);
        }
        res = __hadd2(res, tmp_bias);
      }

      if constexpr (m_block_size_8) {
        ((c_scalar_t*)sh_red)[idx] = res.x;
        ((c_scalar_t*)sh_red)[idx + 8 * c_sh_stride] = res.y;
      } else {
        ((c_scalar_t2*)sh_red)[idx] = res;
      }
    };

    if (threadIdx.x / 32 < tb_n_warps) {
  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
        for (int j = 0; j < (is_a_8bit ? 2 : 4); j++) {
          if constexpr (m_block_size_8) {
            int wr = c_sh_wr + 16 * j;
            write(wr, frag_c[i][j][0][0], frag_c[i][j][0][1],
                  frag_s[j / 2][2 * (j % 2) + 0],
                  frag_bias[j / 2][2 * (j % 2) + 0]);
            write(wr + 8, frag_c[i][j][0][2], frag_c[i][j][0][3],
                  frag_s[j / 2][2 * (j % 2) + 1],
                  frag_bias[j / 2][2 * (j % 2) + 1]);
          } else {
            int wr = c_sh_wr + 8 * j;
            // E6 consumer: rows lane/4 (c0, c1) and lane/4 + 8 (c2, c3) of m-block i
            float r_lo = 1.f, r_hi = 1.f;
#if defined(E6_ABL) && E6_ABL == 2
            if constexpr (false) {  // ablation: no rstd scaling
#else
            if constexpr (E6_MODE >= 2) {
#endif
              r_lo = sh_r[16 * i + (threadIdx.x % 32) / 4];
              r_hi = sh_r[16 * i + (threadIdx.x % 32) / 4 + 8];
            }
            write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0] * r_lo,
                  frag_c[i][j][0][1] * r_lo, frag_s[j / 2][2 * (j % 2) + 0],
                  frag_bias[j / 2][2 * (j % 2) + 0]);
            write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2] * r_hi,
                  frag_c[i][j][0][3] * r_hi, frag_s[j / 2][2 * (j % 2) + 0],
                  frag_bias[j / 2][2 * (j % 2) + 0]);
            write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0] * r_lo,
                  frag_c[i][j][1][1] * r_lo, frag_s[j / 2][2 * (j % 2) + 1],
                  frag_bias[j / 2][2 * (j % 2) + 1]);
            write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2] * r_hi,
                  frag_c[i][j][1][3] * r_hi, frag_s[j / 2][2 * (j % 2) + 1],
                  frag_bias[j / 2][2 * (j % 2) + 1]);
          }
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

#if defined(E6_SILU_SMEM) && E6_SILU_SMEM
    if constexpr (E6_MODE == 3) {
      // SiLU pair straight from the staged tile: every thread produces one 8-column output chunk per pass
      // (g = chunk oc of the gate half, u = same chunk of the up half, both in sh_red row r)
      const int oc = threadIdx.x % thread_n_blocks;
  #pragma unroll
      for (int r = threadIdx.x / thread_n_blocks; r < 16 * thread_m_blocks; r += threads / thread_n_blocks) {
        if (r < prob_m) {
          float g[8], u[8], y[8];
          e6_unpack(sh_red[c_sh_stride * r + oc], g);
          e6_unpack(sh_red[c_sh_stride * r + thread_n_blocks + oc], u);
  #pragma unroll
          for (int e = 0; e < 8; e++) {
  #if defined(E6_SILU) && E6_SILU == 1
            y[e] = __fdividef(g[e], 1.f + __expf(-g[e])) * u[e];
  #elif defined(E6_SILU) && E6_SILU == 3
            y[e] = g[e] * __frcp_rn(1.f + expf(-g[e])) * u[e];
  #else
            y[e] = (g[e] / (1.f + expf(-g[e]))) * u[e];
  #endif
          }
          C[r * (prob_n / 16) + slice_col * thread_n_blocks + oc] = e6_pack(y);
        }
      }
      __syncthreads();
      return;
    }
#endif
  #pragma unroll
    for (int i = 0;
         i < div_ceil(16 * thread_m_blocks, threads / (2 * thread_n_blocks));
         i++) {
      if (c_gl_wr < c_gl_wr_end) {
        if (use_atomic_add && slice_count > 1) {
          c_scalar_t2* C_half2 = reinterpret_cast<c_scalar_t2*>(&C[c_gl_wr]);
          c_scalar_t2* sh_red_half2 =
              reinterpret_cast<c_scalar_t2*>(&sh_red[c_sh_rd]);
  #pragma unroll
          for (int a = 0; a < 4; a++) {
            atomicAdd(&C_half2[a], sh_red_half2[a]);
          }
        } else if constexpr (E6_MODE == 1) {
          e6_producer_store(sh_red[c_sh_rd], c_gl_wr, c_gl_stride, slice_col, e6_rpre[i], e6_wpre);
        } else if constexpr (E6_MODE == 3) {
          e6_silu_store(sh_red[c_sh_rd], c_gl_wr, c_gl_stride, slice_col);
        } else {
          C[c_gl_wr] = sh_red[c_sh_rd];
        }
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
    __syncthreads();
  };

  // Start global fetch and register load pipelines.
  auto start_pipes = [&]() {

  #pragma unroll
    for (int i = 0; i < stages - 1; i++) {
      if constexpr (has_zp && !is_zp_float && group_blocks == -1) {
        if (i == 0) {
          fetch_col_zp_to_shared();
          if constexpr (!dequant_skip_flop) {
            fetch_col_scale_to_shared();
          }
        }
      }
      fetch_to_shared(i, i, i < slice_iters);
    }

    zero_accums();
    wait_for_stage();
    fetch_to_registers(0, 0);
    fetch_scales_to_registers(0, 0);
    fetch_zp_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
  };
  // E6 consumer: per-row rstd from the producer's per-tile partials. Loads issued before the pipeline
  // fill, summed after it; sh_r is read in write_result (>= 1 __syncthreads of the main loop in between).
  constexpr int E6_TPR = threads / 16;  // threads per row
  constexpr int E6_MAXP = 10;           // partials per thread (T_in <= E6_TPR * E6_MAXP)
  auto e6_rstd_load = [&](float* pv) {
    const int row = threadIdx.x / E6_TPR, sub = threadIdx.x % E6_TPR;
  #pragma unroll
    for (int q = 0; q < E6_MAXP; q++) {
      const int t = sub + q * E6_TPR;
#if defined(E6_ABL) && E6_ABL == 1
      pv[q] = (row < prob_m && t < epi.T_in) ? 1.f : 0.f;  // ablation: no partial loads
#else
      pv[q] = (row < prob_m && t < epi.T_in) ? epi.Pin[row * epi.T_in + t] : 0.f;
#endif
    }
  };
  auto e6_rstd_finish = [&](const float* pv) {
    const int row = threadIdx.x / E6_TPR, sub = threadIdx.x % E6_TPR;
    float s = 0.f;
  #pragma unroll
    for (int q = 0; q < E6_MAXP; q++) s += pv[q];
  #pragma unroll
    for (int o = E6_TPR / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (sub == 0) {
      const float rr = row < prob_m ? rsqrtf(s / (float)epi.n_norm + epi.eps) : 0.f;
      sh_r[row] = rr;
      if (epi.Rout != nullptr && blockIdx.x == 0 && row < prob_m) epi.Rout[row] = rr;
    }
  };
  if constexpr (E6_MODE >= 2 && E6_RSTD_AT == 1) {
    float pv[E6_MAXP];
    e6_rstd_load(pv);
    e6_rstd_finish(pv);
  }
  if constexpr (E6_MODE >= 2 && E6_RSTD_AT == 0) {
    float pv[E6_MAXP];
    e6_rstd_load(pv);
    if (slice_iters) {
      start_pipes();
    }
    e6_rstd_finish(pv);
  } else {
    if (slice_iters) {
      start_pipes();
    }
  }

  // Main loop.
  while (slice_iters) {
    // We unroll over both the global fetch and the register load pipeline to
    // ensure all shared memory accesses are static. Note that both pipelines
    // have even length meaning that the next iteration will always start at
    // index 0.

  #pragma unroll
    for (int pipe = 0; pipe < stages;) {
  #pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        fetch_scales_to_registers(k + 1, pipe);
        fetch_zp_to_registers(k + 1, pipe);
        if (k == b_sh_wr_iters - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe,
                          slice_iters >= stages);
          pipe++;
          wait_for_stage();
        }

        if constexpr (!is_a_8bit) {
          matmul(k, pipe - (k >= b_sh_wr_iters - 2 ? 1 : 0));
        } else {
          static_assert(group_blocks != 0 && group_blocks != 1);
          matmul_a8(k, pipe - (k >= b_sh_wr_iters - 2 ? 1 : 0));
        }
      }
      slice_iters--;
      if (slice_iters == 0) {
        break;
      }
    }

    a_gl_rd += a_gl_rd_delta_o * stages;

    // Process results and, if necessary, proceed to the next column slice.
    // While this pattern may not be the most readable, other ways of writing
    // the loop seemed to noticeably worse performance after compilation.
    if (slice_iters == 0) {
      // convert fp16 accum to fp32 for reduction
      if constexpr (use_fp16_accum) {
  #pragma unroll
        for (int i = 0; i < (thread_m_blocks * (is_a_8bit ? 2 : 4) * 2); i++) {
          float* frag_c_part_float = reinterpret_cast<float*>(frag_c) + i * 4;
          scalar_t* frag_c_part_half =
              reinterpret_cast<scalar_t*>(frag_c_part_float);

  #pragma unroll
          for (int i = 3; i >= 0; i--) {
            frag_c_part_float[i] = Cdtype::num2float(frag_c_part_half[i]);
          }
        }
      }

      mp_flush_fold();
      if constexpr (is_a_8bit) {
        float frag_a_s[2 * thread_m_blocks];

        for (int i = 0; i < 2 * thread_m_blocks; i++)
          frag_a_s[i] = sh_a_s[i * 8 + (threadIdx.x % 32) / 4];

  #pragma unroll
        for (int j = 0; j < 2; j++) {
  #pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
            for (int g = 0; g < 4; g++) {
              float c_val = frag_c[i][j][0][g];

              if constexpr (a_type == vllm::kS8) {
                c_val = __int2float_rn(*reinterpret_cast<int32_t*>(&c_val));
              }
              float s_val = frag_a_s[i * 2 + g / 2];
              frag_c[i][j][0][g] = c_val * s_val;
            }
  #pragma unroll
            for (int g = 0; g < 4; g++) {
              float c_val = frag_c[i][j][1][g];

              if constexpr (a_type == vllm::kS8) {
                c_val = __int2float_rn(*reinterpret_cast<int32_t*>(&c_val));
              }
              float s_val = frag_a_s[i * 2 + g / 2];
              frag_c[i][j][1][g] = c_val * s_val;
            }
          }
        }
      }

      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      // For per-column scales, we only fetch them here in the final step before
      // write-out
      if constexpr (group_blocks == -1 &&
                    (has_zp && dequant_skip_flop || !has_zp)) {
        if (b_type.size_bits() == 8 || (last || use_atomic_add) || is_a_8bit) {
          if (s_sh_wr_pred) {
            cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
          }
          cp_async_fence();
        }
      }

      thread_block_reduce();

      if (has_bias && last) {
        __syncthreads();
        cp_async4_pred(&sh_bias[bias_sh_wr], &b_bias_ptr[bias_gl_rd],
                       threadIdx.x < 16 * thread_n_blocks / 8);
        cp_async_fence();
      }

      if constexpr (group_blocks == -1 &&
                    (has_zp && dequant_skip_flop || !has_zp || is_a_8bit)) {
        if constexpr (is_a_8bit) {
          cp_async_wait<0>();
          __syncthreads();
          if (threadIdx.x / 32 < tb_n_warps) {
            reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
          }
        } else if (b_type.size_bits() == 8 || (last || use_atomic_add)) {
          cp_async_wait<0>();
          __syncthreads();
          if (threadIdx.x / 32 < tb_n_warps) {
            reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
            reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
            if constexpr (m_block_size_8) {
              int idx = (threadIdx.x / 4) % 2;
              c_scalar_t2* frag_s_half2 =
                  reinterpret_cast<c_scalar_t2*>(frag_s);
  #pragma unroll
              for (int i = 0; i < 8; i++) {
                frag_s_half2[i] = Cdtype::num2num2(
                    reinterpret_cast<c_scalar_t*>(&frag_s_half2[i])[idx]);
              }
            }
          }
        }
      }

      // For 8-bit channelwise, we apply the scale before the global reduction
      // that converts the fp32 results to fp16 (so that we avoid possible
      // overflow in fp16)
      if constexpr (group_blocks == -1 && is_a_8bit) {
  #pragma unroll
        for (int j = 0; j < 2; j++) {
          float2 aa[2];
          aa[0] = Cdtype::num22float2(frag_s[0][j * 2][0]);
          aa[1] = Cdtype::num22float2(frag_s[0][j * 2 + 1][0]);

  #pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
            for (int g = 0; g < 4; g++) {
              float scale = reinterpret_cast<float*>(&aa[0])[g % 2];
              frag_c[i][j][0][g] *= scale;
            }

  #pragma unroll
            for (int g = 0; g < 4; g++) {
              float scale = reinterpret_cast<float*>(&aa[1])[g % 2];
              frag_c[i][j][1][g] *= scale;
            }
          }
        }
      } else if (group_blocks == -1 && b_type.size_bits() == 8 &&
                 (has_zp && dequant_skip_flop || !has_zp)) {
        if (threadIdx.x / 32 < tb_n_warps) {
  #pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
            for (int j = 0; j < 4; j++) {
              scale_float<c_type_id>(
                  reinterpret_cast<float*>(&frag_c[i][j][0][0]),
                  frag_s[j / 2][2 * (j % 2) + 0]);
              scale_float<c_type_id>(
                  reinterpret_cast<float*>(&frag_c[i][j][0][2]),
                  frag_s[j / 2][2 * (j % 2) + (m_block_size_8 ? 1 : 0)]);

              if constexpr (!m_block_size_8) {
                scale_float<c_type_id>(
                    reinterpret_cast<float*>(&frag_c[i][j][1][0]),
                    frag_s[j / 2][2 * (j % 2) + 1]);
                scale_float<c_type_id>(
                    reinterpret_cast<float*>(&frag_c[i][j][1][2]),
                    frag_s[j / 2][2 * (j % 2) + 1]);
              }
            }
          }
        }
      }

      if constexpr (E6_MODE == 1) {
        if (last) e6_prefetch();
      }
#if defined(MP_ABL) && MP_ABL == 3
      if (false) {  // ablation (numerically wrong): no split-K reduction
#else
      if (slice_count > 1 && !use_atomic_add) {
#endif
        // only globally reduce if there is more than one block in a slice
#if defined(MP_OVLCHAIN) && MP_OVLCHAIN
        if (use_fp32_reduce && slice_count <= MP_OVL_MAXS &&
            (slice_count - 1) * (thread_m_blocks * 16 * thread_n_blocks * 16 * 4 / 16) <= sh_size_b_red_max) {
          ovl_reduce_fp32(last);
        } else
#endif
        {
        barrier_acquire(&locks[locks_off], slice_idx);
        if (use_fp32_reduce) {
          global_reduce_fp32(slice_idx == 0, last);
        } else {
          global_reduce_fp16(slice_idx == 0, last);
        }
        barrier_release(&locks[locks_off], last);
        }
      }

      if (has_bias && last) {
        cp_async_wait<0>();
        __syncthreads();
        reinterpret_cast<int4*>(&frag_bias)[0] = sh_bias[bias_sh_rd];
        if constexpr (!is_a_8bit)
          reinterpret_cast<int4*>(&frag_bias)[1] = sh_bias[bias_sh_rd + 4];
        __syncthreads();
      }

      if (use_atomic_add && slice_count > 1 && slice_idx != 0)
        wait_negative_and_add(&locks[locks_off]);
      if (last || use_atomic_add)
        // only the last block in a slice actually writes the result
        write_result(last);
      slice_row = 0;
      if (!in_part2) {
        slice_col_par += gridDim.x;
      } else {
        slice_col_par++;
        slice_col++;
      }
      is_first_matmul_in_slice = true;
      init_slice();

      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                  (threadIdx.x % a_gl_rd_delta_o);
        a_gl_rd += a_gl_rd_delta_o * slice_row;
        b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) +
                  (threadIdx.x % b_sh_stride);
        b_gl_rd += b_sh_stride * slice_col + b_gl_rd_delta_o * slice_row;

        bias_gl_rd = (thread_n_blocks * 16 / 8) * slice_col + threadIdx.x;
        if constexpr (group_blocks == -1) {
          s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
          zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
        } else if constexpr (group_blocks >= thread_k_blocks) {
          s_gl_rd =
              s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
              s_sh_stride * slice_col + threadIdx.x;
          zp_gl_rd =
              zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
              zp_sh_stride * slice_col + threadIdx.x;
        } else {
          s_gl_rd =
              s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                             threadIdx.x / s_sh_stride) +
              s_sh_stride * slice_col + threadIdx.x % s_sh_stride;
          zp_gl_rd =
              zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                              threadIdx.x / zp_sh_stride) +
              zp_sh_stride * slice_col + threadIdx.x % zp_sh_stride;
        }
        start_pipes();
      }
    }
  }
}

#define MP_TEMPLATE_HEAD                                                     \
  template <const vllm::ScalarTypeId a_type_id,                              \
            const vllm::ScalarTypeId b_type_id,                              \
            const vllm::ScalarTypeId c_type_id,                              \
            const vllm::ScalarTypeId s_type_id, const int threads,           \
            const int thread_m_blocks, const int thread_n_blocks,            \
            const int thread_k_blocks, const bool m_block_size_8,            \
            const int stages, const int group_blocks, const bool is_zp_float>
#define MP_TEMPLATE_ARGS                                                   \
  a_type_id, b_type_id, c_type_id, s_type_id, threads, thread_m_blocks,    \
      thread_n_blocks, thread_k_blocks, m_block_size_8, stages,            \
      group_blocks, is_zp_float

// stock entry point: one GEMM per launch
MP_TEMPLATE_HEAD
__global__ void Marlin(MARLIN_KERNEL_PARAMS) {
  MarlinBody<MP_TEMPLATE_ARGS>(A, B, C, C_tmp, b_bias_ptr, a_scales_ptr,
                               scales_ptr, global_scale_ptr, zp_ptr, prob_m,
                               prob_n, prob_k, lda, locks, has_bias,
                               use_atomic_add, use_fp32_reduce, max_shared_mem,
                               A_src, lda_src, grid_bar, a_gscale_ptr, epi);
}

// P1 stage 2: a chain of dependent GEMMs in one persistent launch. Job j+1's A
// may be job j's C; a grid barrier separates the jobs, and the launch /
// pipeline fill / drain between the GEMMs is gone. All jobs share one tile
// config (template) and the same M.
MP_TEMPLATE_HEAD
__global__ void MarlinChain(MpJobs jobs, int njobs, int4* __restrict__ C_tmp,
                            int* locks, int* bar, bool use_fp32_reduce,
                            int max_shared_mem) {
  for (int j = 0; j < njobs; j++) {
    const MpJob& jb = jobs.j[j];
    if (jb.type == 0) {
      MarlinBody<MP_TEMPLATE_ARGS>(
          jb.A, jb.B, jb.C, C_tmp, nullptr, jb.a_s, jb.s, nullptr, nullptr,
          jb.m, jb.n, jb.k, jb.lda, locks, false, false, use_fp32_reduce,
          max_shared_mem, nullptr, 0, nullptr, nullptr, MpEpi{});
    } else if (jb.type == 1) {
      mp_norm_phase<threads>(jb.A, jb.lda, jb.B, jb.C, jb.aux,
                             reinterpret_cast<const __nv_bfloat16*>(jb.w), jb.m,
                             jb.k, jb.eps);
    } else if (jb.type == 2) {
      mp_silu_phase<threads>(jb.A, jb.lda, jb.C, jb.m, jb.n);
    }
    if (j + 1 < njobs) mp_grid_barrier(bar);
  }
}

}  // namespace MARLIN_NAMESPACE_NAME

#endif
