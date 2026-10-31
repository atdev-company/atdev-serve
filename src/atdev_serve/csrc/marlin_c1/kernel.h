
#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

#include "marlin.cuh"
#include "marlin_dtypes.cuh"
#include "core/scalar_type.hpp"

// marlin_c1: fused epilogues for the c=1 W4A16 verify batch (8 < M <= 16)
//   mode 0: stock
//   mode 1: producer -- x = fp32(bf16(acc)) + fp32(R); C2 = bf16(x) (residual out); C = bf16(x * (1 + W))
//           (Gemma-norm input without the row rstd); P[row * T + tile] = per-tile sum of x^2 (fixed order)
//   mode 2: consumer -- acc[row] *= rstd[row], rstd = rsqrt(sum_t Pin[row * T_in + t] / n_norm + eps)
//   mode 3: consumer + SiLU pair: mode 2, then the tile's two 64-column halves [g | u] -> C = bf16(silu(g) * u)
//           (C is [M, N / 2]; needs the gate/up weight columns interleaved per 64 and thread_n = 128)
//   modes 2/3 optionally write the row rstd to Rout (fp32 [16]) for another consumer (in_proj_ba)
struct MpEpi {
  int mode;
  const int4* R;
  const int4* W;
  int4* C2;
  float* P;
  const float* Pin;
  int T_in;
  int n_norm;
  float eps;
  float* Rout;  // consumer: optional rstd[16] out (written by CTA 0) for a second consumer of the same A
};

#define MARLIN_KERNEL_PARAMS                                               \
  const int4 *__restrict__ A, const int4 *__restrict__ B,                  \
      int4 *__restrict__ C, int4 *__restrict__ C_tmp,                      \
      const int4 *__restrict__ b_bias_ptr,                                 \
      const float *__restrict__ a_scales_ptr,                              \
      const int4 *__restrict__ scales_ptr,                                 \
      const float *__restrict__ global_scale_ptr,                          \
      const int4 *__restrict__ zp_ptr, int prob_m, int prob_n, int prob_k, \
      int lda, int *locks, bool has_bias, bool use_atomic_add,             \
      bool use_fp32_reduce, int max_shared_mem,                            \
      const int4 *__restrict__ A_src, int lda_src, int *grid_bar,          \
      const float *__restrict__ a_gscale_ptr, MpEpi epi

namespace MARLIN_NAMESPACE_NAME {
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
__global__ void Marlin(MARLIN_KERNEL_PARAMS);

}
