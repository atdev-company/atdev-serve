
#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

#include "marlin.cuh"
#include "marlin_dtypes.cuh"
#include "core/scalar_type.hpp"

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
      const float *__restrict__ a_gscale_ptr

#define MP_MAX_CHAIN 8

namespace MARLIN_NAMESPACE_NAME {
// type 0: GEMM (A [m,k] -> C [m,n], B/s Marlin weights, a_s int8 row scales)
// type 1: residual add + GemmaRMSNorm: A = h [m,k] bf16, B = residual in (or
//         null), C = x out bf16 [m,k], aux = residual out bf16, w = weight bf16 [k]
// type 2: SiLU(g)*u: A = gate_up [m, 2n] bf16 (lda), C = out [m, n] bf16
struct MpJob {
  const int4* A;
  const int4* B;
  int4* C;
  const int4* s;
  const float* a_s;
  int m, n, k, lda;
  int type;
  int4* aux;
  const void* w;
  float eps;
};
struct MpJobs {
  MpJob j[MP_MAX_CHAIN];
};

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

template <const vllm::ScalarTypeId a_type_id, const vllm::ScalarTypeId b_type_id,
          const vllm::ScalarTypeId c_type_id, const vllm::ScalarTypeId s_type_id,
          const int threads, const int thread_m_blocks, const int thread_n_blocks,
          const int thread_k_blocks, const bool m_block_size_8, const int stages,
          const int group_blocks, const bool is_zp_float>
__global__ void MarlinChain(MpJobs jobs, int njobs, int4* __restrict__ C_tmp,
                            int* locks, int* bar, bool use_fp32_reduce,
                            int max_shared_mem);

}
