// declaration of the W4A4 kernel template (definition: mp_a4.cuh)
#pragma once
#include "mp_a4_ne.h"
namespace MARLIN_NAMESPACE_NAME {
template <int threads, int thread_m_blocks, int thread_n_blocks, int stages,
          int MW, int DEFER, int MINB>
__global__ void MarlinA4(MARLIN_KERNEL_PARAMS A4_XPARAMS);
}
namespace MARLIN_NAMESPACE_NAME {
template <int threads, int thread_m_blocks, int thread_n_blocks, int MW>
__global__ void A4Reduce(const int4* __restrict__ C_tmp, int4* __restrict__ C, int prob_m, int prob_n,
                         const int* __restrict__ cols, int par_ms);
}
namespace MARLIN_NAMESPACE_NAME {
template <int threads_c, int thread_m_blocks, int thread_n_blocks, int stages, int MW, int MINB>
__global__ void MarlinA4WS(MARLIN_KERNEL_PARAMS);
}
