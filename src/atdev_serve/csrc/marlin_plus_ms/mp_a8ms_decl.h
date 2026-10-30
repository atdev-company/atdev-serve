// declaration of the M-split W4A8 kernel template (definition: mp_a8ms.cuh)
#pragma once
namespace MARLIN_NAMESPACE_NAME {
template <int threads, int thread_m_blocks, int thread_n_blocks,
          int thread_k_blocks, int stages, int MW, int DEFER, int MINB, int U8>
__global__ void MarlinA8MS(MARLIN_KERNEL_PARAMS);
}
