// declaration of the W4A4 kernel template (definition: mp_a4.cuh)
#pragma once
namespace MARLIN_NAMESPACE_NAME {
template <int threads, int thread_m_blocks, int thread_n_blocks, int stages,
          int MW, int DEFER, int MINB>
__global__ void MarlinA4(MARLIN_KERNEL_PARAMS);
}
