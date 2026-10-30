// declaration of the W3 kernel template (defined in gen/marlin_template_w3.h, instantiated in gen/w3_kernels_*.cu)
#pragma once
namespace MARLIN_NAMESPACE_NAME {
template <const vllm::ScalarTypeId a_type_id, const vllm::ScalarTypeId b_type_id,
          const vllm::ScalarTypeId c_type_id, const vllm::ScalarTypeId s_type_id,
          const int threads, const int thread_m_blocks, const int thread_n_blocks,
          const int thread_k_blocks, const bool m_block_size_8, const int stages,
          const int group_blocks, const bool is_zp_float>
__global__ void MarlinW3(MARLIN_KERNEL_PARAMS);
}
