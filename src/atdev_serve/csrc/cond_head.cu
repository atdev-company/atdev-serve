// E5-(a): sync-free exact verify head. Parent CUDA graph:
//   child(G1: certified screen -> logits_out, need flag) -> set_cond(need) -> IF(cond) { child(G2: stock head
//   -> logits_out) }
// so the stock bf16 lm_head GEMM runs only when some row lacks a strict certificate, without a host sync.
// G1 / G2 are torch CUDAGraphs captured with keep_graph=True (raw cudaGraph_t passed in).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

__global__ void set_cond_kernel(cudaGraphConditionalHandle h, const int* need) {
  if (threadIdx.x == 0) cudaGraphSetConditional(h, *need != 0 ? 1u : 0u);
}

struct Entry {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
};
std::vector<Entry> g_entries;

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    TORCH_CHECK(e_ == cudaSuccess, #x " failed: ", cudaGetErrorString(e_));            \
  } while (0)

}  // namespace

int64_t build(int64_t g1_raw, int64_t g2_raw, torch::Tensor need) {
  TORCH_CHECK(need.is_cuda() && need.scalar_type() == at::kInt);
  cudaGraph_t g;
  CK(cudaGraphCreate(&g, 0));
  cudaGraphConditionalHandle h;
  CK(cudaGraphConditionalHandleCreate(&h, g, 0, cudaGraphCondAssignDefault));
  cudaGraphNode_t n1;
  CK(cudaGraphAddChildGraphNode(&n1, g, nullptr, 0, reinterpret_cast<cudaGraph_t>(g1_raw)));
  const int* needp = need.data_ptr<int>();
  void* args[2] = {&h, &needp};
  cudaKernelNodeParams kp{};
  kp.func = reinterpret_cast<void*>(set_cond_kernel);
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(32);
  kp.sharedMemBytes = 0;
  kp.kernelParams = args;
  kp.extra = nullptr;
  cudaGraphNode_t nk;
  CK(cudaGraphAddKernelNode(&nk, g, &n1, 1, &kp));
  cudaGraphNodeParams cp{};
  cp.type = cudaGraphNodeTypeConditional;
  cp.conditional.handle = h;
  cp.conditional.type = cudaGraphCondTypeIf;
  cp.conditional.size = 1;
  cudaGraphNode_t nc;
  CK(cudaGraphAddNode(&nc, g, &nk, nullptr, 1, &cp));
  cudaGraph_t body = cp.conditional.phGraph_out[0];
  cudaGraphNode_t n2;
  CK(cudaGraphAddChildGraphNode(&n2, body, nullptr, 0, reinterpret_cast<cudaGraph_t>(g2_raw)));
  cudaGraphExec_t ex;
  CK(cudaGraphInstantiate(&ex, g, 0));
  g_entries.push_back({g, ex});
  return (int64_t)g_entries.size() - 1;
}

void launch(int64_t id) {
  TORCH_CHECK(id >= 0 && id < (int64_t)g_entries.size());
  CK(cudaGraphLaunch(g_entries[id].exec, at::cuda::getCurrentCUDAStream()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build", &build);
  m.def("launch", &launch);
}
