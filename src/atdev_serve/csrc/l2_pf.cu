// L2 prefetch of indexed slot regions (ATDEV_GDN_L2PF): one thread per 128-byte line issues
// prefetch.global.L2::evict_last for  base + idx[0] * stride + [0, len)  of up to 24 regions; the slot
// index is read on the device (CUDA-graph safe), idx <= 0 (null / padding slot) skips the region, a null
// idx pointer means slot 0 offset. Pure hint: no data is written -> consumers stay bit-identical.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

struct Reg {
  const char* base;
  const int* idx;
  long long stride;
  long long lines;
};

struct Regs {
  Reg r[24];
  int n;
  int mode;   // 0: prefetch.global.L2::evict_last, 1: ld.global.cg (data really pulled into L2)
  int* sink;  // mode 1: never written in practice (keeps the loads alive)
};

__global__ void l2pf_kernel(Regs R) {
  long long line = (long long)blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
  for (int i = 0; i < 24; ++i) {
    if (i >= R.n) return;
    if (line < R.r[i].lines) {
      long long s = 0;
      if (R.r[i].idx != nullptr) {
        s = (long long)__ldg(R.r[i].idx);
        if (s <= 0) return;
      }
      const char* p = R.r[i].base + s * R.r[i].stride + (line << 7);
      if (R.mode == 0) {
        asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p));
      } else {
        // one 4-byte load per 32-byte sector (L2 fills sectors on demand)
        unsigned v0, v1, v2, v3;
        asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(v0) : "l"(p));
        asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(v1) : "l"(p + 32));
        asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(v2) : "l"(p + 64));
        asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(v3) : "l"(p + 96));
        unsigned v = v0 ^ v1 ^ v2 ^ v3;
        if (v == 0x7f7f7f7fu && line == 0x7fffffffLL) *R.sink = (int)v;
      }
      return;
    }
    line -= R.r[i].lines;
  }
}

}  // namespace

// base / idx: data_ptr ints (idx 0 = none); stride / len in bytes
void l2pf(std::vector<int64_t> base, std::vector<int64_t> idx, std::vector<int64_t> stride,
          std::vector<int64_t> len, int64_t threads, int64_t mode, int64_t sink) {
  TORCH_CHECK(base.size() >= 1 && base.size() <= 24 && idx.size() == base.size() &&
              stride.size() == base.size() && len.size() == base.size(), "l2pf: 1..24 regions");
  Regs R;
  R.n = (int)base.size();
  R.mode = (int)mode;
  R.sink = reinterpret_cast<int*>(sink);
  long long tot = 0;
  for (int i = 0; i < R.n; ++i) {
    R.r[i].base = reinterpret_cast<const char*>(base[i]);
    R.r[i].idx = reinterpret_cast<const int*>(idx[i]);
    R.r[i].stride = stride[i];
    R.r[i].lines = (len[i] + 127) >> 7;
    tot += R.r[i].lines;
  }
  if (tot == 0) return;
  int t = (int)threads;
  long long blocks = (tot + t - 1) / t;
  l2pf_kernel<<<(unsigned)blocks, t, 0, at::cuda::getCurrentCUDAStream()>>>(R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("l2pf", &l2pf, "L2 prefetch of indexed slot regions"); }
