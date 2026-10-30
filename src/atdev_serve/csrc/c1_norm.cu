// E6 (docs/research/E6-c1-norms.md): low-latency residual-add + Gemma RMSNorm and SiLU*mul for the c=1
// verify batch (M = 16 rows, hidden 5120, MLP intermediate 17408).
//
// The Inductor kernels at these shapes are latency bound (one program per row, two passes over the row,
// 3 loop iterations each). Here one CTA owns a row and every thread holds 8 contiguous elements in
// registers (N / 8 threads; 640 for N = 5120): one 16-B load per input, one fixed-order reduction, one
// 16-B store per output.
//
// Numerics follow the Inductor graph of vllm.ir fused_add_rms_norm (fp32 sum, no bf16 rounding of the
// fp32 residual carried between the two norms of a piece, Gemma weight (w + 1) in fp32):
//   x   = a + (b + c)            (HAS_C)   or   a + b
//   res = bf16(x)                (STORE_RES)
//   out = bf16(x * rsqrt(sum(x^2) / N + eps) * (w + 1))
// Only the fp32 order of sum(x^2) differs (fixed: per-thread sequential, xor-butterfly per warp, warps in
// index order) -> deterministic, not bit-identical to Inductor.
//
// Split variants (bench only, ATDEV E6 microbench): S CTAs per row; partials combined with an arrival
// counter, either all CTAs wait (spin) or the last arriver normalises the whole row (no spin).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ void unpack8(const uint4& v, float* f) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 t = __bfloat1622float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

__device__ __forceinline__ uint4 pack8(const float* f) {
  uint4 v;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  return v;
}

__device__ __forceinline__ uint4 ldg_nc(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// fixed-order block reduction; returns the total in every thread
template <int NW>
__device__ __forceinline__ float block_sum(float v, float* sm) {
  v = warp_sum(v);
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) sm[wid] = v;
  __syncthreads();
  if (wid == 0) {
    float t = lane < NW ? sm[lane] : 0.f;
    t = warp_sum(t);
    if (lane == 0) sm[32] = t;
  }
  __syncthreads();
  return sm[32];
}

// one barrier: every thread sums the warp partials in index order (same order as block_sum's lane loop
// is NOT guaranteed -- different but fixed)
template <int NW>
__device__ __forceinline__ float block_sum1(float v, float* sm) {
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int k = 0; k < NW; ++k) t += sm[k];
  return t;
}

template <bool HAS_C>
__device__ __forceinline__ void load_x(const bf16* a, const bf16* b, const bf16* c, size_t off, float* x) {
  float fa[8], fb[8];
  unpack8(ldg_nc(a + off), fa);
  unpack8(ldg_nc(b + off), fb);
  if (HAS_C) {
    float fc[8];
    unpack8(ldg_nc(c + off), fc);
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = fa[i] + (fb[i] + fc[i]);
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = fa[i] + fb[i];
  }
}

// one CTA per row, NT = N / 8 threads
template <int NT, bool HAS_C, bool STORE_RES>
__global__ void __launch_bounds__(NT) add_norm_row(const bf16* __restrict__ a, const bf16* __restrict__ b,
                                                   const bf16* __restrict__ c, const bf16* __restrict__ w,
                                                   bf16* __restrict__ out, bf16* __restrict__ res, int lda,
                                                   int ldb, int ldc, int ldo, int ldr, float eps) {
  constexpr int N = NT * 8;
  constexpr int NW = (NT + 31) / 32;
  __shared__ float sm[33];
  const int row = blockIdx.x;
  const int col = threadIdx.x * 8;
  float x[8], fw[8];
  load_x<HAS_C>(a + (size_t)row * lda, b + (size_t)row * ldb, HAS_C ? c + (size_t)row * ldc : nullptr, col, x);
  unpack8(ldg_nc(w + col), fw);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) s = fmaf(x[i], x[i], s);
#ifdef E6_ONE_BARRIER
  const float tot = block_sum1<NW>(s, sm);
#else
  const float tot = block_sum<NW>(s, sm);
#endif
  const float r = rsqrtf(tot / (float)N + eps);
  if (STORE_RES) *reinterpret_cast<uint4*>(res + (size_t)row * ldr + col) = pack8(x);
  float y[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) y[i] = (x[i] * r) * (fw[i] + 1.f);
  *reinterpret_cast<uint4*>(out + (size_t)row * ldo + col) = pack8(y);
}

// S CTAs per row, NT = N / (8 S) threads each. cnt[row] grows by S per launch (launches are stream-ordered).
// part layout [row][S]. SPIN: every CTA waits for the row's S partials and normalises its own chunk.
// !SPIN: the last arriving CTA normalises the whole row (re-reads the inputs, L2-hot).
template <int NT, int S, bool HAS_C, bool STORE_RES, bool SPIN>
__global__ void __launch_bounds__(NT) add_norm_split(const bf16* __restrict__ a, const bf16* __restrict__ b,
                                                     const bf16* __restrict__ c, const bf16* __restrict__ w,
                                                     bf16* __restrict__ out, bf16* __restrict__ res,
                                                     unsigned* cnt, float* part, int ld, float eps) {
  constexpr int N = NT * 8 * S;
  constexpr int NW = (NT + 31) / 32;
  __shared__ float sm[33];
  __shared__ unsigned s_old;
  const int row = blockIdx.x / S, sl = blockIdx.x % S;
  const int col = sl * NT * 8 + threadIdx.x * 8;
  const bf16* ar = a + (size_t)row * ld;
  const bf16* br = b + (size_t)row * ld;
  const bf16* cr = HAS_C ? c + (size_t)row * ld : nullptr;
  float x[8];
  load_x<HAS_C>(ar, br, cr, col, x);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) s = fmaf(x[i], x[i], s);
  const float ps = block_sum<NW>(s, sm);
  if (STORE_RES && SPIN) *reinterpret_cast<uint4*>(res + (size_t)row * ld + col) = pack8(x);
  if (threadIdx.x == 0) {
    volatile float* vp = part + row * S + sl;
    *vp = ps;
    __threadfence();
    s_old = atomicAdd(cnt + row, 1u);
  }
  __syncthreads();
  const unsigned old = s_old;
  const unsigned target = (old / S + 1) * S;
  if (SPIN) {
    if (threadIdx.x == 0) {
      volatile unsigned* vc = cnt + row;
      while (*vc < target) {
      }
      __threadfence();
      float t = 0.f;
      volatile float* vp = part + row * S;
#pragma unroll
      for (int k = 0; k < S; ++k) t += vp[k];
      sm[32] = t;
    }
    __syncthreads();
    const float r = rsqrtf(sm[32] / (float)N + eps);
    float fw[8], y[8];
    unpack8(ldg_nc(w + col), fw);
#pragma unroll
    for (int i = 0; i < 8; ++i) y[i] = (x[i] * r) * (fw[i] + 1.f);
    *reinterpret_cast<uint4*>(out + (size_t)row * ld + col) = pack8(y);
  } else {
    if (old + 1 != target) return;  // not the last arriver of this row
    __threadfence();
    if (threadIdx.x == 0) {
      float t = 0.f;
      volatile float* vp = part + row * S;
#pragma unroll
      for (int k = 0; k < S; ++k) t += vp[k];
      sm[32] = t;
    }
    __syncthreads();
    const float r = rsqrtf(sm[32] / (float)N + eps);
#pragma unroll
    for (int k = 0; k < S; ++k) {
      const int cc = k * NT * 8 + threadIdx.x * 8;
      float xx[8], fw[8], y[8];
      load_x<HAS_C>(ar, br, cr, cc, xx);  // (plain ld.nc is safe: inputs are read-only in this kernel)
      unpack8(ldg_nc(w + cc), fw);
      if (STORE_RES) *reinterpret_cast<uint4*>(res + (size_t)row * ld + cc) = pack8(xx);
#pragma unroll
      for (int i = 0; i < 8; ++i) y[i] = (xx[i] * r) * (fw[i] + 1.f);
      *reinterpret_cast<uint4*>(out + (size_t)row * ld + cc) = pack8(y);
    }
  }
}

// y[m, j] = bf16(silu(g) * u), g = gu[m, j], u = gu[m, N + j]; silu(g) = g / (1 + exp(-g)) in fp32 (Inductor's
// formula, no bf16 rounding of silu(g)). 8 outputs per thread.
__global__ void __launch_bounds__(256) silu_mul8(const bf16* __restrict__ gu, bf16* __restrict__ y, int N,
                                                 int ldg, int ldy, int total8) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= total8) return;
  const int n8 = N / 8;
  const int m = t / n8, j = (t % n8) * 8;
  float g[8], u[8], o[8];
  unpack8(ldg_nc(gu + (size_t)m * ldg + j), g);
  unpack8(ldg_nc(gu + (size_t)m * ldg + N + j), u);
#pragma unroll
  for (int i = 0; i < 8; ++i) o[i] = (g[i] / (1.f + expf(-g[i]))) * u[i];
  *reinterpret_cast<uint4*>(y + (size_t)m * ldy + j) = pack8(o);
}

__global__ void empty_kernel() {}

// floor probe: row-CTA, load a, b -> store a + b (no reduction, no barrier)
__global__ void __launch_bounds__(640) copy2_row(const bf16* __restrict__ a, const bf16* __restrict__ b,
                                                 bf16* __restrict__ out) {
  const size_t off = (size_t)blockIdx.x * 5120 + threadIdx.x * 8;
  float x[8];
  load_x<false>(a, b, nullptr, off, x);
  *reinterpret_cast<uint4*>(out + off) = pack8(x);
}

#define CHECK_BF16(t) TORCH_CHECK((t).is_cuda() && (t).scalar_type() == at::kBFloat16 && (t).stride(-1) == 1, #t)

template <int NT>
void launch_row(const torch::Tensor& a, const torch::Tensor& b, const c10::optional<torch::Tensor>& c,
                const torch::Tensor& w, torch::Tensor& out, const c10::optional<torch::Tensor>& res, double eps) {
  const int M = a.size(0);
  auto st = at::cuda::getCurrentCUDAStream();
  const bf16* pa = reinterpret_cast<const bf16*>(a.data_ptr());
  const bf16* pb = reinterpret_cast<const bf16*>(b.data_ptr());
  const bf16* pc = c ? reinterpret_cast<const bf16*>(c->data_ptr()) : nullptr;
  const bf16* pw = reinterpret_cast<const bf16*>(w.data_ptr());
  bf16* po = reinterpret_cast<bf16*>(out.data_ptr());
  bf16* pr = res ? reinterpret_cast<bf16*>(res->data_ptr()) : nullptr;
  const int ldc = c ? c->stride(0) : 0, ldr = res ? res->stride(0) : 0;
#define L(HC, SR) add_norm_row<NT, HC, SR><<<M, NT, 0, st>>>(pa, pb, pc, pw, po, pr, a.stride(0), b.stride(0), ldc, out.stride(0), ldr, (float)eps)
  if (c) {
    if (res) L(true, true); else L(true, false);
  } else {
    if (res) L(false, true); else L(false, false);
  }
#undef L
}

}  // namespace

// out = norm(a + (b [+ c])) ; res = bf16 of the fp32 sum (optional). All [M, N] bf16 rows contiguous.
void add_norm(torch::Tensor a, torch::Tensor b, c10::optional<torch::Tensor> c, torch::Tensor w,
              torch::Tensor out, c10::optional<torch::Tensor> res, double eps) {
  CHECK_BF16(a); CHECK_BF16(b); CHECK_BF16(w); CHECK_BF16(out);
  if (c) CHECK_BF16(*c);
  if (res) CHECK_BF16(*res);
  const int N = a.size(1);
  TORCH_CHECK(w.numel() == N && w.is_contiguous());
  const int M = a.size(0);
  if (M == 0) return;
  switch (N) {
    case 5120: launch_row<640>(a, b, c, w, out, res, eps); break;
    case 4096: launch_row<512>(a, b, c, w, out, res, eps); break;
    case 2048: launch_row<256>(a, b, c, w, out, res, eps); break;
    default: TORCH_CHECK(false, "add_norm: unsupported N ", N);
  }
}

// bench-only split variant (N = 5120, contiguous rows of equal stride)
void add_norm_split_bench(torch::Tensor a, torch::Tensor b, c10::optional<torch::Tensor> c, torch::Tensor w,
                          torch::Tensor out, c10::optional<torch::Tensor> res, torch::Tensor cnt, torch::Tensor part,
                          int64_t S, bool spin, double eps) {
  const int M = a.size(0);
  TORCH_CHECK(a.size(1) == 5120);
  auto st = at::cuda::getCurrentCUDAStream();
  const bf16* pa = reinterpret_cast<const bf16*>(a.data_ptr());
  const bf16* pb = reinterpret_cast<const bf16*>(b.data_ptr());
  const bf16* pc = c ? reinterpret_cast<const bf16*>(c->data_ptr()) : nullptr;
  const bf16* pw = reinterpret_cast<const bf16*>(w.data_ptr());
  bf16* po = reinterpret_cast<bf16*>(out.data_ptr());
  bf16* pr = res ? reinterpret_cast<bf16*>(res->data_ptr()) : nullptr;
  unsigned* pcnt = reinterpret_cast<unsigned*>(cnt.data_ptr());
  float* pp = part.data_ptr<float>();
  const int ld = a.stride(0);
#define L(NT, SS, SP)                                                                                    \
  do {                                                                                                   \
    if (c && res) add_norm_split<NT, SS, true, true, SP><<<M * SS, NT, 0, st>>>(pa, pb, pc, pw, po, pr, pcnt, pp, ld, (float)eps); \
    else if (c) add_norm_split<NT, SS, true, false, SP><<<M * SS, NT, 0, st>>>(pa, pb, pc, pw, po, pr, pcnt, pp, ld, (float)eps); \
    else if (res) add_norm_split<NT, SS, false, true, SP><<<M * SS, NT, 0, st>>>(pa, pb, pc, pw, po, pr, pcnt, pp, ld, (float)eps); \
    else add_norm_split<NT, SS, false, false, SP><<<M * SS, NT, 0, st>>>(pa, pb, pc, pw, po, pr, pcnt, pp, ld, (float)eps); \
  } while (0)
  if (spin) {
    if (S == 2) L(320, 2, true); else if (S == 4) L(160, 4, true); else if (S == 5) L(128, 5, true);
    else if (S == 10) L(64, 10, true); else TORCH_CHECK(false, "S");
  } else {
    if (S == 2) L(320, 2, false); else if (S == 4) L(160, 4, false); else if (S == 5) L(128, 5, false);
    else if (S == 10) L(64, 10, false); else TORCH_CHECK(false, "S");
  }
#undef L
}

void silu_mul(torch::Tensor gu, torch::Tensor y) {
  CHECK_BF16(gu); CHECK_BF16(y);
  const int M = gu.size(0), N = y.size(1);
  TORCH_CHECK(gu.size(1) == 2 * N && N % 8 == 0 && y.size(0) == M);
  const int total8 = M * N / 8;
  if (total8 == 0) return;
  auto st = at::cuda::getCurrentCUDAStream();
  silu_mul8<<<(total8 + 255) / 256, 256, 0, st>>>(reinterpret_cast<const bf16*>(gu.data_ptr()),
                                                  reinterpret_cast<bf16*>(y.data_ptr()), N, gu.stride(0),
                                                  y.stride(0), total8);
}

void copy2_bench(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  copy2_row<<<a.size(0), 640, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(a.data_ptr()), reinterpret_cast<const bf16*>(b.data_ptr()),
      reinterpret_cast<bf16*>(out.data_ptr()));
}

void empty_launch() { empty_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_norm", &add_norm);
  m.def("add_norm_split_bench", &add_norm_split_bench);
  m.def("silu_mul", &silu_mul);
  m.def("empty_launch", &empty_launch);
  m.def("copy2_bench", &copy2_bench);
}
