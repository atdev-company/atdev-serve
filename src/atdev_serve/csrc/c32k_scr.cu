// C32k: CUDA replacement of draft_vocab._screen_chunk_kernel (Triton), bit-identical outputs.
// Triton semantics per program (row r, chunk c), S rounds over v = float(x[r, c*CH + j]) (masked: -inf, NaN -> -inf):
//   mx = max(v) ; i = min{offs : v == mx & ok} (none -> c*CH) ; store (mx, i) ; v[offs == i] = -inf
// Equivalent closed form used here:
//   * finite-or-+inf elements (v > -inf) are taken in order (value desc, index asc), +-0 compare equal
//     (IEEE == in the index pick), each one exactly once;
//   * once no such element remains: mx = -inf and i = c*CH (all ok elements are -inf then and ok is a prefix,
//     so the lowest ok index is c*CH; an all-masked chunk also gives c*CH);
//   * the stored mx of a zero pick is the max of the remaining zeros under max.f32 (+0 > -0): +0 while any
//     unpicked +0 remains, else -0 (the test checks this against the Triton kernel).
// One CTA per (row, chunk), NT threads, CH / NT elements per thread loaded as 16-byte vectors.
// Key (32 bit) = ordered bf16 bits (-0 canonicalised to +0) << 16 | (0xffff - chunk-local index); key 0 = empty.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

// 32-bit key: ordered bf16 bits (-0 -> +0) << 16 | (0xffff - chunk-local index); 0 = empty (NaN / -inf / masked)
__device__ __forceinline__ unsigned make_key(unsigned b, unsigned lidx) {
  if (b == 0x8000u) b = 0u;
  const unsigned o = (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u);
  const bool bad = ((b & 0x7f80u) == 0x7f80u && (b & 0x7fu)) || b == 0xff80u;
  return bad ? 0u : ((o << 16) | (0xffffu - lidx));
}
__device__ __forceinline__ float key_val(unsigned k) {
  const unsigned o = k >> 16;
  const unsigned b = (o & 0x8000u) ? (o & 0x7fffu) : (~o & 0xffffu);
  return __uint_as_float(b << 16);
}
__device__ __forceinline__ unsigned warp_max32(unsigned v) {
#pragma unroll
  for (int m = 16; m; m >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, m));
  return v;
}

template <int S, int CH, int NT, bool FAST>
__global__ void __launch_bounds__(NT) scr_chunk_kernel(const __nv_bfloat16* __restrict__ x, long sx,
                                                       float* __restrict__ cv, int* __restrict__ ci, int N, int NC) {
  constexpr int NW = NT / 32, VPT = CH / NT / 8;  // 16-byte vectors per thread
  static_assert(CH % (NT * 8) == 0 && NW * S <= 32 && CH <= 65536, "cfg");
  __shared__ unsigned sk[NW * S];
  __shared__ int spz[NW];
  const int r = blockIdx.x, c = blockIdx.y, tx = threadIdx.x, lane = tx & 31, w = tx >> 5;
  const int base = c * CH;
  unsigned t[S];
#pragma unroll
  for (int s = 0; s < S; ++s) t[s] = 0u;
  int npz = 0;  // +0 elements of this thread
  const __nv_bfloat16* xr = x + (long)r * sx;
  uint4 buf[VPT];
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int p = base + (j * NT + tx) * 8;
    if (p < N) buf[j] = __ldg(reinterpret_cast<const uint4*>(xr + p));
    else buf[j] = make_uint4(0xff80ff80u, 0xff80ff80u, 0xff80ff80u, 0xff80ff80u);  // -inf
  }
  if constexpr (FAST) {
    // Threshold path: T = 4th largest per-thread max (multiset over threads). The 4 threads with the largest maxima
    // hold 4 distinct elements >= T, so the chunk's top-4 values are all >= T and {x >= T} contains the exact top-4
    // incl. every tie at the boundary. Taken only when T > 0 (then no NaN / -inf / zero can be selected); otherwise
    // or on candidate overflow the block falls through to the exact path below (block-uniform branches).
    constexpr int CAP = 64;
    __shared__ unsigned st[NW * S];
    __shared__ unsigned sth, scnt;
    __shared__ unsigned cand[CAP];
    __nv_bfloat162 m2 = *reinterpret_cast<const __nv_bfloat162*>(&buf[0].x);
#pragma unroll
    for (int j = 0; j < VPT; ++j) {
      const unsigned u[4] = {buf[j].x, buf[j].y, buf[j].z, buf[j].w};
#pragma unroll
      for (int q = 0; q < 4; ++q) m2 = __hmax2(m2, *reinterpret_cast<const __nv_bfloat162*>(&u[q]));
    }
    const __nv_bfloat16 mh = __hmax(__low2bfloat16(m2), __high2bfloat16(m2));
    const unsigned tk = make_key(__bfloat16_as_ushort(mh), 0xffffu) >> 16;  // ordered bits, 0 = NaN / -inf
    unsigned kk = (tk << 16) | (0xffffu - (unsigned)tx);
    if (tk == 0u) kk = 0u;
#pragma unroll
    for (int s = 0; s < S; ++s) {
      const unsigned m = warp_max32(kk);
      if (kk == m) kk = 0u;
      if (lane == 0) st[w * S + s] = m;
    }
    if (tx == 0) scnt = 0u;
    __syncthreads();
    if (w == 0) {
      unsigned k = lane < NW * S ? st[lane] : 0u;
      unsigned m = 0u;
#pragma unroll
      for (int s = 0; s < S; ++s) {
        m = warp_max32(k);
        if (k == m) k = 0u;
      }
      if (lane == 0) sth = m >> 16;  // 4th largest thread max (0 if fewer than 4 valid)
    }
    __syncthreads();
    const unsigned T16 = sth;
    if (T16 > 0x8000u) {
      const unsigned short tb = (unsigned short)(T16 & 0x7fffu);
      const __nv_bfloat162 t2 = __halves2bfloat162(__ushort_as_bfloat16(tb), __ushort_as_bfloat16(tb));
#pragma unroll
      for (int j = 0; j < VPT; ++j) {
        const unsigned l0 = (unsigned)((j * NT + tx) * 8);
        const unsigned u[4] = {buf[j].x, buf[j].y, buf[j].z, buf[j].w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const unsigned msk = __hge2_mask(*reinterpret_cast<const __nv_bfloat162*>(&u[q]), t2);
          if (msk) {
            if (msk & 0xffffu) {
              const unsigned id = atomicAdd(&scnt, 1u);
              if (id < CAP) cand[id] = make_key(u[q] & 0xffffu, l0 + 2 * q);
            }
            if (msk >> 16) {
              const unsigned id = atomicAdd(&scnt, 1u);
              if (id < CAP) cand[id] = make_key(u[q] >> 16, l0 + 2 * q + 1);
            }
          }
        }
      }
      __syncthreads();
      const unsigned n = scnt;
      if (n <= CAP) {
        if (w == 0) {
          unsigned k0 = lane < n ? cand[lane] : 0u, k1 = lane + 32 < n ? cand[lane + 32] : 0u;
          if (k1 > k0) { const unsigned x_ = k0; k0 = k1; k1 = x_; }
          const long ob = ((long)r * NC + c) * S;
          float ov = 0.f;
          int oi = 0;
#pragma unroll
          for (int s = 0; s < S; ++s) {
            const unsigned m = warp_max32(k0);
            if (k0 == m) { k0 = k1; k1 = 0u; }
            if (lane == s) { ov = key_val(m); oi = base + (int)(0xffffu - (m & 0xffffu)); }
          }
          if (lane < S) { cv[ob + lane] = ov; ci[ob + lane] = oi; }
        }
        return;
      }
    }
  }
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const unsigned l0 = (unsigned)((j * NT + tx) * 8);
    const unsigned u[4] = {buf[j].x, buf[j].y, buf[j].z, buf[j].w};
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const unsigned b = (u[q >> 1] >> ((q & 1) * 16)) & 0xffffu;
      npz += (b == 0u);
      unsigned k = make_key(b, l0 + q);
#pragma unroll
      for (int s = 0; s < S; ++s) {  // branch-free sorted insert (descending)
        const unsigned hi = max(t[s], k);
        k = min(t[s], k);
        t[s] = hi;
      }
    }
  }
  // warp top-S
#pragma unroll
  for (int s = 0; s < S; ++s) {
    const unsigned m = warp_max32(t[0]);
    if (m != 0u && t[0] == m) {
#pragma unroll
      for (int q = 0; q < S - 1; ++q) t[q] = t[q + 1];
      t[S - 1] = 0u;
    }
    if (lane == 0) sk[w * S + s] = m;
  }
#pragma unroll
  for (int m = 16; m; m >>= 1) npz += __shfl_xor_sync(0xffffffffu, npz, m);
  if (lane == 0) spz[w] = npz;
  __syncthreads();
  if (w == 0) {
    int P = 0;
#pragma unroll
    for (int q = 0; q < NW; ++q) P += spz[q];
    unsigned k = lane < NW * S ? sk[lane] : 0u;
    const long ob = ((long)r * NC + c) * S;
    float ov = 0.f;
    int oi = 0;
#pragma unroll
    for (int s = 0; s < S; ++s) {
      const unsigned m = warp_max32(k);
      if (m != 0u && k == m) k = 0u;
      float v;
      int i;
      if (m == 0u) {
        v = __uint_as_float(0xff800000u);
        i = base;
      } else {
        i = base + (int)(0xffffu - (m & 0xffffu));
        v = key_val(m);
        if (v == 0.0f) {  // stored max of the remaining zeros under max.f32: +0 while an unpicked +0 remains
          const bool pos = __bfloat16_as_ushort(xr[i]) == 0;
          v = P > 0 ? 0.0f : -0.0f;
          P -= pos;
        }
      }
      if (lane == s) { ov = v; oi = i; }
    }
    if (lane < S) { cv[ob + lane] = ov; ci[ob + lane] = oi; }
  }
}

template <int S, int CH, int NT, bool FAST>
void launch(const torch::Tensor& x, torch::Tensor& cv, torch::Tensor& ci, int n, int nc) {
  const int m = x.size(0);
  dim3 grid(m, nc);
  scr_chunk_kernel<S, CH, NT, FAST><<<grid, NT, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0), cv.data_ptr<float>(), ci.data_ptr<int>(),
      n, nc);
}

}  // namespace

// nt > 0: exact path only; nt < 0: threshold fast path (|nt| threads) with in-kernel exact fallback
// x [M, >= N] bf16 (stride(0) % 8 == 0, 16-byte aligned), cv f32 / ci i32 [M, NC * S]
void scr_chunk(torch::Tensor x, torch::Tensor cv, torch::Tensor ci, int64_t n, int64_t nc, int64_t s, int64_t ch,
               int64_t nt) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.stride(0) % 8 == 0 && n % 8 == 0);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0);
  if (x.size(0) == 0) return;
#define C32K_L(S_, CH_, NT_) \
  if (s == S_ && ch == CH_ && nt == NT_) { launch<S_, CH_, NT_, false>(x, cv, ci, (int)n, (int)nc); return; } \
  if (s == S_ && ch == CH_ && nt == -NT_) { launch<S_, CH_, NT_, true>(x, cv, ci, (int)n, (int)nc); return; }
  C32K_L(4, 4096, 256)
  C32K_L(4, 4096, 128)
  C32K_L(4, 1024, 128)
  C32K_L(4, 1024, 64)
  C32K_L(4, 1024, 32)
#undef C32K_L
  TORCH_CHECK(false, "c32k scr_chunk: unsupported cfg");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("scr_chunk", &scr_chunk); }
