// CUDA tree-verify GDN kernel for the c=1 DFlash2 tree verify. Replaces, per GDN layer, the Triton
// _conv_tree_vec_kernel + _gdn_tree_kernel (+ the fused-norm ticket tail) of gdn_tree.py.
//
// Same slot / log / parity semantics as gdn_tree._gdn_tree_kernel (single-state layout, BV = 32):
//   h <- slot s0 rows [iv*32, iv*32+32) of value head hv (bf16), replay log entries 0..a-2,
//   tree pass (ancestor-masked WY form, N <= 16 nodes), outputs for all nodes,
//   slot s0 <- state after the root, log nodes 1..T-1 into buffer 1-p, parity flip.
//
// CTA = (value head hv, V-block iv), 128 threads; the 12 CTAs (3 value heads x 4 V-blocks) of a
// key head hk form a group with rank = (hv % G) * 4 + iv. Requires every CTA of the grid to be
// resident (1 request: 192 CTAs, 2 CTAs/SM) -- the host wrapper enforces R == 1.
//
// Structure:
//  * CONV: each CTA convolves only its own channels: its 2-3 chunks of 8 q/k channels of the key
//    head (32 chunks / 12 ranks) and its 32 v channels; it alone rewrites their conv state (no
//    cross-CTA hazard). The q/k slice is published to a global scratch with a release flag tagged
//    with the launch generation; warps 2-3 poll the 12 flags of the group and load the full q/k
//    rows while warps 0-1 replay the log (which does not need q/k).
//  * generation = (per-(request, key head) arrival counter read before this CTA's own arrival) / 12
//    + 1 -- every launch adds exactly 12, so no reset and no atomic round trip on the critical path.
//  * NORM: per-node partial sums of squares are exchanged as (value, generation) 8-byte stores
//    (single-copy atomic: no fence); the 4 V-block CTAs of a head poll each other's 16 pairs.
//  warps 0-1: state rows 16w..16w+15 in registers in the mma A-fragment layout: replay,
//             KSt = h k^T / QSt = h q^T (bf16x3 split of h, exact products, fp32 accumulate),
//             rhs, out = e^G QSt + rhs Z^T, partial sums, state commit, norm, store;
//  warp 2:    KK = k k^T (mma), X, P = (I - X)^-1 by forward substitution, Z = Pq P;
//  warp 3:    QK = q k^T (mma), Pq = M o ratio o QK, token log.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define GT_SPLITS 3  // bf16 pieces of the replayed fp32 state in KSt/QSt (~2^-24 relative)
#ifndef GT_BAP
// 1: b / a arrive as the UNSCALED fp32 in_proj_ba accumulator (c1_norm.ba_tot, computed on
// a side stream concurrently with the qkvz GEMM) plus the qkvz consumer's rstd; the kernel applies
// bf16(tot * rstd) itself (= c1_norm._ba_gemv_r_kernel's epilogue: fp32 mul RN, bf16 RN) -> bit-identical.
#define GT_BAP 0
#endif

namespace {

typedef __nv_bfloat16 bf16;

constexpr int kK = 128;
constexpr int kV = 128;
constexpr int kBV = 32;
constexpr int kNV = kV / kBV;
constexpr int kNP = 16;
constexpr int kThreads = 128;
constexpr int kPad = 136;   // bf16 pitch of [*, 128] tiles (272 B: conflict-free fragment loads)
constexpr int kCQK = 256;   // q + k channels of one key head
constexpr int kMaxSl = 24;  // max q/k slice channels per CTA (3 chunks of 8)
constexpr int kCX = kMaxSl + kBV;  // own conv channels: [slice | v]
constexpr int kXP = 64;     // bf16 pitch of the raw conv rows
constexpr int kRP = 20;     // fp32 pitch of the rhs tile
constexpr int kNPP = 20;  // fp32 pitch of XT / Pq / Z (20: rows 2t apart hit distinct banks)
constexpr int kOP = 40;  // bf16 pitch of the output tile

struct Args {
  const bf16* q; const bf16* k; const bf16* v; int64_t sq, sk, sv;
  const bf16* x; int64_t sx;
  const bf16* cw; const bf16* cb; bf16* cs; int64_t cs_slot, cs_dim, cs_tok; int cs_len;
  const int* csrc; int64_t csrc_req;
  const bf16* a; const bf16* b; int64_t sa, sb;
#if GT_BAP
  const float* af; const float* bf; const float* bar_p;  // fp32 unscaled b / a (strides sa / sb) + rstd [T]
#endif
  const void* A_log; int A_log_bf16; const void* dt_bias; int dt_bias_bf16;
  unsigned char* state; int64_t slot_bytes;
  const int* idx; int64_t idx_row; const int* cu; const int* acc;
  const int64_t* anc; int64_t anc_row;
  bf16* out; int64_t so;
  const bf16* z; int64_t sz; const void* nw; int nw_bf16;
  unsigned long long* cnt;  // [R * H] arrival counters
  float* part;              // [R * HV * 4 * 16] (value, gen) pairs as 2 x 32 bit + prof area
  bf16* qkx;                // [R * H][16][256] conv output exchange
  unsigned* qkf;            // [R * H][16] slice flags (gen)
  int H, HV, G, max_log;
  int64_t log_base, buf, v_off, d_off, b_off;  // fp32 elements inside slot s1
  float scale, eps, sp_beta, sp_thr;
};

struct __align__(16) Smem {
  bf16 h[kBV][kPad];
  bf16 q[kNP][kPad];
  bf16 k[kNP][kPad];
  float lk[kNP][kK];
  float lv[kNP][kBV];
  float XT[kNP][kNPP];
  float Pq[kNP][kNPP];
  float Z[kNP][kNPP];
  float rhs[kBV][kRP];
  bf16 x[kNP + 3][kXP];  // raw conv rows of the own channels; rows 16..18: window (oldest..newest)
  bf16 v[kNP][kBV];
  bf16 zg[kNP][kBV];
  bf16 o[kNP][kOP];
  float qs[kBV][12];  // QS of nodes 8..15 (warps 0-1 -> warps 2-3)
  float cw[kCX][4];
  float cb[kCX];
  float nw[kBV];
  float ld[kNP], lb[kNP];
  float rq[kNP], rk[kNP], gg[kNP], beta[kNP], G[kNP], eG[kNP], av[kNP], bv[kNP];
  float tot[kNV][kNP];
  float ss[2][kNP];
  unsigned anc[kNP];
  int srow[kNP][4];
  float Alog, dtb;
  unsigned gen;
  int swap;
};

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ unsigned long long ld_relaxed_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ int ld_meta(const int* p) {
  int v;
  asm volatile("ld.global.s32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ unsigned ld_acquire_u32(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_u32(unsigned* p, unsigned v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint2 ld_relaxed_v2(const uint2* p) {
  uint2 v;
  asm volatile("ld.relaxed.gpu.global.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_v2(uint2* p, uint2 v) {
  asm volatile("st.relaxed.gpu.global.v2.u32 [%0], {%1, %2};" ::"l"(p), "r"(v.x), "r"(v.y) : "memory");
}
__device__ __forceinline__ float rcp_approx(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float sigmoid_f(float x) { return rcp_approx(1.0f + __expf(-x)); }
__device__ __forceinline__ float ldf(const void* p, int i, int is_bf16) {
  return is_bf16 ? __bfloat162float(reinterpret_cast<const bf16*>(p)[i]) : reinterpret_cast<const float*>(p)[i];
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 t = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&t);
}
__device__ __forceinline__ uint32_t lds32(const bf16* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ float2 bf2f(uint32_t u) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u));
}

// D += A (16x16 bf16, row) * B (16x8 bf16, col), fp32 accumulate
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 16x16 node matrix C = A B^T over K = 128 (A, B row-major bf16 [16][kPad]); C-fragment layout of two
// n8 tiles: c[nt][e] = C[g + 8*(e>>1)][nt*8 + 2t + (e&1)]. Two accumulators per tile (even/odd k-steps).
__device__ __forceinline__ void node_mma(const bf16 (*A)[kPad], const bf16 (*B)[kPad], int g, int t,
                                         float (&c)[2][4]) {
  float c2[2][2][4];
#pragma unroll
  for (int u = 0; u < 2; ++u)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) c2[u][nt][e] = 0.f;
#pragma unroll
  for (int s = 0; s < kK / 16; ++s) {
    const int col = 16 * s + 2 * t;
    uint32_t a[4] = {lds32(&A[g][col]), lds32(&A[g + 8][col]), lds32(&A[g][col + 8]), lds32(&A[g + 8][col + 8])};
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
      mma16816(c2[s & 1][nt], a, lds32(&B[nt * 8 + g][col]), lds32(&B[nt * 8 + g][col + 8]));
  }
#pragma unroll
  for (int nt = 0; nt < 2; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) c[nt][e] = c2[0][nt][e] + c2[1][nt][e];
}

// global q/k channel (within the q / k halves of mixed_qkv) of the CTA's conv slice chunk
__device__ __forceinline__ int slice_chunk0(int rank) { return rank < 8 ? rank * 3 : 24 + (rank - 8) * 2; }
__device__ __forceinline__ int slice_nchunk(int rank) { return rank < 8 ? 3 : 2; }

template <bool CONV, bool NORM, bool SIGMOID, bool ACT>
__global__ __launch_bounds__(kThreads, 2) void gdn_tree_kernel(Args p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>(smem_raw);
  const int n = blockIdx.y;
  const int hv = blockIdx.x / kNV, iv = blockIdx.x % kNV;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int hk = hv / p.G;
  const int rank = (hv % p.G) * kNV + iv;
  const int vb = iv * kBV;
  // all metadata words in one round trip, issued before any branch (the compiler otherwise chains
  // cu -> exit test -> idx -> acc as three dependent global round trips)
  const int bos = ld_meta(p.cu + n);
  const int cu1 = ld_meta(p.cu + n + 1);
  const int s0 = ld_meta(p.idx + (int64_t)n * p.idx_row), s1 = ld_meta(p.idx + (int64_t)n * p.idx_row + 1);
  const int acc_pre = ld_meta(p.acc + n);
  const int T = cu1 - bos;
  // one branch consuming all five words -> the loads are issued together (acc > 2^31-16 never occurs)
  if (T <= 0 || s0 <= 0 || s1 <= 0 || T > p.max_log + 1 || T > kNP || acc_pre > 0x7ffffff0) {
    if (T > 0)
      for (int e = tid; e < T * kBV; e += kThreads)
        p.out[(int64_t)(bos + e / kBV) * p.so + hv * kV + vb + e % kBV] = __float2bfloat16(0.f);
    return;
  }
  if (s0 <= 0 || s1 <= 0 || T > p.max_log + 1 || T > kNP) {
    for (int e = tid; e < T * kBV; e += kThreads)
      p.out[(int64_t)(bos + e / kBV) * p.so + hv * kV + vb + e % kBV] = __float2bfloat16(0.f);
    return;
  }
  const int acc_n = acc_pre;
  int n_rep = acc_n - 1;
  n_rep = n_rep < 0 ? 0 : (n_rep > p.max_log ? p.max_log : n_rep);
  unsigned long long* cnt = p.cnt + (int64_t)n * p.H + hk;
  const int nb = p.G * kNV;  // CTAs per key-head group (12)
  const int ch0 = slice_chunk0(rank) * 8, nsl = slice_nchunk(rank) * 8;  // q/k slice [ch0, ch0+nsl)

  // ---------------- prologue: every independent load in flight at once
  unsigned char* hsrc = p.state + (int64_t)s0 * p.slot_bytes + ((int64_t)hv * kV + vb) * kK * 2;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int e = tid + r * kThreads;  // 512 16-B chunks
    cp_async_16(&s.h[e >> 4][(e & 15) * 8], hsrc + (e >> 4) * (kK * 2) + (e & 15) * 16);
  }
  if (CONV) {
    // own raw rows: slice chunks (x cols 0..23) + v chunks (x cols 24..55)
    if (tid < kNP * 7) {
      const int i = tid / 7, c = tid % 7;
      const int nslc = nsl / 8;
      if (c < nslc || c >= 3) {
        bf16* dst = c < 3 ? &s.x[i][c * 8] : &s.x[i][kMaxSl + (c - 3) * 8];
        if (i < T) {
          const bf16* row = p.x + (int64_t)(bos + i) * p.sx;
          const int qc = ch0 + c * 8;  // q/k channel 0..255
          const bf16* src = c < 3 ? (qc < 128 ? row + hk * kK + qc : row + p.H * kK + hk * kK + (qc - 128))
                                  : row + 2 * p.H * kK + hv * kV + vb + (c - 3) * 8;
          cp_async_16(dst, src);
        } else {
          *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
      }
    }
  } else {
    for (int e = tid; e < kNP * 36; e += kThreads) {
      const int i = e / 36, c = e % 36;
      void* dst = c < 16 ? (void*)&s.q[i][c * 8] : c < 32 ? (void*)&s.k[i][(c - 16) * 8] : (void*)&s.v[i][(c - 32) * 8];
      if (i < T) {
        const bf16* src = c < 16 ? p.q + (int64_t)(bos + i) * p.sq + hk * kK + c * 8
                        : c < 32 ? p.k + (int64_t)(bos + i) * p.sk + hk * kK + (c - 16) * 8
                                 : p.v + (int64_t)(bos + i) * p.sv + hv * kV + vb + (c - 32) * 8;
        cp_async_16(dst, src);
      } else {
        *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
      }
    }
  }
  if (NORM && tid < kNP * 4) {
    const int i = tid >> 2, c = tid & 3;
    if (i < T) cp_async_16(&s.zg[i][c * 8], p.z + (int64_t)(bos + i) * p.sz + hv * kV + vb + c * 8);
    else *reinterpret_cast<uint4*>(&s.zg[i][c * 8]) = make_uint4(0, 0, 0, 0);
  }
  // plain loads into registers first, smem stores afterwards
  float* lslot = reinterpret_cast<float*>(p.state + (int64_t)s1 * p.slot_bytes);
  int* parw = reinterpret_cast<int*>(lslot) + hv * kNV + iv;
  const int parv = *parw;
  unsigned long long t0 = 0;
  if ((CONV || NORM) && tid == 0) t0 = ld_relaxed_u64(cnt);
  float r_a = 0.f, r_b = 0.f, r_m0 = 0.f, r_m1 = 0.f, r_nw = 0.f;
  unsigned r_anc = 0u;
  int r_src = 0;
  if (tid < kNP) {
    if (tid < T) {
#if GT_BAP
      {
        const float rr = p.bar_p[bos + tid];
        r_a = __bfloat162float(__float2bfloat16_rn(__fmul_rn(p.af[(int64_t)(bos + tid) * p.sa + hv], rr)));
        r_b = __bfloat162float(__float2bfloat16_rn(__fmul_rn(p.bf[(int64_t)(bos + tid) * p.sb + hv], rr)));
      }
#else
      r_a = __bfloat162float(p.a[(int64_t)(bos + tid) * p.sa + hv]);
      r_b = __bfloat162float(p.b[(int64_t)(bos + tid) * p.sb + hv]);
#endif
      r_anc = (unsigned)p.anc[n * p.anc_row + tid];
    }
  } else if (tid < kNP + 48) {
    const int e = tid - kNP, i = e / 3, j = e % 3;
    if (CONV && i < T) r_src = p.csrc[n * p.csrc_req + i * 3 + j];
  } else if (tid < 96) {
    if (tid == 64) { r_m0 = ldf(p.A_log, hv, p.A_log_bf16); r_m1 = ldf(p.dt_bias, hv, p.dt_bias_bf16); }
  } else if (NORM) {
    r_nw = ldf(p.nw, vb + tid - 96, p.nw_bf16);
  }
  // conv window (3 columns at offset a-1) + weights of the own channels (thread = x column)
  bf16 r_win[3];
  uint2 r_w = make_uint2(0, 0);
  bf16 r_cb = __float2bfloat16(0.f);
  int chg = -1;
  if (CONV && tid < kCX) {
    if (tid < nsl) {
      const int qc = ch0 + tid;
      chg = qc < 128 ? hk * kK + qc : p.H * kK + hk * kK + (qc - 128);
    } else if (tid >= kMaxSl) {
      chg = 2 * p.H * kK + hv * kV + vb + (tid - kMaxSl);
    }
    if (chg >= 0) {
      const int off = acc_n - 1;
      const bf16* base = p.cs + (int64_t)s0 * p.cs_slot + (int64_t)chg * p.cs_dim;
#pragma unroll
      for (int j = 0; j < 3; ++j) r_win[j] = base[(int64_t)(off + j) * p.cs_tok];
      r_w = *reinterpret_cast<const uint2*>(p.cw + (int64_t)chg * 4);
      if (p.cb) r_cb = p.cb[chg];
    }
  }
  // replay log (dependent on the parity word)
  const int pb = n_rep > 0 ? (parv & 1) : 0;
  const float* rbuf = lslot + p.log_base + (int64_t)pb * p.buf;
  // the replay log (one more dependent round trip: parity -> log) is a separate cp.async group of warps 0-1
  // only, waited for just before the replay; the prologue barrier waits for the rest only
  cp_async_commit();
  float r_ld = 0.f, r_lb = 0.f;
  if (tid < 64) {
    for (int e = tid; e < n_rep * 40; e += 64) {
      const int j = e / 40, c = e % 40;
      if (c < 32) cp_async_16(&s.lk[j][c * 4], rbuf + (int64_t)(j * p.H + hk) * kK + c * 4);
      else cp_async_16(&s.lv[j][(c - 32) * 4], rbuf + p.v_off + (int64_t)(j * p.HV + hv) * kV + vb + (c - 32) * 4);
    }
    if (tid < n_rep) {
      r_ld = rbuf[p.d_off + tid * p.HV + hv];
      r_lb = rbuf[p.b_off + tid * p.HV + hv];
    }
  }
  cp_async_commit();
  // ---- smem stores of the plain loads
  if (tid < kNP) {
    s.av[tid] = r_a; s.bv[tid] = r_b; s.anc[tid] = r_anc;
  } else if (tid < kNP + 48) {
    const int e = tid - kNP, i = e / 3, j = e % 3;
    s.srow[i][j] = r_src >= 0 ? r_src : kNP + 3 + r_src;  // window rows 16..18 = oldest..newest
  } else if (tid == 64) {
    s.Alog = r_m0; s.dtb = r_m1;
  }
  if (tid == 0) {
    s.gen = (unsigned)(t0 / (unsigned long long)nb) + 1u;  // t0: before this CTA's arrival
    s.swap = 0;
  }
  if (tid >= 96) {
    if (NORM) s.nw[tid - 96] = r_nw;
  }
  if (CONV && chg >= 0) {
#pragma unroll
    for (int j = 0; j < 3; ++j) s.x[kNP + j][tid] = r_win[j];
    const float2 w01 = bf2f(r_w.x), w23 = bf2f(r_w.y);
    s.cw[tid][0] = w01.x; s.cw[tid][1] = w01.y; s.cw[tid][2] = w23.x; s.cw[tid][3] = w23.y;
    s.cb[tid] = __bfloat162float(r_cb);
  }
  asm volatile("cp.async.wait_group 1;\n" ::: "memory");  // all but the log group
  __syncthreads();
  // generation of this launch (t0 was read before this CTA's arrival; every launch adds exactly nb)
  const unsigned gen = s.gen;
  if ((CONV || NORM) && tid == 0) asm volatile("red.relaxed.gpu.global.add.u64 [%0], 1;" ::"l"(cnt) : "memory");


  // ---------------- conv of the own channels: unit = (channel pair, 4 nodes)
  if (CONV) {
    const int npair = kCX / 2;  // 28 pairs: 12 slice pairs (only nsl/2 live) + 16 v pairs
    if (tid < npair * 4) {
      const int cp = tid % npair, i0 = (tid / npair) * 4;
      const int c = 2 * cp;
      const bool live_c = c < nsl || c >= kMaxSl;
      if (live_c) {
        const float4 wa = *reinterpret_cast<const float4*>(&s.cw[c][0]);
        const float4 wb = *reinterpret_cast<const float4*>(&s.cw[c + 1][0]);
        const float ba = s.cb[c], bb = s.cb[c + 1];
        uint32_t xv[4][4];
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
          const int i = i0 + ii;
          const int4 sr = *reinterpret_cast<const int4*>(&s.srow[i][0]);
          xv[ii][0] = lds32(&s.x[i][c]);
          xv[ii][1] = lds32(&s.x[sr.x][c]);
          xv[ii][2] = lds32(&s.x[sr.y][c]);
          xv[ii][3] = lds32(&s.x[sr.z][c]);
        }
        uint32_t yo[4];
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
          const float2 f0 = bf2f(xv[ii][0]), f1 = bf2f(xv[ii][1]), f2 = bf2f(xv[ii][2]), f3 = bf2f(xv[ii][3]);
          float y0 = fmaf(wa.w, f0.x, ba), y1 = fmaf(wb.w, f0.y, bb);
          y0 = fmaf(wa.z, f1.x, y0); y1 = fmaf(wb.z, f1.y, y1);
          y0 = fmaf(wa.y, f2.x, y0); y1 = fmaf(wb.y, f2.y, y1);
          y0 = fmaf(wa.x, f3.x, y0); y1 = fmaf(wb.x, f3.y, y1);
          if (ACT) { y0 = y0 * sigmoid_f(y0); y1 = y1 * sigmoid_f(y1); }
          const bool live = i0 + ii < T;
          yo[ii] = pack_bf16(live ? y0 : 0.f, live ? y1 : 0.f);
        }
        if (c >= kMaxSl) {
#pragma unroll
          for (int ii = 0; ii < 4; ++ii) *reinterpret_cast<uint32_t*>(&s.v[i0 + ii][c - kMaxSl]) = yo[ii];
        } else {
          bf16* dst = p.qkx + ((int64_t)n * p.H + hk) * (kNP * kCQK) + ch0 + c;
#pragma unroll
          for (int ii = 0; ii < 4; ++ii) *reinterpret_cast<uint32_t*>(dst + (i0 + ii) * kCQK) = yo[ii];
          __threadfence();  // this writer's slice stores visible GPU-wide before the CTA's flag
        }
      }
    }
    __syncthreads();  // slice stores issued by all writers (observed by thread 0 at CTA scope)
    if (tid == 0) {
      st_release_u32(p.qkf + ((int64_t)n * p.H + hk) * 16 + rank, gen);
    }
    // conv state of the own channels (thread = channel): [h2, h3, x_0..x_{T-1}]
    if (chg >= 0) {
      bf16* base = p.cs + (int64_t)s0 * p.cs_slot + (int64_t)chg * p.cs_dim;
      const int ncol = 2 + T < p.cs_len ? 2 + T : p.cs_len;
      bf16 vals[kNP + 2];
#pragma unroll
      for (int cc = 0; cc < kNP + 2; ++cc) vals[cc] = cc < 2 ? s.x[kNP + 1 + cc][tid] : s.x[cc - 2][tid];
#pragma unroll
      for (int cc = 0; cc < kNP + 2; ++cc)
        if (cc < ncol) base[(int64_t)cc * p.cs_tok] = vals[cc];
    }
  }
  const int rw = warp ^ (s.swap ? 2 : 0);  // role warp
  const int rt = rw * 32 + lane;           // role thread

  if (rw >= 2) {
    // ================= warps 2-3: q/k rows of the group, norms, node parameters, node matrices
    if (rw == 3 && lane < kNP) {  // node parameters (need only the prologue's a / b / A_log / dt_bias / anc)
      const int i = lane;
      const float A = __expf(s.Alog);
      const float xx = s.av[i] + s.dtb;
      const float sp = p.sp_beta * xx <= p.sp_thr ? rcp_approx(p.sp_beta) * __logf(1.0f + __expf(p.sp_beta * xx)) : xx;
      const float gi = i < T ? -A * sp : 0.f;
      const float bi = i < T ? sigmoid_f(s.bv[i]) : 0.f;
      const unsigned an = s.anc[i];
      float Gi = 0.f;
#pragma unroll
      for (int j = 0; j < kNP; ++j) {
        const float gj = __shfl_sync(0x0000ffffu, gi, j, 16);
        Gi += ((an >> j) & 1u) ? gj : 0.f;
      }
      s.gg[i] = gi; s.beta[i] = bi; s.G[i] = Gi; s.eG[i] = __expf(Gi);
    }
    if (CONV) {
      if (rw == 2) {
        const unsigned* fl = p.qkf + ((int64_t)n * p.H + hk) * 16;
        bool ok = false;
        while (!ok) {
          const unsigned f = lane < nb ? ld_acquire_u32(fl + lane) : gen;
          ok = __all_sync(0xffffffffu, f == gen);
        }
      }
      bar_sync(6, 64);
      // q/k rows [16][256] -> s.q / s.k (all writers' stores are visible after rw 2's acquire + bar)
      const bf16* src = p.qkx + ((int64_t)n * p.H + hk) * (kNP * kCQK);
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int e = (rt - 64) + r * 64;  // 512 chunks of 8 channels
        const int i = e >> 5, c = e & 31;
        bf16* dst = c < 16 ? &s.q[i][c * 8] : &s.k[i][(c - 16) * 8];
        cp_async_16(dst, src + i * kCQK + c * 8);
      }
      cp_async_commit();
      cp_async_wait_all();
      bar_sync(6, 64);
      bar_arrive(4, 128);  // q / k rows in smem: warps 0-1 may start KSt / QSt
    }
    float c[2][4];
    // node matrices on the raw (un-normalised) q / k rows: independent of the l2 norms
    if (rw == 2) node_mma(s.k, s.k, g, t, c);
    else node_mma(s.q, s.k, g, t, c);
    {
      // l2 norms: rw 2 -> q rows, rw 3 -> k rows (16 each); reduce-scatter over the 32 lanes
      const bf16 (*M)[kPad] = rw == 2 ? s.q : s.k;
      float a[16];
#pragma unroll
      for (int r = 0; r < 16; ++r) {
        const uint2 u = *reinterpret_cast<const uint2*>(&M[r][lane * 4]);
        const float2 x01 = bf2f(u.x), x23 = bf2f(u.y);
        a[r] = x01.x * x01.x + x01.y * x01.y + x23.x * x23.x + x23.y * x23.y;
      }
      {
        // after the step with offset o (16, 8, 4, 2) the lane keeps the half of its rows selected by lane bit o
#define NRS_STEP(HW, O)                                                     \
  {                                                                         \
    const bool hi = (lane & (O)) != 0;                                      \
    _Pragma("unroll") for (int q = 0; q < (HW); ++q) {                      \
      const float keep = hi ? a[q + (HW)] : a[q];                           \
      const float send = hi ? a[q] : a[q + (HW)];                           \
      a[q] = keep + __shfl_xor_sync(0xffffffffu, send, (O));                \
    }                                                                       \
  }
        NRS_STEP(8, 16) NRS_STEP(4, 8) NRS_STEP(2, 4) NRS_STEP(1, 2)
#undef NRS_STEP
        a[0] += __shfl_xor_sync(0xffffffffu, a[0], 1);  // lane holds row (lane >> 1)
      }
      if ((lane & 1) == 0) {
        const float rs = rsqrtf(a[0] + 1e-6f);
        if (rw == 2) s.rq[lane >> 1] = rs; else s.rk[lane >> 1] = rs;
      }
    }
    bar_sync(6, 64);
    bar_arrive(5, 128);  // q/k, norms, node params ready for warps 0-1
    // ---------------- node matrices
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = g + 8 * (e >> 1), j = nt * 8 + 2 * t + (e & 1);
        const bool m = (s.anc[i] >> j) & 1u;
        const float ratio = !m ? 0.f : (i == j ? 1.f : __expf(s.G[i] - s.G[j]));
        if (rw == 2) {
          const float kk = c[nt][e] * (s.rk[i] * s.rk[j]);
          s.XT[j][i] = -((m && i != j) ? ratio * kk : 0.f) * s.beta[i];
        } else {
          const float qk = c[nt][e] * ((s.rq[i] * p.scale) * s.rk[j]);
          s.Pq[i][j] = m ? ratio * qk : 0.f;
        }
      }
    if (rw == 2) {
      __syncwarp();
      // P = (I - X)^-1, X strictly lower: P[i][j] = d_ij + sum_{m<i} X[i][m] P[m][j]; lane -> column j
      const int j = lane & 15;
      float pc[kNP], ac[kNP];
#pragma unroll
      for (int i = 0; i < kNP; ++i) ac[i] = 0.f;
#pragma unroll
      for (int m = 0; m < kNP; ++m) {
        pc[m] = (m == j ? 1.f : 0.f) + ac[m];
        float xr[kNP];
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
          const float4 v4 = *reinterpret_cast<const float4*>(&s.XT[m][q4 * 4]);
          xr[q4 * 4] = v4.x; xr[q4 * 4 + 1] = v4.y; xr[q4 * 4 + 2] = v4.z; xr[q4 * 4 + 3] = v4.w;
        }
#pragma unroll
        for (int i = m + 1; i < kNP; ++i) ac[i] = fmaf(xr[i], pc[m], ac[i]);
      }
      bar_sync(1, 64);  // Pq from rw 3
      // Z = Pq P: lane -> column j, rows 8*(lane>>4) .. +7
      const int i0 = (lane >> 4) * 8;
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int i = i0 + r;
        float za = 0.f, zb = 0.f;
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
          const float4 v4 = *reinterpret_cast<const float4*>(&s.Pq[i][q4 * 4]);
          if (q4 & 1) { zb = fmaf(v4.x, pc[q4 * 4], zb); zb = fmaf(v4.y, pc[q4 * 4 + 1], zb);
                        zb = fmaf(v4.z, pc[q4 * 4 + 2], zb); zb = fmaf(v4.w, pc[q4 * 4 + 3], zb); }
          else { za = fmaf(v4.x, pc[q4 * 4], za); za = fmaf(v4.y, pc[q4 * 4 + 1], za);
                 za = fmaf(v4.z, pc[q4 * 4 + 2], za); za = fmaf(v4.w, pc[q4 * 4 + 3], za); }
        }
        s.Z[i][j] = za + zb;
      }
      bar_arrive(2, 96);
      __syncwarp();
      bar_arrive(9, 64);
    } else {
      bar_arrive(1, 64);
      // ---------------- token log: nodes 1..T-1 -> entries 0..T-2 of buffer 1-p
      float* wbuf = lslot + p.log_base + (int64_t)(1 - pb) * p.buf;
      if (iv == 0 && (hv % p.G) == 0) {
        for (int i = 1; i < T; ++i) {
          const uint2 u = *reinterpret_cast<const uint2*>(&s.k[i][lane * 4]);
          const float2 x01 = bf2f(u.x), x23 = bf2f(u.y);
          const float r = s.rk[i];
          *reinterpret_cast<float4*>(wbuf + (int64_t)((i - 1) * p.H + hk) * kK + lane * 4) =
              make_float4(x01.x * r, x01.y * r, x23.x * r, x23.y * r);
        }
      }
      for (int i = 1; i < T; ++i)
        wbuf[p.v_off + (int64_t)((i - 1) * p.HV + hv) * kV + vb + lane] = __bfloat162float(s.v[i][lane]);
      if (iv == 0 && lane + 1 < T) {
        wbuf[p.d_off + lane * p.HV + hv] = __expf(s.gg[lane + 1]);
        wbuf[p.b_off + lane * p.HV + hv] = s.beta[lane + 1];
      }
      if (lane == 0) *parw = 1 - pb;
    }
    {
      if (rw == 3) bar_sync(9, 64);  // Z from rw 2
      bar_sync(8, 128);              // rhs + QS(nodes 8..15) from warps 0-1
      const int pw = rw - 2;
      const int r0 = 16 * pw + g, r1 = r0 + 8;
      float rr0[kNP], rr1[kNP];
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        const float4 a4 = *reinterpret_cast<const float4*>(&s.rhs[r0][q4 * 4]);
        const float4 b4 = *reinterpret_cast<const float4*>(&s.rhs[r1][q4 * 4]);
        rr0[q4 * 4] = a4.x; rr0[q4 * 4 + 1] = a4.y; rr0[q4 * 4 + 2] = a4.z; rr0[q4 * 4 + 3] = a4.w;
        rr1[q4 * 4] = b4.x; rr1[q4 * 4 + 1] = b4.y; rr1[q4 * 4 + 2] = b4.z; rr1[q4 * 4 + 3] = b4.w;
      }
      float ps[2];
#pragma unroll
      for (int c2 = 0; c2 < 2; ++c2) {
        const int i = 8 + 2 * t + c2;
        float z0 = 0.f, z1 = 0.f;
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
          const float4 zz = *reinterpret_cast<const float4*>(&s.Z[i][q4 * 4]);
          z0 = fmaf(rr0[q4 * 4], zz.x, z0); z0 = fmaf(rr0[q4 * 4 + 1], zz.y, z0);
          z0 = fmaf(rr0[q4 * 4 + 2], zz.z, z0); z0 = fmaf(rr0[q4 * 4 + 3], zz.w, z0);
          z1 = fmaf(rr1[q4 * 4], zz.x, z1); z1 = fmaf(rr1[q4 * 4 + 1], zz.y, z1);
          z1 = fmaf(rr1[q4 * 4 + 2], zz.z, z1); z1 = fmaf(rr1[q4 * 4 + 3], zz.w, z1);
        }
        const float eg = s.eG[i];
        const float o0 = __bfloat162float(__float2bfloat16(fmaf(eg, s.qs[r0][2 * t + c2], z0)));
        const float o1 = __bfloat162float(__float2bfloat16(fmaf(eg, s.qs[r1][2 * t + c2], z1)));
        s.o[i][r0] = __float2bfloat16(o0);
        s.o[i][r1] = __float2bfloat16(o1);
        ps[c2] = o0 * o0 + o1 * o1;
      }
      if (NORM) {
#pragma unroll
        for (int o = 4; o < 32; o <<= 1)
#pragma unroll
          for (int c2 = 0; c2 < 2; ++c2) ps[c2] += __shfl_xor_sync(0xffffffffu, ps[c2], o);
        if (g == 0) {
#pragma unroll
          for (int c2 = 0; c2 < 2; ++c2) s.ss[pw][8 + 2 * t + c2] = ps[c2];
        }
      }
      bar_arrive(10, 128);  // o / ss of nodes 8..15 ready
    }
    return;  // warps 2-3 done
  }

  // ================= warps 0-1: state rows r0 = 16*rw + g, r1 = r0 + 8
  const int r0 = 16 * rw + g, r1 = r0 + 8;
  float h[8][2][4];  // [k-step][row][col 2t, 2t+1, 2t+8, 2t+9]
#pragma unroll
  for (int st = 0; st < 8; ++st) {
    const int col = 16 * st + 2 * t;
    const float2 a0 = bf2f(lds32(&s.h[r0][col])), a1 = bf2f(lds32(&s.h[r0][col + 8]));
    const float2 b0 = bf2f(lds32(&s.h[r1][col])), b1 = bf2f(lds32(&s.h[r1][col + 8]));
    h[st][0][0] = a0.x; h[st][0][1] = a0.y; h[st][0][2] = a1.x; h[st][0][3] = a1.y;
    h[st][1][0] = b0.x; h[st][1][1] = b0.y; h[st][1][2] = b1.x; h[st][1][3] = b1.y;
  }
  cp_async_wait_all();
  if (tid < n_rep) { s.ld[tid] = r_ld; s.lb[tid] = r_lb; }
  bar_sync(7, 64);  // warps 0-1: log in smem
  // ---------------- replay (single-state order: decay, delta = (v - h k) beta, h += delta k^T)
  for (int j = 0; j < n_rep; ++j) {
    const float d = s.ld[j], bt = s.lb[j];
    const float v0 = s.lv[j][r0], v1 = s.lv[j][r1];
    float kv[8][4];
#pragma unroll
    for (int st = 0; st < 8; ++st) {
      const float2 k0 = *reinterpret_cast<const float2*>(&s.lk[j][16 * st + 2 * t]);
      const float2 k1 = *reinterpret_cast<const float2*>(&s.lk[j][16 * st + 2 * t + 8]);
      kv[st][0] = k0.x; kv[st][1] = k0.y; kv[st][2] = k1.x; kv[st][3] = k1.y;
    }
    float d0[8], d1[8];
#pragma unroll
    for (int st = 0; st < 8; ++st) {
      d0[st] = 0.f; d1[st] = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        h[st][0][e] *= d;
        h[st][1][e] *= d;
        d0[st] = fmaf(h[st][0][e], kv[st][e], d0[st]);
        d1[st] = fmaf(h[st][1][e], kv[st][e], d1[st]);
      }
    }
    float dot0 = ((d0[0] + d0[1]) + (d0[2] + d0[3])) + ((d0[4] + d0[5]) + (d0[6] + d0[7]));
    float dot1 = ((d1[0] + d1[1]) + (d1[2] + d1[3])) + ((d1[4] + d1[5]) + (d1[6] + d1[7]));
    dot0 += __shfl_xor_sync(0xffffffffu, dot0, 1);
    dot1 += __shfl_xor_sync(0xffffffffu, dot1, 1);
    dot0 += __shfl_xor_sync(0xffffffffu, dot0, 2);
    dot1 += __shfl_xor_sync(0xffffffffu, dot1, 2);
    const float del0 = (v0 - dot0) * bt, del1 = (v1 - dot1) * bt;
#pragma unroll
    for (int st = 0; st < 8; ++st)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        h[st][0][e] = fmaf(del0, kv[st][e], h[st][0][e]);
        h[st][1][e] = fmaf(del1, kv[st][e], h[st][1][e]);
      }
  }
  if (CONV) bar_sync(4, 128); // q / k rows loaded by warps 2-3
  // ---------------- KSt = h k^T, QSt = h q^T (bf16x3 split of h: exact products)
  float aK[3][2][4], aQ[3][2][4];
#pragma unroll
  for (int sp = 0; sp < 3; ++sp)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) { aK[sp][nt][e] = 0.f; aQ[sp][nt][e] = 0.f; }
#pragma unroll
  for (int st = 0; st < 8; ++st) {
    float hs[3][2][4];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float x0 = h[st][r][e];
        const float x1 = __bfloat162float(__float2bfloat16(x0));
        const float rr = x0 - x1;
        const float x2 = __bfloat162float(__float2bfloat16(rr));
        hs[0][r][e] = x1; hs[1][r][e] = x2; hs[2][r][e] = rr - x2;
      }
    const int col = 16 * st + 2 * t;
    uint32_t bk[2][2], bq[2][2];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      bk[nt][0] = lds32(&s.k[nt * 8 + g][col]); bk[nt][1] = lds32(&s.k[nt * 8 + g][col + 8]);
      bq[nt][0] = lds32(&s.q[nt * 8 + g][col]); bq[nt][1] = lds32(&s.q[nt * 8 + g][col + 8]);
    }
#pragma unroll
    for (int sp = 0; sp < GT_SPLITS; ++sp) {
      if (sp > 0 && n_rep == 0) break;  // no replay: h is exactly bf16, one piece is exact
      uint32_t a[4] = {pack_bf16(hs[sp][0][0], hs[sp][0][1]), pack_bf16(hs[sp][1][0], hs[sp][1][1]),
                       pack_bf16(hs[sp][0][2], hs[sp][0][3]), pack_bf16(hs[sp][1][2], hs[sp][1][3])};
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        mma16816(aK[sp][nt], a, bk[nt][0], bk[nt][1]);
        mma16816(aQ[sp][nt], a, bq[nt][0], bq[nt][1]);
      }
    }
  }
  bar_sync(5, 128);  // norms + node params from warps 2-3 (q / k were in smem already)
  // C layout: [nt][e] -> row (e<2 ? r0 : r1), node j = nt*8 + 2t + (e&1)
  float QS[2][4], RH[2][4];
#pragma unroll
  for (int nt = 0; nt < 2; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int j = nt * 8 + 2 * t + (e & 1);
      const int row = e < 2 ? r0 : r1;
      const float ks = ((aK[2][nt][e] + aK[1][nt][e]) + aK[0][nt][e]) * s.rk[j];
      QS[nt][e] = ((aQ[2][nt][e] + aQ[1][nt][e]) + aQ[0][nt][e]) * (s.rq[j] * p.scale);
      const float vv = __bfloat162float(s.v[j][row]);
      RH[nt][e] = s.beta[j] * (vv - s.eG[j] * ks);
      s.rhs[row][j] = RH[nt][e];
      if (nt == 1) s.qs[row][2 * t + (e & 1)] = QS[1][e];
    }
  __syncwarp();
  bar_arrive(8, 128);
  // ---------------- state commit: h_root = h e^{g0} + w0 k0^T (w0 = rhs[:, 0] since P[0][j] = d_0j)
  auto commit = [&]() {
    const float e0 = __expf(s.gg[0]);
    const float w00 = __shfl_sync(0xffffffffu, RH[0][0], g * 4);
    const float w01 = __shfl_sync(0xffffffffu, RH[0][2], g * 4);
    const float rk0 = s.rk[0];
#pragma unroll
    for (int st = 0; st < 8; ++st) {
      const int col = 16 * st + 2 * t;
      const float2 ka = bf2f(lds32(&s.k[0][col])), kb = bf2f(lds32(&s.k[0][col + 8]));
      const float kz[4] = {ka.x * rk0, ka.y * rk0, kb.x * rk0, kb.y * rk0};
      float hn[2][4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        hn[0][e] = fmaf(w00, kz[e], h[st][0][e] * e0);
        hn[1][e] = fmaf(w01, kz[e], h[st][1][e] * e0);
      }
      *reinterpret_cast<uint32_t*>(&s.h[r0][col]) = pack_bf16(hn[0][0], hn[0][1]);
      *reinterpret_cast<uint32_t*>(&s.h[r0][col + 8]) = pack_bf16(hn[0][2], hn[0][3]);
      *reinterpret_cast<uint32_t*>(&s.h[r1][col]) = pack_bf16(hn[1][0], hn[1][1]);
      *reinterpret_cast<uint32_t*>(&s.h[r1][col + 8]) = pack_bf16(hn[1][2], hn[1][3]);
    }
    __syncwarp();
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      const int e = lane + 32 * m;  // 256 chunks = 16 rows x 16
      const int row = 16 * rw + (e >> 4);
      *reinterpret_cast<uint4*>(hsrc + row * (kK * 2) + (e & 15) * 16) =
          *reinterpret_cast<const uint4*>(&s.h[row][(e & 15) * 8]);
    }
  };
  bar_sync(2, 96);  // Z from rw 2
  // ---------------- out[v][i] = e^G_i QSt[v][i] + sum_j rhs[v][j] Z[i][j]
  float rr0[kNP], rr1[kNP];
#pragma unroll
  for (int q4 = 0; q4 < 4; ++q4) {
    const float4 a4 = *reinterpret_cast<const float4*>(&s.rhs[r0][q4 * 4]);
    const float4 b4 = *reinterpret_cast<const float4*>(&s.rhs[r1][q4 * 4]);
    rr0[q4 * 4] = a4.x; rr0[q4 * 4 + 1] = a4.y; rr0[q4 * 4 + 2] = a4.z; rr0[q4 * 4 + 3] = a4.w;
    rr1[q4 * 4] = b4.x; rr1[q4 * 4 + 1] = b4.y; rr1[q4 * 4 + 2] = b4.z; rr1[q4 * 4 + 3] = b4.w;
  }
  float ob[2][4] = {};
  constexpr int kNT = 1;  // node halves computed by warps 0-1
#pragma unroll
  for (int nt = 0; nt < kNT; ++nt)
#pragma unroll
    for (int c2 = 0; c2 < 2; ++c2) {
      const int i = nt * 8 + 2 * t + c2;
      float z0 = 0.f, z1 = 0.f;
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        const float4 zz = *reinterpret_cast<const float4*>(&s.Z[i][q4 * 4]);
        z0 = fmaf(rr0[q4 * 4], zz.x, z0); z0 = fmaf(rr0[q4 * 4 + 1], zz.y, z0);
        z0 = fmaf(rr0[q4 * 4 + 2], zz.z, z0); z0 = fmaf(rr0[q4 * 4 + 3], zz.w, z0);
        z1 = fmaf(rr1[q4 * 4], zz.x, z1); z1 = fmaf(rr1[q4 * 4 + 1], zz.y, z1);
        z1 = fmaf(rr1[q4 * 4 + 2], zz.z, z1); z1 = fmaf(rr1[q4 * 4 + 3], zz.w, z1);
      }
      const float eg = s.eG[i];
      ob[nt][c2] = __bfloat162float(__float2bfloat16(fmaf(eg, QS[nt][c2], z0)));
      ob[nt][2 + c2] = __bfloat162float(__float2bfloat16(fmaf(eg, QS[nt][2 + c2], z1)));
      s.o[i][r0] = __float2bfloat16(ob[nt][c2]);
      s.o[i][r1] = __float2bfloat16(ob[nt][2 + c2]);
    }
  uint2* parts = reinterpret_cast<uint2*>(p.part) + ((int64_t)n * p.HV + hv) * (kNV * kNP);
  if (NORM) {
    float ps[2][2];
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt)
#pragma unroll
      for (int c2 = 0; c2 < 2; ++c2) ps[nt][c2] = ob[nt][c2] * ob[nt][c2] + ob[nt][2 + c2] * ob[nt][2 + c2];
#pragma unroll
    for (int o = 4; o < 32; o <<= 1)
#pragma unroll
      for (int nt = 0; nt < kNT; ++nt)
#pragma unroll
        for (int c2 = 0; c2 < 2; ++c2) ps[nt][c2] += __shfl_xor_sync(0xffffffffu, ps[nt][c2], o);
    if (g == 0) {
#pragma unroll
      for (int nt = 0; nt < kNT; ++nt)
#pragma unroll
        for (int c2 = 0; c2 < 2; ++c2) s.ss[rw][nt * 8 + 2 * t + c2] = ps[nt][c2];
    }
    bar_sync(10, 128);
    if (rt < kNP) {  // (value, generation): one 8-byte single-copy-atomic store, no fence needed
      st_relaxed_v2(parts + iv * kNP + rt, make_uint2(__float_as_uint(s.ss[0][rt] + s.ss[1][rt]), gen));
    }
  }
  commit();
  // ---------------- gated RMSNorm: sums of squares of the bf16 outputs of the 4 V-blocks
  if (NORM) {
    // thread -> (V-block jv, node i): poll the tagged pair until it carries this launch's generation
    const int jv = rt >> 4, i = rt & 15;
    uint2 pr;
    do {
      pr = ld_relaxed_v2(parts + jv * kNP + i);
    } while (pr.y != gen);
    s.tot[jv][i] = __uint_as_float(pr.x);
    bar_sync(3, 64);
    // thread -> node i = rt >> 2 (16 nodes x 4 threads), 8 v each
    const int in = rt >> 2, v0 = (rt & 3) * 8;
    if (in < T) {
      const float tot = ((s.tot[0][in] + s.tot[1][in]) + s.tot[2][in]) + s.tot[3][in];
      const float rstd = rsqrtf(tot * (1.0f / (float)kV) + p.eps);
      uint32_t w4[4];
#pragma unroll
      for (int e = 0; e < 8; e += 2) {
        float y[2];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const float xo = __bfloat162float(s.o[in][v0 + e + u]);
          const float zz = __bfloat162float(s.zg[in][v0 + e + u]);
          float yy = (xo * rstd) * s.nw[v0 + e + u];
          yy *= SIGMOID ? sigmoid_f(zz) : zz * sigmoid_f(zz);
          y[u] = yy;
        }
        w4[e >> 1] = pack_bf16(y[0], y[1]);
      }
      *reinterpret_cast<uint4*>(p.out + (int64_t)(bos + in) * p.so + hv * kV + vb + v0) =
          make_uint4(w4[0], w4[1], w4[2], w4[3]);
    }
  } else {
    bar_sync(10, 128);
    const int i = rt >> 2, v0 = (rt & 3) * 8;
    if (i < T)
      *reinterpret_cast<uint4*>(p.out + (int64_t)(bos + i) * p.so + hv * kV + vb + v0) =
          *reinterpret_cast<const uint4*>(&s.o[i][v0]);
  }
}

constexpr int kSmemBytes = (int)sizeof(Smem);

template <bool CONV, bool NORM, bool SIGMOID, bool ACT>
void launch(const Args& a, int R, cudaStream_t st) {
  auto kern = gdn_tree_kernel<CONV, NORM, SIGMOID, ACT>;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    attr = true;
  }
  dim3 grid(a.HV * kNV, R);
  kern<<<grid, kThreads, kSmemBytes, st>>>(a);
}

}  // namespace

void gdn_tree_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor x,
                  torch::Tensor conv_w, torch::Tensor conv_b, torch::Tensor conv_state,
                  torch::Tensor conv_src, bool conv, bool act,
                  torch::Tensor a, torch::Tensor b, torch::Tensor A_log, torch::Tensor dt_bias,
                  torch::Tensor state, torch::Tensor idx, torch::Tensor cu, torch::Tensor acc,
                  torch::Tensor anc, torch::Tensor out, torch::Tensor z, torch::Tensor nw,
                  bool norm, bool sigmoid, torch::Tensor cnt, torch::Tensor part,
                  torch::Tensor qkx, torch::Tensor qkf,
                  int64_t H, int64_t R, double scale, double eps, double sp_beta, double sp_thr,
                  int64_t max_log, int64_t log_base, int64_t buf, int64_t v_off, int64_t d_off,
                  int64_t b_off
#if GT_BAP
                  , torch::Tensor bar
#endif
                  ) {
  TORCH_CHECK(state.scalar_type() == at::kBFloat16 && state.size(2) == kV && state.size(3) == kK);
#if GT_BAP
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat &&
              bar.scalar_type() == at::kFloat && bar.stride(0) == 1 && a.stride(1) == 1 && b.stride(1) == 1);
#else
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && a.scalar_type() == at::kBFloat16 &&
              b.scalar_type() == at::kBFloat16);
#endif
  TORCH_CHECK(max_log + 1 <= kNP);
  TORCH_CHECK(state.size(1) / H * kNV <= 16, "group of > 16 CTAs per key head");
  Args p{};
  if (conv) {
    p.x = reinterpret_cast<const bf16*>(x.data_ptr()); p.sx = x.stride(0);
    p.cw = reinterpret_cast<const bf16*>(conv_w.data_ptr());
    p.cb = conv_b.numel() ? reinterpret_cast<const bf16*>(conv_b.data_ptr()) : nullptr;
    p.cs = reinterpret_cast<bf16*>(conv_state.data_ptr());
    p.cs_slot = conv_state.stride(0); p.cs_dim = conv_state.stride(1); p.cs_tok = conv_state.stride(2);
    p.cs_len = (int)conv_state.size(2);
    p.csrc = conv_src.data_ptr<int>(); p.csrc_req = conv_src.stride(0);
  } else {
    p.q = reinterpret_cast<const bf16*>(q.data_ptr()); p.sq = q.stride(0);
    p.k = reinterpret_cast<const bf16*>(k.data_ptr()); p.sk = k.stride(0);
    p.v = reinterpret_cast<const bf16*>(v.data_ptr()); p.sv = v.stride(0);
  }
#if GT_BAP
  p.af = a.data_ptr<float>(); p.sa = a.stride(0);
  p.bf = b.data_ptr<float>(); p.sb = b.stride(0);
  p.bar_p = bar.data_ptr<float>();
#else
  p.a = reinterpret_cast<const bf16*>(a.data_ptr()); p.sa = a.stride(0);
  p.b = reinterpret_cast<const bf16*>(b.data_ptr()); p.sb = b.stride(0);
#endif
  p.A_log = A_log.data_ptr(); p.A_log_bf16 = A_log.scalar_type() == at::kBFloat16;
  p.dt_bias = dt_bias.data_ptr(); p.dt_bias_bf16 = dt_bias.scalar_type() == at::kBFloat16;
  p.state = reinterpret_cast<unsigned char*>(state.data_ptr());
  p.slot_bytes = state.stride(0) * 2;
  p.idx = idx.data_ptr<int>(); p.idx_row = idx.stride(0);
  p.cu = cu.data_ptr<int>(); p.acc = acc.data_ptr<int>();
  p.anc = anc.data_ptr<int64_t>(); p.anc_row = anc.stride(0);
  p.out = reinterpret_cast<bf16*>(out.data_ptr()); p.so = out.stride(0);
  if (norm) {
    p.z = reinterpret_cast<const bf16*>(z.data_ptr()); p.sz = z.stride(0);
    p.nw = nw.data_ptr(); p.nw_bf16 = nw.scalar_type() == at::kBFloat16;
  }
  p.cnt = reinterpret_cast<unsigned long long*>(cnt.data_ptr());
  p.part = part.data_ptr<float>();
  p.qkx = reinterpret_cast<bf16*>(qkx.data_ptr());
  p.qkf = reinterpret_cast<unsigned*>(qkf.data_ptr());
  p.HV = (int)state.size(1); p.H = (int)H; p.G = p.HV / p.H; p.max_log = (int)max_log;
  p.log_base = log_base; p.buf = buf; p.v_off = v_off; p.d_off = d_off; p.b_off = b_off;
  p.scale = (float)scale; p.eps = (float)eps; p.sp_beta = (float)sp_beta; p.sp_thr = (float)sp_thr;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
#define L(C, N, S, A) launch<C, N, S, A>(p, (int)R, st)
  if (conv) {
    if (norm) { if (sigmoid) { if (act) L(true, true, true, true); else L(true, true, true, false); }
                else { if (act) L(true, true, false, true); else L(true, true, false, false); } }
    else { if (act) L(true, false, false, true); else L(true, false, false, false); }
  } else {
    if (norm) { if (sigmoid) L(false, true, true, false); else L(false, true, false, false); }
    else L(false, false, false, false);
  }
#undef L
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<int64_t> occupancy() {
  auto kern = gdn_tree_kernel<true, true, false, true>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
  cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  int nb = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, kThreads, kSmemBytes);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, kern);
  return {nb, (int64_t)kSmemBytes, fa.numRegs, (int64_t)fa.localSizeBytes};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_tree_fwd", &gdn_tree_fwd);
  m.def("occupancy", &occupancy);
  m.attr("ba_pre") = (bool)GT_BAP;
}
