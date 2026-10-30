// W3 stream-K GEMM for the c=1 verify batch (M <= 16) on the w3_mixed planar 3-bit weights, with the c1_fuse
// E6 epilogues (atdev_serve/w3_sk.py, ATDEV_C1_W3_SK=1). Same packed weights / Marlin-permuted scales / gate_up
// interleave as the marlin_c1 W3 V2 kernel (marlin_c1_w3_build.py); same V2 bit tricks to build bf16 operands,
// different GEMM schedule and scale application:
//
//  * tile = 128 output columns (two Marlin n64 blocks; for gate_up = one interleaved gate|up pair); work units =
//    (tile, k16 slab), linearised tile-major: U = (N / 128) * (K / 16).
//  * persistent grid (1 CTA / SM, 8 warps): CTA c owns the contiguous unit range [c U / G, (c + 1) U / G) (stream-K,
//    no partial waves). The range is cut into per-tile segments; every segment is split evenly (in slabs) over the
//    8 warps (intra-CTA split-K, each warp a contiguous k range).
//  * each warp covers all 128 columns of its tile (16 n8 mma tiles + 1 zero-point mma per slab, one A fragment
//    reused 17x) and streams B / scales / A with 16 B cp.async into a warp-private D-stage smem ring (steps of 4
//    slabs, no __syncthreads in the main loop); the ring runs across segment boundaries.
//  * offset dequant: codes -> bf16 128 + u / 8 + u (one LOP3 per 2 weights, no subtract, no bf16 scale multiply);
//    per 128-group: y += s (accg - Z) in fp32, Z = the zero-point row sums from the extra mma -> exact (u - 4) s
//    products (V2 rounds them to bf16): rounding-level different from V2, closer to fp64.
//  * segment end: fp32 partials through smem, two-phase fixed-order reduction (w + w+4, then 0..3).
//    Tile ownership (stream-K fixup): the CTA holding a tile's first slab owns it; a CTA whose first segment starts
//    mid-tile publishes that partial (fp32, global workspace, release flag); the owner adds its peers' partials in
//    CTA order after its own (fixed order -> deterministic run to run) and runs the epilogue.
//    Only a CTA's last segment can have peers (they publish their FIRST segment) -> no wait cycles; all CTAs are
//    co-resident (grid <= #SMs, 1 CTA / SM). Flags are reset by their consumer -> CUDA-graph replay safe.
//  * W3SK_ABL / W3SK_PROF: dev-only ablation / timeline builds (wrong results), see w3_sk.py.
//
// Epilogues (E6 contract of csrc/marlin_c1/kernel.h): mode 0 plain; 1 producer (x = fp32(bf16(acc)) + R,
// C2 = bf16(x), C = bf16(x (1 + W)), P[row, tile] = sum_tile x^2 with T = N / 128 tiles); 2 consumer
// (acc *= rstd[row], rstd = rsqrt(sum_t Pin[row, t] / n_norm + eps)); 3 consumer + SiLU pair (gate | up halves of
// the 128-column tile, C [M, N / 2] = bf16(silu(bf16 g) * bf16 u), E6_SILU=1 math). Modes 2/3 write rstd to Rout
// (CTA 0) when given.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define DEV __device__ __forceinline__
#ifndef W3SK_ABL
  #define W3SK_ABL 0
#endif
#ifndef W3SK_PROF
  #define W3SK_PROF 0
#endif

namespace w3sk {

constexpr int TILE_N = 128;
constexpr int RED_LD = TILE_N + 8;  // padded fp32 row of a warp partial in smem
constexpr int SLOT = 16 * RED_LD;   // floats per warp partial
constexpr int KK = 4;              // slabs (k16) per pipeline step (A: 16 rows x 128 B = 16 full lines per step)
constexpr int STAGE_B = KK * 1280 + 256;  // bytes per warp ring stage (B KK x 768 + S 256 + A KK x 512)
constexpr int NSLOT = 4;           // reduction slots (two-phase: warps 4-7 into 0-3, then 0-3 add theirs)

struct Epi {
  const __nv_bfloat16* R;   // mode 1: residual in [M, N]
  const __nv_bfloat16* Wn;  // mode 1: norm weight [N]
  __nv_bfloat16* C2;        // mode 1: residual out [M, N]
  float* P;                 // mode 1: per-tile sum of squares [M, N / 128]
  const float* Pin;         // mode 2/3: producer partials [M, T_in]
  int T_in;
  int n_norm;
  float eps;
  float* Rout;  // mode 2/3: optional rstd out [M]
};

DEV float4 ldcg4(const float* p) {
  float4 v;
  asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  return v;
}
DEV void stcg4(float* p, float4 v) {
  asm volatile("st.global.cg.v4.f32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
}
DEV void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}
DEV void cp_async16_zfill(uint32_t dst, const void* src, int src_size) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_size) : "memory");
}
DEV void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int n>
DEV void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n) : "memory");
}
DEV void ldsm4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
DEV int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}
DEV void st_release(int* p, int v) { asm volatile("st.global.release.gpu.b32 [%0], %1;\n" ::"l"(p), "r"(v)); }

template <int lut>
DEV uint32_t lop3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c), "n"(lut));
  return r;
}

DEV void mma0(float* c, const uint32_t* a, const uint32_t* b) {  // C = 0
  const float z = 0.f;
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(z));
}
DEV void mma(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

DEV void unpack8(const uint4& v, float* f) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int e = 0; e < 4; e++) {
    float2 q = __bfloat1622float2(h[e]);
    f[2 * e] = q.x;
    f[2 * e + 1] = q.y;
  }
}
DEV uint4 pack8(const float* f) {
  uint4 v;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
  for (int e = 0; e < 4; e++) h[e] = __floats2bfloat162_rn(f[2 * e], f[2 * e + 1]);
  return v;
}

#if W3SK_PROF
// dev timeline (W3SK_PROF=1 builds): per (CTA, warp) 16 u64 stamps: 0 globaltimer at entry, then clock64:
// 1 entry, 2 prologue issued, 3 + 2 sg main loop of segment sg done, 4 + 2 sg seg_end done (sg < 5), 15 exit
__device__ unsigned long long g_prof[256 * 8 * 16];
DEV unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
  #define PROF(i) \
    if ((threadIdx.x & 31) == 0 && (i) < 16) g_prof[(blockIdx.x * 8 + (threadIdx.x >> 5)) * 16 + (i)] = clock64();
#else
  #define PROF(i)
#endif

template <int N, int K, int MODE, int W, int D>
__global__ void __launch_bounds__(W * 32, 1)
    sk_kernel(const __nv_bfloat16* __restrict__ A, const uint32_t* __restrict__ B, const __nv_bfloat16* __restrict__ S,
              __nv_bfloat16* __restrict__ C, int M, int lda, Epi epi, float* __restrict__ ws, int* __restrict__ flags) {
  constexpr int KS = K / 16, NT = N / TILE_N, U = NT * KS;
  static_assert(N % TILE_N == 0 && K % 128 == 0, "shape");
  static_assert(W * 32 == 256, "epilogue thread map assumes 256 threads (16 rows x 16 chunks of 8 columns)");
  static_assert(KK == 4 && W == 8, "chunk / slot maps below are written for 4 slabs per step, 8 warps");
  extern __shared__ float4 smem4[];
  float* red = reinterpret_cast<float*>(smem4);
  __shared__ float sh_rstd[16];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int gid = lane >> 2, tq = lane & 3;
  const int G = gridDim.x, cta = blockIdx.x;
  const int c0 = (int)((long long)cta * U / G), c1 = (int)((long long)(cta + 1) * U / G);
  if (c0 >= c1) return;
#if W3SK_PROF
  if ((threadIdx.x & 31) == 0) g_prof[(blockIdx.x * 8 + (threadIdx.x >> 5)) * 16] = gtime();
#endif
  PROF(1)
  const int t_first = c0 / KS, nseg = (c1 - 1) / KS - t_first + 1;

  // warp sub-range [s, e) of segment sg (slab indices inside tile t)
  auto seg_range = [&](int sg, int& t, int& s, int& e) {
    t = t_first + sg;
    const int a = max(c0, t * KS) - t * KS, b = min(c1, (t + 1) * KS) - t * KS, L = b - a;
    s = a + (warp * L) / W;
    e = a + ((warp + 1) * L) / W;
  };

  // ---- consumer rstd: issue the partial loads first (finished after the ring prologue)
  float pv[5];
  const int rr_row = tid >> 4, rr_sub = tid & 15;
  if constexpr (MODE >= 2) {
#pragma unroll
    for (int q = 0; q < 5; q++) {
      const int t = rr_sub + 16 * q;
      pv[q] = (rr_row < M && t < epi.T_in) ? __ldg(epi.Pin + rr_row * epi.T_in + t) : 0.f;
    }
  }

  // ---- warp-private cp.async ring, D stages of STAGE_B bytes (after the reduction slots). A step = KK = 4
  // consecutive slabs of one 128-column tile inside one 128-k group (steps are KK-aligned in absolute slab index).
  // stage: [B: KK x 768 B = slab i, segment c * 3 + p (cb c, plane p), 128 B][S: 2 x 128 B (cb c)]
  //        [A: KK x 512 B = slab i, k half h, 16 rows x 16 B]
  // 16 B cp.async per lane: 6 B rows (192 chunks, slab-major), 4 A rows (A row r, 128 contiguous bytes = 8 chunks
  // per 8 lanes: one full line per row), S on lanes 0-15.
  char* ring = reinterpret_cast<char*>(red + NSLOT * SLOT) + warp * (D * STAGE_B);
  const uint32_t ring_s = (uint32_t)__cvta_generic_to_shared(ring);
  int b_off[6], b_sl[6];
#pragma unroll
  for (int i = 0; i < 6; i++) {
    const int q = 32 * i + lane, sl = q / 48, r = q % 48, sgm = r >> 3, part = r & 7;
    b_sl[i] = sl;
    b_off[i] = sl * (6 * N) + (sgm % 3) * (2 * N) + (sgm / 3) * 128 + part * 16;
  }
  const int a_c = lane & 7, a_sl = a_c >> 1, a_hh = a_c & 1;  // chunk of the row's 128 B: slab a_sl, half a_hh
  const char* Ag[4];
  int a_dst[4];
  bool a_live[4];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int row = 4 * j + (lane >> 3);
    Ag[j] = reinterpret_cast<const char*>(A) + (size_t)min(row, M - 1) * lda * 2 + a_c * 16;
    a_dst[j] = KK * 768 + 256 + a_sl * 512 + a_hh * 256 + row * 16;
    a_live[j] = row < M;
#if W3SK_ABL == 17  // dev: read A replica (cta % 8) of an [M, 8 K] buffer (lda = 8 K) -- hot-line probe
    Ag[j] += (size_t)(cta & 7) * K * 2;
#endif
  }
  const int s_off = (lane >> 3) * 128 + (lane & 7) * 16;
  const char* Bg = reinterpret_cast<const char*>(B);
  const char* Sg = reinterpret_cast<const char*>(S);

  constexpr bool ISSUE = W3SK_ABL != 5 && W3SK_ABL != 8;
  // dev ablations (no math): 9 only B, 10 only A, 11 only S, 12 B + A
  constexpr bool IB = W3SK_ABL != 10 && W3SK_ABL != 11 && W3SK_ABL != 15,
                 IA = W3SK_ABL != 9 && W3SK_ABL != 11 && W3SK_ABL != 14 && W3SK_ABL != 16,  // 16: full math, no A loads
                 IS = W3SK_ABL != 9 && W3SK_ABL != 10 && W3SK_ABL != 12 && W3SK_ABL != 14 && W3SK_ABL != 15;
  auto issue = [&](int stage, int t, int x, int nv) {
    const uint32_t st = ring_s + stage * STAGE_B;
    if (ISSUE) {
      const char* bp = Bg + (size_t)x * (6 * N) + t * 256;
#if W3SK_ABL == 14  // dev: B only, hot: every warp reads the first 64 slabs of tile 0 (L2-resident)
      bp = Bg + (size_t)(x & 63) * (6 * N);
#endif
#pragma unroll
      for (int i = 0; i < 6; i++)
        if (IB && b_sl[i] < nv) cp_async16(st + (32 * i + lane) * 16, bp + b_off[i]);
      const int sz = a_sl < nv ? 16 : 0;
#pragma unroll
      for (int j = 0; j < 4; j++) {
#if W3SK_ABL == 15  // dev: A-like reads (4 rows x 128 B per instr) from distinct lines of the (L2-cold) weight buffer
        cp_async16_zfill(st + a_dst[j], Bg + ((size_t)(t * KS + x) * 2048 + j * 512 + (lane >> 3) * 128 + a_c * 16) % ((size_t)K * 3 * N / 8), 16);
#else
        if (IA) cp_async16_zfill(st + a_dst[j], Ag[j] + x * 32, a_live[j] ? sz : 0);
#endif
      }
      if (IS && lane < 16) cp_async16(st + KK * 768 + lane * 16, Sg + (size_t)(x >> 3) * (2 * N) + t * 256 + s_off);
    }
  };

  // ---- load cursor over (segment, step): [lx, lx_end) = the next step, inside sub-range [.., le) of tile lt
  int lseg = 0, lt, lx, le;
  seg_range(0, lt, lx, le);
  while (lx >= le && lseg + 1 < nseg) {
    lseg++;
    seg_range(lseg, lt, lx, le);
  }
  auto ladv = [&]() {
    lx = min(le, (lx & ~(KK - 1)) + KK);
    while (lx >= le && lseg + 1 < nseg) {
      lseg++;
      seg_range(lseg, lt, lx, le);
    }
  };
  auto lnv = [&]() { return min(le, (lx & ~(KK - 1)) + KK) - lx; };
#pragma unroll
  for (int d = 0; d < D - 1; d++) {
    if (lx < le) issue(d, lt, lx, lnv());
    cp_async_commit();
    ladv();
  }
  int lstage = D - 1, cstage = 0;
  PROF(2)

  if constexpr (MODE >= 2) {
    float s = 0.f;
#pragma unroll
    for (int q = 0; q < 5; q++) s += pv[q];
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (rr_sub == 0) {
      const float r = rr_row < M ? rsqrtf(s / (float)epi.n_norm + epi.eps) : 0.f;
      sh_rstd[rr_row] = r;
      if (epi.Rout != nullptr && cta == 0 && rr_row < M) epi.Rout[rr_row] = r;
    }
  }

  // ---- math. Offset dequant (no subtract, no bf16 scale multiply): even nibble pairs -> 128 + u (k rows 0-7 of
  // the mma), odd pairs -> 8 + u (k rows 8-15), both exact in bf16. Per 128-group and row,
  //   sum_k a (u - 4) = accg - Z,  Z = sum_{k%16<8} 132 a + sum_{k%16>=8} 12 a  (17th mma on a constant B),
  // and y += s[g, n] (accg - Z) in fp32 at the group end (s = the unrounded bf16 scale): the (u - 4) s products
  // are exact instead of bf16-rounded as in Marlin (more accurate than V2, rounding-level different).
  float acc[16][4], accg[16][4], zacc[4];
#pragma unroll
  for (int j = 0; j < 16; j++)
#pragma unroll
    for (int e = 0; e < 4; e++) acc[j][e] = 0.f;
  const uint32_t zb[2] = {0x43044304u, 0x41404140u};  // bf16x2 (132, 132), (12, 12)
  const int lm_row = (lane & 7) + 8 * ((lane >> 3) & 1), lm_h = lane >> 4;
  const uint32_t lm_off = KK * 768 + 256 + lm_h * 256 + lm_row * 16;

  auto slab_math = [&](const char* st, uint32_t st_s, int i, bool first) {
    uint32_t b[6];
#pragma unroll
    for (int q = 0; q < 6; q++) b[q] = *reinterpret_cast<const uint32_t*>(st + i * 768 + q * 128 + lane * 4);
    uint32_t a[4];
    ldsm4(a, st_s + lm_off + i * 512);
#if W3SK_ABL == 3 || W3SK_ABL == 8 || W3SK_ABL >= 9
    zacc[0] += __uint_as_float((b[0] ^ b[1] ^ b[2] ^ b[3] ^ b[4] ^ b[5] ^ a[0] ^ a[3]) & 0x3fffffu);
    return;
#endif
    if (first) mma0(zacc, a, zb); else mma(zacc, a, zb);
#pragma unroll
    for (int c = 0; c < 2; c++) {
      const uint32_t w0 = b[3 * c], w1 = b[3 * c + 1], w2 = b[3 * c + 2];
      const uint32_t q[4] = {w0, w1, w2,
                             ((w0 >> 3) & 0x11111111u) | ((w1 >> 2) & 0x22222222u) | ((w2 >> 1) & 0x44444444u)};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        uint32_t f0[2], f1[2];
        const uint32_t q8 = q[j] >> 8;
        f0[0] = lop3<(0xf0 & 0xcc) | 0xaa>(q[j], 0x00070007u, 0x43004300u);
        f0[1] = lop3<(0xf0 & 0xcc) | 0xaa>(q[j], 0x00700070u, 0x41004100u);
        f1[0] = lop3<(0xf0 & 0xcc) | 0xaa>(q8, 0x00070007u, 0x43004300u);
        f1[1] = lop3<(0xf0 & 0xcc) | 0xaa>(q8, 0x00700070u, 0x41004100u);
        if (first) {
          mma0(accg[c * 8 + 2 * j], a, f0);
          mma0(accg[c * 8 + 2 * j + 1], a, f1);
        } else {
          mma(accg[c * 8 + 2 * j], a, f0);
          mma(accg[c * 8 + 2 * j + 1], a, f1);
        }
      }
    }
  };
  auto group_end = [&](const char* st) {
#if W3SK_ABL == 3 || W3SK_ABL == 8 || W3SK_ABL >= 9
    acc[0][0] += zacc[0];
    return;
#endif
    const float z0 = zacc[0], z1 = zacc[2];
    // C fragment columns of this lane: 2 tq, 2 tq + 1 of every n8 tile; the Marlin-permuted scale row of an n64
    // block holds column c of tile j at int4 c, bf16 j -> two int4 per cb (columns 2 tq, 2 tq + 1)
#pragma unroll
    for (int c = 0; c < 2; c++) {
      const uint4 sv0 = *reinterpret_cast<const uint4*>(st + KK * 768 + c * 128 + (2 * tq) * 16);
      const uint4 sv1 = *reinterpret_cast<const uint4*>(st + KK * 768 + c * 128 + (2 * tq + 1) * 16);
      const uint32_t s0[4] = {sv0.x, sv0.y, sv0.z, sv0.w}, s1[4] = {sv1.x, sv1.y, sv1.z, sv1.w};
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const float f0 = __uint_as_float((e & 1) ? (s0[e >> 1] & 0xffff0000u) : (s0[e >> 1] << 16));
        const float f1 = __uint_as_float((e & 1) ? (s1[e >> 1] & 0xffff0000u) : (s1[e >> 1] << 16));
        float* ac = acc[c * 8 + e];
        const float* ag = accg[c * 8 + e];
        ac[0] = fmaf(f0, ag[0] - z0, ac[0]);
        ac[1] = fmaf(f1, ag[1] - z0, ac[1]);
        ac[2] = fmaf(f0, ag[2] - z1, ac[2]);
        ac[3] = fmaf(f1, ag[3] - z1, ac[3]);
      }
    }
  };

  // ---- segment end: smem partials -> fixed-order reduction -> publish (peer) or fixup + epilogue (owner)
  const int er = tid >> 4, ec = tid & 15;  // epilogue: row er, columns 8 ec .. 8 ec + 7 of the tile
  auto seg_end = [&](int sg) {
    const int t = t_first + sg;
    // two-phase fixed-order reduction: slot w = acc_w + acc_{w+4} (w < 4), then v = ((s0 + s1) + s2) + s3
    float* slot = red + (warp & 3) * SLOT;
    if (warp >= 4) {
#pragma unroll
      for (int j = 0; j < 16; j++) {
        *reinterpret_cast<float2*>(slot + gid * RED_LD + 8 * j + 2 * tq) = make_float2(acc[j][0], acc[j][1]);
        *reinterpret_cast<float2*>(slot + (gid + 8) * RED_LD + 8 * j + 2 * tq) = make_float2(acc[j][2], acc[j][3]);
      }
    }
    __syncthreads();
    if (warp < 4) {
#pragma unroll
      for (int j = 0; j < 16; j++) {
        float2* p0 = reinterpret_cast<float2*>(slot + gid * RED_LD + 8 * j + 2 * tq);
        float2* p1 = reinterpret_cast<float2*>(slot + (gid + 8) * RED_LD + 8 * j + 2 * tq);
        const float2 u0 = *p0, u1 = *p1;
        *p0 = make_float2(acc[j][0] + u0.x, acc[j][1] + u0.y);
        *p1 = make_float2(acc[j][2] + u1.x, acc[j][3] + u1.y);
      }
    }
#pragma unroll
    for (int j = 0; j < 16; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) acc[j][e] = 0.f;
    __syncthreads();
    float v[8];
    {
      const float* p = red + er * RED_LD + 8 * ec;
      float4 x0 = *reinterpret_cast<const float4*>(p), x1 = *reinterpret_cast<const float4*>(p + 4);
      v[0] = x0.x; v[1] = x0.y; v[2] = x0.z; v[3] = x0.w; v[4] = x1.x; v[5] = x1.y; v[6] = x1.z; v[7] = x1.w;
#pragma unroll
      for (int w = 1; w < NSLOT; w++) {
        x0 = *reinterpret_cast<const float4*>(p + w * SLOT);
        x1 = *reinterpret_cast<const float4*>(p + w * SLOT + 4);
        v[0] += x0.x; v[1] += x0.y; v[2] += x0.z; v[3] += x0.w;
        v[4] += x1.x; v[5] += x1.y; v[6] += x1.z; v[7] += x1.w;
      }
    }
    const bool owner = t * KS >= c0;
    if (!owner) {  // peer: publish this (first) segment's partial
      float* wp = ws + (size_t)cta * (16 * TILE_N) + er * TILE_N + 8 * ec;
      stcg4(wp, make_float4(v[0], v[1], v[2], v[3]));
      stcg4(wp + 4, make_float4(v[4], v[5], v[6], v[7]));
      __syncthreads();
      if (tid == 0) st_release(flags + cta, 1);  // release (cumulative over the CTA's writes via bar.sync)
      return;
    }
    const int col = t * TILE_N + 8 * ec;
    // epilogue operands issued before any peer wait
    uint4 rv, wv;
    if constexpr (MODE == 1) {
      rv = er < M ? *reinterpret_cast<const uint4*>(epi.R + (size_t)er * N + col) : make_uint4(0, 0, 0, 0);
      wv = *reinterpret_cast<const uint4*>(epi.Wn + col);
    }
    const bool peers = (t + 1) * KS > c1;
    int pe = cta + 1;
    if (peers) {
      // peers: CTAs cta+1 .. whose range starts inside tile t
      int pend = cta + 1;
      while (pend < G && (int)((long long)pend * U / G) < (t + 1) * KS) pend++;
      if (tid == 0) {
        for (int j = cta + 1; j < pend; j++)
          while (ld_acquire(flags + j) == 0) {
          }
      }
      __syncthreads();
      for (int j = cta + 1; j < pend; j++) {
        const float* rp = ws + (size_t)j * (16 * TILE_N) + er * TILE_N + 8 * ec;
        const float4 x0 = ldcg4(rp), x1 = ldcg4(rp + 4);
        v[0] += x0.x; v[1] += x0.y; v[2] += x0.z; v[3] += x0.w;
        v[4] += x1.x; v[5] += x1.y; v[6] += x1.z; v[7] += x1.w;
      }
      pe = pend;
    }
    if constexpr (MODE == 0) {
      if (er < M) *reinterpret_cast<uint4*>(C + (size_t)er * N + col) = pack8(v);
    } else if constexpr (MODE == 1) {
      float h[8], r[8], wn[8], x[8], y[8];
      const uint4 hv = pack8(v);
      unpack8(hv, h);
      unpack8(rv, r);
      unpack8(wv, wn);
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        x[e] = h[e] + r[e];
        ss = fmaf(x[e], x[e], ss);
        y[e] = x[e] * (wn[e] + 1.f);
      }
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      if (er < M) {
        *reinterpret_cast<uint4*>(epi.C2 + (size_t)er * N + col) = pack8(x);
        *reinterpret_cast<uint4*>(C + (size_t)er * N + col) = pack8(y);
        if (ec == 0) epi.P[er * NT + t] = ss;
      }
    } else if constexpr (MODE == 2) {
      const float rs_ = er < M ? sh_rstd[er] : 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) v[e] *= rs_;
      if (er < M) *reinterpret_cast<uint4*>(C + (size_t)er * N + col) = pack8(v);
    } else {  // MODE 3: chunks 0-7 gate, 8-15 up of the same output columns
      const float rs_ = er < M ? sh_rstd[er] : 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) v[e] *= rs_;
      uint4 hv = pack8(v), uv;
      uv.x = __shfl_xor_sync(0xffffffffu, hv.x, 8);
      uv.y = __shfl_xor_sync(0xffffffffu, hv.y, 8);
      uv.z = __shfl_xor_sync(0xffffffffu, hv.z, 8);
      uv.w = __shfl_xor_sync(0xffffffffu, hv.w, 8);
      if (ec < 8 && er < M) {
        float g[8], u[8], y[8];
        unpack8(hv, g);
        unpack8(uv, u);
#pragma unroll
        for (int e = 0; e < 8; e++) y[e] = __fdividef(g[e], 1.f + __expf(-g[e])) * u[e];
        *reinterpret_cast<uint4*>(C + (size_t)er * (N / 2) + t * (TILE_N / 2) + 8 * ec) = pack8(y);
      }
    }
    __syncthreads();  // smem partials consumed (next segment may overwrite); peer partials read
    if (peers && tid == 0)
      for (int j = cta + 1; j < pe; j++) flags[j] = 0;
  };

  // ---- main loop: per segment, this warp's steps; the ring carries across segments
#if W3SK_PROF
  long long wait_cyc = 0;
  int nsteps = 0;
#endif
  for (int sg = 0; sg < nseg; sg++) {
    int t_, s_, e_;
    seg_range(sg, t_, s_, e_);
    for (int x = s_; x < e_;) {
      const int xe = min(e_, (x & ~(KK - 1)) + KK);
#if W3SK_PROF
      const long long tw0 = clock64();
#endif
      cp_async_wait<D - 2>();
      __syncwarp();  // all lanes' chunks of cstage landed; everyone is done reading lstage (previous step)
#if W3SK_PROF
      wait_cyc += clock64() - tw0;
      nsteps++;
#endif
      if (lx < le) issue(lstage, lt, lx, lnv());
      cp_async_commit();
      ladv();
      lstage = lstage + 1 == D ? 0 : lstage + 1;
      const char* st = ring + cstage * STAGE_B;
      const uint32_t st_s = ring_s + cstage * STAGE_B;
      const bool gfirst = (x & 7) == 0 || x == s_;
      if (gfirst) slab_math(st, st_s, 0, true); else slab_math(st, st_s, 0, false);
#pragma unroll
      for (int i = 1; i < KK; i++) slab_math(st, st_s, i, false);  // slabs >= xe - x: A zero-filled -> 0
      if ((xe & 7) == 0 || xe == e_) group_end(st);
      cstage = cstage + 1 == D ? 0 : cstage + 1;
      x = xe;
    }
    PROF(3 + 2 * sg)
    seg_end(sg);
    PROF(4 + 2 * sg)
  }
  cp_async_wait<0>();
  PROF(15)
#if W3SK_PROF
  if ((threadIdx.x & 31) == 0) {
    g_prof[(blockIdx.x * 8 + (threadIdx.x >> 5)) * 16 + 13] = wait_cyc;
    g_prof[(blockIdx.x * 8 + (threadIdx.x >> 5)) * 16 + 14] = nsteps;
  }
#endif
}

template <int N, int K, int MODE, int W, int D>
static void launch(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& S, torch::Tensor& C, int M,
                   const Epi& epi, torch::Tensor& ws, torch::Tensor& flags, int grid) {
  auto k = sk_kernel<N, K, MODE, W, D>;
  const int smem = NSLOT * SLOT * 4 + W * D * STAGE_B;
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess,
                "w3_sk: ", smem, " B dynamic smem not available");
    attr = true;
  }
  k<<<grid, W * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()), reinterpret_cast<const uint32_t*>(B.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(S.data_ptr()), reinterpret_cast<__nv_bfloat16*>(C.data_ptr()), M, (int)A.stride(0), epi,
      ws.data_ptr<float>(), flags.data_ptr<int>());
}

#define W3SK_D(N_, K_, MO_, D_)                                                  \
  if (n == N_ && k == K_ && mode == MO_ && depth == D_) {                        \
    launch<N_, K_, MO_, 8, D_>(A, B, S, C, M, epi, ws, flags, (int)grid);        \
    return;                                                                      \
  }
#define W3SK_S(N_, K_, MO_) W3SK_D(N_, K_, MO_, 2) W3SK_D(N_, K_, MO_, 3)

bool supported(int64_t n, int64_t k, int64_t mode) {
  const bool sh = (n == 34816 && k == 5120) || (n == 5120 && k == 17408) || (n == 5120 && k == 6144) ||
                  (n == 16384 && k == 5120) || (n == 14336 && k == 5120);
  if (!sh) return false;
  if (mode == 0) return true;
  if (n == 34816) return mode == 3;
  if (n == 5120) return mode == 1;
  return mode == 2;
}

void gemm(torch::Tensor A, torch::Tensor B, torch::Tensor S, torch::Tensor C, int64_t n, int64_t k, int64_t mode,
          c10::optional<torch::Tensor> R, c10::optional<torch::Tensor> Wn, c10::optional<torch::Tensor> C2,
          c10::optional<torch::Tensor> P, c10::optional<torch::Tensor> Pin, int64_t T_in, int64_t n_norm,
          double eps, c10::optional<torch::Tensor> Rout, torch::Tensor ws, torch::Tensor flags, int64_t grid,
          int64_t depth) {
  const int M = (int)A.size(0);
  TORCH_CHECK(M >= 1 && M <= 16, "w3_sk: 1 <= M <= 16");
  TORCH_CHECK(A.scalar_type() == at::kBFloat16 && A.stride(1) == 1 && A.size(1) == k && A.stride(0) % 64 == 0,
              "w3_sk: A [M, K] bf16, unit column stride, row stride a multiple of 64 elements");
  TORCH_CHECK(B.scalar_type() == at::kInt && B.is_contiguous() && B.numel() == k / 16 * 3 * (n / 2),
              "w3_sk: B planar [K/16, 3, N/2] int32");
  TORCH_CHECK(S.scalar_type() == at::kBFloat16 && S.is_contiguous() && S.numel() == k / 128 * n, "w3_sk: S");
  TORCH_CHECK(C.is_contiguous() && C.scalar_type() == at::kBFloat16 && C.numel() == (int64_t)M * (mode == 3 ? n / 2 : n),
              "w3_sk: C");
  TORCH_CHECK(supported(n, k, mode), "w3_sk: shape / mode not instantiated ", n, " ", k, " ", mode);
  TORCH_CHECK(grid >= 1 && ws.numel() >= grid * 16 * TILE_N && flags.numel() >= grid, "w3_sk: workspace");
  Epi epi{};
  if (mode == 1) {
    TORCH_CHECK(R && Wn && C2 && P, "w3_sk mode 1: R, W, C2, P");
    TORCH_CHECK(P->numel() == (int64_t)M * (n / TILE_N) && P->scalar_type() == at::kFloat, "w3_sk: P [M, N/128] fp32");
    TORCH_CHECK(R->is_contiguous() && C2->is_contiguous() && Wn->numel() == n, "w3_sk mode 1 operands");
    epi.R = reinterpret_cast<const __nv_bfloat16*>(R->data_ptr());
    epi.Wn = reinterpret_cast<const __nv_bfloat16*>(Wn->data_ptr());
    epi.C2 = reinterpret_cast<__nv_bfloat16*>(C2->data_ptr());
    epi.P = P->data_ptr<float>();
  }
  if (mode >= 2) {
    TORCH_CHECK(Pin && Pin->is_contiguous() && T_in >= 1 && T_in <= 80 && Pin->numel() >= (int64_t)M * T_in,
                "w3_sk mode 2/3: Pin [M, T_in <= 80]");
    epi.Pin = Pin->data_ptr<float>();
    epi.T_in = (int)T_in;
    epi.n_norm = (int)n_norm;
    epi.eps = (float)eps;
    epi.Rout = (Rout && Rout->numel() >= M) ? Rout->data_ptr<float>() : nullptr;
  }
  W3SK_S(34816, 5120, 3)
  W3SK_S(5120, 17408, 1)
  W3SK_S(5120, 6144, 1)
  W3SK_S(16384, 5120, 2)
  W3SK_S(14336, 5120, 2)
  W3SK_S(34816, 5120, 0)
  W3SK_S(5120, 17408, 0)
  W3SK_S(5120, 6144, 0)
  W3SK_S(16384, 5120, 0)
  W3SK_S(14336, 5120, 0)
  TORCH_CHECK(false, "w3_sk: depth ", depth, " not instantiated");
}

}  // namespace w3sk

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &w3sk::gemm, py::arg("A"), py::arg("B"), py::arg("S"), py::arg("C"), py::arg("n"), py::arg("k"),
        py::arg("mode"), py::arg("R") = py::none(), py::arg("W") = py::none(), py::arg("C2") = py::none(),
        py::arg("P") = py::none(), py::arg("Pin") = py::none(), py::arg("T_in") = 0, py::arg("n_norm") = 0,
        py::arg("eps") = 0.0, py::arg("Rout") = py::none(), py::arg("ws"), py::arg("flags"), py::arg("grid"),
        py::arg("depth") = 4);
  m.def("supported", &w3sk::supported);
#if W3SK_PROF
  m.def("prof", []() {
    auto t = torch::empty({256 * 8 * 16}, torch::dtype(torch::kInt64));
    cudaMemcpyFromSymbol(t.data_ptr(), w3sk::g_prof, sizeof(unsigned long long) * 256 * 8 * 16);
    return t;
  });
  m.def("prof_clear", []() {
    static unsigned long long z[256 * 8 * 16] = {0};
    cudaMemcpyToSymbol(w3sk::g_prof, z, sizeof(z));
  });
#endif
}
