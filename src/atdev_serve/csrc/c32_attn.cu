// c=32 varlen decode / MTP-verify full attention (ATDEV_OPT=c32_attn, src/atdev_serve/c32_attn.py).
//
// Replaces vLLM 0.31.0 TRITON_ATTN's 2D kernel_unified_attention for batches of short query rows per request
// (MTP k=2 verify: 3 rows / request; any per-request query length via cu_seqlens_q, ql * G <= 16 MT rows) with a
// bf16 paged KV cache, head 256, causal, no sinks / alibi / softcap / window / qq_bias / fp8.
//
// CTA = (request, kv head) [, split]: all ql * G query rows of the request that share the kv head (Qwen3.8: 3 x 6 =
// 18 rows, padded to MT = 2 m16 tiles) read each K / V tile ONCE (Triton: BLOCK_M 16 / BLOCK_Q 2 -> 2 programs per
// (request, kv head), each reading the whole K / V). 64 MT threads: warp pair per m16 tile; per 32-key tile warp kh
// computes S for keys [16 kh, 16 kh + 16), the pair exchanges row max / bf16 P / row-sum partials through smem and
// each warp accumulates O for one 128-dim half over all 32 keys. cp.async K / V ring (ST stages), Q fragments in
// registers. Requests are served longest-first (blockIdx -> rank by seq len, computed on the GPU) so the long CTAs
// start in the first wave. Graph-capturable: every length is read on the GPU, idle CTAs exit.
//
// EXACT (nsplit 1): bit-identical to TRITON_ATTN 2D (TILE_SIZE 32, BLOCK_M 16, 4 warps, compiled for sm_80):
//   S = fma(scale, mma chain over the 16 dim chunks from 0, 0); key j > ctx + tok -> -inf;
//   m = max(M, rowmax S); m = m > -inf ? m : 0; P = ex2((S - m) * log2e) (ex2.approx, no ftz);
//   l = per 8-key group (p[2t] + p[2t+1]) -> + bfly 2 -> + bfly 1, groups combined as C32_LORDER;
//   alpha = ex2((M - m) * log2e); L = fma(L, alpha, l) (L0 = 1, M0 = -inf); acc = mma(bf16 P, V, acc * alpha)
//   (k16 steps keys 0-15 then 16-31); out = bf16(div.full(acc, L)).
// SPLIT (nsplit > 1, rounding-level): the tile range of a request is cut into nsplit_r = cdiv(nt, tps) pieces
//   (tps = max(C32_TPS_MIN, cdiv(nt, nsplit))); each piece runs the same recurrence from (-inf, 1, 0) and stores
//   (O, m, l) fp32 partials; the last-arriving CTA of the (request, kv head) merges them (self-resetting counter).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace {

typedef __nv_bfloat16 bf16;
constexpr int kHS = 256;
constexpr int kPitch = kHS + 8;  // 528 B rows: ldmatrix conflict-free
constexpr int kTile = 32;
#ifndef C32_LORDER
#define C32_LORDER 2  // 8-key group sums z0..z3 -> l: 0 ((z0+z1)+z2)+z3, 1 (z0+z1)+(z2+z3), 2 (z0+z2)+(z1+z3)
#endif
#ifndef C32_TPS_MIN
#define C32_TPS_MIN 4  // split mode: minimum 32-key tiles per split
#endif
#ifndef C32_SORT
#define C32_SORT 1  // 1: longest request first (blockIdx -> length rank)
#endif
#ifndef C32_ABL
#define C32_ABL 0  // timing ablation (wrong results): 1 no K/V loads after the prologue, 2 no QK mma, 4 no PV mma
#endif
#ifndef C32_MAXR
#define C32_MAXR 256  // max requests per call (smem rank table)
#endif

struct Args {
  const bf16* q; int64_t sq_tok, sq_head;
  const bf16* kc; int64_t sk_blk, sk_tok, sk_head;
  const bf16* vc; int64_t sv_blk, sv_tok, sv_head;
  const int* bt; int64_t sbt;
  const int* seqlen;   // [R]
  const int* cuq;      // [R + 1]
  int R, page, G, QH, kvh, nsplit;
  int xs;              // ATT2 hybrid: max requests split over 2 dim-half CTAs
  float scale;
  bf16* out; int64_t so_tok, so_head;
  float* opart;        // split: [R][kvh][nsplit][16 MT][256]
  float* ml;           // split: [R][kvh][nsplit][16 MT][2]
  unsigned* cnt;       // split: [R][kvh] arrival counters (zeroed once, self-resetting)
};

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem, bool pred) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 t = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&t);
}
__device__ __forceinline__ float ex2_tr(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float exp_tr(float x) { return ex2_tr(__fmul_rn(x, __int_as_float(0x3FB8AA3B))); }
__device__ __forceinline__ float div_full(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

// MT m16 tiles (16 MT query rows), ST ring stages (2: Q staged in stage 1, 2 CTAs / SM; 3: 1 CTA / SM)
template <int MT, int ST, bool SPLIT>
__global__ __launch_bounds__(64 * MT, ST == 2 ? 2 : 1) void c32_attn_kernel(Args p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int NT = 64 * MT;
  constexpr int ROWS = 16 * MT;
  static_assert(ROWS <= 2 * kTile, "Q staging must fit one ring stage");
  auto kt = [&](int b) -> bf16 (*)[kPitch] { return reinterpret_cast<bf16 (*)[kPitch]>(smem_raw + b * 2 * kTile * kPitch * 2); };
  auto vt = [&](int b) -> bf16 (*)[kPitch] { return kt(b) + kTile; };
  bf16 (*Qs)[kPitch] = kt(1);
  unsigned char* tail = smem_raw + ST * 2 * kTile * kPitch * 2;
  float (*xm)[2][2][32] = reinterpret_cast<float (*)[2][2][32]>(tail);                 // [mt][kh][r][lane]
  uint4 (*xp)[2][32] = reinterpret_cast<uint4 (*)[2][32]>(tail + MT * 512);           // [mt][kh][lane]
  float (*xl)[2][2][2][32] = reinterpret_cast<float (*)[2][2][2][32]>(tail + MT * 1536);  // [mt][kh][nt][r][lane]
  int* rank = reinterpret_cast<int*>(tail + MT * 2560);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int mt = warp >> 1, kh = warp & 1;
  const int h = blockIdx.x % p.kvh;
  int r = blockIdx.x / p.kvh;
#if C32_SORT
  {
    // rank of request i = #longer requests (+ #equal with a lower index); CTA rank r serves rank[r]
    int* lens = rank + C32_MAXR;
    for (int i = tid; i < p.R; i += NT) lens[i] = p.seqlen[i] * (p.cuq[i + 1] > p.cuq[i]);
    __syncthreads();
    for (int i = tid; i < p.R; i += NT) {
      const int li = lens[i];
      int k = 0;
      for (int j = 0; j < p.R; ++j) {
        const int lj = lens[j];
        k += (lj > li) || (lj == li && j < i);
      }
      rank[k] = i;
    }
    __syncthreads();
    r = rank[r];
  }
#endif
  const int q0 = p.cuq[r], ql = p.cuq[r + 1] - q0;
  const int L = p.seqlen[r];
  if (ql <= 0 || L <= 0) return;
  const int ctx = L - ql;
  const int nrows = ql * p.G;  // host guarantees <= ROWS
  const int nt_all = (L + kTile - 1) / kTile;
  int j0 = 0, j1 = nt_all, split = 0, nsp = 1;
  if (SPLIT) {
    int tps = (nt_all + p.nsplit - 1) / p.nsplit;
    tps = tps < C32_TPS_MIN ? C32_TPS_MIN : tps;
    nsp = (nt_all + tps - 1) / tps;
    split = blockIdx.y;
    if (split >= nsp) return;
    j0 = split * tps;
    j1 = min(j0 + tps, nt_all);
  }
  const int nld = j1 - j0;
  const int* btr = p.bt + (int64_t)r * p.sbt;
  const int lc = tid & 31, lr = tid >> 5;
  auto load_tile = [&](int ld, int buf) {
    const int jt = (j0 + ld) * kTile;
    const int pj0 = jt / p.page, off0 = jt - pj0 * p.page;
    const bool one_page = off0 + kTile <= p.page;
    const int blk0 = btr[pj0];
    const bf16* kb0 = p.kc + (int64_t)blk0 * p.sk_blk + (int64_t)h * p.sk_head + lc * 8;
    const bf16* vb0 = p.vc + (int64_t)blk0 * p.sv_blk + (int64_t)h * p.sv_head + lc * 8;
#pragma unroll
    for (int rr = 0; rr < kTile / (NT / 32); ++rr) {
      const int rw = lr + rr * (NT / 32);
      const int j = jt + rw;
      const bool ok = j < L;
      const bf16 *ks, *vs;
      if (one_page || !ok) {
        ks = kb0 + (int64_t)(off0 + (ok ? rw : 0)) * p.sk_tok;
        vs = vb0 + (int64_t)(off0 + (ok ? rw : 0)) * p.sv_tok;
      } else {
        const int pj = j / p.page, off = j - pj * p.page, blk = btr[pj];
        ks = p.kc + (int64_t)blk * p.sk_blk + (int64_t)off * p.sk_tok + (int64_t)h * p.sk_head + lc * 8;
        vs = p.vc + (int64_t)blk * p.sv_blk + (int64_t)off * p.sv_tok + (int64_t)h * p.sv_head + lc * 8;
      }
      cp_async_16(&kt(buf)[rw][lc * 8], ks, ok);
      cp_async_16(&vt(buf)[rw][lc * 8], vs, ok);
    }
  };
  load_tile(0, 0);
  cp_async_commit();
  // Q rows: row = tok * G + head-in-group (Triton order); rows >= nrows are zero
  for (int e = tid; e < ROWS * (kHS / 8); e += NT) {
    const int row = e >> 5, c = e & 31;
    const bool ok = row < nrows;
    const int tok = ok ? row / p.G : 0, hl = ok ? row - tok * p.G : 0;
    const bf16* src = p.q + (int64_t)(q0 + tok) * p.sq_tok + (int64_t)(h * p.G + hl) * p.sq_head + c * 8;
    cp_async_16(&Qs[row][c * 8], src, ok);
  }
  cp_async_commit();
  if (ST == 3 && 1 < nld) load_tile(1, 2);
  if (ST == 3) cp_async_commit();
  // per-thread rows g, g + 8 of m-tile mt: causal limit ctx + tok (pad rows: -1 -> all masked)
  int lim[2];
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    const int row = mt * 16 + g + 8 * rh;
    lim[rh] = row < nrows ? ctx + row / p.G : -1;
  }
  float o[16][4];
#pragma unroll
  for (int d = 0; d < 16; ++d)
#pragma unroll
    for (int e = 0; e < 4; ++e) o[d][e] = 0.f;
  float m_r[2] = {-INFINITY, -INFINITY}, l_r[2] = {1.f, 1.f};
  const int bar_id = 1 + mt;
  uint32_t qf[kHS / 16][4];
  {
    if (ST == 3) cp_async_wait<1>(); else cp_async_wait<0>();
    __syncthreads();
    const bf16* qa = &Qs[mt * 16 + (lane & 15)][(lane >> 4) * 8];
#pragma unroll
    for (int ks = 0; ks < kHS / 16; ++ks) ldsm_x4(qf[ks], qa + ks * 16);
  }
  // ST 3 ring: tile 0 -> stage 0, tile 1 -> stage 2 (Q staging = stage 1 is free after the Q fragment load)
  int buf = 0;
  for (int ld = 0; ld < nld; ++ld) {
    if (ST == 2) {
      __syncthreads();
      if (ld + 1 < nld && !(C32_ABL & 1)) load_tile(ld + 1, (ld + 1) & 1);
      cp_async_commit();
      cp_async_wait<1>();
      __syncthreads();
      buf = ld & 1;
    } else {
      cp_async_wait<1>();
      __syncthreads();
      // stages in use: ld -> s(ld), ld+1 -> s(ld+1); s(i) = {0, 2, 1, 0, 2, 1, ...}
      buf = (3 - (ld % 3)) % 3;
      if (ld + 2 < nld) load_tile(ld + 2, (3 - ((ld + 2) % 3)) % 3);
      cp_async_commit();
    }
    float s[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) s[nt][e] = 0.f;
    {
      const bf16* kb = &kt(buf)[kh * 16 + (lane & 7) + ((lane >> 4) << 3)][((lane >> 3) & 1) * 8];
      uint32_t b[2][4];
      ldsm_x4(b[0], kb);
#pragma unroll
      for (int ks = 0; ks < kHS / 16; ++ks) {
        const int c = ks & 1;
        if (ks + 1 < kHS / 16) ldsm_x4(b[c ^ 1], kb + (ks + 1) * 16);
        if (!(C32_ABL & 2)) mma16816(s[0], qf[ks], b[c][0], b[c][1]);
        if (!(C32_ABL & 2)) mma16816(s[1], qf[ks], b[c][2], b[c][3]);
      }
    }
    const int jbase = (j0 + ld) * kTile + kh * 16;
    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int j = jbase + nt * 8 + 2 * t + (e & 1);
        float v = __fmaf_rn(p.scale, s[nt][e], 0.f);
        if (j > lim[e >> 1]) v = -INFINITY;
        s[nt][e] = v;
        mx[e >> 1] = fmaxf(mx[e >> 1], v);
      }
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      mx[rh] = fmaxf(mx[rh], __shfl_xor_sync(0xffffffffu, mx[rh], 1));
      mx[rh] = fmaxf(mx[rh], __shfl_xor_sync(0xffffffffu, mx[rh], 2));
      xm[mt][kh][rh][lane] = mx[rh];
    }
    bar_sync(bar_id, 64);
    float al[2], mn[2];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      float m = fmaxf(m_r[rh], fmaxf(mx[rh], xm[mt][kh ^ 1][rh][lane]));
      m = m > -INFINITY ? m : 0.f;
      al[rh] = exp_tr(__fsub_rn(m_r[rh], m));
      mn[rh] = m;
      m_r[rh] = m;
    }
    uint32_t own[4];
    float z[2][2];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const float p0 = exp_tr(__fsub_rn(s[nt][0], mn[0]));
      const float p1 = exp_tr(__fsub_rn(s[nt][1], mn[0]));
      const float p2 = exp_tr(__fsub_rn(s[nt][2], mn[1]));
      const float p3 = exp_tr(__fsub_rn(s[nt][3], mn[1]));
      own[nt * 2 + 0] = pack_bf16(p0, p1);
      own[nt * 2 + 1] = pack_bf16(p2, p3);
      float x0 = __fadd_rn(p0, p1), x1 = __fadd_rn(p2, p3);
      x0 = __fadd_rn(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
      x1 = __fadd_rn(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
      x0 = __fadd_rn(x0, __shfl_xor_sync(0xffffffffu, x0, 1));
      x1 = __fadd_rn(x1, __shfl_xor_sync(0xffffffffu, x1, 1));
      z[nt][0] = x0; z[nt][1] = x1;
      xl[mt][kh][nt][0][lane] = x0;
      xl[mt][kh][nt][1][lane] = x1;
    }
    xp[mt][kh][lane] = make_uint4(own[0], own[1], own[2], own[3]);
    bar_sync(bar_id, 64);
    uint32_t pa[2][4];
    {
      const uint4 u = xp[mt][kh ^ 1][lane];
      const uint32_t oth[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        pa[0][i] = kh ? oth[i] : own[i];
        pa[1][i] = kh ? own[i] : oth[i];
      }
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        const float y0 = xl[mt][kh ^ 1][0][rh][lane], y1 = xl[mt][kh ^ 1][1][rh][lane];
        const float z0 = kh ? y0 : z[0][rh], z1 = kh ? y1 : z[1][rh];
        const float z2 = kh ? z[0][rh] : y0, z3 = kh ? z[1][rh] : y1;
        float l;
#if C32_LORDER == 0
        l = __fadd_rn(__fadd_rn(__fadd_rn(z0, z1), z2), z3);
#elif C32_LORDER == 1
        l = __fadd_rn(__fadd_rn(z0, z1), __fadd_rn(z2, z3));
#else
        l = __fadd_rn(__fadd_rn(z0, z2), __fadd_rn(z1, z3));
#endif
        l_r[rh] = __fmaf_rn(l_r[rh], al[rh], l);
      }
    }
    {
      const bf16* vb = &vt(buf)[(lane & 7) + (((lane >> 3) & 1) << 3)][kh * 128 + (lane >> 4) * 8];
#pragma unroll
      for (int d = 0; d < 16; ++d) {
        o[d][0] = __fmul_rn(o[d][0], al[0]); o[d][1] = __fmul_rn(o[d][1], al[0]);
        o[d][2] = __fmul_rn(o[d][2], al[1]); o[d][3] = __fmul_rn(o[d][3], al[1]);
      }
      uint32_t b[2][4], b2[2][4];
      ldsm_x4_t(b[0], vb);
      ldsm_x4_t(b2[0], vb + 16 * kPitch);
#pragma unroll
      for (int dp = 0; dp < 8; ++dp) {
        const int c = dp & 1;
        if (dp + 1 < 8) {
          ldsm_x4_t(b[c ^ 1], vb + (dp + 1) * 16);
          ldsm_x4_t(b2[c ^ 1], vb + 16 * kPitch + (dp + 1) * 16);
        }
        if (!(C32_ABL & 4)) mma16816(o[2 * dp], pa[0], b[c][0], b[c][1]);
        if (!(C32_ABL & 4)) mma16816(o[2 * dp + 1], pa[0], b[c][2], b[c][3]);
        if (!(C32_ABL & 4)) mma16816(o[2 * dp], pa[1], b2[c][0], b2[c][1]);
        if (!(C32_ABL & 4)) mma16816(o[2 * dp + 1], pa[1], b2[c][2], b2[c][3]);
      }
    }
  }
  cp_async_wait<0>();
  if (!SPLIT) {
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const int row = mt * 16 + g + 8 * rh;
      if (row >= nrows) continue;
      const int tok = row / p.G, hl = row - tok * p.G;
      bf16* op = p.out + (int64_t)(q0 + tok) * p.so_tok + (int64_t)(h * p.G + hl) * p.so_head + kh * 128 + 2 * t;
#pragma unroll
      for (int d = 0; d < 16; ++d)
        *reinterpret_cast<uint32_t*>(op + d * 8) =
            pack_bf16(div_full(o[d][2 * rh], l_r[rh]), div_full(o[d][2 * rh + 1], l_r[rh]));
    }
    return;
  }
  // ---- split: store partials, last arrival merges
  const int64_t pb = (((int64_t)r * p.kvh + h) * p.nsplit + split) * ROWS;
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    const int row = mt * 16 + g + 8 * rh;
    float* op = p.opart + (pb + row) * kHS + kh * 128 + 2 * t;
#pragma unroll
    for (int d = 0; d < 16; ++d) __stcg(reinterpret_cast<float2*>(op + d * 8), make_float2(o[d][2 * rh], o[d][2 * rh + 1]));
    if (kh == 0 && t == 0) __stcg(reinterpret_cast<float2*>(p.ml + (pb + row) * 2), make_float2(m_r[rh], l_r[rh]));
  }
  __threadfence();
  __syncthreads();
  int* last = rank;  // reuse
  if (tid == 0) {
    unsigned* c = p.cnt + (int64_t)r * p.kvh + h;
    const unsigned old = atomicAdd(c, 1u);
    const bool is_last = old == (unsigned)(nsp - 1);
    if (is_last) *c = 0u;
    last[0] = is_last;
  }
  __syncthreads();
  if (!last[0]) return;
  __threadfence();
  // merge: thread -> (row, 8 dims); rows < nrows
  const int64_t pb0 = ((int64_t)r * p.kvh + h) * p.nsplit * ROWS;
  for (int e = tid; e < nrows * (kHS / 8); e += NT) {
    const int row = e >> 5, d = (e & 31) * 8;
    float M = -INFINITY;
    for (int sp = 0; sp < nsp; ++sp) M = fmaxf(M, __ldcg(p.ml + (pb0 + (int64_t)sp * ROWS + row) * 2));
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, ls = 0.f;
    for (int sp = 0; sp < nsp; ++sp) {
      const int64_t ix = pb0 + (int64_t)sp * ROWS + row;
      const float2 mlv = __ldcg(reinterpret_cast<const float2*>(p.ml + ix * 2));
      const float w = exp_tr(__fsub_rn(mlv.x, M));
      ls = __fmaf_rn(w, mlv.y, ls);
      const float4 a = __ldcg(reinterpret_cast<const float4*>(p.opart + ix * kHS + d));
      const float4 b = __ldcg(reinterpret_cast<const float4*>(p.opart + ix * kHS + d + 4));
      acc[0] = __fmaf_rn(w, a.x, acc[0]); acc[1] = __fmaf_rn(w, a.y, acc[1]);
      acc[2] = __fmaf_rn(w, a.z, acc[2]); acc[3] = __fmaf_rn(w, a.w, acc[3]);
      acc[4] = __fmaf_rn(w, b.x, acc[4]); acc[5] = __fmaf_rn(w, b.y, acc[5]);
      acc[6] = __fmaf_rn(w, b.z, acc[6]); acc[7] = __fmaf_rn(w, b.w, acc[7]);
    }
    const int tok = row / p.G, hl = row - tok * p.G;
    bf16* op = p.out + (int64_t)(q0 + tok) * p.so_tok + (int64_t)(h * p.G + hl) * p.so_head + d;
    *reinterpret_cast<uint4*>(op) = make_uint4(pack_bf16(div_full(acc[0], ls), div_full(acc[1], ls)),
                                               pack_bf16(div_full(acc[2], ls), div_full(acc[3], ls)),
                                               pack_bf16(div_full(acc[4], ls), div_full(acc[5], ls)),
                                               pack_bf16(div_full(acc[6], ls), div_full(acc[7], ls)));
  }
}

// ---------------------------------------------------------------------------------------------------------------
// ATT2 PIPE (stages 2x: KS = x): EXACT, software-pipelined. Separate K / V rings of KS 32-key slots; K runs one tile
// ahead of V so QK(i+1) (independent of tile i) is interleaved with PV(i) mma by mma, hiding the 16-deep QK mma
// chain under PV. Iteration i: wait K(i+1), V(i) -> one __syncthreads -> issue K(i+KS), V(i+KS-1) -> softmax(i)
// -> PV(i) || QK(i+1). Per-row fp32 op sequence identical to the EXACT kernel (only cross-tile reordering).
// KS 2: same 67.6 KB smem as ST 2 (2 CTAs / SM); KS 3: 1 CTA / SM, one more tile in flight.
// DSPL (C32_DSPL > 1 at launch): exact head-dim split for under-filled batches: blockIdx.y = dim part; each CTA
// computes the full S / P / l (identical ops) and PV only for its 256 / DSPL output dims (V read partially).
// rank prologue shared by the pipe kernels: rank[k] = request of length rank k (longest first), lens[i] = active len
template <int NT>
__device__ __forceinline__ void c32_rank_table(const Args& p, int* rank) {
  int* lens = rank + C32_MAXR;
  const int tid = threadIdx.x;
  for (int i = tid; i < p.R; i += NT) lens[i] = p.seqlen[i] * (p.cuq[i + 1] > p.cuq[i]);
  __syncthreads();
  for (int i = tid; i < p.R; i += NT) {
    const int li = lens[i];
    int k = 0;
    for (int j = 0; j < p.R; ++j) {
      const int lj = lens[j];
      k += (lj > li) || (lj == li && j < i);
    }
    rank[k] = i;
  }
  __syncthreads();
}

template <int MT, int KS, int DSPL>
__device__ __forceinline__ void c32_pipe_body(const Args& p, unsigned char* smem_raw, int r, int h, int dpart) {
  constexpr int NT = 64 * MT;
  constexpr int ROWS = 16 * MT;
  constexpr int DW = kHS / DSPL;      // output dims per CTA
  constexpr int DWW = DW / 2;         // per warp of the pair
  constexpr int NDP = DWW / 16;       // 16-dim steps per warp (8 at DSPL 1)
  static_assert(ROWS <= kTile, "Q staging must fit one slot");
  static_assert(NDP >= 1, "DSPL too large");
  auto ks_ = [&](int b) -> bf16 (*)[kPitch] { return reinterpret_cast<bf16 (*)[kPitch]>(smem_raw + b * kTile * kPitch * 2); };
  auto vs_ = [&](int b) -> bf16 (*)[kPitch] { return ks_(KS + b); };
  bf16 (*Qs)[kPitch] = vs_(KS - 1);
  unsigned char* tail = smem_raw + 2 * KS * kTile * kPitch * 2;
  float (*xm)[2][2][32] = reinterpret_cast<float (*)[2][2][32]>(tail);
  uint4 (*xp)[2][32] = reinterpret_cast<uint4 (*)[2][32]>(tail + MT * 512);
  float (*xl)[2][2][2][32] = reinterpret_cast<float (*)[2][2][2][32]>(tail + MT * 1536);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int mt = warp >> 1, kh = warp & 1;
  const int dbase = dpart * DW;
  const int q0 = p.cuq[r], ql = p.cuq[r + 1] - q0;
  const int L = p.seqlen[r];
  if (ql <= 0 || L <= 0) return;
  const int ctx = L - ql;
  const int nrows = ql * p.G;
  const int nld = (L + kTile - 1) / kTile;
  const int* btr = p.bt + (int64_t)r * p.sbt;
  // V loads only the CTA's dims: chunks [dbase / 8, dbase / 8 + DW / 8)
  // unrolled loader: CPR 16-B chunks per row (K 32, V DW / 8), NT / CPR rows per pass
  auto load_kv = [&](int ld, int buf, auto isv_c) {
    constexpr bool isv = decltype(isv_c)::value;
    if (ld >= nld) return;
#if C32_ABL & 1
    if (ld >= KS) return;
#endif
    constexpr int CPR = isv ? DW / 8 : 32;
    constexpr int RPP = NT / CPR;
    static_assert(RPP >= 1 && kTile % RPP == 0, "loader");
    const int c = (tid % CPR) + (isv ? dbase / 8 : 0), rw0 = tid / CPR;
    const int jt = ld * kTile;
    const int pj0 = jt / p.page, off0 = jt - pj0 * p.page;
    const bool one_page = off0 + kTile <= p.page;
    const int blk0 = btr[pj0];
    const bf16* base = isv ? p.vc : p.kc;
    const int64_t sblk = isv ? p.sv_blk : p.sk_blk, stok = isv ? p.sv_tok : p.sk_tok, shd = isv ? p.sv_head : p.sk_head;
    bf16 (*dst)[kPitch] = isv ? vs_(buf) : ks_(buf);
    const bf16* b0 = base + (int64_t)blk0 * sblk + (int64_t)h * shd + c * 8;
#pragma unroll
    for (int rr = 0; rr < kTile / RPP; ++rr) {
      const int rw = rw0 + rr * RPP;
      const int j = jt + rw;
      const bool ok = j < L;
      const bf16* src;
      if (one_page || !ok) {
        src = b0 + (int64_t)(off0 + (ok ? rw : 0)) * stok;
      } else {
        const int pj = j / p.page, off = j - pj * p.page, blk = btr[pj];
        src = base + (int64_t)blk * sblk + (int64_t)off * stok + (int64_t)h * shd + c * 8;
      }
      cp_async_16(&dst[rw][c * 8], src, ok);
    }
  };
  using KF = std::integral_constant<bool, false>;
  using VT = std::integral_constant<bool, true>;
  // prologue groups: A = {K(0), Q}; P_j = {K(j), V(j-1)} j = 1 .. KS-1
  load_kv(0, 0, KF{});
  for (int e = tid; e < ROWS * (kHS / 8); e += NT) {
    const int row = e >> 5, c = e & 31;
    const bool ok = row < nrows;
    const int tok = ok ? row / p.G : 0, hl = ok ? row - tok * p.G : 0;
    const bf16* src = p.q + (int64_t)(q0 + tok) * p.sq_tok + (int64_t)(h * p.G + hl) * p.sq_head + c * 8;
    cp_async_16(&Qs[row][c * 8], src, ok);
  }
  cp_async_commit();
#pragma unroll
  for (int j = 1; j < KS; ++j) {
    load_kv(j, j, KF{});
    load_kv(j - 1, j - 1, VT{});
    cp_async_commit();
  }
  int lim[2];
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    const int row = mt * 16 + g + 8 * rh;
    lim[rh] = row < nrows ? ctx + row / p.G : -1;
  }
  float o[2 * NDP][4];
#pragma unroll
  for (int d = 0; d < 2 * NDP; ++d)
#pragma unroll
    for (int e = 0; e < 4; ++e) o[d][e] = 0.f;
  float m_r[2] = {-INFINITY, -INFINITY}, l_r[2] = {1.f, 1.f};
  const int bar_id = 1 + mt;
  uint32_t qf[kHS / 16][4];
  float s[2][4];
  const int krow = kh * 16 + (lane & 7) + ((lane >> 4) << 3), kcol = ((lane >> 3) & 1) * 8;
  {
    cp_async_wait<KS - 1>();
    __syncthreads();
    const bf16* qa = &Qs[mt * 16 + (lane & 15)][(lane >> 4) * 8];
#pragma unroll
    for (int ks = 0; ks < kHS / 16; ++ks) ldsm_x4(qf[ks], qa + ks * 16);
    // QK(0)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) s[nt][e] = 0.f;
    const bf16* kb = &ks_(0)[krow][kcol];
    uint32_t b[2][4];
    ldsm_x4(b[0], kb);
#pragma unroll
    for (int ks = 0; ks < kHS / 16; ++ks) {
      const int c = ks & 1;
      if (ks + 1 < kHS / 16) ldsm_x4(b[c ^ 1], kb + (ks + 1) * 16);
      if (!(C32_ABL & 2)) mma16816(s[0], qf[ks], b[c][0], b[c][1]);
      if (!(C32_ABL & 2)) mma16816(s[1], qf[ks], b[c][2], b[c][3]);
    }
  }
  for (int ld = 0; ld < nld; ++ld) {
    // wait {K(ld+1), V(ld)} (KS - 2 newer groups may stay in flight), then refill the slots freed by iteration ld-1
#if C32_PIPE_LATEWAIT
    cp_async_wait<KS - 2>();
    __syncthreads();
    load_kv(ld + KS, (ld + KS) % KS, KF{});
    load_kv(ld + KS - 1, (ld + KS - 1) % KS, VT{});
    cp_async_commit();
#else
    // keep the load stream busy: refill first (WAR guarded by the first barrier), then wait {K(ld+1), V(ld)}
    __syncthreads();
    load_kv(ld + KS, (ld + KS) % KS, KF{});
    load_kv(ld + KS - 1, (ld + KS - 1) % KS, VT{});
    cp_async_commit();
    cp_async_wait<KS - 1>();
    __syncthreads();
#endif
    const int jbase = ld * kTile + kh * 16;
    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int j = jbase + nt * 8 + 2 * t + (e & 1);
        float v = __fmaf_rn(p.scale, s[nt][e], 0.f);
        if (j > lim[e >> 1]) v = -INFINITY;
        s[nt][e] = v;
        mx[e >> 1] = fmaxf(mx[e >> 1], v);
      }
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      mx[rh] = fmaxf(mx[rh], __shfl_xor_sync(0xffffffffu, mx[rh], 1));
      mx[rh] = fmaxf(mx[rh], __shfl_xor_sync(0xffffffffu, mx[rh], 2));
      xm[mt][kh][rh][lane] = mx[rh];
    }
    bar_sync(bar_id, 64);
    float al[2], mn[2];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      float m = fmaxf(m_r[rh], fmaxf(mx[rh], xm[mt][kh ^ 1][rh][lane]));
      m = m > -INFINITY ? m : 0.f;
      al[rh] = exp_tr(__fsub_rn(m_r[rh], m));
      mn[rh] = m;
      m_r[rh] = m;
    }
    uint32_t own[4];
    float z[2][2];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const float p0 = exp_tr(__fsub_rn(s[nt][0], mn[0]));
      const float p1 = exp_tr(__fsub_rn(s[nt][1], mn[0]));
      const float p2 = exp_tr(__fsub_rn(s[nt][2], mn[1]));
      const float p3 = exp_tr(__fsub_rn(s[nt][3], mn[1]));
      own[nt * 2 + 0] = pack_bf16(p0, p1);
      own[nt * 2 + 1] = pack_bf16(p2, p3);
      float x0 = __fadd_rn(p0, p1), x1 = __fadd_rn(p2, p3);
      x0 = __fadd_rn(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
      x1 = __fadd_rn(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
      x0 = __fadd_rn(x0, __shfl_xor_sync(0xffffffffu, x0, 1));
      x1 = __fadd_rn(x1, __shfl_xor_sync(0xffffffffu, x1, 1));
      z[nt][0] = x0; z[nt][1] = x1;
      xl[mt][kh][nt][0][lane] = x0;
      xl[mt][kh][nt][1][lane] = x1;
    }
    xp[mt][kh][lane] = make_uint4(own[0], own[1], own[2], own[3]);
    bar_sync(bar_id, 64);
    uint32_t pa[2][4];
    {
      const uint4 u = xp[mt][kh ^ 1][lane];
      const uint32_t oth[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        pa[0][i] = kh ? oth[i] : own[i];
        pa[1][i] = kh ? own[i] : oth[i];
      }
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        const float y0 = xl[mt][kh ^ 1][0][rh][lane], y1 = xl[mt][kh ^ 1][1][rh][lane];
        const float z0 = kh ? y0 : z[0][rh], z1 = kh ? y1 : z[1][rh];
        const float z2 = kh ? z[0][rh] : y0, z3 = kh ? z[1][rh] : y1;
        float l;
#if C32_LORDER == 0
        l = __fadd_rn(__fadd_rn(__fadd_rn(z0, z1), z2), z3);
#elif C32_LORDER == 1
        l = __fadd_rn(__fadd_rn(z0, z1), __fadd_rn(z2, z3));
#else
        l = __fadd_rn(__fadd_rn(z0, z2), __fadd_rn(z1, z3));
#endif
        l_r[rh] = __fmaf_rn(l_r[rh], al[rh], l);
      }
    }
#pragma unroll
    for (int d = 0; d < 2 * NDP; ++d) {
      o[d][0] = __fmul_rn(o[d][0], al[0]); o[d][1] = __fmul_rn(o[d][1], al[0]);
      o[d][2] = __fmul_rn(o[d][2], al[1]); o[d][3] = __fmul_rn(o[d][3], al[1]);
    }
    // PV(ld) interleaved with QK(ld+1)
    const bool nxt = ld + 1 < nld;
    const bf16* vb = &vs_(ld % KS)[(lane & 7) + (((lane >> 3) & 1) << 3)][dbase + kh * DWW + (lane >> 4) * 8];
    const bf16* kb = &ks_((ld + 1) % KS)[krow][kcol];
    float sn[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) sn[nt][e] = 0.f;
    uint32_t b[2][4], b2[2][4], kf[2][2][4];
    ldsm_x4_t(b[0], vb);
    ldsm_x4_t(b2[0], vb + 16 * kPitch);
    if (nxt) { ldsm_x4(kf[0][0], kb); ldsm_x4(kf[0][1], kb + 16); }
    constexpr int KPD = (kHS / 16) / NDP;  // QK k-steps per PV dim step
    static_assert(KPD % 2 == 0, "KPD");
#pragma unroll
    for (int dp = 0; dp < NDP; ++dp) {
      const int c = dp & 1;
      if (dp + 1 < NDP) {
        ldsm_x4_t(b[c ^ 1], vb + (dp + 1) * 16);
        ldsm_x4_t(b2[c ^ 1], vb + 16 * kPitch + (dp + 1) * 16);
      }
#pragma unroll
      for (int kq = 0; kq < KPD; kq += 2) {
        const int ks0 = dp * KPD + kq;
        // K fragments ks0, ks0+1 are in kf[(ks0/2)&1]; prefetch ks0+2, ks0+3
        const int cb = (ks0 >> 1) & 1;
        if (nxt && ks0 + 2 < kHS / 16) {
          ldsm_x4(kf[cb ^ 1][0], kb + (ks0 + 2) * 16);
          ldsm_x4(kf[cb ^ 1][1], kb + (ks0 + 3) * 16);
        }
        if (kq == 0) {
          if (!(C32_ABL & 4)) mma16816(o[2 * dp], pa[0], b[c][0], b[c][1]);
          if (nxt && !(C32_ABL & 2)) mma16816(sn[0], qf[ks0], kf[cb][0][0], kf[cb][0][1]);
          if (!(C32_ABL & 4)) mma16816(o[2 * dp + 1], pa[0], b[c][2], b[c][3]);
          if (nxt && !(C32_ABL & 2)) mma16816(sn[1], qf[ks0], kf[cb][0][2], kf[cb][0][3]);
          if (!(C32_ABL & 4)) mma16816(o[2 * dp], pa[1], b2[c][0], b2[c][1]);
          if (nxt && !(C32_ABL & 2)) mma16816(sn[0], qf[ks0 + 1], kf[cb][1][0], kf[cb][1][1]);
          if (!(C32_ABL & 4)) mma16816(o[2 * dp + 1], pa[1], b2[c][2], b2[c][3]);
          if (nxt && !(C32_ABL & 2)) mma16816(sn[1], qf[ks0 + 1], kf[cb][1][2], kf[cb][1][3]);
        } else if (nxt) {
          if (!(C32_ABL & 2)) mma16816(sn[0], qf[ks0], kf[cb][0][0], kf[cb][0][1]);
          if (!(C32_ABL & 2)) mma16816(sn[1], qf[ks0], kf[cb][0][2], kf[cb][0][3]);
          if (!(C32_ABL & 2)) mma16816(sn[0], qf[ks0 + 1], kf[cb][1][0], kf[cb][1][1]);
          if (!(C32_ABL & 2)) mma16816(sn[1], qf[ks0 + 1], kf[cb][1][2], kf[cb][1][3]);
        }
      }
    }
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) s[nt][e] = sn[nt][e];
  }
  cp_async_wait<0>();
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    const int row = mt * 16 + g + 8 * rh;
    if (row >= nrows) continue;
    const int tok = row / p.G, hl = row - tok * p.G;
    bf16* op = p.out + (int64_t)(q0 + tok) * p.so_tok + (int64_t)(h * p.G + hl) * p.so_head + dbase + kh * DWW + 2 * t;
#pragma unroll
    for (int d = 0; d < 2 * NDP; ++d)
      *reinterpret_cast<uint32_t*>(op + d * 8) =
          pack_bf16(div_full(o[d][2 * rh], l_r[rh]), div_full(o[d][2 * rh + 1], l_r[rh]));
  }
}

#ifndef C32_XA
#define C32_XA 60  // hybrid: split a request over 2 dim-half CTAs iff len >= C32_XA % of the longest (and >= 8 tiles)
#endif
template <int MT, int KS, int DSPL>
__global__ __launch_bounds__(64 * MT, KS == 2 ? 2 : 1) void c32_pipe_kernel(Args p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  int* rank = reinterpret_cast<int*>(smem_raw + 2 * KS * kTile * kPitch * 2 + MT * 2560);
  const int h = blockIdx.x % p.kvh;
  int r = blockIdx.x / p.kvh;
#if C32_SORT
  c32_rank_table<64 * MT>(p, rank);
  r = rank[r];
#endif
  c32_pipe_body<MT, KS, DSPL>(p, smem_raw, r, h, DSPL > 1 ? blockIdx.y : 0);
}

// ATT2 hybrid: grid (R + xs) * kvh; CTA idx < R serves length rank idx; the xs longest requests whose length is
// >= C32_XA % of the longest run as two dim-half CTAs (idx = rank and idx = R + rank), the rest whole (exact).
template <int MT, int KS>
__global__ __launch_bounds__(64 * MT, KS == 2 ? 2 : 1) void c32_hyb_kernel(Args p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  int* rank = reinterpret_cast<int*>(smem_raw + 2 * KS * kTile * kPitch * 2 + MT * 2560);
  const int* lens = rank + C32_MAXR;
  const int h = blockIdx.x % p.kvh;
  const int idx = blockIdx.x / p.kvh;
  c32_rank_table<64 * MT>(p, rank);
  // block order: [rank 0 part 0, rank 0 part 1, ..., rank xs-1 part 1, rank xs, rank xs+1, ...] (long first)
  const int xs = p.xs;
  const int k = idx < 2 * xs ? idx >> 1 : idx - xs;
  const int part = idx < 2 * xs ? idx & 1 : 0;
  if (k >= p.R) return;
  const int r = rank[k];
  const int lmax = lens[rank[0]], lr = lens[r];
  const bool spl = k < xs && lr >= 8 * kTile && lr * 100 >= C32_XA * lmax;
  if (spl) c32_pipe_body<MT, KS, 2>(p, smem_raw, r, h, part);
  else if (part == 0) c32_pipe_body<MT, KS, 1>(p, smem_raw, r, h, 0);
}

template <int MT, int KS>
void launch_hyb(const Args& a, cudaStream_t st) {
  auto kern = c32_hyb_kernel<MT, KS>;
#ifndef C32_HYB_OCC1
#define C32_HYB_OCC1 0  // 1: pad smem to force 1 CTA / SM (long split CTAs never share an SM)
#endif
  const int smem0 = 2 * KS * kTile * kPitch * 2 + MT * 2560 + C32_MAXR * 8;
  const int smem = C32_HYB_OCC1 && smem0 < 120 * 1024 ? 120 * 1024 : smem0;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr = true;
  }
  kern<<<dim3((a.R + a.xs) * a.kvh), 64 * MT, smem, st>>>(a);
}

template <int MT, int KS, int DSPL>
void launch_pipe(const Args& a, cudaStream_t st) {
  auto kern = c32_pipe_kernel<MT, KS, DSPL>;
  const int smem = 2 * KS * kTile * kPitch * 2 + MT * 2560 + C32_MAXR * 8;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr = true;
  }
  kern<<<dim3(a.R * a.kvh, DSPL), 64 * MT, smem, st>>>(a);
}

template <int MT, int ST, bool SPLIT>
void launch(const Args& a, cudaStream_t st) {
  auto kern = c32_attn_kernel<MT, ST, SPLIT>;
  const int smem = ST * 2 * kTile * kPitch * 2 + MT * 2560 + C32_MAXR * 8;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr = true;
  }
  kern<<<dim3(a.R * a.kvh, SPLIT ? a.nsplit : 1), 64 * MT, smem, st>>>(a);
}

}  // namespace

// q [N, QH, 256]; kc / vc [blocks, page, kvh, 256] views; bt [R, >= pages] int32; seqlen [R] int32 (incl. the query
// rows); cuq [R + 1] int32; out [N, QH, 256]; opart / ml / cnt split scratch (empty for nsplit 1).
void c32_attn_fwd(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor bt, torch::Tensor seqlen,
                  torch::Tensor cuq, torch::Tensor out, double scale, int64_t max_ql, int64_t nsplit, int64_t stages,
                  torch::Tensor opart, torch::Tensor ml, torch::Tensor cnt) {
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && kc.scalar_type() == at::kBFloat16 &&
              vc.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(q.size(2) == kHS && q.stride(2) == 1 && kc.stride(3) == 1 && vc.stride(3) == 1 && out.stride(2) == 1);
  TORCH_CHECK(bt.scalar_type() == at::kInt && seqlen.scalar_type() == at::kInt && cuq.scalar_type() == at::kInt);
  TORCH_CHECK(bt.stride(1) == 1 && seqlen.stride(0) == 1 && cuq.stride(0) == 1);
  const int QH = (int)q.size(1), kvh = (int)kc.size(2);
  TORCH_CHECK(QH % kvh == 0);
  const int G = QH / kvh;
  const int R = (int)seqlen.size(0);
  TORCH_CHECK(R >= 1 && R <= C32_MAXR && cuq.size(0) >= R + 1);
  const int rows = (int)max_ql * G;
  TORCH_CHECK(rows >= 1 && rows <= 32, "c32_attn: max_ql * G must be <= 32");
  Args p{};
  p.q = reinterpret_cast<const bf16*>(q.data_ptr()); p.sq_tok = q.stride(0); p.sq_head = q.stride(1);
  p.kc = reinterpret_cast<const bf16*>(kc.data_ptr()); p.sk_blk = kc.stride(0); p.sk_tok = kc.stride(1); p.sk_head = kc.stride(2);
  p.vc = reinterpret_cast<const bf16*>(vc.data_ptr()); p.sv_blk = vc.stride(0); p.sv_tok = vc.stride(1); p.sv_head = vc.stride(2);
  p.bt = bt.data_ptr<int>(); p.sbt = bt.stride(0);
  p.seqlen = seqlen.data_ptr<int>(); p.cuq = cuq.data_ptr<int>();
  p.R = R; p.page = (int)kc.size(1); p.G = G; p.QH = QH; p.kvh = kvh; p.nsplit = (int)nsplit;
  p.scale = (float)scale;
  p.out = reinterpret_cast<bf16*>(out.data_ptr()); p.so_tok = out.stride(0); p.so_head = out.stride(1);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const bool split = nsplit > 1;
  if (split) {
    TORCH_CHECK(opart.numel() >= (int64_t)R * kvh * nsplit * 32 * kHS && ml.numel() >= (int64_t)R * kvh * nsplit * 32 * 2 &&
                cnt.numel() >= (int64_t)R * kvh);
    p.opart = opart.data_ptr<float>(); p.ml = ml.data_ptr<float>(); p.cnt = reinterpret_cast<unsigned*>(cnt.data_ptr());
  }
  const bool mt1 = rows <= 16;
  if (!split && (stages % 100) / 10 == 9) {  // ATT2 hybrid: stages = 100 * xs + 90 + KS (xs 0: auto = free 2/SM slots)
    const int ks = stages % 10;
    int xs = (int)(stages / 100);
    if (xs == 0) {
      int dev = 0, sms = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
      xs = (sms * (ks == 2 ? 2 : 1) - R * kvh) / kvh;
    }
    p.xs = xs < 0 ? 0 : (xs > R ? R : xs);
    TORCH_CHECK(ks == 2 || ks == 3, "c32_attn hybrid: bad stages code");
    if (ks == 2) { if (mt1) launch_hyb<1, 2>(p, st); else launch_hyb<2, 2>(p, st); }
    else { if (mt1) launch_hyb<1, 3>(p, st); else launch_hyb<2, 3>(p, st); }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (!split && (stages % 100) / 10 == 2) {  // ATT2 pipe: stages = 100 * dspl + 20 + KS
    const int ks = stages % 10, ds = stages >= 100 ? (int)(stages / 100) : 1;
    TORCH_CHECK((ks == 2 || ks == 3) && (ds == 1 || ds == 2 || ds == 4), "c32_attn pipe: bad stages code");
#define C32_P(KS_, DS_) (mt1 ? launch_pipe<1, KS_, DS_>(p, st) : launch_pipe<2, KS_, DS_>(p, st))
    if (ks == 2) { if (ds == 1) C32_P(2, 1); else if (ds == 2) C32_P(2, 2); else C32_P(2, 4); }
    else { if (ds == 1) C32_P(3, 1); else if (ds == 2) C32_P(3, 2); else C32_P(3, 4); }
#undef C32_P
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
#define C32_L(MT_, ST_) (split ? launch<MT_, ST_, true>(p, st) : launch<MT_, ST_, false>(p, st))
  if (stages == 3) { if (mt1) C32_L(1, 3); else C32_L(2, 3); }
  else { if (mt1) C32_L(1, 2); else C32_L(2, 2); }
#undef C32_L
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("c32_attn_fwd", &c32_attn_fwd); }
