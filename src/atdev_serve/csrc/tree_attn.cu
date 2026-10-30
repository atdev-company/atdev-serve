// c=1 tree-verify full attention (ATDEV_OPT=tree_attn, src/atdev_serve/tree_attn.py).
//
// One request, T <= 16 query tokens (the verify tree), G = q heads per kv head (Qwen3.8: 24 / 4 = 6),
// head dim 256, bf16 paged KV cache (vLLM TRITON_ATTN layout, any strides with a contiguous head dim).
// Mask = vLLM unified_attention semantics: key j < ctx = L - T always visible; key j = ctx + jj of the
// tree visible to token i iff jj <= i, plus the additive qq_bias[i][jj] (0 / -inf ancestor mask).
//
// split kernel: grid (kv heads x hspl, NSPLIT); a CTA holds GH = G / hspl q heads (16 GH query rows, row =
// w * 16 + token) of one kv head, so each K/V tile is read once per head group (hspl = 2 by default). 2 GH warps:
// warp pair per q head (Q A-fragments in registers); per 32-key tile the pair splits the QK^T keys, exchanges row
// max and P through smem, and each warp accumulates O for one 128-dim half. 3-stage cp.async ring (the Q staging
// buffer becomes stage 2), mma.m16n8k16 bf16 -> fp32, register online softmax (exp2), the CTA's contiguous tile
// range [s * tps, (s + 1) * tps), tps = ceil(ntiles / NSPLIT) from the GPU seq len (graph-capturable: fixed grid,
// idle splits exit). Partials: unnormalised O (fp32) + (m, l).
// combine kernel: CTA per output row (token, q head), 4 warps over the splits; rounds to bf16 and (optionally)
// applies the attention output gate y = bf16(bf16(o) * sigmoid(gate)) (the two roundings of attention -> Inductor
// mul_sigmoid), and (optionally) stores the tree K / V rows into their cache slots (fused reshape_and_cache).
// KS-c1 profile notes (TA_PROF): per 32-key tile ~3.5k cycles with loads in flight vs ~1.9k without (smem / MIO
// pipe shared by cp.async and ldmatrix); a warp-specialised producer and an L2 prefetch of the key range were
// slower / neutral (TA_L2PF, results in the KS-c1 report), the in-place fused merge (TA_FUSE) slower than the
// combine launch.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

typedef __nv_bfloat16 bf16;
constexpr int kHS = 256;
constexpr int kPitch = kHS + 8;  // bf16 pitch of smem rows (528 B: ldmatrix conflict-free)
constexpr int kTile = 32;        // keys per tile
constexpr int kTok = 16;
#ifndef TA_PROF
#define TA_PROF 0  // 1: printf clock64 phase stamps of CTA (0, 0) thread 0 (debug only)
#endif
#ifndef TA_QREG
#define TA_QREG 1  // 1: each warp keeps its head's Q A-fragments in registers (no per-tile Q smem reads)
#endif
#if !TA_QREG
#error "tree_attn: the 3-stage KV ring reuses the Q staging buffer (TA_QREG=1 required)"
#endif
#ifndef TA_PBF16
#define TA_PBF16 0  // 1: split partials O stored as bf16 (half the partial traffic; extra rounding)
#endif
#ifndef TA_FUSE
#define TA_FUSE 0  // 1: no combine kernel -- the split CTAs of a (kv head, head group) meet at a self-resetting
                   // arrival counter and merge the partials in place (each CTA a slice of the rows x dims);
                   // needs every CTA of the grid co-resident (host: grid <= SMs)
#endif
#ifndef TA_L2PF
#define TA_L2PF 0  // 1: at kernel start, L2-prefetch every K / V line of the CTA's tiles beyond the first two
#endif
#ifndef TA_LAUNCH
#define TA_LAUNCH 3  // timing ablation: 1 split only, 2 combine only
#endif
#ifndef TA_ABL
#define TA_ABL 0  // ablation bits (timing only, wrong results): 1 no K/V tile loads, 2 no QK mma, 4 no PV mma
#endif

struct Args {
  const bf16* q; int64_t sq_tok, sq_head;
  const bf16* kc; int64_t sk_blk, sk_tok, sk_head;
  const bf16* vc; int64_t sv_blk, sv_tok, sv_head;
  const int* bt;
  const int* seqlen;
  const float* qq; int64_t sqq;  // [T][T] additive tree bias (nullptr: causal only)
  int T, page, nsplit, G, QH, hspl;
  int xf;  // ATT2 TA_XF fused merge engaged for this launch  // hspl: CTAs per kv head along the q heads (G / hspl heads each)
  float scale_log2;
  float scale;  // exact mode: the plain softmax scale (TRITON_ATTN: S = fma(scale, QK^T, 0))
  float* opart;  // [kvh][nsplit][16 G][256]
  float* ml;     // [kvh][nsplit][16 G][2]
  bf16* out; int64_t so_tok, so_head;
  const bf16* gate; int64_t sg_tok;  // gate[tok][qh * 256 + d] (nullptr: no gate)
  unsigned* sync;                    // TA_FUSE: [kvh * hspl][2] (arrivals, generation), zeroed once
  // fused KV-cache write (nullptr: the tree K/V are already in the cache): the split kernel reads the T tree keys
  // from kn / vn, the combine kernel stores them to the cache slots slot[tok] (< 0: skipped)
  const bf16* kn; int64_t skn_tok, skn_head;
  const bf16* vn; int64_t svn_tok, svn_head;
  const int64_t* slot;
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
__device__ __forceinline__ unsigned ld_acquire_u32(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_u32(unsigned* p, unsigned v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 t = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&t);
}
__device__ __forceinline__ float fexp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ int tiles_per_split(int ntiles, int nsplit) { return (ntiles + nsplit - 1) / nsplit; }

// merge of one output item (row gr of kv head h = token gr % 16 of q head h G + gr / 16, dims [d, d + 8)) over the
// nact split partials; writes bf16 (optionally gated). Shared by the fused split kernel and the combine kernel.
__device__ __forceinline__ void load_part8(const Args& p, int64_t off, float (&v)[8]) {
#if TA_PBF16
  const uint4 u = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const bf16*>(p.opart) + off));
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(&w[e]);
    v[2 * e] = __low2float(b2); v[2 * e + 1] = __high2float(b2);
  }
#else
  const float4 a = __ldcg(reinterpret_cast<const float4*>(p.opart + off));
  const float4 b = __ldcg(reinterpret_cast<const float4*>(p.opart + off + 4));
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
#endif
}

__device__ __forceinline__ void store_out8(const Args& p, int tok, int qh, int d, float (&y)[8]) {
  if (p.gate) {
    const uint4 gu = *reinterpret_cast<const uint4*>(p.gate + (int64_t)tok * p.sg_tok + (int64_t)qh * kHS + d);
    const uint32_t gw[4] = {gu.x, gu.y, gu.z, gu.w};
#pragma unroll
    for (int e = 0; e < 8; e += 2) {
      const __nv_bfloat162 g2 = *reinterpret_cast<const __nv_bfloat162*>(&gw[e >> 1]);
      const float ga = __low2float(g2), gb = __high2float(g2);
      y[e] = __bfloat162float(__float2bfloat16(y[e])) * (1.0f / (1.0f + __expf(-ga)));
      y[e + 1] = __bfloat162float(__float2bfloat16(y[e + 1])) * (1.0f / (1.0f + __expf(-gb)));
    }
  }
  *reinterpret_cast<uint4*>(p.out + (int64_t)tok * p.so_tok + (int64_t)qh * p.so_head + d) =
      make_uint4(pack_bf16(y[0], y[1]), pack_bf16(y[2], y[3]), pack_bf16(y[4], y[5]), pack_bf16(y[6], y[7]));
}

__device__ __forceinline__ void merge_item(const Args& p, int h, int gr, int d, int nact) {
  const int64_t rows = 16 * p.G;
  const float2* mlp = reinterpret_cast<const float2*>(p.ml) + (int64_t)h * p.nsplit * rows + gr;
  const int64_t base = ((int64_t)h * p.nsplit * rows + gr) * kHS + d;
  float M = -INFINITY;
#pragma unroll 8
  for (int s = 0; s < nact; ++s) M = fmaxf(M, __ldcg(mlp + (int64_t)s * rows).x);
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, ls = 0.f;
  for (int s0 = 0; s0 < nact; s0 += 4) {
    float v[4][8];
    float2 ml[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      if (s0 + u < nact) {
        ml[u] = __ldcg(mlp + (int64_t)(s0 + u) * rows);
        load_part8(p, base + (int64_t)(s0 + u) * rows * kHS, v[u]);
      } else {
        ml[u] = make_float2(-INFINITY, 0.f);
      }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      if (ml[u].x == -INFINITY) continue;
      const float w = fexp2(ml[u].x - M);
      ls += w * ml[u].y;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] = fmaf(w, v[u][e], acc[e]);
    }
  }
  const float il = 1.f / ls;
  float y[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) y[e] = acc[e] * il;
  store_out8(p, gr & 15, h * p.G + (gr >> 4), d, y);
}

__device__ __forceinline__ void kv_write_row(const Args& p, int h, int tok, int c, bool isv) {
  const int64_t sl = p.slot[tok];
  if (sl < 0) return;
  const int64_t blk = sl / p.page, off = sl % p.page;
  if (isv) {
    const uint4 x = *reinterpret_cast<const uint4*>(p.vn + (int64_t)tok * p.svn_tok + (int64_t)h * p.svn_head + c * 8);
    *reinterpret_cast<uint4*>(const_cast<bf16*>(p.vc) + blk * p.sv_blk + off * p.sv_tok + (int64_t)h * p.sv_head + c * 8) = x;
  } else {
    const uint4 x = *reinterpret_cast<const uint4*>(p.kn + (int64_t)tok * p.skn_tok + (int64_t)h * p.skn_head + c * 8);
    *reinterpret_cast<uint4*>(const_cast<bf16*>(p.kc) + blk * p.sk_blk + off * p.sk_tok + (int64_t)h * p.sk_head + c * 8) = x;
  }
}

// 2 G warps: warp w -> q head hw = w >> 1 (16 token rows), half dh = w & 1. Per 32-key tile the pair splits the
// QK^T keys (warp dh: keys [16 dh, 16 dh + 16)), exchanges row max and its bf16 P fragment through smem (pair
// barrier), and each warp accumulates O for its 128-dim half over all 32 keys: 64 mma per warp, no duplicated work,
// O = 64 registers, 3 warps per SM sub-partition.
template <int GH>
__global__ __launch_bounds__(64 * GH, 1) void tree_attn_split_kernel(Args p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  // [Q staging = KV stage 2 after Q went to registers | stage 0 K, V | stage 1 K, V | tail]
  constexpr int QROWS = 16 * GH > 2 * kTile ? 16 * GH : 2 * kTile;
  bf16 (*Qs)[kPitch] = reinterpret_cast<bf16 (*)[kPitch]>(smem_raw);
  auto kt = [&](int b) -> bf16 (*)[kPitch] {
    return reinterpret_cast<bf16 (*)[kPitch]>(smem_raw + (b == 2 ? 0 : (QROWS + b * 2 * kTile) * kPitch * 2));
  };
  auto vt = [&](int b) -> bf16 (*)[kPitch] { return kt(b) + kTile; };
  unsigned char* tail = smem_raw + (QROWS + 4 * kTile) * kPitch * 2;
  float (*bias)[kTok] = reinterpret_cast<float (*)[kTok]>(tail);
  float (*xm)[2][2][32] = reinterpret_cast<float (*)[2][2][32]>(tail + kTok * kTok * 4);        // [pair][half][r][lane]
  uint4 (*xp)[2][32] = reinterpret_cast<uint4 (*)[2][32]>(tail + kTok * kTok * 4 + GH * 512);  // [pair][half][lane]
  constexpr int NT = 64 * GH;
  const int h = blockIdx.x / p.hspl, hg0 = (blockIdx.x % p.hspl) * GH, split = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int hw = warp >> 1, dh = warp & 1;
  const int L = *p.seqlen;
  const int T = p.T;
  const int ctx = L - T;
  const int ntiles = (L + kTile - 1) / kTile;
  const int tps = tiles_per_split(ntiles, p.nsplit);
  const int t0 = split * tps;
  const int t1 = min(t0 + tps, ntiles);
  if (t0 >= t1) return;
#if TA_PROF
  long long ts[40]; int nts = 0;
  ts[nts++] = clock64();
#endif
  // thread -> 16-B column chunk lc of rows lr, lr + NT/32, ... of a K / V tile
  const int lc = tid & 31, lr = tid >> 5;
  auto load_tile = [&](int tile, int buf) {
    if (TA_ABL & 1) return;
    const int jt = tile * kTile;
    const int pj0 = jt / p.page, off0 = jt - pj0 * p.page;
    const bool one_page = off0 + kTile <= p.page;
    const int blk0 = p.bt[pj0];
    const bf16* kb0 = p.kc + (int64_t)blk0 * p.sk_blk + (int64_t)h * p.sk_head + lc * 8;
    const bf16* vb0 = p.vc + (int64_t)blk0 * p.sv_blk + (int64_t)h * p.sv_head + lc * 8;
#pragma unroll
    for (int rr = 0; rr < (kTile + NT / 32 - 1) / (NT / 32); ++rr) {
      const int r = lr + rr * (NT / 32);
      if (r < kTile) {
        const int j = jt + r;
        const bool ok = j < L;
        const bf16 *ks, *vs;
        if (p.kn && j >= ctx && ok) {  // tree token: straight from this step's k / v (cache written by combine)
          ks = p.kn + (int64_t)(j - ctx) * p.skn_tok + (int64_t)h * p.skn_head + lc * 8;
          vs = p.vn + (int64_t)(j - ctx) * p.svn_tok + (int64_t)h * p.svn_head + lc * 8;
        } else if (one_page || !ok) {
          ks = kb0 + (int64_t)(off0 + (ok ? r : 0)) * p.sk_tok;
          vs = vb0 + (int64_t)(off0 + (ok ? r : 0)) * p.sv_tok;
        } else {
          const int pj = j / p.page, off = j - pj * p.page, blk = p.bt[pj];
          ks = p.kc + (int64_t)blk * p.sk_blk + (int64_t)off * p.sk_tok + (int64_t)h * p.sk_head + lc * 8;
          vs = p.vc + (int64_t)blk * p.sv_blk + (int64_t)off * p.sv_tok + (int64_t)h * p.sv_head + lc * 8;
        }
        cp_async_16(&kt(buf)[r][lc * 8], ks, ok);
        cp_async_16(&vt(buf)[r][lc * 8], vs, ok);
      }
    }
  };
  load_tile(t0, 0);
  cp_async_commit();  // group: tile t0
#if TA_L2PF
  // the whole key range of the CTA streams from DRAM at once; the ring's cp.async then hit L2
  for (int e = tid; e < (t1 - t0 - 2) * kTile * 8; e += NT) {
    const int r = e >> 3, part = e & 7;  // 8 = 4 x 128-B lines of K + 4 of V per key row
    const int j = (t0 + 2) * kTile + r;
    if (j < L && !(p.kn && j >= ctx)) {
      const int pj = j / p.page, off = j - pj * p.page, blk = p.bt[pj];
      const bf16* src = (part < 4 ? p.kc + (int64_t)blk * p.sk_blk + (int64_t)off * p.sk_tok + (int64_t)h * p.sk_head
                                  : p.vc + (int64_t)blk * p.sv_blk + (int64_t)off * p.sv_tok + (int64_t)h * p.sv_head) +
                        (part & 3) * 64;
      asm volatile("prefetch.global.L2 [%0];" ::"l"(src));
    }
  }
#endif
  // ---- Q rows (row = w * 16 + tok) -> smem
  for (int e = tid; e < 16 * GH * (kHS / 8); e += NT) {
    const int row = e >> 5, c = e & 31;
    const int w = row >> 4, tok = row & 15;
    const bool ok = tok < T;
    const bf16* src = p.q + (int64_t)(ok ? tok : 0) * p.sq_tok + (int64_t)(h * p.G + hg0 + w) * p.sq_head + c * 8;
    cp_async_16(&Qs[row][c * 8], src, ok);
  }
  cp_async_commit();  // group: Q
  if (t0 + 1 < t1) load_tile(t0 + 1, 1);
  cp_async_commit();  // group: tile t0 + 1
  for (int e = tid; e < kTok * kTok; e += NT) {
    const int i = e >> 4, jj = e & 15;
    float b = 0.f;
    if (i < T && jj < T) b = jj > i ? -INFINITY : (p.qq ? p.qq[(int64_t)i * p.sqq + jj] : 0.f);
    bias[i][jj] = b * 1.4426950408889634f;
  }

  float o[16][4];
#pragma unroll
  for (int d = 0; d < 16; ++d)
#pragma unroll
    for (int e = 0; e < 4; ++e) o[d][e] = 0.f;
  float m_r[2] = {-INFINITY, -INFINITY}, l_r[2] = {0.f, 0.f};
  const int qrow0 = hw * 16;
  const int bar_id = 1 + hw;  // named barrier of the warp pair (0 = __syncthreads)
#if TA_QREG
  uint32_t qf[kHS / 16][4];
  {
    cp_async_wait<1>();  // tile t0 + Q groups landed
    __syncthreads();
    const bf16* qa = &Qs[qrow0 + (lane & 15)][(lane >> 4) * 8];
#pragma unroll
    for (int ks = 0; ks < kHS / 16; ++ks) ldsm_x4(qf[ks], qa + ks * 16);
  }
#endif

  int buf = 0, nbuf = 2;  // stage of `tile`, stage receiving tile + 2 (ring 0 -> 1 -> 2 -> 0)
  for (int tile = t0; tile < t1; ++tile) {
#if TA_PROF
    if (nts < 36) ts[nts++] = clock64();
#endif
    cp_async_wait<1>();
    __syncthreads();  // tile data visible; every warp is done with tile - 1 (its stage gets tile + 2)
#if TA_PROF
    if (nts < 36) ts[nts++] = clock64();
#endif
    if (tile + 2 < t1) load_tile(tile + 2, nbuf);
    cp_async_commit();
    // ---- S = Q K^T for this warp's 16 keys (2 n-tiles), two accumulator sets (even / odd k-steps)
    float s[2][4], s2[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) { s[nt][e] = 0.f; s2[nt][e] = 0.f; }
    if (!(TA_ABL & 2)) {
      const bf16* kb = &kt(buf)[dh * 16 + (lane & 7) + ((lane >> 4) << 3)][((lane >> 3) & 1) * 8];
#if TA_QREG
      uint32_t b[2][4];
      ldsm_x4(b[0], kb);
#pragma unroll
      for (int ks = 0; ks < kHS / 16; ++ks) {
        const int c = ks & 1;
        if (ks + 1 < kHS / 16) ldsm_x4(b[c ^ 1], kb + (ks + 1) * 16);
        float (*acc)[4] = c ? s2 : s;
        mma16816(acc[0], qf[ks], b[c][0], b[c][1]);
        mma16816(acc[1], qf[ks], b[c][2], b[c][3]);
      }
#else
      const bf16* qa = &Qs[qrow0 + (lane & 15)][(lane >> 4) * 8];
      uint32_t a[2][4], b[2][4];
      ldsm_x4(a[0], qa);
      ldsm_x4(b[0], kb);
#pragma unroll
      for (int ks = 0; ks < kHS / 16; ++ks) {
        const int c = ks & 1;
        if (ks + 1 < kHS / 16) {
          ldsm_x4(a[c ^ 1], qa + (ks + 1) * 16);
          ldsm_x4(b[c ^ 1], kb + (ks + 1) * 16);
        }
        float (*acc)[4] = c ? s2 : s;
        mma16816(acc[0], a[c], b[c][0], b[c][1]);
        mma16816(acc[1], a[c], b[c][2], b[c][3]);
      }
#endif
    }
    // ---- mask (log2 domain), local row max
    const int jbase = tile * kTile + dh * 16;
    float mx[2] = {-INFINITY, -INFINITY};
    if (jbase + 16 <= ctx) {  // pure context keys: no mask
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float v = (s[nt][e] + s2[nt][e]) * p.scale_log2;
          s[nt][e] = v;
          mx[e >> 1] = fmaxf(mx[e >> 1], v);
        }
    } else {
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int i = g + 8 * (e >> 1);
          const int j = jbase + nt * 8 + 2 * t + (e & 1);
          float v = (s[nt][e] + s2[nt][e]) * p.scale_log2;
          if (j >= L) v = -INFINITY;
          else if (j >= ctx) v += bias[i][j - ctx];
          s[nt][e] = v;
          mx[e >> 1] = fmaxf(mx[e >> 1], v);
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
      xm[hw][dh][r][lane] = mx[r];
    }
    bar_sync(bar_id, 64);
    float alpha[2], ref[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float mn = fmaxf(m_r[r], fmaxf(mx[r], xm[hw][dh ^ 1][r][lane]));
      ref[r] = mn == -INFINITY ? 0.f : mn;
      alpha[r] = fexp2(m_r[r] - ref[r]);
      m_r[r] = mn;
    }
    float rs[2] = {0.f, 0.f};
    uint32_t pa[2][4];  // [key half kk][a0..a3]
    uint32_t own[4];
    {
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const float p0 = fexp2(s[nt][0] - ref[0]), p1 = fexp2(s[nt][1] - ref[0]);
        const float p2 = fexp2(s[nt][2] - ref[1]), p3 = fexp2(s[nt][3] - ref[1]);
        rs[0] += p0 + p1;
        rs[1] += p2 + p3;
        own[nt * 2 + 0] = pack_bf16(p0, p1);
        own[nt * 2 + 1] = pack_bf16(p2, p3);
      }
      xp[hw][dh][lane] = make_uint4(own[0], own[1], own[2], own[3]);
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) l_r[r] = l_r[r] * alpha[r] + rs[r];
#pragma unroll
    for (int d = 0; d < 16; ++d) {
      o[d][0] *= alpha[0]; o[d][1] *= alpha[0];
      o[d][2] *= alpha[1]; o[d][3] *= alpha[1];
    }
    bar_sync(bar_id, 64);
    {
      const uint4 u = xp[hw][dh ^ 1][lane];
      const uint32_t oth[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        pa[0][i] = dh ? oth[i] : own[i];
        pa[1][i] = dh ? own[i] : oth[i];
      }
    }
    // ---- O[:, dh half] += P V (32 keys)
    if (!(TA_ABL & 4)) {
      const bf16* vb = &vt(buf)[(lane & 7) + (((lane >> 3) & 1) << 3)][dh * 128 + (lane >> 4) * 8];
      uint32_t b[2][4];
      ldsm_x4_t(b[0], vb);
#pragma unroll
      for (int st = 0; st < 16; ++st) {  // st = kk * 8 + dp
        const int kk = st >> 3, dp = st & 7, c = st & 1;
        if (st + 1 < 16) {
          const int k2 = (st + 1) >> 3, d2 = (st + 1) & 7;
          ldsm_x4_t(b[c ^ 1], vb + k2 * 16 * kPitch + d2 * 16);
        }
        mma16816(o[2 * dp], pa[kk], b[c][0], b[c][1]);
        mma16816(o[2 * dp + 1], pa[kk], b[c][2], b[c][3]);
      }
    }
    buf = buf == 2 ? 0 : buf + 1;
    nbuf = nbuf == 2 ? 0 : nbuf + 1;
  }
  cp_async_wait<0>();
#if TA_PROF
  ts[nts++] = clock64();
#endif

  // ---- partials: l of the row = sum over both key halves (same running max)
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    l_r[r] += __shfl_xor_sync(0xffffffffu, l_r[r], 1);
    l_r[r] += __shfl_xor_sync(0xffffffffu, l_r[r], 2);
    xm[hw][dh][r][lane] = l_r[r];
  }
  bar_sync(bar_id, 64);
  const int64_t pbase = ((int64_t)h * p.nsplit + split) * (16 * p.G) + hg0 * 16;
#if TA_PBF16
  bf16* op = reinterpret_cast<bf16*>(p.opart) + (pbase + qrow0) * kHS + dh * 128;
#pragma unroll
  for (int d = 0; d < 16; ++d) {
    const int col = d * 8 + 2 * t;
    *reinterpret_cast<uint32_t*>(op + (int64_t)g * kHS + col) = pack_bf16(o[d][0], o[d][1]);
    *reinterpret_cast<uint32_t*>(op + (int64_t)(g + 8) * kHS + col) = pack_bf16(o[d][2], o[d][3]);
  }
#else
  float* op = p.opart + (pbase + qrow0) * kHS + dh * 128;
#pragma unroll
  for (int d = 0; d < 16; ++d) {
    const int col = d * 8 + 2 * t;
    *reinterpret_cast<float2*>(op + (int64_t)g * kHS + col) = make_float2(o[d][0], o[d][1]);
    *reinterpret_cast<float2*>(op + (int64_t)(g + 8) * kHS + col) = make_float2(o[d][2], o[d][3]);
  }
#endif
  if (dh == 0 && t == 0) {
    const float l0 = l_r[0] + xm[hw][1][0][lane], l1 = l_r[1] + xm[hw][1][1][lane];
    *reinterpret_cast<float2*>(p.ml + (pbase + qrow0 + g) * 2) = make_float2(m_r[0], l0);
    *reinterpret_cast<float2*>(p.ml + (pbase + qrow0 + g + 8) * 2) = make_float2(m_r[1], l1);
  }
#if TA_FUSE
  {
    // ---- meet the other split CTAs of this (kv head, head group), then merge a slice in place
    const int nact = (ntiles + tps - 1) / tps;
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      unsigned* arr = p.sync + 2 * blockIdx.x;
      const unsigned g0 = ld_acquire_u32(arr + 1);  // generation before this CTA's arrival
      const unsigned old = atomicAdd(arr, 1u);
      if (old == (unsigned)nact - 1u) {
        atomicExch(arr, 0u);
        __threadfence();
        st_release_u32(arr + 1, g0 + 1u);
      } else {
        while (ld_acquire_u32(arr + 1) == g0) {
        }
      }
    }
    __syncthreads();
    if (p.kn && hg0 == 0 && split == 0) {  // fused cache write of the T tree tokens of kv head h
      for (int e = tid; e < T * 64; e += NT) kv_write_row(p, h, e >> 6, e & 31, (e >> 5) & 1);
    }
    const int items = 16 * GH * (kHS / 8);
    const int per = (items + nact - 1) / nact;
    for (int it = split * per + tid; it < min(items, (split + 1) * per); it += NT) {
      const int gr = hg0 * 16 + (it >> 5), d = (it & 31) * 8;
      if ((gr & 15) < T) merge_item(p, h, gr, d, nact);
    }
  }
#endif
#if TA_PROF
  ts[nts++] = clock64();
  if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
    for (int k = 1; k < nts; ++k) printf("ta_prof %d %lld\n", k, ts[k] - ts[k - 1]);
  }
#endif
}

// CTA per output row (tok, q head), 4 warps; warp q merges splits q, q + 4, ... (all its loads in flight together,
// online max), the 4 warp results are merged through smem by warp 0; lane -> dims [8 lane, 8 lane + 8).
constexpr int kCombWarps = 4;
__device__ __forceinline__ void load_part(const Args& p, int64_t off, float (&v)[8]) {
#if TA_PBF16
  const uint4 u = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const bf16*>(p.opart) + off));
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(&w[e]);
    v[2 * e] = __low2float(b2); v[2 * e + 1] = __high2float(b2);
  }
#else
  const float4 a = __ldcg(reinterpret_cast<const float4*>(p.opart + off));
  const float4 b = __ldcg(reinterpret_cast<const float4*>(p.opart + off + 4));
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
#endif
}

__global__ __launch_bounds__(32 * kCombWarps) void tree_attn_combine_kernel(Args p) {
  const int lane = threadIdx.x & 31, q = threadIdx.x >> 5;
  const int rid = blockIdx.x;  // = tok * QH + qh
  const int tok = rid / p.QH, qh = rid % p.QH;
  const int h = qh / p.G, w = qh % p.G;
  const int L = *p.seqlen;
  const int ntiles = (L + kTile - 1) / kTile;
  const int tps = tiles_per_split(ntiles, p.nsplit);
  const int nact = (ntiles + tps - 1) / tps;  // <= 64
  const int row = w * 16 + tok;
  const int d = lane * 8;
  const int64_t rows = 16 * p.G;
  const int64_t base = ((int64_t)h * p.nsplit * rows + row) * kHS + d;
  const float2* mlp = reinterpret_cast<const float2*>(p.ml) + (int64_t)h * p.nsplit * rows + row;
  __shared__ float sm_m[kCombWarps], sm_l[kCombWarps];
  __shared__ float sm_acc[kCombWarps][kHS];
  uint4 gu = make_uint4(0, 0, 0, 0);
  if (q == 0 && p.gate) gu = *reinterpret_cast<const uint4*>(p.gate + (int64_t)tok * p.sg_tok + (int64_t)qh * kHS + d);
  if (q == 1 && p.kn && w == 0) {  // fused cache write of (tok, kv head h): lane -> 16 B of the K row and the V row
    const int64_t sl = p.slot[tok];
    if (sl >= 0) {
      const int64_t blk = sl / p.page, off = sl % p.page;
      const uint4 kv4 = *reinterpret_cast<const uint4*>(p.kn + (int64_t)tok * p.skn_tok + (int64_t)h * p.skn_head + d);
      const uint4 vv4 = *reinterpret_cast<const uint4*>(p.vn + (int64_t)tok * p.svn_tok + (int64_t)h * p.svn_head + d);
      *reinterpret_cast<uint4*>(const_cast<bf16*>(p.kc) + blk * p.sk_blk + off * p.sk_tok + (int64_t)h * p.sk_head + d) = kv4;
      *reinterpret_cast<uint4*>(const_cast<bf16*>(p.vc) + blk * p.sv_blk + off * p.sv_tok + (int64_t)h * p.sv_head + d) = vv4;
    }
  }
  float mloc = -INFINITY, lloc = 0.f, acc[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;
  for (int s0 = q; s0 < nact; s0 += 4 * kCombWarps) {
    float2 ml[4];
    float v[4][8];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int s = s0 + u * kCombWarps;
      if (s < nact) {
        ml[u] = __ldcg(mlp + (int64_t)s * rows);
        load_part(p, base + (int64_t)s * rows * kHS, v[u]);
      } else {
        ml[u] = make_float2(-INFINITY, 0.f);
#pragma unroll
        for (int e = 0; e < 8; ++e) v[u][e] = 0.f;
      }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      if (ml[u].x == -INFINITY) continue;
      const float mn = fmaxf(mloc, ml[u].x);
      const float a = mloc == -INFINITY ? 0.f : fexp2(mloc - mn), b = fexp2(ml[u].x - mn);
      lloc = lloc * a + ml[u].y * b;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] = acc[e] * a + v[u][e] * b;
      mloc = mn;
    }
  }
  if (lane == 0) { sm_m[q] = mloc; sm_l[q] = lloc; }
#pragma unroll
  for (int e = 0; e < 8; ++e) sm_acc[q][d + e] = acc[e];
  __syncthreads();
  if (q != 0) return;
  float M = sm_m[0];
#pragma unroll
  for (int k = 1; k < kCombWarps; ++k) M = fmaxf(M, sm_m[k]);
  float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, ls = 0.f;
#pragma unroll
  for (int k = 0; k < kCombWarps; ++k) {
    const float wk = sm_m[k] == -INFINITY ? 0.f : fexp2(sm_m[k] - M);
    ls += wk * sm_l[k];
#pragma unroll
    for (int e = 0; e < 8; ++e) y[e] += wk * sm_acc[k][d + e];
  }
  const float il = 1.f / ls;
#pragma unroll
  for (int e = 0; e < 8; ++e) y[e] *= il;
  if (p.gate) {
    const uint32_t gw[4] = {gu.x, gu.y, gu.z, gu.w};
#pragma unroll
    for (int e = 0; e < 8; e += 2) {
      const __nv_bfloat162 g2 = *reinterpret_cast<const __nv_bfloat162*>(&gw[e >> 1]);
      const float ga = __low2float(g2), gb = __high2float(g2);
      y[e] = __bfloat162float(__float2bfloat16(y[e])) * (1.0f / (1.0f + __expf(-ga)));
      y[e + 1] = __bfloat162float(__float2bfloat16(y[e + 1])) * (1.0f / (1.0f + __expf(-gb)));
    }
  }
  *reinterpret_cast<uint4*>(p.out + (int64_t)tok * p.so_tok + (int64_t)qh * p.so_head + d) =
      make_uint4(pack_bf16(y[0], y[1]), pack_bf16(y[2], y[3]), pack_bf16(y[4], y[5]), pack_bf16(y[6], y[7]));
}

// =====================================================================================================================
// EXACT mode (ATDEV_TREE_ATTN_EXACT=1): bit-identical to vLLM 0.31.0 TRITON_ATTN 3D (kernel_unified_attention IS_3D +
// reduce_segments, the attn_split_verify path) for a bf16 cache, head 256, BLOCK_M 16 (G <= 16), TILE_SIZE 16.
// Reproduced from the compiled PTX (Triton 3.x, sm_80, 4 warps):
//   segments: NSEG (16) per row, tps = cdiv(L, NSEG * 16) 16-key tiles each; segment s < act = cdiv(L, tps * 16);
//     the row of token t runs the tiles [s tps, min((s + 1) tps, nt_t)), nt_t = cdiv(min(ctx + (t / BQ) BQ +
//     15 / G + 1, L), 16), BQ = 16 / G (its Triton q block's prefix); state init M = -inf, L = 1, acc = 0
//   tile: S = fma(scale, mma-chain over the 16 dim chunks from 0, 0); masked -inf; S += bias (qq / 0);
//     m = max(M, rowmax(S)); m = m > -inf ? m : 0; P = ex2((S - m) * log2e) (ex2.approx, no ftz);
//     l = per 8-key half (p[2t] + p[2t+1]) -> + bfly 2 -> + bfly 1, then half0 + half1;
//     alpha = ex2((M - m) * log2e); L = fma(L, alpha, l); acc = mma(bf16(P), V, acc * alpha) (one k16 step)
//   reduce: e_s = ex2((M_s - max_s M_s) * log2e); D = per lane-half butterfly of fma(e_j, L_j, e_{j^8} L_{j^8});
//     out[d] = bf16(div.full(T0 + T1, D)) with T_{0/1} the per-thread fma chain over segments s0 + {2, 0, 4, ..}
//   gate (Inductor mul_sigmoid): bf16(div.full(1, ex2((0 - g) * log2e) + 1) * o)
// Split CTA = (kv head, head group, segment): the KV tile ring loads 32 keys = two 16-key softmax steps (warp dh
// computes S of step dh; both warps of a pair apply both steps to their dim half in order).
// =====================================================================================================================
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
constexpr int kNSeg = 16;

__device__ __forceinline__ void combine_exact_row(const Args& p, int rid, int i);
#ifndef TA_XF
#define TA_XF 0  // ATT2: exact fused merge -- the active segment CTAs of a (kv head, head group) meet at a
                 // generation barrier (grid <= 2 CTAs / SM, co-resident) and each runs combine_exact_row for a
                 // slice of the GH x T output rows (same op sequence as the combine kernel -> bit-identical)
#endif
// ST = ring stages: 3 = [Q staging = stage 2 | stage 0 | stage 1] (prefetch 2 ahead, 1 CTA / SM at GH 3);
// 2 = [stage 0 | stage 1 = Q staging] (prefetch 1 ahead, ~75 KB smem: 2 CTAs / SM, needs 16 GH <= 64)
template <int GH, int ST>
__global__ __launch_bounds__(64 * GH, ST == 2 ? 2 : 1) void tree_attn_split_exact_kernel(Args p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  static_assert(ST == 3 || (ST == 2 && 16 * GH <= 2 * kTile), "tree_attn exact: stages");
  constexpr int QROWS = ST == 2 ? 0 : (16 * GH > 2 * kTile ? 16 * GH : 2 * kTile);
  bf16 (*Qs)[kPitch] = reinterpret_cast<bf16 (*)[kPitch]>(smem_raw + (ST == 2 ? 2 * kTile * kPitch * 2 : 0));
  auto kt = [&](int b) -> bf16 (*)[kPitch] {
    if (ST == 2) return reinterpret_cast<bf16 (*)[kPitch]>(smem_raw + b * 2 * kTile * kPitch * 2);
    return reinterpret_cast<bf16 (*)[kPitch]>(smem_raw + (b == 2 ? 0 : (QROWS + b * 2 * kTile) * kPitch * 2));
  };
  auto vt = [&](int b) -> bf16 (*)[kPitch] { return kt(b) + kTile; };
  unsigned char* tail = smem_raw + (QROWS + 4 * kTile) * kPitch * 2;
  float (*bias)[kTok] = reinterpret_cast<float (*)[kTok]>(tail);
  float (*xm)[2][2][32] = reinterpret_cast<float (*)[2][2][32]>(tail + kTok * kTok * 4);        // [pair][half][r][lane]
  uint4 (*xp)[2][32] = reinterpret_cast<uint4 (*)[2][32]>(tail + kTok * kTok * 4 + GH * 512);  // [pair][half][lane]
  float (*xl)[2][2][32] = reinterpret_cast<float (*)[2][2][32]>(tail + kTok * kTok * 4 + GH * 1536);
  constexpr int NT = 64 * GH;
  const int h = blockIdx.x / p.hspl, hg0 = (blockIdx.x % p.hspl) * GH, seg = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int hw = warp >> 1, dh = warp & 1;
  const int L = *p.seqlen;
  const int T = p.T;
  const int ctx = L - T;
  const int tps = (L + p.nsplit * 16 - 1) / (p.nsplit * 16);
  if (seg * tps * 16 >= L) return;
  const int j0 = seg * tps, j1 = min(j0 + tps, (L + 15) >> 4);  // 16-key tiles of this segment
  const int nld = (j1 - j0 + 1) >> 1;                              // 32-key ring loads
  const int key0 = j0 * 16;
  // per-row tile limit of the row's Triton q block (rows g, g + 8 of this thread = tokens g, g + 8)
  const int bq = 16 / p.G, pre = 15 / p.G + 1;
  int nt_r[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int tok = g + 8 * r;
    nt_r[r] = (min(ctx + (tok / bq) * bq + pre, L) + 15) >> 4;
  }
  const int lc = tid & 31, lr = tid >> 5;
  auto load_tile = [&](int ld, int buf) {
    const int jt = key0 + ld * kTile;
    const int pj0 = jt / p.page, off0 = jt - pj0 * p.page;
    const bool one_page = off0 + kTile <= p.page;
    const int blk0 = p.bt[pj0];
    const bf16* kb0 = p.kc + (int64_t)blk0 * p.sk_blk + (int64_t)h * p.sk_head + lc * 8;
    const bf16* vb0 = p.vc + (int64_t)blk0 * p.sv_blk + (int64_t)h * p.sv_head + lc * 8;
#pragma unroll
    for (int rr = 0; rr < (kTile + NT / 32 - 1) / (NT / 32); ++rr) {
      const int r = lr + rr * (NT / 32);
      if (r < kTile) {
        const int j = jt + r;
        const bool ok = j < L;
        const bf16 *ks, *vs;
        if (p.kn && j >= ctx && ok) {
          ks = p.kn + (int64_t)(j - ctx) * p.skn_tok + (int64_t)h * p.skn_head + lc * 8;
          vs = p.vn + (int64_t)(j - ctx) * p.svn_tok + (int64_t)h * p.svn_head + lc * 8;
        } else if (one_page || !ok) {
          ks = kb0 + (int64_t)(off0 + (ok ? r : 0)) * p.sk_tok;
          vs = vb0 + (int64_t)(off0 + (ok ? r : 0)) * p.sv_tok;
        } else {
          const int pj = j / p.page, off = j - pj * p.page, blk = p.bt[pj];
          ks = p.kc + (int64_t)blk * p.sk_blk + (int64_t)off * p.sk_tok + (int64_t)h * p.sk_head + lc * 8;
          vs = p.vc + (int64_t)blk * p.sv_blk + (int64_t)off * p.sv_tok + (int64_t)h * p.sv_head + lc * 8;
        }
        cp_async_16(&kt(buf)[r][lc * 8], ks, ok);
        cp_async_16(&vt(buf)[r][lc * 8], vs, ok);
      }
    }
  };
  load_tile(0, 0);
  cp_async_commit();
  for (int e = tid; e < 16 * GH * (kHS / 8); e += NT) {
    const int row = e >> 5, c = e & 31;
    const int w = row >> 4, tok = row & 15;
    const bool ok = tok < T;
    const bf16* src = p.q + (int64_t)(ok ? tok : 0) * p.sq_tok + (int64_t)(h * p.G + hg0 + w) * p.sq_head + c * 8;
    cp_async_16(&Qs[row][c * 8], src, ok);
  }
  cp_async_commit();
  if (ST == 3 && 1 < nld) load_tile(1, 1);
  if (ST == 3) cp_async_commit();
  for (int e = tid; e < kTok * kTok; e += NT) {
    const int i = e >> 4, jj = e & 15;
    bias[i][jj] = (i < T && jj < T && p.qq) ? p.qq[(int64_t)i * p.sqq + jj] : 0.f;
  }

  float o[16][4];
#pragma unroll
  for (int d = 0; d < 16; ++d)
#pragma unroll
    for (int e = 0; e < 4; ++e) o[d][e] = 0.f;
  float m_r[2] = {-INFINITY, -INFINITY}, l_r[2] = {1.f, 1.f};
  const int qrow0 = hw * 16;
  const int bar_id = 1 + hw;
  uint32_t qf[kHS / 16][4];
  {
    if (ST == 3) cp_async_wait<1>(); else cp_async_wait<0>();
    __syncthreads();
    const bf16* qa = &Qs[qrow0 + (lane & 15)][(lane >> 4) * 8];
#pragma unroll
    for (int ks = 0; ks < kHS / 16; ++ks) ldsm_x4(qf[ks], qa + ks * 16);
  }

  int buf = 0, nbuf = 2;
  for (int ld = 0; ld < nld; ++ld) {
    if (ST == 2) {
      __syncthreads();  // every warp done with tile ld - 1 (and with the Q staging = stage 1)
      if (ld + 1 < nld) load_tile(ld + 1, (ld + 1) & 1);
      cp_async_commit();
      cp_async_wait<1>();
      __syncthreads();
      buf = ld & 1;
    } else {
      cp_async_wait<1>();
      __syncthreads();
      if (ld + 2 < nld) load_tile(ld + 2, nbuf);
      cp_async_commit();
    }
    const int jA = j0 + 2 * ld;
    const bool validB = jA + 1 < j1;  // step B inside this segment (uniform)
    // ---- S of this warp's step (keys [16 dh, 16 dh + 16) of the load): one mma chain per n-tile, as Triton
    float s[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) s[nt][e] = 0.f;
    if (dh == 0 || validB) {
      const bf16* kb = &kt(buf)[dh * 16 + (lane & 7) + ((lane >> 4) << 3)][((lane >> 3) & 1) * 8];
      uint32_t b[2][4];
      ldsm_x4(b[0], kb);
#pragma unroll
      for (int ks = 0; ks < kHS / 16; ++ks) {
        const int c = ks & 1;
        if (ks + 1 < kHS / 16) ldsm_x4(b[c ^ 1], kb + (ks + 1) * 16);
        mma16816(s[0], qf[ks], b[c][0], b[c][1]);
        mma16816(s[1], qf[ks], b[c][2], b[c][3]);
      }
    }
    const int jbase = key0 + ld * kTile + dh * 16;
    float mx[2] = {-INFINITY, -INFINITY};
    if (jbase + 16 <= ctx) {
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float v = __fmaf_rn(p.scale, s[nt][e], 0.f);
          s[nt][e] = v;
          mx[e >> 1] = fmaxf(mx[e >> 1], v);
        }
    } else {
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int i = g + 8 * (e >> 1);
          const int j = jbase + nt * 8 + 2 * t + (e & 1);
          float v = __fmaf_rn(p.scale, s[nt][e], 0.f);
          if (j >= L || j > ctx + i) v = -INFINITY;
          else if (j >= ctx) v = __fadd_rn(bias[i][j - ctx], v);
          s[nt][e] = v;
          mx[e >> 1] = fmaxf(mx[e >> 1], v);
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
      xm[hw][dh][r][lane] = mx[r];
    }
    bar_sync(bar_id, 64);
    // ---- the two softmax steps (A = keys 0-15, B = keys 16-31), identical in both warps of the pair
    float alA[2], alB[2], mine[2];
    bool actA[2], actB[2], myact[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float oth = xm[hw][dh ^ 1][r][lane];
      const float rA = dh ? oth : mx[r], rB = dh ? mx[r] : oth;
      actA[r] = jA < nt_r[r];
      actB[r] = validB && jA + 1 < nt_r[r];
      float mA = m_r[r], mB;
      alA[r] = 1.f;
      if (actA[r]) {
        mA = fmaxf(m_r[r], rA);
        mA = mA > -INFINITY ? mA : 0.f;
        alA[r] = exp_tr(__fsub_rn(m_r[r], mA));
      }
      mB = mA;
      alB[r] = 1.f;
      if (actB[r]) {
        mB = fmaxf(mA, rB);
        mB = mB > -INFINITY ? mB : 0.f;
        alB[r] = exp_tr(__fsub_rn(mA, mB));
      }
      mine[r] = dh ? mB : mA;
      myact[r] = dh ? actB[r] : actA[r];
      m_r[r] = mB;
    }
    uint32_t own[4];
    float lsum[2];
    {
      float z[2][2];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const float p0 = myact[0] ? exp_tr(__fsub_rn(s[nt][0], mine[0])) : 0.f;
        const float p1 = myact[0] ? exp_tr(__fsub_rn(s[nt][1], mine[0])) : 0.f;
        const float p2 = myact[1] ? exp_tr(__fsub_rn(s[nt][2], mine[1])) : 0.f;
        const float p3 = myact[1] ? exp_tr(__fsub_rn(s[nt][3], mine[1])) : 0.f;
        own[nt * 2 + 0] = pack_bf16(p0, p1);
        own[nt * 2 + 1] = pack_bf16(p2, p3);
        float x0 = __fadd_rn(p0, p1), x1 = __fadd_rn(p2, p3);
        x0 = __fadd_rn(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
        x1 = __fadd_rn(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
        x0 = __fadd_rn(x0, __shfl_xor_sync(0xffffffffu, x0, 1));
        x1 = __fadd_rn(x1, __shfl_xor_sync(0xffffffffu, x1, 1));
        z[nt][0] = x0; z[nt][1] = x1;
      }
      lsum[0] = __fadd_rn(z[0][0], z[1][0]);
      lsum[1] = __fadd_rn(z[0][1], z[1][1]);
      xp[hw][dh][lane] = make_uint4(own[0], own[1], own[2], own[3]);
      xl[hw][dh][0][lane] = lsum[0];
      xl[hw][dh][1][lane] = lsum[1];
    }
    bar_sync(bar_id, 64);
    uint32_t pa[2][4];
    {
      const uint4 u = xp[hw][dh ^ 1][lane];
      const uint32_t oth[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        pa[0][i] = dh ? oth[i] : own[i];
        pa[1][i] = dh ? own[i] : oth[i];
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const float lo = xl[hw][dh ^ 1][r][lane];
        const float lA = dh ? lo : lsum[r], lB = dh ? lsum[r] : lo;
        if (actA[r]) l_r[r] = __fmaf_rn(l_r[r], alA[r], lA);
        if (actB[r]) l_r[r] = __fmaf_rn(l_r[r], alB[r], lB);
      }
    }
    // ---- O[:, dh half]: acc = mma(P_A, V_A, acc * alA); acc = mma(P_B, V_B, acc * alB)
    {
      const bf16* vb = &vt(buf)[(lane & 7) + (((lane >> 3) & 1) << 3)][dh * 128 + (lane >> 4) * 8];
#pragma unroll
      for (int d = 0; d < 16; ++d) {
        o[d][0] = __fmul_rn(o[d][0], alA[0]); o[d][1] = __fmul_rn(o[d][1], alA[0]);
        o[d][2] = __fmul_rn(o[d][2], alA[1]); o[d][3] = __fmul_rn(o[d][3], alA[1]);
      }
      uint32_t b[2][4];
      ldsm_x4_t(b[0], vb);
#pragma unroll
      for (int dp = 0; dp < 8; ++dp) {
        const int c = dp & 1;
        if (dp + 1 < 8) ldsm_x4_t(b[c ^ 1], vb + (dp + 1) * 16);
        mma16816(o[2 * dp], pa[0], b[c][0], b[c][1]);
        mma16816(o[2 * dp + 1], pa[0], b[c][2], b[c][3]);
      }
      if (validB) {
#pragma unroll
        for (int d = 0; d < 16; ++d) {
          o[d][0] = __fmul_rn(o[d][0], alB[0]); o[d][1] = __fmul_rn(o[d][1], alB[0]);
          o[d][2] = __fmul_rn(o[d][2], alB[1]); o[d][3] = __fmul_rn(o[d][3], alB[1]);
        }
        const bf16* vb2 = vb + 16 * kPitch;
        ldsm_x4_t(b[0], vb2);
#pragma unroll
        for (int dp = 0; dp < 8; ++dp) {
          const int c = dp & 1;
          if (dp + 1 < 8) ldsm_x4_t(b[c ^ 1], vb2 + (dp + 1) * 16);
          mma16816(o[2 * dp], pa[1], b[c][0], b[c][1]);
          mma16816(o[2 * dp + 1], pa[1], b[c][2], b[c][3]);
        }
      }
    }
    if (ST == 3) {
      buf = buf == 2 ? 0 : buf + 1;
      nbuf = nbuf == 2 ? 0 : nbuf + 1;
    }
  }
  cp_async_wait<0>();
  const int64_t pbase = ((int64_t)h * p.nsplit + seg) * (16 * p.G) + hg0 * 16;
  float* op = p.opart + (pbase + qrow0) * kHS + dh * 128;
#pragma unroll
  for (int d = 0; d < 16; ++d) {
    const int col = d * 8 + 2 * t;
    *reinterpret_cast<float2*>(op + (int64_t)g * kHS + col) = make_float2(o[d][0], o[d][1]);
    *reinterpret_cast<float2*>(op + (int64_t)(g + 8) * kHS + col) = make_float2(o[d][2], o[d][3]);
  }
  if (dh == 0 && t == 0) {
    *reinterpret_cast<float2*>(p.ml + (pbase + qrow0 + g) * 2) = make_float2(m_r[0], l_r[0]);
    *reinterpret_cast<float2*>(p.ml + (pbase + qrow0 + g + 8) * 2) = make_float2(m_r[1], l_r[1]);
  }
#if TA_XF
  if (p.xf) {
    const int act = (L + tps * 16 - 1) / (tps * 16);
    __syncthreads();
    if (tid == 0) {
      unsigned* cnt = p.sync + 2 * blockIdx.x;
      volatile unsigned* gen = cnt + 1;
      const unsigned g0 = *gen;
      __threadfence();
      const unsigned old = atomicAdd(cnt, 1u);
      if (old == (unsigned)(act - 1)) {
        *cnt = 0u;
        __threadfence();
        atomicAdd(cnt + 1, 1u);
      } else {
        while (*gen == g0) __nanosleep(32);
      }
      __threadfence();
    }
    __syncthreads();
    const int nu = GH * T;  // output rows (w, tok) of this head group, 64 threads each
    for (int u = seg * (NT / 64) + (tid >> 6); u < nu; u += act * (NT / 64)) {
      const int w = u / T, tok = u - w * T;
      combine_exact_row(p, tok * p.QH + h * p.G + hg0 + w, tid & 63);
    }
  }
#endif
}

// reduce_segments + (optional) the Inductor gate, CTA per (tok, q head), 64 threads x 4 dims.
__device__ __forceinline__ void combine_exact_row(const Args& p, int rid, int i) {
  const int tok = rid / p.QH, qh = rid % p.QH;
  const int h = qh / p.G, w = qh % p.G;
  const int L = *p.seqlen;
  const int tps = (L + kNSeg * 16 - 1) / (kNSeg * 16);
  const int act = (L + tps * 16 - 1) / (tps * 16);
  const int row = w * 16 + tok;
  const int64_t rows = 16 * p.G;
  const int d = i * 4;
  if (p.kn && w == 0) {  // fused cache write of (tok, kv head h): thread -> 16 B of the K row (i < 32) or V row
    const int64_t sl = p.slot[tok];
    if (sl >= 0) {
      const int64_t blk = sl / p.page, off = sl % p.page;
      const int c = (i & 31) * 8;
      if (i < 32) {
        const uint4 x = *reinterpret_cast<const uint4*>(p.kn + (int64_t)tok * p.skn_tok + (int64_t)h * p.skn_head + c);
        *reinterpret_cast<uint4*>(const_cast<bf16*>(p.kc) + blk * p.sk_blk + off * p.sk_tok + (int64_t)h * p.sk_head + c) = x;
      } else {
        const uint4 x = *reinterpret_cast<const uint4*>(p.vn + (int64_t)tok * p.svn_tok + (int64_t)h * p.svn_head + c);
        *reinterpret_cast<uint4*>(const_cast<bf16*>(p.vc) + blk * p.sv_blk + off * p.sv_tok + (int64_t)h * p.sv_head + c) = x;
      }
    }
  }
  uint2 gu = make_uint2(0, 0);
  if (p.gate) gu = *reinterpret_cast<const uint2*>(p.gate + (int64_t)tok * p.sg_tok + (int64_t)qh * kHS + d);
  const float2* mlp = reinterpret_cast<const float2*>(p.ml) + (int64_t)h * kNSeg * rows + row;
  const float* opp = p.opart + ((int64_t)h * kNSeg * rows + row) * kHS + d;
  float Ms[kNSeg], Ls[kNSeg];
  float4 ov[kNSeg];
#pragma unroll
  for (int s = 0; s < kNSeg; ++s) {
    if (s < act) {
      const float2 ml = __ldcg(mlp + (int64_t)s * rows);
      Ms[s] = ml.x; Ls[s] = ml.y;
      ov[s] = __ldcg(reinterpret_cast<const float4*>(opp + (int64_t)s * rows * kHS));
    } else {
      Ms[s] = -INFINITY; Ls[s] = 0.f;
      ov[s] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
  float M = Ms[0];
#pragma unroll
  for (int s = 1; s < kNSeg; ++s) M = fmaxf(M, Ms[s]);
  float e[kNSeg];
#pragma unroll
  for (int s = 0; s < kNSeg; ++s) e[s] = exp_tr(__fsub_rn(Ms[s], M));
  float a[kNSeg];
#pragma unroll
  for (int j = 0; j < kNSeg; ++j) a[j] = __fmaf_rn(e[j], Ls[j], __fmul_rn(e[j ^ 8], Ls[j ^ 8]));
  // Triton: the thread dividing dims [4i, 4i + 4) holds the expsum of lane half (i >> 3) & 1
  const float Dlo = __fadd_rn(__fadd_rn(__fadd_rn(a[0], a[4]), __fadd_rn(a[2], a[6])),
                              __fadd_rn(__fadd_rn(a[1], a[5]), __fadd_rn(a[3], a[7])));
  const float Dhi = __fadd_rn(__fadd_rn(__fadd_rn(a[8], a[12]), __fadd_rn(a[10], a[14])),
                              __fadd_rn(__fadd_rn(a[9], a[13]), __fadd_rn(a[11], a[15])));
  const float D = ((i >> 3) & 1) ? Dhi : Dlo;
  float y[4];
  {
    float t0[4], t1[4];
    const float* o2 = reinterpret_cast<const float*>(&ov[2]);
    const float* o3 = reinterpret_cast<const float*>(&ov[3]);
#pragma unroll
    for (int c = 0; c < 4; ++c) { t0[c] = __fmul_rn(e[2], o2[c]); t1[c] = __fmul_rn(e[3], o3[c]); }
#pragma unroll
    for (int s = 0; s < kNSeg; s += 2) {
      if (s == 2) continue;
      const float* oa = reinterpret_cast<const float*>(&ov[s]);
      const float* ob = reinterpret_cast<const float*>(&ov[s + 1]);
#pragma unroll
      for (int c = 0; c < 4; ++c) { t0[c] = __fmaf_rn(e[s], oa[c], t0[c]); t1[c] = __fmaf_rn(e[s + 1], ob[c], t1[c]); }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) y[c] = D == 0.f ? 0.f : div_full(__fadd_rn(t0[c], t1[c]), D);
  }
  uint32_t outw[2];
#pragma unroll
  for (int c2 = 0; c2 < 2; ++c2) {
    __nv_bfloat162 ob = __floats2bfloat162_rn(y[2 * c2], y[2 * c2 + 1]);
    if (p.gate) {
      const uint32_t gw = c2 ? gu.y : gu.x;
      const __nv_bfloat162 g2 = *reinterpret_cast<const __nv_bfloat162*>(&gw);
      const float ga = __low2float(g2), gb = __high2float(g2);
      const float sa = div_full(1.f, __fadd_rn(exp_tr(__fsub_rn(0.f, ga)), 1.f));
      const float sb = div_full(1.f, __fadd_rn(exp_tr(__fsub_rn(0.f, gb)), 1.f));
      ob = __floats2bfloat162_rn(__fmul_rn(sa, __low2float(ob)), __fmul_rn(sb, __high2float(ob)));
    }
    outw[c2] = *reinterpret_cast<uint32_t*>(&ob);
  }
  *reinterpret_cast<uint2*>(p.out + (int64_t)tok * p.so_tok + (int64_t)qh * p.so_head + d) =
      make_uint2(outw[0], outw[1]);
}
__global__ __launch_bounds__(64) void tree_attn_combine_exact_kernel(Args p) {
  combine_exact_row(p, blockIdx.x, threadIdx.x);
}

template <int GH, int ST>
void launch_split_exact(const Args& a, int kvh, cudaStream_t st) {
  auto kern = tree_attn_split_exact_kernel<GH, ST>;
  constexpr int QROWS = ST == 2 ? 0 : (16 * GH > 2 * kTile ? 16 * GH : 2 * kTile);
  const int smem = (QROWS + 4 * kTile) * kPitch * 2 + kTok * kTok * 4 + GH * 512 + GH * 1024 + GH * 512;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr = true;
  }
  kern<<<dim3(kvh * a.hspl, a.nsplit), 64 * GH, smem, st>>>(a);
}

template <int GH>
void launch_split(const Args& a, int kvh, cudaStream_t st) {
  auto kern = tree_attn_split_kernel<GH>;
  constexpr int QROWS = 16 * GH > 2 * kTile ? 16 * GH : 2 * kTile;
  const int smem = (QROWS + 4 * kTile) * kPitch * 2 + kTok * kTok * 4 + GH * 512 + GH * 1024;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr = true;
  }
  kern<<<dim3(kvh * a.hspl, a.nsplit), 64 * GH, smem, st>>>(a);
}


// ATT2 TA_PIPE=1 (exact 2 path): software-pipelined variant of tree_attn_split_exact_kernel<GH, 2> (GH <= 2).
// Separate K / V rings of 2 slots each (same 67.6 KB as ST 2 -> 2 CTAs / SM); K runs one load ahead of V, so the QK
// mma chain of load ld+1 (independent of load ld) is interleaved mma by mma with PV(ld). One __syncthreads per load
// (wait {K(ld+1), V(ld)} -> sync -> issue K(ld+2), V(ld+1)). Per-row fp32 op sequence identical to the exact kernel.
#ifndef TA_PIPE
#define TA_PIPE 0
#endif
template <int GH>
__global__ __launch_bounds__(64 * GH, 2) void tree_attn_split_exact_pipe_kernel(Args p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  static_assert(16 * GH <= kTile, "pipe: Q staging must fit one V slot");
  auto ks_ = [&](int b) -> bf16 (*)[kPitch] { return reinterpret_cast<bf16 (*)[kPitch]>(smem_raw + b * kTile * kPitch * 2); };
  auto vs_ = [&](int b) -> bf16 (*)[kPitch] { return ks_(2 + b); };
  bf16 (*Qs)[kPitch] = vs_(1);
  unsigned char* tail = smem_raw + 4 * kTile * kPitch * 2;
  float (*bias)[kTok] = reinterpret_cast<float (*)[kTok]>(tail);
  float (*xm)[2][2][32] = reinterpret_cast<float (*)[2][2][32]>(tail + kTok * kTok * 4);
  uint4 (*xp)[2][32] = reinterpret_cast<uint4 (*)[2][32]>(tail + kTok * kTok * 4 + GH * 512);
  float (*xl)[2][2][32] = reinterpret_cast<float (*)[2][2][32]>(tail + kTok * kTok * 4 + GH * 1536);
  constexpr int NT = 64 * GH;
  const int h = blockIdx.x / p.hspl, hg0 = (blockIdx.x % p.hspl) * GH, seg = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int hw = warp >> 1, dh = warp & 1;
  const int L = *p.seqlen;
  const int T = p.T;
  const int ctx = L - T;
  const int tps = (L + p.nsplit * 16 - 1) / (p.nsplit * 16);
  if (seg * tps * 16 >= L) return;
  const int j0 = seg * tps, j1 = min(j0 + tps, (L + 15) >> 4);
  const int nld = (j1 - j0 + 1) >> 1;
  const int key0 = j0 * 16;
  const int bq = 16 / p.G, pre = 15 / p.G + 1;
  int nt_r[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int tok = g + 8 * r;
    nt_r[r] = (min(ctx + (tok / bq) * bq + pre, L) + 15) >> 4;
  }
  const int lc = tid & 31, lr = tid >> 5;
  auto load_kv = [&](int ld, int buf, bool isv) {
    if (ld >= nld) return;
    const int jt = key0 + ld * kTile;
    const int pj0 = jt / p.page, off0 = jt - pj0 * p.page;
    const bool one_page = off0 + kTile <= p.page;
    const int blk0 = p.bt[pj0];
    const bf16* base = isv ? p.vc : p.kc;
    const int64_t sblk = isv ? p.sv_blk : p.sk_blk, stok = isv ? p.sv_tok : p.sk_tok, shd = isv ? p.sv_head : p.sk_head;
    const bf16* nb = isv ? p.vn : p.kn;
    const int64_t sntok = isv ? p.svn_tok : p.skn_tok, snhd = isv ? p.svn_head : p.skn_head;
    bf16 (*dst)[kPitch] = isv ? vs_(buf) : ks_(buf);
    const bf16* b0 = base + (int64_t)blk0 * sblk + (int64_t)h * shd + lc * 8;
#pragma unroll
    for (int rr = 0; rr < (kTile + NT / 32 - 1) / (NT / 32); ++rr) {
      const int r = lr + rr * (NT / 32);
      if (r < kTile) {
        const int j = jt + r;
        const bool ok = j < L;
        const bf16* src;
        if (nb && j >= ctx && ok) {
          src = nb + (int64_t)(j - ctx) * sntok + (int64_t)h * snhd + lc * 8;
        } else if (one_page || !ok) {
          src = b0 + (int64_t)(off0 + (ok ? r : 0)) * stok;
        } else {
          const int pj = j / p.page, off = j - pj * p.page, blk = p.bt[pj];
          src = base + (int64_t)blk * sblk + (int64_t)off * stok + (int64_t)h * shd + lc * 8;
        }
        cp_async_16(&dst[r][lc * 8], src, ok);
      }
    }
  };
  // group A = {K(0), Q}; group P1 = {K(1), V(0)}
  load_kv(0, 0, false);
  for (int e = tid; e < 16 * GH * (kHS / 8); e += NT) {
    const int row = e >> 5, c = e & 31;
    const int w = row >> 4, tok = row & 15;
    const bool ok = tok < T;
    const bf16* src = p.q + (int64_t)(ok ? tok : 0) * p.sq_tok + (int64_t)(h * p.G + hg0 + w) * p.sq_head + c * 8;
    cp_async_16(&Qs[row][c * 8], src, ok);
  }
  cp_async_commit();
  load_kv(1, 1, false);
  load_kv(0, 0, true);
  cp_async_commit();
  for (int e = tid; e < kTok * kTok; e += NT) {
    const int i = e >> 4, jj = e & 15;
    bias[i][jj] = (i < T && jj < T && p.qq) ? p.qq[(int64_t)i * p.sqq + jj] : 0.f;
  }
  float o[16][4];
#pragma unroll
  for (int d = 0; d < 16; ++d)
#pragma unroll
    for (int e = 0; e < 4; ++e) o[d][e] = 0.f;
  float m_r[2] = {-INFINITY, -INFINITY}, l_r[2] = {1.f, 1.f};
  const int qrow0 = hw * 16;
  const int bar_id = 1 + hw;
  uint32_t qf[kHS / 16][4];
  const int krow = dh * 16 + (lane & 7) + ((lane >> 4) << 3), kcol = ((lane >> 3) & 1) * 8;
  float s[2][4];
#pragma unroll
  for (int nt = 0; nt < 2; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) s[nt][e] = 0.f;
  {
    cp_async_wait<1>();
    __syncthreads();
    const bf16* qa = &Qs[qrow0 + (lane & 15)][(lane >> 4) * 8];
#pragma unroll
    for (int ks = 0; ks < kHS / 16; ++ks) ldsm_x4(qf[ks], qa + ks * 16);
    if (dh == 0 || j0 + 1 < j1) {
      const bf16* kb = &ks_(0)[krow][kcol];
      uint32_t b[2][4];
      ldsm_x4(b[0], kb);
#pragma unroll
      for (int ks = 0; ks < kHS / 16; ++ks) {
        const int c = ks & 1;
        if (ks + 1 < kHS / 16) ldsm_x4(b[c ^ 1], kb + (ks + 1) * 16);
        mma16816(s[0], qf[ks], b[c][0], b[c][1]);
        mma16816(s[1], qf[ks], b[c][2], b[c][3]);
      }
    }
  }
  for (int ld = 0; ld < nld; ++ld) {
    cp_async_wait<0>();
    __syncthreads();  // K(ld+1), V(ld) visible; every warp done with load ld-1 (K slot ld&1, V slot (ld+1)&1 free)
    load_kv(ld + 2, ld & 1, false);
    load_kv(ld + 1, (ld + 1) & 1, true);
    cp_async_commit();
    const int jA = j0 + 2 * ld;
    const bool validB = jA + 1 < j1;
    const int jbase = key0 + ld * kTile + dh * 16;
    float mx[2] = {-INFINITY, -INFINITY};
    if (jbase + 16 <= ctx) {
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float v = __fmaf_rn(p.scale, s[nt][e], 0.f);
          s[nt][e] = v;
          mx[e >> 1] = fmaxf(mx[e >> 1], v);
        }
    } else {
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int i = g + 8 * (e >> 1);
          const int j = jbase + nt * 8 + 2 * t + (e & 1);
          float v = __fmaf_rn(p.scale, s[nt][e], 0.f);
          if (j >= L || j > ctx + i) v = -INFINITY;
          else if (j >= ctx) v = __fadd_rn(bias[i][j - ctx], v);
          s[nt][e] = v;
          mx[e >> 1] = fmaxf(mx[e >> 1], v);
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
      xm[hw][dh][r][lane] = mx[r];
    }
    bar_sync(bar_id, 64);
    float alA[2], alB[2], mine[2];
    bool actA[2], actB[2], myact[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float oth = xm[hw][dh ^ 1][r][lane];
      const float rA = dh ? oth : mx[r], rB = dh ? mx[r] : oth;
      actA[r] = jA < nt_r[r];
      actB[r] = validB && jA + 1 < nt_r[r];
      float mA = m_r[r], mB;
      alA[r] = 1.f;
      if (actA[r]) {
        mA = fmaxf(m_r[r], rA);
        mA = mA > -INFINITY ? mA : 0.f;
        alA[r] = exp_tr(__fsub_rn(m_r[r], mA));
      }
      mB = mA;
      alB[r] = 1.f;
      if (actB[r]) {
        mB = fmaxf(mA, rB);
        mB = mB > -INFINITY ? mB : 0.f;
        alB[r] = exp_tr(__fsub_rn(mA, mB));
      }
      mine[r] = dh ? mB : mA;
      myact[r] = dh ? actB[r] : actA[r];
      m_r[r] = mB;
    }
    uint32_t own[4];
    float lsum[2];
    {
      float z[2][2];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const float p0 = myact[0] ? exp_tr(__fsub_rn(s[nt][0], mine[0])) : 0.f;
        const float p1 = myact[0] ? exp_tr(__fsub_rn(s[nt][1], mine[0])) : 0.f;
        const float p2 = myact[1] ? exp_tr(__fsub_rn(s[nt][2], mine[1])) : 0.f;
        const float p3 = myact[1] ? exp_tr(__fsub_rn(s[nt][3], mine[1])) : 0.f;
        own[nt * 2 + 0] = pack_bf16(p0, p1);
        own[nt * 2 + 1] = pack_bf16(p2, p3);
        float x0 = __fadd_rn(p0, p1), x1 = __fadd_rn(p2, p3);
        x0 = __fadd_rn(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
        x1 = __fadd_rn(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
        x0 = __fadd_rn(x0, __shfl_xor_sync(0xffffffffu, x0, 1));
        x1 = __fadd_rn(x1, __shfl_xor_sync(0xffffffffu, x1, 1));
        z[nt][0] = x0; z[nt][1] = x1;
      }
      lsum[0] = __fadd_rn(z[0][0], z[1][0]);
      lsum[1] = __fadd_rn(z[0][1], z[1][1]);
      xp[hw][dh][lane] = make_uint4(own[0], own[1], own[2], own[3]);
      xl[hw][dh][0][lane] = lsum[0];
      xl[hw][dh][1][lane] = lsum[1];
    }
    bar_sync(bar_id, 64);
    uint32_t pa[2][4];
    {
      const uint4 u = xp[hw][dh ^ 1][lane];
      const uint32_t oth[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        pa[0][i] = dh ? oth[i] : own[i];
        pa[1][i] = dh ? own[i] : oth[i];
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const float lo = xl[hw][dh ^ 1][r][lane];
        const float lA = dh ? lo : lsum[r], lB = dh ? lsum[r] : lo;
        if (actA[r]) l_r[r] = __fmaf_rn(l_r[r], alA[r], lA);
        if (actB[r]) l_r[r] = __fmaf_rn(l_r[r], alB[r], lB);
      }
    }
    // ---- PV(ld) (steps A, B) interleaved with QK(ld + 1) (k-steps 0-7 under A, 8-15 under B)
    const bool nxt = ld + 1 < nld;
    const bool qkn = nxt && (dh == 0 || jA + 3 < j1);  // this warp's step of load ld+1 is inside the segment
    float sn[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) sn[nt][e] = 0.f;
    const bf16* vb = &vs_(ld & 1)[(lane & 7) + (((lane >> 3) & 1) << 3)][dh * 128 + (lane >> 4) * 8];
    const bf16* kb = &ks_((ld + 1) & 1)[krow][kcol];
    uint32_t kf[2][4];
    if (qkn) ldsm_x4(kf[0], kb);
#pragma unroll
    for (int stp = 0; stp < 2; ++stp) {
      if (stp == 1 && !validB) break;
      const float* al = stp ? alB : alA;
#pragma unroll
      for (int d = 0; d < 16; ++d) {
        o[d][0] = __fmul_rn(o[d][0], al[0]); o[d][1] = __fmul_rn(o[d][1], al[0]);
        o[d][2] = __fmul_rn(o[d][2], al[1]); o[d][3] = __fmul_rn(o[d][3], al[1]);
      }
      const bf16* vbs = vb + stp * 16 * kPitch;
      uint32_t b[2][4];
      ldsm_x4_t(b[0], vbs);
#pragma unroll
      for (int dp = 0; dp < 8; ++dp) {
        const int c = dp & 1;
        const int ks = stp * 8 + dp;
        if (dp + 1 < 8) ldsm_x4_t(b[c ^ 1], vbs + (dp + 1) * 16);
        if (qkn && ks + 1 < kHS / 16) ldsm_x4(kf[(ks + 1) & 1], kb + (ks + 1) * 16);
        mma16816(o[2 * dp], pa[stp], b[c][0], b[c][1]);
        if (qkn) mma16816(sn[0], qf[ks], kf[ks & 1][0], kf[ks & 1][1]);
        mma16816(o[2 * dp + 1], pa[stp], b[c][2], b[c][3]);
        if (qkn) mma16816(sn[1], qf[ks], kf[ks & 1][2], kf[ks & 1][3]);
      }
    }
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) s[nt][e] = sn[nt][e];
  }
  cp_async_wait<0>();
  const int64_t pbase = ((int64_t)h * p.nsplit + seg) * (16 * p.G) + hg0 * 16;
  float* op = p.opart + (pbase + qrow0) * kHS + dh * 128;
#pragma unroll
  for (int d = 0; d < 16; ++d) {
    const int col = d * 8 + 2 * t;
    *reinterpret_cast<float2*>(op + (int64_t)g * kHS + col) = make_float2(o[d][0], o[d][1]);
    *reinterpret_cast<float2*>(op + (int64_t)(g + 8) * kHS + col) = make_float2(o[d][2], o[d][3]);
  }
  if (dh == 0 && t == 0) {
    *reinterpret_cast<float2*>(p.ml + (pbase + qrow0 + g) * 2) = make_float2(m_r[0], l_r[0]);
    *reinterpret_cast<float2*>(p.ml + (pbase + qrow0 + g + 8) * 2) = make_float2(m_r[1], l_r[1]);
  }
#if TA_XF
  if (p.xf) {
    const int act = (L + tps * 16 - 1) / (tps * 16);
    __syncthreads();
    if (tid == 0) {
      unsigned* cnt = p.sync + 2 * blockIdx.x;
      volatile unsigned* gen = cnt + 1;
      const unsigned g0 = *gen;
      __threadfence();
      const unsigned old = atomicAdd(cnt, 1u);
      if (old == (unsigned)(act - 1)) {
        *cnt = 0u;
        __threadfence();
        atomicAdd(cnt + 1, 1u);
      } else {
        while (*gen == g0) __nanosleep(32);
      }
      __threadfence();
    }
    __syncthreads();
    const int nu = GH * T;  // output rows (w, tok) of this head group, 64 threads each
    for (int u = seg * (NT / 64) + (tid >> 6); u < nu; u += act * (NT / 64)) {
      const int w = u / T, tok = u - w * T;
      combine_exact_row(p, tok * p.QH + h * p.G + hg0 + w, tid & 63);
    }
  }
#endif
}

template <int GH>
void launch_split_exact_pipe(const Args& a, int kvh, cudaStream_t st) {
  auto kern = tree_attn_split_exact_pipe_kernel<GH>;
  const int smem = 4 * kTile * kPitch * 2 + kTok * kTok * 4 + GH * 512 + GH * 1024 + GH * 512;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr = true;
  }
  kern<<<dim3(kvh * a.hspl, a.nsplit), 64 * GH, smem, st>>>(a);
}

}  // namespace

// q [T, QH, 256]; kc / vc [blocks, page, kvh, 256] views; bt [>= ceil(L / page)] int32 (request 0);
// seqlen [1] int32 (incl. the T tree tokens); qq [T, T] fp32 or empty; opart / ml scratch; out [T, QH, 256];
// gate [T, QH * 256] bf16 or empty.
void tree_attn_fwd(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor bt, torch::Tensor seqlen,
                   torch::Tensor qq, torch::Tensor opart, torch::Tensor ml, torch::Tensor out, torch::Tensor gate,
                   int64_t nsplit, double scale, torch::Tensor kn, torch::Tensor vn, torch::Tensor slot,
                   int64_t hspl, torch::Tensor sync, int64_t exact) {
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && kc.scalar_type() == at::kBFloat16 &&
              vc.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(q.size(2) == kHS && q.stride(2) == 1 && kc.stride(3) == 1 && vc.stride(3) == 1 && out.stride(2) == 1);
  const int T = (int)q.size(0), QH = (int)q.size(1), kvh = (int)kc.size(2);
  TORCH_CHECK(T >= 1 && T <= kTok && QH % kvh == 0);
  const int G = QH / kvh;
  TORCH_CHECK(nsplit >= 1 && nsplit <= 64);
  TORCH_CHECK(opart.numel() >= (int64_t)kvh * nsplit * 16 * G * kHS && ml.numel() >= (int64_t)kvh * nsplit * 16 * G * 2);
  Args p{};
  p.q = reinterpret_cast<const bf16*>(q.data_ptr()); p.sq_tok = q.stride(0); p.sq_head = q.stride(1);
  p.kc = reinterpret_cast<const bf16*>(kc.data_ptr()); p.sk_blk = kc.stride(0); p.sk_tok = kc.stride(1); p.sk_head = kc.stride(2);
  p.vc = reinterpret_cast<const bf16*>(vc.data_ptr()); p.sv_blk = vc.stride(0); p.sv_tok = vc.stride(1); p.sv_head = vc.stride(2);
  p.bt = bt.data_ptr<int>();
  p.seqlen = seqlen.data_ptr<int>();
  if (qq.numel()) { TORCH_CHECK(qq.scalar_type() == at::kFloat && qq.stride(1) == 1); p.qq = qq.data_ptr<float>(); p.sqq = qq.stride(0); }
  TORCH_CHECK(hspl >= 1 && G % hspl == 0, "tree_attn: hspl must divide the q heads per kv head");
  if (TA_FUSE) {
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    TORCH_CHECK((int64_t)kvh * hspl * nsplit <= sms, "tree_attn (fused merge): grid must be co-resident (<= SMs)");
    TORCH_CHECK(sync.scalar_type() == at::kInt && sync.numel() >= 2 * kvh * hspl);
    p.sync = reinterpret_cast<unsigned*>(sync.data_ptr());
  }
  p.T = T; p.page = (int)kc.size(1); p.nsplit = (int)nsplit; p.G = G; p.QH = QH; p.hspl = (int)hspl;
  p.scale_log2 = (float)(scale * 1.4426950408889634);
  p.scale = (float)scale;
  p.opart = opart.data_ptr<float>(); p.ml = ml.data_ptr<float>();
  p.out = reinterpret_cast<bf16*>(out.data_ptr()); p.so_tok = out.stride(0); p.so_head = out.stride(1);
  if (gate.numel()) {
    TORCH_CHECK(gate.scalar_type() == at::kBFloat16 && gate.stride(-1) == 1 && gate.size(-1) == (int64_t)QH * kHS);
    p.gate = reinterpret_cast<const bf16*>(gate.data_ptr()); p.sg_tok = gate.stride(0);
  }
  if (kn.numel()) {
    TORCH_CHECK(kn.scalar_type() == at::kBFloat16 && vn.scalar_type() == at::kBFloat16 && kn.dim() == 3 &&
                vn.dim() == 3 && kn.size(0) == T && vn.size(0) == T && kn.size(1) == kvh && vn.size(1) == kvh &&
                kn.size(2) == kHS && vn.size(2) == kHS && kn.stride(2) == 1 && vn.stride(2) == 1);
    TORCH_CHECK((kn.stride(0) * 2) % 16 == 0 && (kn.stride(1) * 2) % 16 == 0 && (vn.stride(0) * 2) % 16 == 0 &&
                (vn.stride(1) * 2) % 16 == 0 && reinterpret_cast<uintptr_t>(kn.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(vn.data_ptr()) % 16 == 0);
    TORCH_CHECK(slot.scalar_type() == at::kLong && slot.numel() >= T && slot.stride(0) == 1);
    p.kn = reinterpret_cast<const bf16*>(kn.data_ptr()); p.skn_tok = kn.stride(0); p.skn_head = kn.stride(1);
    p.vn = reinterpret_cast<const bf16*>(vn.data_ptr()); p.svn_tok = vn.stride(0); p.svn_head = vn.stride(1);
    p.slot = slot.data_ptr<int64_t>();
  }
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (exact) {
    TORCH_CHECK(nsplit == kNSeg && G <= 16 && !TA_FUSE, "tree_attn exact: nsplit must be ", kNSeg);
    const bool st2 = exact == 2;  // 2-stage ring (2 CTAs / SM), GH <= 4
    if (TA_XF) {
      int dev = 0, sms = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
      p.xf = st2 && G / hspl <= 2 && (int64_t)kvh * hspl * nsplit <= 2 * sms && sync.scalar_type() == at::kInt &&
             sync.numel() >= 2 * kvh * hspl;
      if (p.xf) p.sync = reinterpret_cast<unsigned*>(sync.data_ptr());
    }
    if (TA_PIPE && st2 && G / hspl <= 2) {  // ATT2: software-pipelined exact split (same results)
      if (TA_LAUNCH & 1) { if (G / hspl == 1) launch_split_exact_pipe<1>(p, kvh, st); else launch_split_exact_pipe<2>(p, kvh, st); }
    } else if (TA_LAUNCH & 1)
    switch (G / hspl) {
      case 1: st2 ? launch_split_exact<1, 2>(p, kvh, st) : launch_split_exact<1, 3>(p, kvh, st); break;
      case 2: st2 ? launch_split_exact<2, 2>(p, kvh, st) : launch_split_exact<2, 3>(p, kvh, st); break;
      case 3: st2 ? launch_split_exact<3, 2>(p, kvh, st) : launch_split_exact<3, 3>(p, kvh, st); break;
      case 4: st2 ? launch_split_exact<4, 2>(p, kvh, st) : launch_split_exact<4, 3>(p, kvh, st); break;
      case 6: launch_split_exact<6, 3>(p, kvh, st); break;
      default: TORCH_CHECK(false, "tree_attn: unsupported q heads per CTA ", G / hspl);
    }
    if ((TA_LAUNCH & 2) && !p.xf) tree_attn_combine_exact_kernel<<<T * QH, 64, 0, st>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (TA_LAUNCH & 1) {
  switch (G / hspl) {
    case 1: launch_split<1>(p, kvh, st); break;
    case 2: launch_split<2>(p, kvh, st); break;
    case 3: launch_split<3>(p, kvh, st); break;
    case 4: launch_split<4>(p, kvh, st); break;
    case 6: launch_split<6>(p, kvh, st); break;
    default: TORCH_CHECK(false, "tree_attn: unsupported q heads per CTA ", G / hspl);
  }
  }
  if ((TA_LAUNCH & 2) && !TA_FUSE)
  tree_attn_combine_kernel<<<T * QH, 32 * kCombWarps, 0, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("tree_attn_fwd", &tree_attn_fwd); }
