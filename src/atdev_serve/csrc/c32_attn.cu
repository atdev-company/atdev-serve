// c=32 varlen decode / MTP-verify full attention (ATDEV_OPT=c32_attn, src/atdev_serve/c32_attn.py).
//
// Replaces vLLM 0.31.0 TRITON_ATTN's 2D kernel_unified_attention for batches of short query rows per request
// (MTP verify: k+1 rows / request; any per-request query length via cu_seqlens_q, ql * G <= 16 MT rows) with a
// bf16 paged KV cache, head 256, causal, no sinks / alibi / softcap / window / qq_bias / fp8.
//
// CTA = (request, kv head): all ql * G query rows of the request that share the kv head (Qwen3.8: 6 q heads per kv
// head, padded to MT m16 tiles) read each K / V tile ONCE (Triton: BLOCK_M 16 / BLOCK_Q 2 -> several programs per
// (request, kv head), each reading the whole K / V). 64 MT threads: warp pair per m16 tile; per 32-key tile warp kh
// computes S for keys [16 kh, 16 kh + 16), the pair exchanges row max / bf16 P / row-sum partials through smem and
// each warp accumulates O for one 128-dim half over all 32 keys. cp.async K / V ring (ST = 2 stages, 2 CTAs / SM),
// Q fragments in registers. Requests are served longest-first (blockIdx -> rank by seq len, computed on the GPU) so
// the long CTAs start in the first wave. Graph-capturable: every length is read on the GPU, idle CTAs exit.
//
// Bit-identical to TRITON_ATTN 2D (TILE_SIZE 32, BLOCK_M 16, 4 warps, compiled for sm_80):
//   S = fma(scale, mma chain over the 16 dim chunks from 0, 0); key j > ctx + tok -> -inf;
//   m = max(M, rowmax S); m = m > -inf ? m : 0; P = ex2((S - m) * log2e) (ex2.approx, no ftz);
//   l = per 8-key group (p[2t] + p[2t+1]) -> + bfly 2 -> + bfly 1, groups z0..z3 combined as (z0+z2)+(z1+z3);
//   alpha = ex2((M - m) * log2e); L = fma(L, alpha, l) (L0 = 1, M0 = -inf); acc = mma(bf16 P, V, acc * alpha)
//   (k16 steps keys 0-15 then 16-31); out = bf16(div.full(acc, L)).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace {

typedef __nv_bfloat16 bf16;
constexpr int kHS = 256;
constexpr int kMaxReq = 256;  // max requests per call (smem rank table)
constexpr int kPitch = kHS + 8;  // 528 B rows: ldmatrix conflict-free
constexpr int kTile = 32;

struct Args {
  const bf16* q; int64_t sq_tok, sq_head;
  const bf16* kc; int64_t sk_blk, sk_tok, sk_head;
  const bf16* vc; int64_t sv_blk, sv_tok, sv_head;
  const int* bt; int64_t sbt;
  const int* seqlen;   // [R]
  const int* cuq;      // [R + 1]
  int R, page, G, QH, kvh, nsplit;  // nsplit, xs, opart, ml, cnt: unused (kept for the parameter layout)
  int xs;
  float scale;
  bf16* out; int64_t so_tok, so_head;
  float* opart;
  float* ml;
  unsigned* cnt;
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
template <int MT, int ST>
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
  {
    // rank of request i = #longer requests (+ #equal with a lower index); CTA rank r serves rank[r]
    int* lens = rank + kMaxReq;
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
  const int q0 = p.cuq[r], ql = p.cuq[r + 1] - q0;
  const int L = p.seqlen[r];
  if (ql <= 0 || L <= 0) return;
  const int ctx = L - ql;
  const int nrows = ql * p.G;  // host guarantees <= ROWS
  const int nt_all = (L + kTile - 1) / kTile;
  const int j0 = 0, j1 = nt_all;
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
      if (ld + 1 < nld) load_tile(ld + 1, (ld + 1) & 1);
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
        mma16816(s[0], qf[ks], b[c][0], b[c][1]);
        mma16816(s[1], qf[ks], b[c][2], b[c][3]);
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
        l = __fadd_rn(__fadd_rn(z0, z2), __fadd_rn(z1, z3));
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
        mma16816(o[2 * dp], pa[0], b[c][0], b[c][1]);
        mma16816(o[2 * dp + 1], pa[0], b[c][2], b[c][3]);
        mma16816(o[2 * dp], pa[1], b2[c][0], b2[c][1]);
        mma16816(o[2 * dp + 1], pa[1], b2[c][2], b2[c][3]);
      }
    }
  }
  cp_async_wait<0>();
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
}

template <int MT, int ST>
void launch(const Args& a, cudaStream_t st) {
  auto kern = c32_attn_kernel<MT, ST>;
  const int smem = ST * 2 * kTile * kPitch * 2 + MT * 2560 + kMaxReq * 8;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr = true;
  }
  kern<<<dim3(a.R * a.kvh, 1), 64 * MT, smem, st>>>(a);
}

}  // namespace

// q [N, QH, 256]; kc / vc [blocks, page, kvh, 256] views; bt [R, >= pages] int32; seqlen [R] int32 (incl. the query
// rows); cuq [R + 1] int32; out [N, QH, 256].
void c32_attn_fwd(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor bt, torch::Tensor seqlen,
                  torch::Tensor cuq, torch::Tensor out, double scale, int64_t max_ql) {
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && kc.scalar_type() == at::kBFloat16 &&
              vc.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(q.size(2) == kHS && q.stride(2) == 1 && kc.stride(3) == 1 && vc.stride(3) == 1 && out.stride(2) == 1);
  TORCH_CHECK(bt.scalar_type() == at::kInt && seqlen.scalar_type() == at::kInt && cuq.scalar_type() == at::kInt);
  TORCH_CHECK(bt.stride(1) == 1 && seqlen.stride(0) == 1 && cuq.stride(0) == 1);
  const int QH = (int)q.size(1), kvh = (int)kc.size(2);
  TORCH_CHECK(QH % kvh == 0);
  const int G = QH / kvh;
  const int R = (int)seqlen.size(0);
  TORCH_CHECK(R >= 1 && R <= kMaxReq && cuq.size(0) >= R + 1);
  const int rows = (int)max_ql * G;
  TORCH_CHECK(rows >= 1 && rows <= 32, "c32_attn: max_ql * G must be <= 32");
  Args p{};
  p.q = reinterpret_cast<const bf16*>(q.data_ptr()); p.sq_tok = q.stride(0); p.sq_head = q.stride(1);
  p.kc = reinterpret_cast<const bf16*>(kc.data_ptr()); p.sk_blk = kc.stride(0); p.sk_tok = kc.stride(1); p.sk_head = kc.stride(2);
  p.vc = reinterpret_cast<const bf16*>(vc.data_ptr()); p.sv_blk = vc.stride(0); p.sv_tok = vc.stride(1); p.sv_head = vc.stride(2);
  p.bt = bt.data_ptr<int>(); p.sbt = bt.stride(0);
  p.seqlen = seqlen.data_ptr<int>(); p.cuq = cuq.data_ptr<int>();
  p.R = R; p.page = (int)kc.size(1); p.G = G; p.QH = QH; p.kvh = kvh; p.nsplit = 0;
  p.scale = (float)scale;
  p.out = reinterpret_cast<bf16*>(out.data_ptr()); p.so_tok = out.stride(0); p.so_head = out.stride(1);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (rows <= 16) launch<1, 2>(p, st); else launch<2, 2>(p, st);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("c32_attn_fwd", &c32_attn_fwd); }
