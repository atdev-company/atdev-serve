// Single-state GDN spec-verify kernel, CUDA (ATDEV_OPT=gdn_single_state, ATDEV_GDN_IMPL=cuda), with the
// slot semantics of gdn_single_state.py:
//   * read slot s0 = indices[n, 0] (state after token 0 of the previous chunk) and replay the a-1
//     accepted tokens 1..a-1 of the previous chunk from the token log kept in slot s1 = indices[n, 1]
//     (fp32 k, v, decay, beta);
//   * process the T current tokens, write ONLY the state after token 0 back into s0, log tokens 1..T-1
//     into the other log buffer, flip the parity.
// The log layout equals the Triton kernel's with BV = 32, so both can serve the same slots (the Triton
// path handles mixed prefill + spec batches). bf16 state, K = V = 128, k <= 3 (T <= 4).
//
// Optional W4A4 epilogue (gdn_ss_a4, used by gdn_a4_fold): the kernel also writes the out_proj s4
// fragments + group scales of its bf16 output rows (one 128-group = one value head), bit-identical to
// marlin_plus_a4 quant_a4 on the bf16 core rows.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kK = 128;
constexpr int kV = 128;

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
  return x;
}
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float softplus_f(float x) { return x > 20.0f ? x : log1pf(__expf(x)); }

struct Args {
  const __nv_bfloat16* qkv; int64_t qkv_row;
  const __nv_bfloat16* a; int64_t a_row;
  const __nv_bfloat16* b; int64_t b_row;
  const float* a_log; const void* dt_bias; int dt_bias_bf16;
  unsigned char* state; int64_t slot_bytes;  // state base (bytes), slot stride in bytes
  const int* idx; int idx_width;
  const int* cu; const int* acc;
  const __nv_bfloat16* gate; int64_t gate_row;
  const void* norm_w; int norm_w_bf16;
  __nv_bfloat16* out;
  int H, HV, G, max_log, ng, nvp;  // ng: unused (1)
  int64_t log_base, buf, v_off, d_off, b_off;  // fp32 element offsets inside slot s1
  float scale, eps;
  int dbg;  // always 0 (kept for the parameter layout)
  uint32_t* a4_pk; float* a4_s; const float* a4_gs; int a4_tmb;  // W4A4 epilogue (a4_pk null: off)
};

// ---------------------------------------------------------------------------
// gdn_ss4: FlashInfer wide_vec structure under the single-state protocol.
// One 128-thread CTA per (request, value head) (non-persistent grid); token setup (q/k l2-norm,
// decay, beta, replay log, log write, parity) staged in smem once; the 128 state rows are processed
// in 4 passes of 32 rows: 16 lanes cooperate on one row (8 K-elements = one 16 B load per lane),
// 8 row groups x 4 rows (ILP) per pass, all tokens per pass in registers; fused gated norm.
// Memory pipeline: every thread reads the item metadata itself (broadcast loads); the state rows of
// the first pass are cp.async'd to smem right after idx is known, before the parity / log / token-row
// loads and the token-setup barrier (each thread copies exactly the 16 B pieces it later reads, so
// only cp.async.wait_group guards them); after a pass is converted to registers its buffer is
// refilled with the next pass. The decay is kept as a running CTA-uniform scale G (h = G * H), so
// the per-element decay multiply disappears. Dot products use a reduce-scatter over the 16 lanes of
// a row (lane l16 owns local rows grp*4 + (i ^ c), c = 2*(l16&1) + ((l16>>1)&1)), with the same
// pairwise addition tree as a butterfly. CTAs are ordered longest-first by T + n_rep (warp-ballot
// rank; each item's computation is unchanged).
// ---------------------------------------------------------------------------
constexpr int kS4Threads = 128;
constexpr int kS4MaxT = 4, kS4MaxLog = 3, kS4MaxTok = 7;

// s4 fragments of one (row, value head) group: lane holds values d = lane + 32 i (i = 0..3) of the
// bf16-rounded output; formula / layout of marlin_plus_a4 a4_quant_kernel (value v = 32a + 16p +
// 4b + e -> nibble 8e + 4p of word (kb = 2g + a/2, l = (row%8)*4 + b, r = (row%16)/8 + 2(a%2))).
__device__ __forceinline__ void a4_store_head(const float (&y)[4], int lane, int g, int row, const Args& p) {
  float amax = fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float s = fmaxf(amax, 1e-8f) * (1.0f / 7.0f);
  const int chunk_rows = 16 * p.a4_tmb, ch = row / chunk_rows, rr = row - ch * chunk_rows;
  const int K = p.HV * kV;
  uint32_t* pk = p.a4_pk + (size_t)ch * (K / 64) * p.a4_tmb * 32 * 4;
  const int e = lane & 3, pp = lane >> 4, b = (lane >> 2) & 3;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float q = rintf(__fdiv_rn(y[i], s));
    q = fminf(fmaxf(q, -8.f), 7.f);
    uint32_t w = ((uint32_t)((int)q) & 0xFu) << (8 * e + 4 * pp);
    w |= __shfl_xor_sync(0xffffffffu, w, 1);
    w |= __shfl_xor_sync(0xffffffffu, w, 2);
    w |= __shfl_xor_sync(0xffffffffu, w, 16);
    if (e == 0 && pp == 0) {
      const int kb = 2 * g + (i >> 1);
      const int l = (rr & 7) * 4 + b;
      const int r = ((rr & 15) >> 3) + 2 * (i & 1);
      pk[((size_t)(kb * p.a4_tmb + (rr >> 4)) * 32 + l) * 4 + r] = w;
    }
  }
  if (lane == 0) p.a4_s[(size_t)ch * (K / 128) * chunk_rows + (size_t)g * chunk_rows + rr] = s * p.a4_gs[0];
}

constexpr int kLPR = 16;            // lanes per state row
constexpr int kRPT = kLPR / 4;      // rows per thread per 32-row pass
constexpr int kEPT = kK / kLPR;     // K elements per thread per row
constexpr int kSEG = kEPT / 8;      // gdn_ss4 lane segments per thread

// smem position of K element d in k_f / q_f: [seg][half][thread][4] so each float4 read of a
// row's thread group is 16 B-contiguous across threads (no bank conflicts).
__device__ __forceinline__ int kpos(int d) {
  const int L = d >> 3, e = d & 7, m = L / kSEG, sg = L % kSEG;
  return ((sg * 2 + (e >> 2)) * kLPR + m) * 4 + (e & 3);
}
// smem byte offset of the 16 B state piece (local row lr, thread l, segment sg): [row][seg][thread]
__device__ __forceinline__ int spos(int lr, int l, int sg) { return ((lr * kSEG + sg) * kLPR + l) * 16; }
__device__ __forceinline__ float seg_tree(const float (&a)[kSEG]) {
  if constexpr (kSEG == 1) return a[0];
  else if constexpr (kSEG == 2) return a[0] + a[1];
  else return (a[0] + a[1]) + (a[2] + a[3]);
}
__device__ __forceinline__ void st_state(void* ptr, uint4 v) { *reinterpret_cast<uint4*>(ptr) = v; }
__device__ __forceinline__ void cp_async_wait_n(int n) {
  if (n <= 0) asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  else if (n == 1) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
  else if (n == 2) asm volatile("cp.async.wait_group 2;\n" ::: "memory");
  else asm volatile("cp.async.wait_group 3;\n" ::: "memory");
}

template <bool SIGMOID>
__global__ __launch_bounds__(kS4Threads, 6) void gdn_ss4_kernel(Args p, int num_items) {
  __shared__ __align__(16) float k_f[kS4MaxTok][kK];
  __shared__ __align__(16) float q_f[kS4MaxT][kK];
  __shared__ __align__(16) float v_f[kS4MaxTok][kV];
  __shared__ __align__(16) __nv_bfloat16 g_w[kS4MaxT][kV];
  __shared__ __align__(16) __nv_bfloat16 o_s[kS4MaxT][kV];
  __shared__ __align__(16) unsigned char s_st[1][32 * kK * 2];
  __shared__ float dec_s[kS4MaxTok], bet_s[kS4MaxTok];
  __shared__ float nw_s[kV];
  // longest-first: CTA blockIdx -> (request of work-rank blockIdx / HV, head blockIdx % HV), work =
  // T + n_rep (stable by request index). Bit-identical (each item's computation is unchanged).
  if ((int)blockIdx.x >= num_items) return;
  int item = blockIdx.x;
  // warp-level rank (no barrier / smem): every warp redundantly ranks the <= 32 requests by work
  // (T + n_rep) via 8 ballots and picks the request of rank blockIdx / HV
  if (num_items / p.HV <= 32) {
    const int nreq = num_items / p.HV, r = blockIdx.x / p.HV, ln = threadIdx.x & 31;
    int w = -1;
    if (ln < nreq) {
      int rr = p.acc[ln] - 1;
      rr = rr < 0 ? 0 : (rr > p.max_log ? p.max_log : rr);
      w = p.cu[ln + 1] - p.cu[ln] + rr;
      w = w > 15 ? 15 : w;
    }
    int above = 0, rank = 0;
    const unsigned lt = (1u << ln) - 1u;
#pragma unroll
    for (int v = 15; v >= 0; --v) {
      const unsigned m = __ballot_sync(0xffffffffu, w == v);
      if (w == v) rank = above + __popc(m & lt);
      above += __popc(m);
    }
    const unsigned pick = __ballot_sync(0xffffffffu, w >= 0 && rank == r);
    item = (__ffs(pick) - 1) * p.HV + blockIdx.x % p.HV;
  }
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int n = item / p.HV, hv = item % p.HV;
  const int hk = hv / p.G;
  const int bos = p.cu[n], T = p.cu[n + 1] - bos;
  const int s0 = p.idx[n * p.idx_width], s1 = p.idx[n * p.idx_width + 1];
  int r_ = p.acc[n] - 1;
  const int n_rep = r_ < 0 ? 0 : (r_ > p.max_log ? p.max_log : r_);
  const int run = T > 0 && s0 > 0 && s1 > 0 && T <= p.max_log + 1 && T <= kS4MaxT && n_rep <= kS4MaxLog &&
                  n_rep + T <= kS4MaxTok;
  if (!run) {
    for (int i = tid; i < T * kV; i += kS4Threads)
      p.out[((int64_t)(bos + i / kV) * p.HV + hv) * kV + i % kV] = __float2bfloat16(0.f);
    if (p.a4_pk != nullptr) {
      const float z[4] = {0.f, 0.f, 0.f, 0.f};
      for (int t = warp; t < T; t += kS4Threads / 32) a4_store_head(z, lane, hv, bos + t, p);
    }
    return;
  }
  if (p.dbg & 1024) return;  // microbench: launch + metadata only
  const int l16 = tid % kLPR, grp = tid / kLPR;
  static_assert(kLPR == 16 && kRPT == 4, "1: gdn_ss4 thread mapping only");
  const int rsc = 2 * (l16 & 1) + ((l16 >> 1) & 1);  // slot i <-> local row grp*4 + (i ^ rsc)
#define S12_LR(i) (grp * kRPT + ((i) ^ rsc))
  unsigned char* head = p.state + (int64_t)s0 * p.slot_bytes + (int64_t)hv * kV * kK * 2;
  const bool ld = !(p.dbg & 256);
  auto issue = [&](int pass) {
    if (ld) {
      unsigned char* sb = s_st[pass % 1];
#pragma unroll
      for (int i = 0; i < kRPT; ++i) {
        const int lr = S12_LR(i);
#pragma unroll
        for (int sg = 0; sg < kSEG; ++sg)
          cp_async_16(sb + spos(lr, l16, sg),
                      head + ((int64_t)(pass * 32 + lr) * kK + l16 * kEPT + sg * 8) * 2);
      }
    }
    cp_async_commit();
  };
#pragma unroll
  for (int ps = 0; ps < 1; ++ps) issue(ps);

  float* lslot = reinterpret_cast<float*>(p.state + (int64_t)s1 * p.slot_bytes);
  // every independent global load of the token setup is issued before any store (the stores below
  // may alias as far as the compiler knows, which serialised gdn_ss4's loads behind them)
  const bool tw = warp < T;
  const int tt = warp, jt = n_rep + warp;
  const int64_t tok_t = bos + warp;
  const __nv_bfloat16* row_t = p.qkv + tok_t * p.qkv_row;
  __nv_bfloat16 rq[4], rk[4], rv[4], rg[4];
  float av = 0.f, bv = 0.f, dtb = 0.f, alg = 0.f;
  if (tw) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + 32 * i;
      rq[i] = row_t[hk * kK + d];
      rk[i] = row_t[p.H * kK + hk * kK + d];
      rv[i] = row_t[2 * p.H * kK + hv * kV + d];
      rg[i] = p.gate[tok_t * p.gate_row + hv * kV + d];
    }
    if (lane == 0) {
      av = __bfloat162float(p.a[tok_t * p.a_row + hv]);
      bv = __bfloat162float(p.b[tok_t * p.b_row + hv]);
      dtb = p.dt_bias_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(p.dt_bias)[hv])
                           : static_cast<const float*>(p.dt_bias)[hv];
      alg = p.a_log[hv];
    }
  }
  const float nwv = p.norm_w_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(p.norm_w)[tid])
                                  : static_cast<const float*>(p.norm_w)[tid];
  const int pb = n_rep > 0 ? (reinterpret_cast<const int*>(lslot)[hv * p.nvp] & 1) : 0;
  const float* rbuf = lslot + p.log_base + pb * p.buf;
  float* wbuf = lslot + p.log_base + (1 - pb) * p.buf;
  float lk[kS4MaxLog], lv[kS4MaxLog], ld0 = 0.f, lb0 = 0.f;  // kK == kS4Threads: log token j, element tid
#pragma unroll
  for (int jj = 0; jj < kS4MaxLog; ++jj) {
    if (jj < n_rep) {
      lk[jj] = rbuf[(int64_t)(jj * p.H + hk) * kK + tid];
      lv[jj] = rbuf[p.v_off + (int64_t)(jj * p.HV + hv) * kV + tid];
    }
  }
  if (tid < n_rep) { ld0 = rbuf[p.d_off + tid * p.HV + hv]; lb0 = rbuf[p.b_off + tid * p.HV + hv]; }
  // stores
  nw_s[tid] = nwv;
#pragma unroll
  for (int jj = 0; jj < kS4MaxLog; ++jj) {
    if (jj < n_rep) {
      k_f[jj][kpos(tid)] = lk[jj];
      v_f[jj][tid] = __bfloat162float(__float2bfloat16(lv[jj]));
    }
  }
  if (tid < n_rep) { dec_s[tid] = ld0; bet_s[tid] = lb0; }
  if (tw) {
    float qv[4], kv[4], qs = 0.f, ks = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      qv[i] = __bfloat162float(rq[i]);
      kv[i] = __bfloat162float(rk[i]);
      qs += qv[i] * qv[i];
      ks += kv[i] * kv[i];
    }
    qs = warp_sum(qs);
    ks = warp_sum(ks);
    const float qn = rsqrtf(qs + 1e-6f) * p.scale, kn = rsqrtf(ks + 1e-6f);
    const bool log_k = tt >= 1 && (hv % p.G) == 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + 32 * i;
      q_f[tt][kpos(d)] = qv[i] * qn;
      k_f[jt][kpos(d)] = kv[i] * kn;
      v_f[jt][d] = __bfloat162float(rv[i]);
      g_w[tt][d] = rg[i];
      if (log_k) wbuf[(int64_t)((tt - 1) * p.H + hk) * kK + d] = kv[i] * kn;
      if (tt >= 1) wbuf[p.v_off + (int64_t)((tt - 1) * p.HV + hv) * kV + d] = __bfloat162float(rv[i]);
    }
    if (lane == 0) {
      const float dec = __expf(-__expf(alg) * softplus_f(av + dtb));
      const float bet = sigmoid_f(bv);
      dec_s[jt] = dec;
      bet_s[jt] = bet;
      if (tt >= 1) {
        wbuf[p.d_off + (tt - 1) * p.HV + hv] = dec;
        wbuf[p.b_off + (tt - 1) * p.HV + hv] = bet;
      }
    }
  }
  __syncthreads();  // token arrays complete; every thread has read the parity word
  if (tid < p.nvp) reinterpret_cast<int*>(lslot)[hv * p.nvp + tid] = 1 - pb;
  const int TT = (p.dbg & 128) ? 0 : n_rep + T;
#pragma unroll 1
  for (int pass = 0; pass < 4; ++pass) {
    float h[kRPT][kEPT];
    int rows[kRPT];
    {
      const int after = 3 - pass < 1 - 1 ? 3 - pass : 1 - 1;
      cp_async_wait_n(after);
    }
    const unsigned char* sb = s_st[pass % 1];
#pragma unroll
    for (int i = 0; i < kRPT; ++i) {
      rows[i] = pass * 32 + S12_LR(i);
#pragma unroll
      for (int sg = 0; sg < kSEG; ++sg) {
        const uint4 w = *reinterpret_cast<const uint4*>(sb + spos(S12_LR(i), l16, sg));
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const __nv_bfloat162 v2 = *reinterpret_cast<const __nv_bfloat162*>(&ww[e]);
          h[i][sg * 8 + 2 * e] = __bfloat162float(v2.x);
          h[i][sg * 8 + 2 * e + 1] = __bfloat162float(v2.y);
        }
      }
    }
    if (pass + 1 < 4) issue(pass + 1);
    float gG = 1.f;
#pragma unroll 1
    for (int j = 0, gsc = 0; j < TT; ++j) {
      // h = G * H (G uniform per CTA); the per-element decay multiply is folded into G
      (void)gsc;
      const float dec = dec_s[j], bet = bet_s[j];
      gG *= dec;
      if (gG < 1e-30f) {  // renormalise (CTA-uniform branch)
#pragma unroll
        for (int i = 0; i < kRPT; ++i)
#pragma unroll
          for (int e = 0; e < kEPT; ++e) h[i][e] *= gG;
        gG = 1.f;
      }
      const float ginv = 1.0f / gG;
      float kk[kEPT];
#pragma unroll
      for (int e = 0; e < kEPT; e += 4) {
        const float4 a4 = *reinterpret_cast<const float4*>(&k_f[j][((e >> 2) * kLPR + l16) * 4]);
        kk[e] = a4.x; kk[e + 1] = a4.y; kk[e + 2] = a4.z; kk[e + 3] = a4.w;
      }
      float d4[kRPT];
#pragma unroll
      for (int i = 0; i < kRPT; ++i) {
        float ps[kSEG];
#pragma unroll
        for (int sg = 0; sg < kSEG; ++sg) {
          float acc = 0.f;
#pragma unroll
          for (int e = sg * 8; e < sg * 8 + 8; ++e) acc += h[i][e] * kk[e];
          ps[sg] = acc;
        }
        d4[i] = seg_tree(ps);
      }
      {
        // slot 0 (row c) / 1 (row c^1) keep, slots 2 / 3 = the xor-1 partner's slots 0 / 1
        const float e0 = d4[0] + __shfl_xor_sync(0xffffffffu, d4[2], 1);
        const float e1 = d4[1] + __shfl_xor_sync(0xffffffffu, d4[3], 1);
        float f = e0 + __shfl_xor_sync(0xffffffffu, e1, 2);  // xor-2 partner's slot 1 = row c
        f += __shfl_xor_sync(0xffffffffu, f, 4);
        f += __shfl_xor_sync(0xffffffffu, f, 8);
        float dd[kRPT];
        dd[0] = (v_f[j][rows[0]] - gG * f) * bet * ginv;  // this lane's row only
        dd[1] = __shfl_xor_sync(0xffffffffu, dd[0], 2);  // row c^1
        dd[2] = __shfl_xor_sync(0xffffffffu, dd[0], 1);  // row c^2
        dd[3] = __shfl_xor_sync(0xffffffffu, dd[0], 3);  // row c^3
#pragma unroll
        for (int i = 0; i < kRPT; ++i) {
#pragma unroll
          for (int e = 0; e < kEPT; ++e) h[i][e] += kk[e] * dd[i];
        }
      }
      if (j < n_rep) continue;
      const int t = j - n_rep;
#pragma unroll
      for (int e = 0; e < kEPT; e += 4) {
        const float4 a4 = *reinterpret_cast<const float4*>(&q_f[t][((e >> 2) * kLPR + l16) * 4]);
        kk[e] = a4.x; kk[e + 1] = a4.y; kk[e + 2] = a4.z; kk[e + 3] = a4.w;
      }
#pragma unroll
      for (int i = 0; i < kRPT; ++i) {
        float ps[kSEG];
#pragma unroll
        for (int sg = 0; sg < kSEG; ++sg) {
          float acc = 0.f;
#pragma unroll
          for (int e = sg * 8; e < sg * 8 + 8; ++e) acc += h[i][e] * kk[e];
          ps[sg] = acc;
        }
        d4[i] = seg_tree(ps);
      }
      {
        const float e0 = d4[0] + __shfl_xor_sync(0xffffffffu, d4[2], 1);
        const float e1 = d4[1] + __shfl_xor_sync(0xffffffffu, d4[3], 1);
        float f = e0 + __shfl_xor_sync(0xffffffffu, e1, 2);
        f += __shfl_xor_sync(0xffffffffu, f, 4);
        f += __shfl_xor_sync(0xffffffffu, f, 8);
        if (l16 < 4) o_s[t][rows[0]] = __float2bfloat16(gG * f);  // lanes 0..3 cover rows c = 0..3
      }
      if (t == 0 && !(p.dbg & 1)) {
#pragma unroll
        for (int i = 0; i < kRPT; ++i) {
#pragma unroll
          for (int sg = 0; sg < kSEG; ++sg) {
            uint32_t w[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              __nv_bfloat162 v2 = __floats2bfloat162_rn(gG * h[i][sg * 8 + 2 * e], gG * h[i][sg * 8 + 2 * e + 1]);
              w[e] = *reinterpret_cast<uint32_t*>(&v2);
            }
            st_state(head + ((int64_t)rows[i] * kK + l16 * kEPT + sg * 8) * 2, make_uint4(w[0], w[1], w[2], w[3]));
          }
        }
      }
    }
    if (TT == 0 && !(p.dbg & 1)) {  // memory skeleton: write the loaded state back
#pragma unroll
      for (int i = 0; i < kRPT; ++i) {
#pragma unroll
        for (int sg = 0; sg < kSEG; ++sg) {
          uint32_t w[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            __nv_bfloat162 v2 = __floats2bfloat162_rn(h[i][sg * 8 + 2 * e], h[i][sg * 8 + 2 * e + 1]);
            w[e] = *reinterpret_cast<uint32_t*>(&v2);
          }
          *reinterpret_cast<uint4*>(head + ((int64_t)rows[i] * kK + l16 * kEPT + sg * 8) * 2) =
              make_uint4(w[0], w[1], w[2], w[3]);
        }
      }
    }
  }
  __syncthreads();
  if (p.dbg & 2048) return;  // microbench: no gated-norm / a4 epilogue
  if (warp < T) {
    const int t = warp;
    const int64_t tok = bos + t;
    float o[4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { o[i] = __bfloat162float(o_s[t][lane + 32 * i]); ss += o[i] * o[i]; }
    ss = warp_sum(ss);
    const float rstd = rsqrtf(ss / kV + p.eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + 32 * i;
      const float nw = nw_s[d];
      const float z = __bfloat162float(g_w[t][d]);
      const float gt = SIGMOID ? sigmoid_f(z) : z * sigmoid_f(z);
      const __nv_bfloat16 ob = __float2bfloat16(o[i] * rstd * nw * gt);
      p.out[(tok * p.HV + hv) * kV + d] = ob;
      o[i] = __bfloat162float(ob);
    }
    if (p.a4_pk != nullptr) a4_store_head(o, lane, hv, (int)tok, p);
  }
}
}  // namespace

static bool gdn_ss_impl(torch::Tensor qkv, torch::Tensor a, torch::Tensor b, torch::Tensor a_log,
            torch::Tensor dt_bias, torch::Tensor state, torch::Tensor idx, torch::Tensor cu,
            torch::Tensor acc, torch::Tensor gate, torch::Tensor norm_w, torch::Tensor out,
            int64_t H, int64_t num_requests, double scale, double eps, bool sigmoid_gate,
            int64_t log_base, int64_t buf, int64_t v_off, int64_t d_off, int64_t b_off, int64_t nvp,
            uint32_t* a4_pk, float* a4_s, const float* a4_gs, int a4_tmb) {
  TORCH_CHECK(state.scalar_type() == at::kBFloat16, "gdn_ss: bf16 state only");
  TORCH_CHECK(state.size(2) == kV && state.size(3) == kK, "gdn_ss: K = V = 128 only");
  TORCH_CHECK(qkv.stride(1) == 1 && a.stride(-1) == 1 && b.stride(-1) == 1, "gdn_ss: strides");
  TORCH_CHECK(out.is_contiguous(), "gdn_ss: out must be contiguous");
  if (num_requests <= 0) return true;
  Args p;
  p.a4_pk = a4_pk; p.a4_s = a4_s; p.a4_gs = a4_gs; p.a4_tmb = a4_tmb;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()); p.qkv_row = qkv.stride(0);
  p.a = reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()); p.a_row = a.stride(0);
  p.b = reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()); p.b_row = b.stride(0);
  p.a_log = a_log.data_ptr<float>();
  p.dt_bias = dt_bias.data_ptr(); p.dt_bias_bf16 = dt_bias.scalar_type() == at::kBFloat16;
  TORCH_CHECK(dt_bias.scalar_type() == at::kBFloat16 || dt_bias.scalar_type() == at::kFloat);
  p.state = reinterpret_cast<unsigned char*>(state.data_ptr()); p.slot_bytes = state.stride(0) * 2;
  p.idx = idx.data_ptr<int>(); p.idx_width = idx.stride(0);
  p.cu = cu.data_ptr<int>(); p.acc = acc.data_ptr<int>();
  p.gate = reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()); p.gate_row = gate.stride(0);
  p.norm_w = norm_w.data_ptr(); p.norm_w_bf16 = norm_w.scalar_type() == at::kBFloat16;
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.HV = state.size(1); p.H = H; p.G = p.HV / H;
  p.max_log = idx.size(1) - 1; p.ng = 1; p.nvp = nvp;
  p.log_base = log_base; p.buf = buf; p.v_off = v_off; p.d_off = d_off; p.b_off = b_off;
  p.scale = scale; p.eps = eps;
  p.dbg = 0;
  TORCH_CHECK(p.max_log + 1 <= kS4MaxT && p.HV <= 64, "gdn_ss: k <= ", kS4MaxT - 1, " and <= 64 value heads");
  auto stream = at::cuda::getCurrentCUDAStream();
  const int items = static_cast<int>(num_requests * p.HV);
  if (sigmoid_gate) gdn_ss4_kernel<true><<<items, kS4Threads, 0, stream>>>(p, items);
  else gdn_ss4_kernel<false><<<items, kS4Threads, 0, stream>>>(p, items);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

void gdn_ss(torch::Tensor qkv, torch::Tensor a, torch::Tensor b, torch::Tensor a_log,
            torch::Tensor dt_bias, torch::Tensor state, torch::Tensor idx, torch::Tensor cu,
            torch::Tensor acc, torch::Tensor gate, torch::Tensor norm_w, torch::Tensor out,
            int64_t H, int64_t num_requests, double scale, double eps, bool sigmoid_gate,
            int64_t log_base, int64_t buf, int64_t v_off, int64_t d_off, int64_t b_off, int64_t nvp) {
  gdn_ss_impl(qkv, a, b, a_log, dt_bias, state, idx, cu, acc, gate, norm_w, out, H, num_requests,
              scale, eps, sigmoid_gate, log_base, buf, v_off, d_off, b_off, nvp,
              nullptr, nullptr, nullptr, 0);
}

// same + W4A4 out_proj fragments (a4_pk int32 [chunks][K/64][tmb][32][4], a4_s fp32
// [chunks][K/128][16 tmb], gs fp32 [1])
bool gdn_ss_a4(torch::Tensor qkv, torch::Tensor a, torch::Tensor b, torch::Tensor a_log,
               torch::Tensor dt_bias, torch::Tensor state, torch::Tensor idx, torch::Tensor cu,
               torch::Tensor acc, torch::Tensor gate, torch::Tensor norm_w, torch::Tensor out,
               int64_t H, int64_t num_requests, double scale, double eps, bool sigmoid_gate,
               int64_t log_base, int64_t buf, int64_t v_off, int64_t d_off, int64_t b_off, int64_t nvp,
               torch::Tensor a4_pk, torch::Tensor a4_s, torch::Tensor a4_gs, int64_t tmb) {
  TORCH_CHECK(a4_pk.scalar_type() == at::kInt && a4_s.scalar_type() == at::kFloat &&
              a4_gs.scalar_type() == at::kFloat && tmb >= 1);
  return gdn_ss_impl(qkv, a, b, a_log, dt_bias, state, idx, cu, acc, gate, norm_w, out, H,
                     num_requests, scale, eps, sigmoid_gate, log_base, buf, v_off, d_off, b_off, nvp,
                     reinterpret_cast<uint32_t*>(a4_pk.data_ptr()), a4_s.data_ptr<float>(),
                     a4_gs.data_ptr<float>(), (int)tmb);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_ss", &gdn_ss, "single-state GDN verify");
  m.def("gdn_ss_a4", &gdn_ss_a4, "single-state GDN verify + W4A4 out_proj fragments");
}
