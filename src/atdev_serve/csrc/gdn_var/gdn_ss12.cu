// gdn_ss12 (C32g) = gdn_ss10 + S12_RS (default 1): bit-identical reduce-scatter dot products in the
//   GSCALE recurrence. Lane l16 of a 16-lane row group owns local rows grp*4 + (i ^ c), c = 2*(l16&1) +
//   ((l16>>1)&1), so the xor-1 / xor-2 butterfly levels exchange fixed slots (no selects): k-dot 16 SHFL +
//   16 FADD -> 8 SHFL (5 reduce + 3 delta gather) + 5 FADD, delta 4x -> 1x per lane, v_f 4 -> 1 LDS;
//   q-dot 16 + 16 -> 5 + 5, o_s written by lanes l16 < 4. Same pairwise addition tree as the butterfly
//   (FADD is commutative) -> every sum, delta, output and stored state bit-identical to gdn_ss10.
#ifndef S12_RS
#define S12_RS 1
#endif
// gdn_ss10 (C32n-gdn) = gdn_ss4g + S10_LPT (longest-first CTA order by T + n_rep; default 2 = warp-ballot rank, bit-identical) + S10_MAXT (max tokens per request instance)
// gdn_ss4g = gdn_ss4b with S4B_GSCALE=1 (C32-gdn-gs: decay folded into a CTA-uniform scale; rounding-level, gated)
// Single-state GDN spec-verify kernel (ATDEV_OPT=gdn_single_state, ATDEV_GDN_IMPL=cuda).
// gdn_ss4a4 = gdn_ss4 + optional W4A4 epilogue (ATDEV_GDN_A4_FOLD, fused_quant): the gdn_ss4 path
// also writes the out_proj s4 fragments + group scales of its bf16 output rows (one 128-group =
// one value head), bit-identical to marlin_plus_a4 quant_a4 on the bf16 core rows.
//
// Structure follows vLLM 0.31's gdn_decode_post_conv_mtp_kernel
// (csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu: one 256-thread CTA per
// (request, value head), state streamed in four 32-row chunks through
// cp.async double buffering, 4 rows x 4 K-columns of fp32 state per thread,
// fused gated RMSNorm), with the slot semantics of gdn_single_state.py:
//   * read slot s0 = indices[n, 0] (state after token 0 of the previous chunk)
//     and replay the a-1 accepted tokens 1..a-1 of the previous chunk from the
//     token log kept in slot s1 = indices[n, 1] (fp32 k, v, decay, beta);
//   * process the T current tokens, write ONLY the state after token 0 back
//     into s0, log tokens 1..T-1 into the other log buffer, flip the parity.
// Optional int8 storage of s0 (QINT8): each 32-row chunk region of a head's
// native bf16 rows holds [32*128 int8 codes][32*NG fp32 scales][tag] with
// tag = 0x7FC1A3B7 (bf16 pair (number, NaN): never a real state). Untagged
// chunks are read natively (a stock prefill / non-spec write cleared the tag).
// The layout equals the Triton kernel's with BV = 32, so both can serve the
// same slots (the Triton path handles mixed prefill+spec batches).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kK = 128;
constexpr int kV = 128;
constexpr int kThreads = 256;
constexpr int kWarps = 8;
constexpr int kChunk = 32;
constexpr int kNumChunks = kV / kChunk;
constexpr int kRows = kChunk / kWarps;  // 4 rows per warp per chunk
constexpr int kMaxNG = 4;
constexpr int kMagic = 0x7FC1A3B7;
constexpr int kNativeChunkBytes = kChunk * kK * 2;  // bf16
constexpr int kCodeBytes = kChunk * kK;

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }

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
  int H, HV, G, max_log, ng, nvp;
  int64_t log_base, buf, v_off, d_off, b_off;  // fp32 element offsets inside slot s1
  float scale, eps;
  int dbg;  // microbench switches (0 in production)
  uint32_t* a4_pk; float* a4_s; const float* a4_gs; int a4_tmb;  // W4A4 epilogue (a4_pk null: off)
};

template <bool QINT8, bool SIGMOID, int kMaxT, int kMaxTok>
__global__ __launch_bounds__(kThreads, 2) void gdn_ss_kernel(Args p) {
  const int n = blockIdx.x, hv = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int bos = p.cu[n], eos = p.cu[n + 1];
  const int T = eos - bos;
  if (T <= 0) return;
  const int s0 = p.idx[n * p.idx_width], s1 = p.idx[n * p.idx_width + 1];
  if (s0 <= 0 || s1 <= 0 || T > p.max_log + 1 || T > kMaxT) {
    for (int i = tid; i < T * kV; i += kThreads)
      p.out[((int64_t)(bos + i / kV) * p.HV + hv) * kV + i % kV] = __float2bfloat16(0.f);
    return;
  }
  const int hk = hv / p.G;
  int n_rep = p.acc[n] - 1;
  n_rep = n_rep < 0 ? 0 : (n_rep > p.max_log ? p.max_log : n_rep);
  if (n_rep + T > kMaxTok) n_rep = kMaxTok - T;  // unreachable for k <= 7

  float* log_slot = reinterpret_cast<float*>(p.state + (int64_t)s1 * p.slot_bytes);
  int* par = reinterpret_cast<int*>(log_slot) + hv * p.nvp;
  const int par0 = par[0] & 1;
  const int pb = n_rep > 0 ? par0 : 0;
  const float* rbuf = log_slot + p.log_base + pb * p.buf;
  float* wbuf = log_slot + p.log_base + (1 - pb) * p.buf;

  __shared__ alignas(16) unsigned char s_state[2][kNativeChunkBytes];
  __shared__ alignas(16) float s_scale[2][kChunk * kMaxNG];
  __shared__ alignas(16) float s_q[kMaxT][kK];
  __shared__ alignas(16) float s_k[kMaxTok][kK];
  __shared__ alignas(16) __nv_bfloat16 s_v[kMaxTok][kV];  // bf16 values are exact
  __shared__ __nv_bfloat16 s_out[kMaxT][kV];
  __shared__ float s_decay[kMaxTok], s_beta[kMaxTok];
  __shared__ int s_q8[kNumChunks];

  unsigned char* head = p.state + (int64_t)s0 * p.slot_bytes + (int64_t)hv * kV * kK * 2;
  const int ng = p.ng;
  if (QINT8) {
    if (tid < kNumChunks) {
      const int* tag = reinterpret_cast<const int*>(head + tid * kNativeChunkBytes + kCodeBytes +
                                                    kChunk * ng * 4);
      s_q8[tid] = (p.dbg & 2) ? 1 : (*tag == kMagic);
    }
    __syncthreads();
  }
  auto issue = [&](int c) {
    const int st = c & 1;
    unsigned char* src = head + c * kNativeChunkBytes;
    if (QINT8 && s_q8[c]) {
      for (int i = tid; i < kCodeBytes / 16; i += kThreads)
        cp_async_16(&s_state[st][i * 16], src + i * 16);
      for (int i = tid; i < kChunk * ng / 4; i += kThreads)
        cp_async_16(&s_scale[st][i * 4], src + kCodeBytes + i * 16);
    } else {
      for (int i = tid; i < kNativeChunkBytes / 16; i += kThreads)
        cp_async_16(&s_state[st][i * 16], src + i * 16);
    }
    cp_async_commit();
  };
  issue(0);

  // replayed tokens (fp32 from the log)
  for (int i = tid; i < n_rep * kK; i += kThreads) {
    const int j = i / kK, d = i % kK;
    s_k[j][d] = rbuf[(int64_t)(j * p.H + hk) * kK + d];
    s_v[j][d] = __float2bfloat16(rbuf[p.v_off + (int64_t)(j * p.HV + hv) * kV + d]);
  }
  if (tid < n_rep) {
    s_decay[tid] = rbuf[p.d_off + tid * p.HV + hv];
    s_beta[tid] = rbuf[p.b_off + tid * p.HV + hv];
  }
  // current tokens: warp t loads token t
  if (warp < T) {
    const int t = warp, j = n_rep + t, tok = bos + t;
    const __nv_bfloat16* row = p.qkv + (int64_t)tok * p.qkv_row;
    float qv[4], kv[4], qs = 0.f, ks = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + 32 * i;
      qv[i] = __bfloat162float(row[hk * kK + d]);
      kv[i] = __bfloat162float(row[p.H * kK + hk * kK + d]);
      qs += qv[i] * qv[i];
      ks += kv[i] * kv[i];
    }
    qs = warp_sum(qs);
    ks = warp_sum(ks);
    const float qn = rsqrtf(qs + 1e-6f) * p.scale, kn = rsqrtf(ks + 1e-6f);
    const bool log_k = t >= 1 && (hv % p.G) == 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + 32 * i;
      const float vv = __bfloat162float(row[2 * p.H * kK + hv * kV + d]);
      s_q[t][d] = qv[i] * qn;
      s_k[j][d] = kv[i] * kn;
      s_v[j][d] = __float2bfloat16(vv);
      if (log_k) wbuf[(int64_t)((t - 1) * p.H + hk) * kK + d] = kv[i] * kn;
      if (t >= 1) wbuf[p.v_off + (int64_t)((t - 1) * p.HV + hv) * kV + d] = vv;
    }
    if (lane == 0) {
      const float av = __bfloat162float(p.a[(int64_t)tok * p.a_row + hv]);
      const float bv = __bfloat162float(p.b[(int64_t)tok * p.b_row + hv]);
      const float dtb = p.dt_bias_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(p.dt_bias)[hv])
                                       : static_cast<const float*>(p.dt_bias)[hv];
      const float dec = __expf(-__expf(p.a_log[hv]) * softplus_f(av + dtb));
      const float bet = sigmoid_f(bv);
      s_decay[j] = dec;
      s_beta[j] = bet;
      if (t >= 1) {
        wbuf[p.d_off + (t - 1) * p.HV + hv] = dec;
        wbuf[p.b_off + (t - 1) * p.HV + hv] = bet;
      }
    }
  }
  const int TT = n_rep + T;
  const int kb = lane * 4;
  const int gs = kK / ng;  // group size along K

#pragma unroll 1
  for (int c = 0; c < kNumChunks; ++c) {
    cp_async_wait_all();
    __syncthreads();
    if (c + 1 < kNumChunks) issue(c + 1);
    const int st = c & 1;
    const bool q8 = QINT8 && s_q8[c];
    float h[kRows][4];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const int row = warp + r * kWarps;
      if (q8) {
        const char4 cd = *reinterpret_cast<const char4*>(&s_state[st][row * kK + kb]);
        const float sc = s_scale[st][row * ng + kb / gs];
        h[r][0] = cd.x * sc; h[r][1] = cd.y * sc; h[r][2] = cd.z * sc; h[r][3] = cd.w * sc;
      } else {
        const __nv_bfloat162* s2 = reinterpret_cast<const __nv_bfloat162*>(&s_state[st][(row * kK + kb) * 2]);
        const __nv_bfloat162 lo = s2[0], hi = s2[1];
        h[r][0] = __bfloat162float(lo.x); h[r][1] = __bfloat162float(lo.y);
        h[r][2] = __bfloat162float(hi.x); h[r][3] = __bfloat162float(hi.y);
      }
    }
    for (int j = 0; j < TT; ++j) {
      const float4 k4 = *reinterpret_cast<const float4*>(&s_k[j][kb]);
      const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
      const float dec = s_decay[j], bet = s_beta[j];
      float dhk[kRows];
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        dhk[r] = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { h[r][i] *= dec; dhk[r] += h[r][i] * kk[i]; }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) dhk[r] += __shfl_xor_sync(0xffffffffu, dhk[r], o);
      }
      const bool cur = j >= n_rep;
      const float4 q4 = cur ? *reinterpret_cast<const float4*>(&s_q[j - n_rep][kb]) : make_float4(0, 0, 0, 0);
      const float qq[4] = {q4.x, q4.y, q4.z, q4.w};
      float dhq[kRows];
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        const int row = c * kChunk + warp + r * kWarps;
        const float delta = (__bfloat162float(s_v[j][row]) - dhk[r]) * bet;
        dhq[r] = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { h[r][i] += kk[i] * delta; dhq[r] += h[r][i] * qq[i]; }
      }
      if (!cur) continue;
      const int t = j - n_rep;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) dhq[r] += __shfl_xor_sync(0xffffffffu, dhq[r], o);
      }
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) s_out[t][c * kChunk + warp + r * kWarps] = __float2bfloat16(dhq[r]);
      }
      if (t == 0 && !(p.dbg & 1)) {  // the single state write: state after token 0 -> s0
        unsigned char* dst = head + c * kNativeChunkBytes;
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
          const int row = warp + r * kWarps;
          if (QINT8) {
            float am = fmaxf(fmaxf(fabsf(h[r][0]), fabsf(h[r][1])), fmaxf(fabsf(h[r][2]), fabsf(h[r][3])));
            for (int o = gs / 8; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
            const float sc = fmaxf(am, 1e-30f) * (1.0f / 127.0f);
            const float inv = 1.0f / sc;
            char4 cd;
            cd.x = (signed char)max(-127, min(127, __float2int_rn(h[r][0] * inv)));
            cd.y = (signed char)max(-127, min(127, __float2int_rn(h[r][1] * inv)));
            cd.z = (signed char)max(-127, min(127, __float2int_rn(h[r][2] * inv)));
            cd.w = (signed char)max(-127, min(127, __float2int_rn(h[r][3] * inv)));
            *reinterpret_cast<char4*>(dst + row * kK + kb) = cd;
            if (kb % gs == 0)
              reinterpret_cast<float*>(dst + kCodeBytes)[row * ng + kb / gs] = sc;
          } else {
            __nv_bfloat162* d2 = reinterpret_cast<__nv_bfloat162*>(dst + (row * kK + kb) * 2);
            d2[0] = __floats2bfloat162_rn(h[r][0], h[r][1]);
            d2[1] = __floats2bfloat162_rn(h[r][2], h[r][3]);
          }
        }
        if (QINT8 && tid == 0)
          *reinterpret_cast<int*>(dst + kCodeBytes + kChunk * ng * 4) = kMagic;
      }
    }
  }
  __syncthreads();
  if (tid < p.nvp) par[tid] = 1 - pb;

  // gated RMSNorm over the head, as stock
  if (warp < T) {
    const int t = warp, tok = bos + t;
    float o[4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { o[i] = __bfloat162float(s_out[t][lane + 32 * i]); ss += o[i] * o[i]; }
    ss = warp_sum(ss);
    const float rstd = rsqrtf(ss / kV + p.eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + 32 * i;
      const float z = __bfloat162float(p.gate[(int64_t)tok * p.gate_row + hv * kV + d]);
      const float gt = SIGMOID ? sigmoid_f(z) : z * sigmoid_f(z);
      const float w = p.norm_w_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(p.norm_w)[d])
                                    : static_cast<const float*>(p.norm_w)[d];
      p.out[((int64_t)tok * p.HV + hv) * kV + d] = __float2bfloat16(o[i] * rstd * w * gt);
    }
  }
}


// ---------------------------------------------------------------------------
// v2: whole head per CTA in registers, no chunk loop (latency-oriented).
// Thread layout: warp w owns rows [16w, 16w+16); lane = rg*8 + kl, kl = K
// segment (16 elements), rg = row group; thread rows = 16w + rg + 4i, i<4.
// Every state byte is loaded up front (max memory-level parallelism); each
// token costs two 3-level shuffle reductions over 4 independent rows.
// Same slot / log / int8-region layout as v1 (32-row regions).
// ---------------------------------------------------------------------------
constexpr int kSeg = 16;

__device__ __forceinline__ void load_seg_bf16(const unsigned char* p, float* h) {
  const uint4 a = *reinterpret_cast<const uint4*>(p);
  const uint4 b = *reinterpret_cast<const uint4*>(p + 16);
  const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&w[e]);
    h[2 * e] = __bfloat162float(v.x);
    h[2 * e + 1] = __bfloat162float(v.y);
  }
}

template <bool QINT8, bool SIGMOID, int kMaxT, int kMaxTok>
__global__ __launch_bounds__(kThreads, 2) void gdn_ss2_kernel(Args p) {
  const int n = blockIdx.x, hv = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int kl = lane & 7, rg = lane >> 3;
  const int bos = p.cu[n], eos = p.cu[n + 1];
  const int T = eos - bos;
  if (T <= 0) return;
  const int s0 = p.idx[n * p.idx_width], s1 = p.idx[n * p.idx_width + 1];
  if (s0 <= 0 || s1 <= 0 || T > p.max_log + 1 || T > kMaxT) {
    for (int i = tid; i < T * kV; i += kThreads)
      p.out[((int64_t)(bos + i / kV) * p.HV + hv) * kV + i % kV] = __float2bfloat16(0.f);
    return;
  }
  const int hk = hv / p.G;
  int n_rep = p.acc[n] - 1;
  n_rep = n_rep < 0 ? 0 : (n_rep > p.max_log ? p.max_log : n_rep);
  if (n_rep + T > kMaxTok) n_rep = kMaxTok - T;
  const int ng = p.ng;
  const int gs = kK / ng;

  unsigned char* head = p.state + (int64_t)s0 * p.slot_bytes + (int64_t)hv * kV * kK * 2;
  // ---- issue the state loads first (they overlap everything below) ----
  float h[4][kSeg];
  const int region = warp >> 1;              // 32-row region of this warp's rows
  unsigned char* reg_base = head + region * kNativeChunkBytes;
  bool q8 = false;
  if (QINT8) {
    const int tag = *reinterpret_cast<const int*>(reg_base + kCodeBytes + kChunk * ng * 4);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int rr = (warp & 1) * 16 + rg + 4 * i;  // row inside the region
      const uint4 c = *reinterpret_cast<const uint4*>(reg_base + rr * kK + kl * kSeg);
      const float sc = reinterpret_cast<const float*>(reg_base + kCodeBytes)[rr * ng + (kl * kSeg) / gs];
      const uint32_t w[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const char4 b4 = *reinterpret_cast<const char4*>(&w[e]);
        h[i][4 * e] = b4.x * sc; h[i][4 * e + 1] = b4.y * sc;
        h[i][4 * e + 2] = b4.z * sc; h[i][4 * e + 3] = b4.w * sc;
      }
    }
    q8 = (p.dbg & 2) ? true : (tag == kMagic);
  }
  if (!q8) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = warp * 16 + rg + 4 * i;
      load_seg_bf16(head + (row * kK + kl * kSeg) * 2, h[i]);
    }
  }

  float* log_slot = reinterpret_cast<float*>(p.state + (int64_t)s1 * p.slot_bytes);
  int* par = reinterpret_cast<int*>(log_slot) + hv * p.nvp;
  const int pb = n_rep > 0 ? (par[0] & 1) : 0;
  const float* rbuf = log_slot + p.log_base + pb * p.buf;
  float* wbuf = log_slot + p.log_base + (1 - pb) * p.buf;

  __shared__ alignas(16) float s_q[kMaxT][kK];
  __shared__ alignas(16) float s_k[kMaxTok][kK];
  __shared__ alignas(16) __nv_bfloat16 s_v[kMaxTok][kV];
  __shared__ __nv_bfloat16 s_out[kMaxT][kV];
  __shared__ float s_decay[kMaxTok], s_beta[kMaxTok];

  for (int i = tid; i < n_rep * kK; i += kThreads) {
    const int j = i / kK, d = i % kK;
    s_k[j][d] = rbuf[(int64_t)(j * p.H + hk) * kK + d];
    s_v[j][d] = __float2bfloat16(rbuf[p.v_off + (int64_t)(j * p.HV + hv) * kV + d]);
  }
  if (tid < n_rep) {
    s_decay[tid] = rbuf[p.d_off + tid * p.HV + hv];
    s_beta[tid] = rbuf[p.b_off + tid * p.HV + hv];
  }
  if (warp < T) {
    const int t = warp, j = n_rep + t, tok = bos + t;
    const __nv_bfloat16* row = p.qkv + (int64_t)tok * p.qkv_row;
    float qv[4], kv[4], qs = 0.f, ks = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + 32 * i;
      qv[i] = __bfloat162float(row[hk * kK + d]);
      kv[i] = __bfloat162float(row[p.H * kK + hk * kK + d]);
      qs += qv[i] * qv[i];
      ks += kv[i] * kv[i];
    }
    qs = warp_sum(qs);
    ks = warp_sum(ks);
    const float qn = rsqrtf(qs + 1e-6f) * p.scale, kn = rsqrtf(ks + 1e-6f);
    const bool log_k = t >= 1 && (hv % p.G) == 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + 32 * i;
      const __nv_bfloat16 vb = row[2 * p.H * kK + hv * kV + d];
      s_q[t][d] = qv[i] * qn;
      s_k[j][d] = kv[i] * kn;
      s_v[j][d] = vb;
      if (log_k) wbuf[(int64_t)((t - 1) * p.H + hk) * kK + d] = kv[i] * kn;
      if (t >= 1) wbuf[p.v_off + (int64_t)((t - 1) * p.HV + hv) * kV + d] = __bfloat162float(vb);
    }
    if (lane == 0) {
      const float av = __bfloat162float(p.a[(int64_t)tok * p.a_row + hv]);
      const float bv = __bfloat162float(p.b[(int64_t)tok * p.b_row + hv]);
      const float dtb = p.dt_bias_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(p.dt_bias)[hv])
                                       : static_cast<const float*>(p.dt_bias)[hv];
      const float dec = __expf(-__expf(p.a_log[hv]) * softplus_f(av + dtb));
      const float bet = sigmoid_f(bv);
      s_decay[j] = dec;
      s_beta[j] = bet;
      if (t >= 1) {
        wbuf[p.d_off + (t - 1) * p.HV + hv] = dec;
        wbuf[p.b_off + (t - 1) * p.HV + hv] = bet;
      }
    }
  }
  __syncthreads();
  if (tid < p.nvp) par[tid] = 1 - pb;  // all reads of par happened before the barrier

  const int TT = n_rep + T;
  for (int j = 0; j < TT; ++j) {
    const float dec = s_decay[j], bet = s_beta[j];
    float kk[kSeg];
#pragma unroll
    for (int e = 0; e < kSeg; e += 4) {
      const float4 v4 = *reinterpret_cast<const float4*>(&s_k[j][kl * kSeg + e]);
      kk[e] = v4.x; kk[e + 1] = v4.y; kk[e + 2] = v4.z; kk[e + 3] = v4.w;
    }
    float d4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float acc = 0.f;
#pragma unroll
      for (int e = 0; e < kSeg; ++e) { h[i][e] *= dec; acc += h[i][e] * kk[e]; }
      d4[i] = acc;
    }
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) {
#pragma unroll
      for (int i = 0; i < 4; ++i) d4[i] += __shfl_xor_sync(0xffffffffu, d4[i], o);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = warp * 16 + rg + 4 * i;
      const float delta = (__bfloat162float(s_v[j][row]) - d4[i]) * bet;
#pragma unroll
      for (int e = 0; e < kSeg; ++e) h[i][e] += kk[e] * delta;
    }
    if (j < n_rep) continue;
    const int t = j - n_rep;
#pragma unroll
    for (int e = 0; e < kSeg; e += 4) {
      const float4 v4 = *reinterpret_cast<const float4*>(&s_q[t][kl * kSeg + e]);
      kk[e] = v4.x; kk[e + 1] = v4.y; kk[e + 2] = v4.z; kk[e + 3] = v4.w;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float acc = 0.f;
#pragma unroll
      for (int e = 0; e < kSeg; ++e) acc += h[i][e] * kk[e];
      d4[i] = acc;
    }
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) {
#pragma unroll
      for (int i = 0; i < 4; ++i) d4[i] += __shfl_xor_sync(0xffffffffu, d4[i], o);
    }
    if (kl == 0) {
#pragma unroll
      for (int i = 0; i < 4; ++i) s_out[t][warp * 16 + rg + 4 * i] = __float2bfloat16(d4[i]);
    }
    if (t == 0 && !(p.dbg & 1)) {  // the single state write
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = warp * 16 + rg + 4 * i;
        if (QINT8) {
          const int rr = (warp & 1) * 16 + rg + 4 * i;
          float am = 0.f;
#pragma unroll
          for (int e = 0; e < kSeg; ++e) am = fmaxf(am, fabsf(h[i][e]));
          for (int o = 1; o < gs / kSeg; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
          const float sc = fmaxf(am, 1e-30f) * (1.0f / 127.0f);
          const float inv = 1.0f / sc;
          uint32_t w[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            char4 b4;
            b4.x = (signed char)max(-127, min(127, __float2int_rn(h[i][4 * e] * inv)));
            b4.y = (signed char)max(-127, min(127, __float2int_rn(h[i][4 * e + 1] * inv)));
            b4.z = (signed char)max(-127, min(127, __float2int_rn(h[i][4 * e + 2] * inv)));
            b4.w = (signed char)max(-127, min(127, __float2int_rn(h[i][4 * e + 3] * inv)));
            w[e] = *reinterpret_cast<uint32_t*>(&b4);
          }
          *reinterpret_cast<uint4*>(reg_base + rr * kK + kl * kSeg) = make_uint4(w[0], w[1], w[2], w[3]);
          if ((kl * kSeg) % gs == 0)
            reinterpret_cast<float*>(reg_base + kCodeBytes)[rr * ng + (kl * kSeg) / gs] = sc;
        } else {
          uint32_t w[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            __nv_bfloat162 v = __floats2bfloat162_rn(h[i][2 * e], h[i][2 * e + 1]);
            w[e] = *reinterpret_cast<uint32_t*>(&v);
          }
          unsigned char* dst = head + (row * kK + kl * kSeg) * 2;
          *reinterpret_cast<uint4*>(dst) = make_uint4(w[0], w[1], w[2], w[3]);
          *reinterpret_cast<uint4*>(dst + 16) = make_uint4(w[4], w[5], w[6], w[7]);
        }
      }
      if (QINT8 && (warp & 1) == 0 && lane == 0)
        *reinterpret_cast<int*>(reg_base + kCodeBytes + kChunk * ng * 4) = kMagic;
    }
  }
  __syncthreads();
  if (warp < T) {
    const int t = warp, tok = bos + t;
    float o[4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { o[i] = __bfloat162float(s_out[t][lane + 32 * i]); ss += o[i] * o[i]; }
    ss = warp_sum(ss);
    const float rstd = rsqrtf(ss / kV + p.eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + 32 * i;
      const float z = __bfloat162float(p.gate[(int64_t)tok * p.gate_row + hv * kV + d]);
      const float gt = SIGMOID ? sigmoid_f(z) : z * sigmoid_f(z);
      const float w = p.norm_w_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(p.norm_w)[d])
                                    : static_cast<const float*>(p.norm_w)[d];
      p.out[((int64_t)tok * p.HV + hv) * kV + d] = __float2bfloat16(o[i] * rstd * w * gt);
    }
  }
}


// ---------------------------------------------------------------------------
// v3: persistent CTAs with a one-item-ahead cp.async pipeline (c >= 8 regime).
// v1/v2 measured ~100 us at 32 requests even with the state store disabled
// (results/gdn-cuda-var.txt): the per-(request, head) CTA is a serial chain
// (index loads -> state loads -> compute -> norm), not a bandwidth stream.
// Here each CTA walks items b, b+G, b+2G, ... and, while computing item i
// from registers, has every global input of item i+1 (state bytes, raw
// q/k/v/gate rows, a/b words, both log buffers) in flight via cp.async into
// the (single) smem stage it has just drained. Item metadata (cu, acc, idx,
// parity) is loaded two items ahead by thread 0 so no load is on the
// critical path. Register layout and arithmetic as v2. k <= 3 only.
// ---------------------------------------------------------------------------
constexpr int kV3MaxT = 4;
constexpr int kV3MaxLog = 3;
constexpr int kV3MaxTok = kV3MaxT + kV3MaxLog;
constexpr int kV3MaxHV = 64;

struct V3Meta {
  int n, hv, bos, T, n_rep, s0, s1, pb, valid;
};

struct V3Smem {
  alignas(16) unsigned char st[kV * kK * 2];          // state bytes of the next/current item
  alignas(16) __nv_bfloat16 rq[kV3MaxT][kK];
  alignas(16) __nv_bfloat16 rk[kV3MaxT][kK];
  alignas(16) __nv_bfloat16 rv[kV3MaxT][kV];
  alignas(16) __nv_bfloat16 rg[kV3MaxT][kV];          // output gate rows
  alignas(16) uint32_t rab[kV3MaxT][2];               // words holding a[tok, hv], b[tok, hv]
  alignas(16) float lk[2][kV3MaxLog][kK];             // both log buffers (parity resolved later)
  alignas(16) float lv[2][kV3MaxLog][kV];
  alignas(16) float ldb[2][kV3MaxLog][4];             // decay, beta
  alignas(16) float q_f[kV3MaxT][kK];
  alignas(16) float k_f[kV3MaxTok][kK];
  alignas(16) __nv_bfloat16 v_b[kV3MaxTok][kV];
  alignas(16) __nv_bfloat16 out[kV3MaxT][kV];
  alignas(16) __nv_bfloat16 g_w[kV3MaxT][kV];         // gate rows of the current item
  float dec[kV3MaxTok], bet[kV3MaxTok];
  float a_log[kV3MaxHV], dt_bias[kV3MaxHV];
  float norm_w[kV];
  V3Meta meta[3];  // ring: item i, i+1, i+2
};

__device__ __forceinline__ void cp_async_4(void* smem, const void* gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(gmem));
}

__device__ __forceinline__ void v3_basic_meta(const Args& p, int item, int num_items, V3Meta& m) {
  m.valid = item < num_items;
  if (!m.valid) return;
  m.n = item / p.HV;
  m.hv = item % p.HV;
  m.bos = p.cu[m.n];
  m.T = p.cu[m.n + 1] - m.bos;
  int r = p.acc[m.n] - 1;
  m.n_rep = r < 0 ? 0 : (r > p.max_log ? p.max_log : r);
  m.s0 = p.idx[m.n * p.idx_width];
  m.s1 = p.idx[m.n * p.idx_width + 1];
  m.pb = 0;
}

__device__ __forceinline__ bool v3_runnable(const Args& p, const V3Meta& m) {
  return m.valid && m.T > 0 && m.s0 > 0 && m.s1 > 0 && m.T <= p.max_log + 1 && m.T <= kV3MaxT &&
         m.n_rep <= kV3MaxLog && m.n_rep + m.T <= kV3MaxTok;
}

template <bool QINT8>
__device__ __forceinline__ void v3_prefetch(const Args& p, V3Smem& S, const V3Meta& m, int tid) {
  // state bytes of the head
  const unsigned char* head = p.state + (int64_t)m.s0 * p.slot_bytes + (int64_t)m.hv * kV * kK * 2;
  if (QINT8) {
    const int sub = (kCodeBytes + kChunk * p.ng * 4 + 4 + 15) / 16;  // 16B pieces per region
    for (int i = tid; i < kNumChunks * sub; i += kThreads) {
      const int c = i / sub, o = (i % sub) * 16;
      cp_async_16(&S.st[c * kNativeChunkBytes + o], head + c * kNativeChunkBytes + o);
    }
  } else {
    for (int i = tid; i < kV * kK * 2 / 16; i += kThreads) cp_async_16(&S.st[i * 16], head + i * 16);
  }
  // raw token rows: q, k (key head), v, gate (value head): 16 pieces of 16B each
  const int hk = m.hv / p.G;
  for (int i = tid; i < m.T * 64; i += kThreads) {
    const int t = i / 64, w = (i % 64) / 16, o = (i % 16) * 8;  // o in elements
    const int64_t tok = m.bos + t;
    const __nv_bfloat16* row = p.qkv + tok * p.qkv_row;
    if (w == 0) cp_async_16(&S.rq[t][o], row + hk * kK + o);
    else if (w == 1) cp_async_16(&S.rk[t][o], row + p.H * kK + hk * kK + o);
    else if (w == 2) cp_async_16(&S.rv[t][o], row + 2 * p.H * kK + m.hv * kV + o);
    else cp_async_16(&S.rg[t][o], p.gate + tok * p.gate_row + m.hv * kV + o);
  }
  if (tid < m.T * 2) {
    const int t = tid >> 1, which = tid & 1;
    const int64_t tok = m.bos + t;
    const __nv_bfloat16* src = which ? p.b + tok * p.b_row + m.hv : p.a + tok * p.a_row + m.hv;
    cp_async_4(&S.rab[t][which], reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(src) & ~uintptr_t(3)));
  }
  // both log buffers for the replayed tokens
  if (m.n_rep > 0) {
    const float* lslot = reinterpret_cast<const float*>(p.state + (int64_t)m.s1 * p.slot_bytes);
    const int per_tok = 32 + 32 + 2;  // 16B k pieces, 16B v pieces, d, b (4B)
    for (int i = tid; i < 2 * m.n_rep * per_tok; i += kThreads) {
      const int pp = i / (m.n_rep * per_tok), rem = i % (m.n_rep * per_tok);
      const int j = rem / per_tok, q = rem % per_tok;
      const float* buf = lslot + p.log_base + pp * p.buf;
      if (q < 32) cp_async_16(&S.lk[pp][j][q * 4], buf + (int64_t)(j * p.H + hk) * kK + q * 4);
      else if (q < 64) cp_async_16(&S.lv[pp][j][(q - 32) * 4], buf + p.v_off + (int64_t)(j * p.HV + m.hv) * kV + (q - 32) * 4);
      else if (q == 64) cp_async_4(&S.ldb[pp][j][0], buf + p.d_off + j * p.HV + m.hv);
      else cp_async_4(&S.ldb[pp][j][1], buf + p.b_off + j * p.HV + m.hv);
    }
  }
  cp_async_commit();
}

template <bool QINT8, bool SIGMOID>
__global__ __launch_bounds__(kThreads, 2) void gdn_ss3_kernel(Args p, int num_items) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  V3Smem& S = *reinterpret_cast<V3Smem*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int kl = lane & 7, rg = lane >> 3;
  const int G = gridDim.x;
  int item = blockIdx.x;
  if (item >= num_items) return;
  const int ng = p.ng, gs = kK / ng;

  for (int i = tid; i < p.HV; i += kThreads) {
    S.a_log[i] = p.a_log[i];
    S.dt_bias[i] = p.dt_bias_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(p.dt_bias)[i])
                                  : static_cast<const float*>(p.dt_bias)[i];
  }
  for (int i = tid; i < kV; i += kThreads)
    S.norm_w[i] = p.norm_w_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(p.norm_w)[i])
                                : static_cast<const float*>(p.norm_w)[i];
  // prologue: meta(i), meta(i+1) basic, parity of i
  if (tid == 0) {
    v3_basic_meta(p, item, num_items, S.meta[0]);
    v3_basic_meta(p, item + G, num_items, S.meta[1]);
    if (v3_runnable(p, S.meta[0]) && S.meta[0].n_rep > 0)
      S.meta[0].pb = reinterpret_cast<const int*>(p.state + (int64_t)S.meta[0].s1 * p.slot_bytes)[S.meta[0].hv * p.nvp] & 1;
  }
  __syncthreads();
  if (v3_runnable(p, S.meta[0])) v3_prefetch<QINT8>(p, S, S.meta[0], tid);
  int cur = 0;

  for (; item < num_items; item += G) {
    const V3Meta m = S.meta[cur];
    const int nxt = (cur + 1) % 3, nn = (cur + 2) % 3;
    // thread 0: start loading basic meta of item + 2G and the parity of item + G
    V3Meta m2;
    int par_next = 0;
    if (tid == 0) {
      v3_basic_meta(p, item + 2 * G, num_items, m2);
      const V3Meta& m1 = S.meta[nxt];
      if (v3_runnable(p, m1) && m1.n_rep > 0)
        par_next = reinterpret_cast<const int*>(p.state + (int64_t)m1.s1 * p.slot_bytes)[m1.hv * p.nvp];
    }
    const bool run = v3_runnable(p, m);
    if (!run) {
      if (m.valid && m.T > 0)
        for (int i = tid; i < m.T * kV; i += kThreads)
          p.out[((int64_t)(m.bos + i / kV) * p.HV + m.hv) * kV + i % kV] = __float2bfloat16(0.f);
    }
    cp_async_wait_all();
    __syncthreads();  // (A) stage holds item m
    float h[4][kSeg];
    unsigned char* head = p.state + (int64_t)m.s0 * p.slot_bytes + (int64_t)m.hv * kV * kK * 2;
    const int region = warp >> 1;
    if (run) {
      // (b) state -> registers
      bool q8 = false;
      if (QINT8) {
        const unsigned char* rb = &S.st[region * kNativeChunkBytes];
        q8 = (p.dbg & 2) ? true : (*reinterpret_cast<const int*>(rb + kCodeBytes + kChunk * ng * 4) == kMagic);
        if (q8) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int rr = (warp & 1) * 16 + rg + 4 * i;
            const uint4 c = *reinterpret_cast<const uint4*>(rb + rr * kK + kl * kSeg);
            const float sc = reinterpret_cast<const float*>(rb + kCodeBytes)[rr * ng + (kl * kSeg) / gs];
            const uint32_t w[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const char4 b4 = *reinterpret_cast<const char4*>(&w[e]);
              h[i][4 * e] = b4.x * sc; h[i][4 * e + 1] = b4.y * sc;
              h[i][4 * e + 2] = b4.z * sc; h[i][4 * e + 3] = b4.w * sc;
            }
          }
        } else {  // native region but only the int8 subset was prefetched: read global
#pragma unroll
          for (int i = 0; i < 4; ++i)
            load_seg_bf16(head + ((warp * 16 + rg + 4 * i) * kK + kl * kSeg) * 2, h[i]);
        }
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
          load_seg_bf16(&S.st[((warp * 16 + rg + 4 * i) * kK + kl * kSeg) * 2], h[i]);
      }
      // (c) token setup
      const int hk = m.hv / p.G;
      const int pb = m.n_rep > 0 ? m.pb : 0;
      float* lslot = reinterpret_cast<float*>(p.state + (int64_t)m.s1 * p.slot_bytes);
      float* wbuf = lslot + p.log_base + (1 - pb) * p.buf;
      for (int i = tid; i < m.n_rep * kK; i += kThreads) {
        const int j = i / kK, d = i % kK;
        S.k_f[j][d] = S.lk[pb][j][d];
        S.v_b[j][d] = __float2bfloat16(S.lv[pb][j][d]);
      }
      if (tid < m.n_rep) { S.dec[tid] = S.ldb[pb][tid][0]; S.bet[tid] = S.ldb[pb][tid][1]; }
      if (warp < m.T) {
        const int t = warp, j = m.n_rep + t;
        float qv[4], kv[4], qs = 0.f, ks = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int d = lane + 32 * i;
          qv[i] = __bfloat162float(S.rq[t][d]);
          kv[i] = __bfloat162float(S.rk[t][d]);
          qs += qv[i] * qv[i];
          ks += kv[i] * kv[i];
        }
        qs = warp_sum(qs);
        ks = warp_sum(ks);
        const float qn = rsqrtf(qs + 1e-6f) * p.scale, kn = rsqrtf(ks + 1e-6f);
        const bool log_k = t >= 1 && (m.hv % p.G) == 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int d = lane + 32 * i;
          const __nv_bfloat16 vb = S.rv[t][d];
          S.q_f[t][d] = qv[i] * qn;
          S.k_f[j][d] = kv[i] * kn;
          S.v_b[j][d] = vb;
          S.g_w[t][d] = S.rg[t][d];
          if (log_k) wbuf[(int64_t)((t - 1) * p.H + hk) * kK + d] = kv[i] * kn;
          if (t >= 1) wbuf[p.v_off + (int64_t)((t - 1) * p.HV + m.hv) * kV + d] = __bfloat162float(vb);
        }
        if (lane == 0) {
          const int64_t tok = m.bos + t;
          const int sa = ((reinterpret_cast<uintptr_t>(p.a + tok * p.a_row + m.hv) & 3) != 0);
          const int sb = ((reinterpret_cast<uintptr_t>(p.b + tok * p.b_row + m.hv) & 3) != 0);
          const uint32_t wa = S.rab[t][0], wb = S.rab[t][1];
          const unsigned short ua = sa ? (wa >> 16) : (wa & 0xffff), ub = sb ? (wb >> 16) : (wb & 0xffff);
          const float av = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&ua));
          const float bv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&ub));
          const float dec = __expf(-__expf(S.a_log[m.hv]) * softplus_f(av + S.dt_bias[m.hv]));
          const float bet = sigmoid_f(bv);
          S.dec[j] = dec;
          S.bet[j] = bet;
          if (t >= 1) {
            wbuf[p.d_off + (t - 1) * p.HV + m.hv] = dec;
            wbuf[p.b_off + (t - 1) * p.HV + m.hv] = bet;
          }
        }
      }
      if (tid < p.nvp)
        reinterpret_cast<int*>(lslot)[m.hv * p.nvp + tid] = 1 - pb;
    }
    __syncthreads();  // (B) stage fully consumed (state in registers, tokens in work arrays)
    // (d) prefetch item + G into the stage
    {
      const V3Meta& m1 = S.meta[nxt];
      if (v3_runnable(p, m1)) v3_prefetch<QINT8>(p, S, m1, tid);
    }
    if (run) {
      // (e) recurrence from registers
      const int TT = m.n_rep + m.T;
      for (int j = 0; j < TT; ++j) {
        const float dec = S.dec[j], bet = S.bet[j];
        float kk[kSeg];
#pragma unroll
        for (int e = 0; e < kSeg; e += 4) {
          const float4 v4 = *reinterpret_cast<const float4*>(&S.k_f[j][kl * kSeg + e]);
          kk[e] = v4.x; kk[e + 1] = v4.y; kk[e + 2] = v4.z; kk[e + 3] = v4.w;
        }
        float d4[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float acc = 0.f;
#pragma unroll
          for (int e = 0; e < kSeg; ++e) { h[i][e] *= dec; acc += h[i][e] * kk[e]; }
          d4[i] = acc;
        }
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) {
#pragma unroll
          for (int i = 0; i < 4; ++i) d4[i] += __shfl_xor_sync(0xffffffffu, d4[i], o);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float delta = (__bfloat162float(S.v_b[j][warp * 16 + rg + 4 * i]) - d4[i]) * bet;
#pragma unroll
          for (int e = 0; e < kSeg; ++e) h[i][e] += kk[e] * delta;
        }
        if (j < m.n_rep) continue;
        const int t = j - m.n_rep;
#pragma unroll
        for (int e = 0; e < kSeg; e += 4) {
          const float4 v4 = *reinterpret_cast<const float4*>(&S.q_f[t][kl * kSeg + e]);
          kk[e] = v4.x; kk[e + 1] = v4.y; kk[e + 2] = v4.z; kk[e + 3] = v4.w;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float acc = 0.f;
#pragma unroll
          for (int e = 0; e < kSeg; ++e) acc += h[i][e] * kk[e];
          d4[i] = acc;
        }
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) {
#pragma unroll
          for (int i = 0; i < 4; ++i) d4[i] += __shfl_xor_sync(0xffffffffu, d4[i], o);
        }
        if (kl == 0) {
#pragma unroll
          for (int i = 0; i < 4; ++i) S.out[t][warp * 16 + rg + 4 * i] = __float2bfloat16(d4[i]);
        }
        if (t == 0 && !(p.dbg & 1)) {
          unsigned char* reg_base = head + region * kNativeChunkBytes;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            if (QINT8) {
              const int rr = (warp & 1) * 16 + rg + 4 * i;
              float am = 0.f;
#pragma unroll
              for (int e = 0; e < kSeg; ++e) am = fmaxf(am, fabsf(h[i][e]));
              for (int o = 1; o < gs / kSeg; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
              const float sc = fmaxf(am, 1e-30f) * (1.0f / 127.0f);
              const float inv = 1.0f / sc;
              uint32_t w[4];
#pragma unroll
              for (int e = 0; e < 4; ++e) {
                char4 b4;
                b4.x = (signed char)max(-127, min(127, __float2int_rn(h[i][4 * e] * inv)));
                b4.y = (signed char)max(-127, min(127, __float2int_rn(h[i][4 * e + 1] * inv)));
                b4.z = (signed char)max(-127, min(127, __float2int_rn(h[i][4 * e + 2] * inv)));
                b4.w = (signed char)max(-127, min(127, __float2int_rn(h[i][4 * e + 3] * inv)));
                w[e] = *reinterpret_cast<uint32_t*>(&b4);
              }
              *reinterpret_cast<uint4*>(reg_base + rr * kK + kl * kSeg) = make_uint4(w[0], w[1], w[2], w[3]);
              if ((kl * kSeg) % gs == 0)
                reinterpret_cast<float*>(reg_base + kCodeBytes)[rr * ng + (kl * kSeg) / gs] = sc;
            } else {
              uint32_t w[8];
#pragma unroll
              for (int e = 0; e < 8; ++e) {
                __nv_bfloat162 v = __floats2bfloat162_rn(h[i][2 * e], h[i][2 * e + 1]);
                w[e] = *reinterpret_cast<uint32_t*>(&v);
              }
              unsigned char* dst = head + ((warp * 16 + rg + 4 * i) * kK + kl * kSeg) * 2;
              *reinterpret_cast<uint4*>(dst) = make_uint4(w[0], w[1], w[2], w[3]);
              *reinterpret_cast<uint4*>(dst + 16) = make_uint4(w[4], w[5], w[6], w[7]);
            }
          }
          if (QINT8 && (warp & 1) == 0 && lane == 0)
            *reinterpret_cast<int*>(reg_base + kCodeBytes + kChunk * ng * 4) = kMagic;
        }
      }
    }
    // thread 0: publish meta(item + 2G) and parity(item + G) (loads issued at the top)
    if (tid == 0) {
      S.meta[nn] = m2;
      S.meta[nxt].pb = par_next & 1;
    }
    __syncthreads();  // (C) out rows complete; meta visible
    if (run && warp < m.T) {
      const int t = warp;
      const int64_t tok = m.bos + t;
      float o[4], ss = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) { o[i] = __bfloat162float(S.out[t][lane + 32 * i]); ss += o[i] * o[i]; }
      ss = warp_sum(ss);
      const float rstd = rsqrtf(ss / kV + p.eps);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int d = lane + 32 * i;
        const float z = __bfloat162float(S.g_w[t][d]);
        const float gt = SIGMOID ? sigmoid_f(z) : z * sigmoid_f(z);
        p.out[(tok * p.HV + m.hv) * kV + d] = __float2bfloat16(o[i] * rstd * S.norm_w[d] * gt);
      }
    }
    cur = nxt;
  }
}


// ---------------------------------------------------------------------------
// v4 / gdn_ss4 (P3): FlashInfer wide_vec structure under our single-state protocol.
// One 128-thread CTA per (request, value head) (non-persistent grid); token setup (q/k l2-norm,
// decay, beta, replay log, log write, parity) staged in smem once; the 128 state rows are processed
// in 4 passes of 32 rows: 16 lanes cooperate on one row (8 K-elements = one 16 B load per lane),
// 8 row groups x 4 rows (ILP) per pass, all tokens per pass in registers. Same slot / log / parity
// semantics as v3 (state after token 0 -> slot 0, log tokens 1..T-1, flip parity), fused gated norm.
// NOT bit-exact to v3 (16-lane x 8 K-reduction instead of 8-lane x 16).
// ---------------------------------------------------------------------------
constexpr int kS4Threads = 128;
#ifndef S10_LPT
#define S10_LPT 2
#endif
#ifndef S10_MAXT
#define S10_MAXT 4
#endif
constexpr int kS4MaxT = S10_MAXT, kS4MaxLog = S10_MAXT - 1, kS4MaxTok = 2 * S10_MAXT - 1;

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

// gdn_ss4b (C32-gdn): gdn_ss4 arithmetic unchanged (bit-identical), memory pipeline reworked:
//  * every thread reads the item metadata itself (broadcast loads; no thread-0 -> barrier chain);
//  * the state rows of the first S4B_NBUF passes are cp.async'd to smem right after idx is known,
//    BEFORE the parity / log / token-row loads and the token-setup barrier; each thread copies
//    exactly the 16 B pieces it later reads, so only cp.async.wait_group (no barrier) guards them;
//  * after a pass is converted to registers its buffer is refilled with pass + S4B_NBUF.
// dbg bits (microbench only): 128 = skip the recurrence (memory skeleton), 256 = no state loads.
#ifndef S4B_NBUF
#define S4B_NBUF 1
#endif
#ifndef S4B_MINB
#define S4B_MINB 6
#endif
// S4B_LPR = lanes per state row (16 = gdn_ss4). 8 / 4: each thread keeps the 2 / 4 consecutive
// 8-element K segments of gdn_ss4's lanes 2m..2m+1 / 4m..4m+3 as separate sequential partial sums and
// adds them in gdn_ss4's butterfly order before the remaining 3 / 2 xor levels -> bit-identical.
#ifndef S4B_LPR
#define S4B_LPR 16
#endif
#ifndef S4B_HOIST
#define S4B_HOIST 1
#endif
// S4B_GSCALE (NOT bit-identical, rounding-level): keep h = G * H with the running decay product G
// (CTA-uniform scalar) so the per-element decay multiply disappears (4 -> 3 FP32 ops per element and
// current token, 3 -> 2 per replayed token)
#ifndef S4B_GSCALE
#define S4B_GSCALE 1
#endif
constexpr int kLPR = S4B_LPR;
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
#ifndef S4B_ST
#define S4B_ST 0
#endif
// state store flavour (bit-identical either way): 0 plain, 1 st.global.cs (evict first), 2 st.global.cg
__device__ __forceinline__ void st_state(void* ptr, uint4 v) {
  if (S4B_ST == 1)
    asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
  else if (S4B_ST == 2)
    asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
  else
    *reinterpret_cast<uint4*>(ptr) = v;
}
__device__ __forceinline__ void cp_async_wait_n(int n) {
  if (n <= 0) asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  else if (n == 1) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
  else if (n == 2) asm volatile("cp.async.wait_group 2;\n" ::: "memory");
  else asm volatile("cp.async.wait_group 3;\n" ::: "memory");
}

#ifdef S4B_PROF
__device__ unsigned long long g_prof[2048 * 16];
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t;
}
#define PSTAMP(k) do { if (tid == 0 && item < 2048) g_prof[item * 16 + (k)] = gtime(); } while (0)
__device__ __forceinline__ unsigned long long gtime_dep(float x) {
  unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t) : "f"(x)); return t;
}
#define PSTAMPD(k, x) do { if (tid == 0 && item < 2048) g_prof[item * 16 + (k)] = gtime_dep(x); } while (0)
#else
#define PSTAMP(k) do {} while (0)
#define PSTAMPD(k, x) do {} while (0)
#endif
template <bool SIGMOID>
__global__ __launch_bounds__(kS4Threads, S4B_MINB) void gdn_ss4_kernel(Args p, int num_items) {
  __shared__ __align__(16) float k_f[kS4MaxTok][kK];
  __shared__ __align__(16) float q_f[kS4MaxT][kK];
  __shared__ __align__(16) float v_f[kS4MaxTok][kV];
  __shared__ __align__(16) __nv_bfloat16 g_w[kS4MaxT][kV];
  __shared__ __align__(16) __nv_bfloat16 o_s[kS4MaxT][kV];
  __shared__ __align__(16) unsigned char s_st[S4B_NBUF][32 * kK * 2];
  __shared__ float dec_s[kS4MaxTok], bet_s[kS4MaxTok];
  __shared__ float nw_s[kV];
#if S10_LPT
  // longest-first: CTA blockIdx -> (request of work-rank blockIdx / HV, head blockIdx % HV), work =
  // T + n_rep (stable by request index). Bit-identical (each item's computation is unchanged).
  if ((int)blockIdx.x >= num_items) return;
  int item = blockIdx.x;
#if S10_LPT == 2
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
#else
  if (num_items / p.HV <= kS4Threads) {
    __shared__ int s_w[kS4Threads];
    __shared__ int s_n;
    const int nreq = num_items / p.HV, r = blockIdx.x / p.HV, t0 = threadIdx.x;
    if (t0 < nreq) {
      int rr = p.acc[t0] - 1;
      rr = rr < 0 ? 0 : (rr > p.max_log ? p.max_log : rr);
      s_w[t0] = p.cu[t0 + 1] - p.cu[t0] + rr;
    }
    __syncthreads();
    if (t0 < nreq) {
      const int w = s_w[t0];
      int rank = 0;
      for (int m = 0; m < nreq; ++m) {
        const int wm = s_w[m];
        rank += (wm > w) || (wm == w && m < t0);
      }
      if (rank == r) s_n = t0;
    }
    __syncthreads();
    item = s_n * p.HV + blockIdx.x % p.HV;
  }
#endif
#else
  const int item = blockIdx.x;
  if (item >= num_items) return;
#endif
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  PSTAMP(0);
#ifdef S4B_PROF
  if (tid == 0 && item < 2048) { unsigned sm; asm("mov.u32 %0, %smid;" : "=r"(sm)); g_prof[item * 16 + 9] = sm; }
#endif
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
#if S12_RS
  static_assert(kLPR == 16 && kRPT == 4, "S12_RS: gdn_ss4 thread mapping only");
  const int rsc = 2 * (l16 & 1) + ((l16 >> 1) & 1);  // slot i <-> local row grp*4 + (i ^ rsc)
#define S12_LR(i) (grp * kRPT + ((i) ^ rsc))
#else
#define S12_LR(i) (grp * kRPT + (i))
#endif
  unsigned char* head = p.state + (int64_t)s0 * p.slot_bytes + (int64_t)hv * kV * kK * 2;
  const bool ld = !(p.dbg & 256);
  auto issue = [&](int pass) {
    if (ld) {
      unsigned char* sb = s_st[pass % S4B_NBUF];
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
  for (int ps = 0; ps < S4B_NBUF; ++ps) issue(ps);

  float* lslot = reinterpret_cast<float*>(p.state + (int64_t)s1 * p.slot_bytes);
#if S4B_HOIST
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
  PSTAMPD(10, (float)(bos + s0 + s1 + n_rep));
  const int pb = n_rep > 0 ? (reinterpret_cast<const int*>(lslot)[hv * p.nvp] & 1) : 0;
  PSTAMPD(11, (float)pb);
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
  PSTAMPD(12, av + bv + __bfloat162float(rq[0]) + __bfloat162float(rv[3]) + __bfloat162float(rg[3]) + nwv);
  PSTAMPD(13, n_rep > 0 ? lk[0] + lv[0] + ld0 : 0.f);
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
#else
  const int pb = n_rep > 0 ? (reinterpret_cast<const int*>(lslot)[hv * p.nvp] & 1) : 0;
  const float* rbuf = lslot + p.log_base + pb * p.buf;
  float* wbuf = lslot + p.log_base + (1 - pb) * p.buf;
  for (int i = tid; i < n_rep * kK; i += kS4Threads) {
    const int j = i / kK, d = i % kK;
    k_f[j][kpos(d)] = rbuf[(int64_t)(j * p.H + hk) * kK + d];
    v_f[j][d] = __bfloat162float(__float2bfloat16(rbuf[p.v_off + (int64_t)(j * p.HV + hv) * kV + d]));
  }
  if (tid < n_rep) { dec_s[tid] = rbuf[p.d_off + tid * p.HV + hv]; bet_s[tid] = rbuf[p.b_off + tid * p.HV + hv]; }
  if (warp < T) {
    const int t = warp, j = n_rep + t;
    const int64_t tok = bos + t;
    const __nv_bfloat16* row = p.qkv + tok * p.qkv_row;
    float qv[4], kv[4], qs = 0.f, ks = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + 32 * i;
      qv[i] = __bfloat162float(row[hk * kK + d]);
      kv[i] = __bfloat162float(row[p.H * kK + hk * kK + d]);
      qs += qv[i] * qv[i];
      ks += kv[i] * kv[i];
    }
    qs = warp_sum(qs);
    ks = warp_sum(ks);
    const float qn = rsqrtf(qs + 1e-6f) * p.scale, kn = rsqrtf(ks + 1e-6f);
    const bool log_k = t >= 1 && (hv % p.G) == 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + 32 * i;
      const __nv_bfloat16 vb = row[2 * p.H * kK + hv * kV + d];
      q_f[t][kpos(d)] = qv[i] * qn;
      k_f[j][kpos(d)] = kv[i] * kn;
      v_f[j][d] = __bfloat162float(vb);
      g_w[t][d] = p.gate[tok * p.gate_row + hv * kV + d];
      if (log_k) wbuf[(int64_t)((t - 1) * p.H + hk) * kK + d] = kv[i] * kn;
      if (t >= 1) wbuf[p.v_off + (int64_t)((t - 1) * p.HV + hv) * kV + d] = __bfloat162float(vb);
    }
    if (lane == 0) {
      const float av = __bfloat162float(p.a[tok * p.a_row + hv]);
      const float bv = __bfloat162float(p.b[tok * p.b_row + hv]);
      const float dtb = p.dt_bias_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(p.dt_bias)[hv])
                                       : static_cast<const float*>(p.dt_bias)[hv];
      const float dec = __expf(-__expf(p.a_log[hv]) * softplus_f(av + dtb));
      const float bet = sigmoid_f(bv);
      dec_s[j] = dec;
      bet_s[j] = bet;
      if (t >= 1) {
        wbuf[p.d_off + (t - 1) * p.HV + hv] = dec;
        wbuf[p.b_off + (t - 1) * p.HV + hv] = bet;
      }
    }
  }
#endif
  __syncthreads();  // token arrays complete; every thread has read the parity word
  PSTAMP(1);
  if (tid < p.nvp) reinterpret_cast<int*>(lslot)[hv * p.nvp + tid] = 1 - pb;
  const int TT = (p.dbg & 128) ? 0 : n_rep + T;
#pragma unroll 1
  for (int pass = 0; pass < 4; ++pass) {
    float h[kRPT][kEPT];
    int rows[kRPT];
    {
      const int after = 3 - pass < S4B_NBUF - 1 ? 3 - pass : S4B_NBUF - 1;
      cp_async_wait_n(after);
    }
    const unsigned char* sb = s_st[pass % S4B_NBUF];
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
    if (pass + S4B_NBUF < 4) issue(pass + S4B_NBUF);
#if S4B_GSCALE
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
#if S12_RS
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
#else
#pragma unroll
      for (int o = 1; o < kLPR; o <<= 1) {
#pragma unroll
        for (int i = 0; i < kRPT; ++i) d4[i] += __shfl_xor_sync(0xffffffffu, d4[i], o);
      }
#pragma unroll
      for (int i = 0; i < kRPT; ++i) {
        const float delta = (v_f[j][rows[i]] - gG * d4[i]) * bet * ginv;
#pragma unroll
        for (int e = 0; e < kEPT; ++e) h[i][e] += kk[e] * delta;
      }
#endif
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
#if S12_RS
      {
        const float e0 = d4[0] + __shfl_xor_sync(0xffffffffu, d4[2], 1);
        const float e1 = d4[1] + __shfl_xor_sync(0xffffffffu, d4[3], 1);
        float f = e0 + __shfl_xor_sync(0xffffffffu, e1, 2);
        f += __shfl_xor_sync(0xffffffffu, f, 4);
        f += __shfl_xor_sync(0xffffffffu, f, 8);
        if (l16 < 4) o_s[t][rows[0]] = __float2bfloat16(gG * f);  // lanes 0..3 cover rows c = 0..3
      }
#else
#pragma unroll
      for (int o = 1; o < kLPR; o <<= 1) {
#pragma unroll
        for (int i = 0; i < kRPT; ++i) d4[i] += __shfl_xor_sync(0xffffffffu, d4[i], o);
      }
      if (l16 == 0) {
#pragma unroll
        for (int i = 0; i < kRPT; ++i) o_s[t][rows[i]] = __float2bfloat16(gG * d4[i]);
      }
#endif
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
#else
#pragma unroll 1
    for (int j = 0; j < TT; ++j) {
      const float dec = dec_s[j], bet = bet_s[j];
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
          for (int e = sg * 8; e < sg * 8 + 8; ++e) { h[i][e] *= dec; acc += h[i][e] * kk[e]; }
          ps[sg] = acc;
        }
        d4[i] = seg_tree(ps);
      }
#pragma unroll
      for (int o = 1; o < kLPR; o <<= 1) {
#pragma unroll
        for (int i = 0; i < kRPT; ++i) d4[i] += __shfl_xor_sync(0xffffffffu, d4[i], o);
      }
#pragma unroll
      for (int i = 0; i < kRPT; ++i) {
        const float delta = (v_f[j][rows[i]] - d4[i]) * bet;
#pragma unroll
        for (int e = 0; e < kEPT; ++e) h[i][e] += kk[e] * delta;
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
#pragma unroll
      for (int o = 1; o < kLPR; o <<= 1) {
#pragma unroll
        for (int i = 0; i < kRPT; ++i) d4[i] += __shfl_xor_sync(0xffffffffu, d4[i], o);
      }
      if (l16 == 0) {
#pragma unroll
        for (int i = 0; i < kRPT; ++i) o_s[t][rows[i]] = __float2bfloat16(d4[i]);
      }
      if (t == 0 && !(p.dbg & 1)) {
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
            st_state(head + ((int64_t)rows[i] * kK + l16 * kEPT + sg * 8) * 2, make_uint4(w[0], w[1], w[2], w[3]));
          }
        }
      }
    }
#endif
    PSTAMP(2 + pass);
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
  PSTAMP(6);
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
#if S4B_HOIST
      const float nw = nw_s[d];
#else
      const float nw = p.norm_w_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(p.norm_w)[d])
                                     : static_cast<const float*>(p.norm_w)[d];
#endif
      const float z = __bfloat162float(g_w[t][d]);
      const float gt = SIGMOID ? sigmoid_f(z) : z * sigmoid_f(z);
      const __nv_bfloat16 ob = __float2bfloat16(o[i] * rstd * nw * gt);
      p.out[(tok * p.HV + hv) * kV + d] = ob;
      o[i] = __bfloat162float(ob);
    }
    if (p.a4_pk != nullptr) a4_store_head(o, lane, hv, (int)tok, p);
  }
  PSTAMP(7);
}
}  // namespace

static bool gdn_ss_impl(torch::Tensor qkv, torch::Tensor a, torch::Tensor b, torch::Tensor a_log,
            torch::Tensor dt_bias, torch::Tensor state, torch::Tensor idx, torch::Tensor cu,
            torch::Tensor acc, torch::Tensor gate, torch::Tensor norm_w, torch::Tensor out,
            int64_t H, int64_t num_requests, double scale, double eps, bool sigmoid_gate,
            bool qint8, int64_t ng, int64_t log_base, int64_t buf, int64_t v_off, int64_t d_off,
            int64_t b_off, int64_t nvp, int64_t dbg, uint32_t* a4_pk, float* a4_s,
            const float* a4_gs, int a4_tmb) {
  TORCH_CHECK(state.scalar_type() == at::kBFloat16, "gdn_ss: bf16 state only");
  TORCH_CHECK(state.size(2) == kV && state.size(3) == kK, "gdn_ss: K = V = 128 only");
  TORCH_CHECK(ng >= 1 && ng <= kMaxNG && (kK / ng) % 4 == 0, "gdn_ss: bad ng");
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
  p.max_log = idx.size(1) - 1; p.ng = ng; p.nvp = nvp;
  p.log_base = log_base; p.buf = buf; p.v_off = v_off; p.d_off = d_off; p.b_off = b_off;
  p.scale = scale; p.eps = eps;
  const dim3 grid(num_requests, p.HV);
  auto stream = at::cuda::getCurrentCUDAStream();
  p.dbg = dbg;
  const bool small = p.max_log + 1 <= 4;
  const bool v2 = !(dbg & 8);  // dbg 8 = the chunked v1 kernel (microbench only)
  const bool v3 = v2 && !(dbg & 4) && small && p.HV <= kV3MaxHV && p.max_log <= kV3MaxLog;
  if (v3 && !qint8 && !(dbg & 64)) {  // P3 v4 / gdn_ss4 (dbg 64 = old v3 path)
    const int items = static_cast<int>(num_requests * p.HV);
    if (sigmoid_gate) gdn_ss4_kernel<true><<<items, kS4Threads, 0, stream>>>(p, items);
    else gdn_ss4_kernel<false><<<items, kS4Threads, 0, stream>>>(p, items);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
  }
  p.a4_pk = nullptr;  // other kernel variants: no W4A4 epilogue
  if (v3) {
    static int num_sms = 0;
    if (num_sms == 0) {
      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, qkv.get_device());
    }
    const int items = static_cast<int>(num_requests * p.HV);
    const int ctas = items < 2 * num_sms ? items : 2 * num_sms;
    const int smem = static_cast<int>(sizeof(V3Smem));
#define ATDEV_V3(Q, SG)                                                                           \
    {                                                                                             \
      static bool attr = false;                                                                   \
      if (!attr) {                                                                                \
        cudaFuncSetAttribute(gdn_ss3_kernel<Q, SG>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
        attr = true;                                                                              \
      }                                                                                           \
      gdn_ss3_kernel<Q, SG><<<ctas, kThreads, smem, stream>>>(p, items);                          \
    }
    if (qint8) {
      if (sigmoid_gate) ATDEV_V3(true, true) else ATDEV_V3(true, false)
    } else {
      if (sigmoid_gate) ATDEV_V3(false, true) else ATDEV_V3(false, false)
    }
#undef ATDEV_V3
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return false;
  }
#define ATDEV_LAUNCH(Q, SG)                                                              \
  if (v2) {                                                                              \
    if (small) gdn_ss2_kernel<Q, SG, 4, 8><<<grid, kThreads, 0, stream>>>(p);           \
    else gdn_ss2_kernel<Q, SG, 8, 16><<<grid, kThreads, 0, stream>>>(p);                \
  } else {                                                                               \
    if (small) gdn_ss_kernel<Q, SG, 4, 8><<<grid, kThreads, 0, stream>>>(p);            \
    else gdn_ss_kernel<Q, SG, 8, 16><<<grid, kThreads, 0, stream>>>(p);                 \
  }
  if (qint8) {
    if (sigmoid_gate) { ATDEV_LAUNCH(true, true) } else { ATDEV_LAUNCH(true, false) }
  } else {
    if (sigmoid_gate) { ATDEV_LAUNCH(false, true) } else { ATDEV_LAUNCH(false, false) }
  }
#undef ATDEV_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return false;
}

void gdn_ss(torch::Tensor qkv, torch::Tensor a, torch::Tensor b, torch::Tensor a_log,
            torch::Tensor dt_bias, torch::Tensor state, torch::Tensor idx, torch::Tensor cu,
            torch::Tensor acc, torch::Tensor gate, torch::Tensor norm_w, torch::Tensor out,
            int64_t H, int64_t num_requests, double scale, double eps, bool sigmoid_gate,
            bool qint8, int64_t ng, int64_t log_base, int64_t buf, int64_t v_off, int64_t d_off,
            int64_t b_off, int64_t nvp, int64_t dbg) {
  gdn_ss_impl(qkv, a, b, a_log, dt_bias, state, idx, cu, acc, gate, norm_w, out, H, num_requests,
              scale, eps, sigmoid_gate, qint8, ng, log_base, buf, v_off, d_off, b_off, nvp, dbg,
              nullptr, nullptr, nullptr, 0);
}

// same + W4A4 out_proj fragments (a4_pk int32 [chunks][K/64][tmb][32][4], a4_s fp32
// [chunks][K/128][16 tmb], gs fp32 [1]); returns false if the taken kernel variant has no
// a4 epilogue (caller then runs quant_a4 on `out`)
bool gdn_ss_a4(torch::Tensor qkv, torch::Tensor a, torch::Tensor b, torch::Tensor a_log,
               torch::Tensor dt_bias, torch::Tensor state, torch::Tensor idx, torch::Tensor cu,
               torch::Tensor acc, torch::Tensor gate, torch::Tensor norm_w, torch::Tensor out,
               int64_t H, int64_t num_requests, double scale, double eps, bool sigmoid_gate,
               bool qint8, int64_t ng, int64_t log_base, int64_t buf, int64_t v_off, int64_t d_off,
               int64_t b_off, int64_t nvp, int64_t dbg, torch::Tensor a4_pk, torch::Tensor a4_s,
               torch::Tensor a4_gs, int64_t tmb) {
  TORCH_CHECK(a4_pk.scalar_type() == at::kInt && a4_s.scalar_type() == at::kFloat &&
              a4_gs.scalar_type() == at::kFloat && tmb >= 1);
  return gdn_ss_impl(qkv, a, b, a_log, dt_bias, state, idx, cu, acc, gate, norm_w, out, H,
                     num_requests, scale, eps, sigmoid_gate, qint8, ng, log_base, buf, v_off, d_off,
                     b_off, nvp, dbg, reinterpret_cast<uint32_t*>(a4_pk.data_ptr()),
                     a4_s.data_ptr<float>(), a4_gs.data_ptr<float>(), (int)tmb);
}

#ifdef S4B_PROF
torch::Tensor prof_dump() {
  auto t = torch::empty({2048, 16}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  cudaMemcpyFromSymbol(t.data_ptr(), g_prof, sizeof(unsigned long long) * 2048 * 16, 0, cudaMemcpyDeviceToDevice);
  return t;
}
#endif
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef S4B_PROF
  m.def("prof_dump", &prof_dump);
#endif
  m.def("gdn_ss", &gdn_ss, "single-state GDN verify");
  m.def("gdn_ss_a4", &gdn_ss_a4, "single-state GDN verify + W4A4 out_proj fragments");
}
