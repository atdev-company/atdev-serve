// Single-state GDN spec-verify kernel (ATDEV_OPT=gdn_single_state, ATDEV_GDN_IMPL=cuda).
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

}  // namespace

void gdn_ss(torch::Tensor qkv, torch::Tensor a, torch::Tensor b, torch::Tensor a_log,
            torch::Tensor dt_bias, torch::Tensor state, torch::Tensor idx, torch::Tensor cu,
            torch::Tensor acc, torch::Tensor gate, torch::Tensor norm_w, torch::Tensor out,
            int64_t H, int64_t num_requests, double scale, double eps, bool sigmoid_gate,
            bool qint8, int64_t ng, int64_t log_base, int64_t buf, int64_t v_off, int64_t d_off,
            int64_t b_off, int64_t nvp, int64_t dbg) {
  TORCH_CHECK(state.scalar_type() == at::kBFloat16, "gdn_ss: bf16 state only");
  TORCH_CHECK(state.size(2) == kV && state.size(3) == kK, "gdn_ss: K = V = 128 only");
  TORCH_CHECK(ng >= 1 && ng <= kMaxNG && (kK / ng) % 4 == 0, "gdn_ss: bad ng");
  TORCH_CHECK(qkv.stride(1) == 1 && a.stride(-1) == 1 && b.stride(-1) == 1, "gdn_ss: strides");
  TORCH_CHECK(out.is_contiguous(), "gdn_ss: out must be contiguous");
  if (num_requests <= 0) return;
  Args p;
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
    return;
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
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gdn_ss", &gdn_ss, "single-state GDN verify"); }
