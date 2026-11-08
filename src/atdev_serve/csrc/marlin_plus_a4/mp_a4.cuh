// marlin_plus P3: W4A4 (s4 activations x uint4b8 weights, group 128 on both
// sides) inside the marlin_plus_ms pipeline.
//
// Keeps the M-split W4A8 kernel (mp_a8ms.cuh) unchanged in everything that
// streams the weights: persistent striped split-K tile schedule, cp.async
// multi-stage ring for B / group scales, lock-ordered fp32 C_tmp reduction,
// bf16 epilogue through smem.  What changes:
//   * mma: m16n8k64 s4.s4 (2x the int8 rate on sm80).  The int8-path Marlin
//     weight word (8 nibbles) is used directly as a B fragment: uint4b8 -> s4
//     is one `xor 0x88888888` (u - 8 mod 16 flips the top bit), no lop3
//     dequant.  The nibble order of the int8 path (nibble i <-> k32-local
//     (i%2)*16 + (lane%4)*4 + i/2) is a k permutation; the activation packer
//     (a4_quant_kernel) applies the same permutation so the dot product is
//     unchanged.
//   * A: s4, pre-packed in mma fragment order [K/64][tmb][32 lanes][4 words]
//     by the quant kernel; one int4 smem load per (k64, m-block) per thread,
//     no ldmatrix.  Half the A bytes of the int8 path.
//   * scales: per-(row, 128-group) activation scales (x input_global_scale)
//     and the int8 path's int16 group weight scales.  The group fold is in
//     fp32: the group's mma chain starts from C = 0x4B400000 so the int32
//     accumulator *is* the float 12582912 + P (|P| <= 8*8*128 < 2^22), and
//     acc += (float(acc_i) - 12582912) * s16 * a_s  (FADD, FMUL, FFMA; no
//     I2F, which is quarter rate).
// thread_k = 128 = one group per pipeline stage.
// Derived from vLLM v0.31.0 csrc marlin_template.h (Apache-2.0).
#pragma once
#include "marlin_template.h"
#include "mp_a4_ne.h"

// Ablation / variant switches (ATDEV_A4_DEFINES=...; each builds a separate module):
//   A4_ABL_NOREDUCE  skip the split-K reduction (wrong result; timing of the lock chain)
//   A4_ABL_NOEPI     skip the bf16 epilogue write
//   A4_ABL_NOMMA     skip mma + fold (weight-stream only)
//   A4_FOLD2         group fold in 2 FP32 ops/element (fma with -magic*s) instead of 3; bit-identical
//   A4_PAR_REDUCE    parallel split-K reduction: every slice stores its fp32 partial to
//                    its own C_tmp slot, takes a ticket; the last arriver sums all slots
//                    in slice order (deterministic) and writes the result. Depth 2
//                    instead of slice_count. Slots per column (MS) arrive in `lda`.
#ifndef A4_ABL_NOREDUCE
  #define A4_ABL_NOREDUCE 0
#endif
#ifndef A4_ABL_NOEPI
  #define A4_ABL_NOEPI 0
#endif
#ifndef A4_ABL_NOMMA
  #define A4_ABL_NOMMA 0
#endif
#ifndef A4_PAR_REDUCE
  #define A4_PAR_REDUCE 0
#endif
#ifndef A4_FF  // FF-port: staged first fill (1 = sleep after stage 0, 2 = after every prologue stage)
  #define A4_FF 0
#endif
#ifndef A4_FF_NS
  #define A4_FF_NS 256
#endif
#ifndef A4_S0  // FF-port: slice 0 skips its split-K acquire poll (bit-identical)
  #define A4_S0 0
#endif
#ifndef A4_ABL_BFRAC  // timing ablation (wrong result): stream only BFRAC/4 of each stage's weight bytes
  #define A4_ABL_BFRAC 4
#endif
#ifndef A4_OVLCHAIN  // overlapped, bit-exact serial-order split-K chain (see marlin_template.h MP_OVLCHAIN)
  #define A4_OVLCHAIN 0
#endif
#ifndef A4_OVL_MAXS
  #define A4_OVL_MAXS 6
#endif
#ifndef A4_FOLD2  // 2-op group fold, bit-identical to the 3-op one (see fold below)
  #define A4_FOLD2 0
#endif
#ifndef A4_ABL_NOFOLD  // timing ablation (wrong result): mma chain runs across groups, one fold per slice
  #define A4_ABL_NOFOLD 0
#endif
#ifndef A4_ABL_FOLDONLY  // timing ablation (wrong result): no mma; partials = cheap fn of the fragments, fold kept
  #define A4_ABL_FOLDONLY 0
#endif
#ifndef A4_ABL_NOA  // timing ablation (wrong result): skip the activation (A + AS) cp.async loads
  #define A4_ABL_NOA 0
#endif
#ifndef A4_MAGICQ  // 1: group-start mma reads C from one persistent magic quad (no per-mma MOV of the
  #define A4_MAGICQ 0  // constant); bit-identical (same C value)
#endif
#ifndef A4_PIPE_UNROLL1  // 1: do not unroll the stage loop (fewer accumulator register shuffles); bit-identical
  #define A4_PIPE_UNROLL1 0
#endif
#ifndef A4_SCALAR_CTMP  // 1: split-K partial store as 4 scalar stores (no register quads for acc)
  #define A4_SCALAR_CTMP 0
#endif
#ifndef A4_SPLITFOLD  // 1 (DEFER=0 only): half of each group fold runs after the next stage barrier,
  #define A4_SPLITFOLD 0  // overlapping the next group's mma; bit-identical
#endif
#ifndef A4_PTRINC  // C32-ic: cp.async sources from 64-bit pointers kept per slice (immediate offsets per stage,
  #define A4_PTRINC 0  // pointer bumps instead of per-load IMAD.WIDE); 2 = + predicated cp.async (no branch). bit-identical
#endif
#ifndef A4_SLATE  // C32-ic: next tile's group / activation scales loaded AFTER the current fold into the single
  #define A4_SLATE 0  // buffer [0] (no frag_s / frag_as double-buffer copies); DEFER=0 only. bit-identical
#endif
#ifndef A4_PTRINC_TMB  // C32b: apply A4_PTRINC only to instances with this thread_m_blocks (0 = all)
  #define A4_PTRINC_TMB 0
#endif
#ifndef A4_PTRINC_ST  // C32b: ... and this stage count (0 = all)
  #define A4_PTRINC_ST 0
#endif
#ifndef A4_S0_TMB  // C32b: apply A4_S0 only to instances with this thread_m_blocks (0 = all)
  #define A4_S0_TMB 0
#endif
#ifndef A4_S0_ST  // C32b: ... and this stage count (0 = all)
  #define A4_S0_ST 0
#endif
#ifndef A4_C32C_DEFREL  // C32c: defer slice 0's split-K release (bit-identical in the serial chain)
  #define A4_C32C_DEFREL 0
#endif
#ifndef A4_C32C_ABL  // C32c gather timing ablations (wrong results): 1 no release fence, 2 no partial store, 4 last no wait/read
  #define A4_C32C_ABL 0
#endif
#ifndef A4_C32C_VST  // C32c gather: non-last partial stored as int4 (L2-only) stores
  #define A4_C32C_VST 0
#endif
#ifndef A4_C32C_RD  // C32c: serial chain partial read as one batch of L2-direct loads (bit-identical)
  #define A4_C32C_RD 0
#endif
#ifndef A4_C32C_PB  // C32d gather (A4_PAR_REDUCE=6) partial format: 0 fp32, 1 bf16, 2 fp16 x per-thread pow2 scale
  #define A4_C32C_PB 0
#endif
#ifndef A4_C32C_PB2  // C32d gather + PB: last slice loads 2 half-width slots per batch (same register budget)
  #define A4_C32C_PB2 0
#endif
#ifndef A4_PAR_DBG  // 1: last arriver skips the slot sum; 2: no __threadfence
  #define A4_PAR_DBG 0
#endif

namespace MARLIN_NAMESPACE_NAME {

__device__ __forceinline__ void a4_mma(const uint32_t* a, uint32_t b0,
                                       uint32_t b1, int* c) {
  MP_ASM_PURE(
      "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void a4_mma_magic(const uint32_t* a, uint32_t b0,
                                             uint32_t b1, int* c) {
  const int m = 0x4B400000;
  MP_ASM_PURE(
      "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
      : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(m));
}

__device__ __forceinline__ void a4_mma_q(const uint32_t* a, uint32_t b0, uint32_t b1, int* c,
                                         const int* q) {
  MP_ASM_PURE(
      "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(q[0]), "r"(q[1]),
        "r"(q[2]), "r"(q[3]));
}

#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800)

// A4_PAR_REDUCE=3: out-of-line parallel reduction (same protocol as =1). Returns 1 if this CTA is the
// last arriver and `v` now holds the column total (slices summed in slice order).
static __device__ __noinline__ int a4_par_reduce_ni(float* v, int nf, int4* C_tmp, int* locks, int lock,
                                             int slice_idx, int slice_count, int par_ms, int c_size,
                                             int nthreads) {
  __shared__ int s_last;
  const int th = nf / 4;
  int4* slot = C_tmp + (size_t)(lock * par_ms + slice_idx) * c_size;
  for (int k = 0; k < th; k++) slot[nthreads * k + threadIdx.x] = reinterpret_cast<int4*>(v)[k];
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) s_last = atomicAdd(&locks[lock], 1) == slice_count - 1;
  __syncthreads();
  if (!s_last) return 0;
  __threadfence();
  for (int i = 0; i < nf; i++) v[i] = 0.f;
  const int4* base = C_tmp + (size_t)(lock * par_ms) * c_size;
  for (int sidx = 0; sidx < slice_count; sidx++)
    for (int k = 0; k < th; k++) {
      int4 x = __ldcg(&base[(size_t)sidx * c_size + nthreads * k + threadIdx.x]);
      v[4 * k] += __int_as_float(x.x); v[4 * k + 1] += __int_as_float(x.y);
      v[4 * k + 2] += __int_as_float(x.z); v[4 * k + 3] += __int_as_float(x.w);
    }
  __syncthreads();
  if (threadIdx.x == 0) locks[lock] = 0;
  return 1;
}

#ifndef A4_NE_ABL  // C32-NE timing ablations (wrong results): 1 no pair quant / rstd, 2 no S/ro/xb/P stores,
  #define A4_NE_ABL 0  // 4 no fences / tickets (pair work done by every finisher), 8 no ne_pack8 (loads only)
#endif
#if A4_NE == 1 || A4_NE == 3
// C32-NE helpers (same fp32 op sequence as c32_prod.cu / fused_quant._a4_pack_store)
__device__ __forceinline__ float ne_div_rn_ftz(float a, float b) {
  float r;
  asm("div.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
// 8 consecutive values y[0..7] of a 128-group held by lane s = (thread % 16) of a 16-lane group
// (values 8s .. 8s+7); all 32 lanes of the warp must call it. Writes the s4 fragments + group scale.
__device__ __forceinline__ void ne_pack8(const float (&y)[8], int s, int g, int row, int m_pad, int tmb,
                                         float gs, uint32_t* __restrict__ apk, float* __restrict__ as_) {
  float am = 0.0f;
  #pragma unroll
  for (int i = 0; i < 8; i++) am = fmaxf(am, fabsf(y[i]));
  #pragma unroll
  for (int o = 1; o < 16; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
  const float sc = __fmul_rn(fmaxf(am, 9.99999993922529029077850282192230224609375e-9f),
                             0.142857149243354797363281250f);
  const int p = (s >> 1) & 1, a = s >> 2;
  uint32_t part[2] = {0u, 0u};
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    float q = rintf(ne_div_rn_ftz(y[i], sc));
    q = fminf(fmaxf(q, -8.0f), 7.0f);
    const uint32_t nib = (uint32_t)((int)q) & 15u;
    part[i >> 2] |= nib << (8 * (i & 3) + 4 * p);
  }
  const uint32_t o0 = __shfl_xor_sync(0xffffffffu, part[0], 2);
  const uint32_t o1 = __shfl_xor_sync(0xffffffffu, part[1], 2);
  if (p == 0) {
    const int b0 = 2 * (s & 1);
    const int kb = 2 * g + (a >> 1);
    const int rr = (row & 15) / 8 + 2 * (a & 1);
    const int base = (kb * tmb + row / 16) * 32 + (row & 7) * 4;
    apk[(base + b0) * 4 + rr] = part[0] | o0;
    apk[(base + b0 + 1) * 4 + rr] = part[1] | o1;
  }
  if (s == 0) as_[g * m_pad + row] = __fmul_rn(sc, gs);
}
#if A4_NE == 3
__device__ __forceinline__ float ne_div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float ne_rsqrt_approx_ftz(float a) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
// A4_NE=3 (C32-NE "GB"): after a grid-wide barrier at the end of the producer GEMM, CTA b runs the deployed
// norm of row b (residual add + Gemma RMSNorm -> ro, optional xb, s4 fragments + scales of the consumer),
// reading the GEMM's bf16 output C from L2. 128 threads emulate the 256-thread Triton reduction tree of
// fused_quant._norm_a4_chunk_kernel (thread t = virtual threads t and t + 128), so sum(x^2) / rstd are bit-equal
// to it (c32_prod.cu norm_a4 math: only the bf16 rounding of (rstd x)(w + 1) can differ on rare fp32 ties).
constexpr int NE_NP = 5;  // chunks of 1024 columns: N = 5120 (host-checked)
struct NePre {
  uint4 r[NE_NP], w[NE_NP];
};
// residual / norm weight loads of this CTA's row, issued before the grid barrier (they do not depend on C)
__device__ __forceinline__ void ne_gb_prefetch(const A4Ne& ne, int prob_m, int prob_n, NePre& pre) {
  const int t = threadIdx.x, row = blockIdx.x;
  #pragma unroll
  for (int p = 0; p < NE_NP; p++) {
    const int c = p * 1024 + 8 * t;
    pre.r[p] = make_uint4(0, 0, 0, 0);
    pre.w[p] = *reinterpret_cast<const uint4*>(ne.w + c);
    if (row < prob_m) pre.r[p] = *reinterpret_cast<const uint4*>(ne.r + (size_t)row * ne.ldr + c);
  }
}
template <int threads>
__device__ __forceinline__ void ne_gb_rownorm(const A4Ne& ne, const __nv_bfloat16* __restrict__ C, int prob_m,
                                              int prob_n, float* red, const NePre& pre) {
  static_assert(threads == 128, "C32-NE GB: 128-thread CTAs");
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int N = prob_n;  // 5120 (host checks)
  {
    const int row = blockIdx.x;  // grid >= m_pad (host-checked); CTAs past m_pad return
    if (row >= ne.m_pad) return;
    const bool valid = row < prob_m;
    // chunk p = columns p * 1024 + 8 t .. + 7 (= virtual thread t + 128 (p & 1), k = p / 2)
    constexpr int NP = 8;
    float v[NP][8];
  #pragma unroll
    for (int p = 0; p < NP; p++) {
      uint4 xa = make_uint4(0, 0, 0, 0), ra = xa;
      const int c = p * 1024 + 8 * t;
      if (p < NE_NP && valid) {
        xa = __ldcg(reinterpret_cast<const uint4*>(C + (size_t)row * N + c));
        ra = pre.r[p < NE_NP ? p : 0];
      }
      const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xa);
      const __nv_bfloat16* rh = reinterpret_cast<const __nv_bfloat16*>(&ra);
  #pragma unroll
      for (int j = 0; j < 8; j++) v[p][j] = __fadd_rn(__bfloat162float(xh[j]), __bfloat162float(rh[j]));
    }
    // per virtual thread, Triton order: fma(e0, e0, e1 * e1), e2 .. e7, then the next k
    float acc[2];
  #pragma unroll
    for (int h = 0; h < 2; h++) {
      float a = __fmul_rn(v[h][1], v[h][1]);
      a = __fmaf_rn(v[h][0], v[h][0], a);
  #pragma unroll
      for (int j = 2; j < 8; j++) a = __fmaf_rn(v[h][j], v[h][j], a);
  #pragma unroll
      for (int k = 1; k < NP / 2; k++)
  #pragma unroll
        for (int j = 0; j < 8; j++) a = __fmaf_rn(v[2 * k + h][j], v[2 * k + h][j], a);
      acc[h] = a;
    }
  #pragma unroll
    for (int h = 0; h < 2; h++)
  #pragma unroll
      for (int o = 16; o > 0; o >>= 1) acc[h] = __fadd_rn(acc[h], __shfl_xor_sync(0xffffffffu, acc[h], o));
    if (lane == 0) {
      red[warp] = acc[0];
      red[warp + 4] = acc[1];
    }
    __syncthreads();
    if (warp == 0) {
      float sm = lane < 8 ? red[lane] : 0.0f;
  #pragma unroll
      for (int o = 4; o > 0; o >>= 1) sm = __fadd_rn(sm, __shfl_xor_sync(0xffffffffu, sm, o));
      if (lane == 0) red[8] = sm;
    }
    __syncthreads();
    const float rstd = ne_rsqrt_approx_ftz(__fadd_rn(ne.eps, ne_div_full(red[8], (float)N)));
    const float gs = *ne.gs;
  #pragma unroll
    for (int p = 0; p < NP; p++) {
      const int c = p * 1024 + 8 * t;
      if (p * 1024 >= N) break;
      const uint4 wa = pre.w[p < NE_NP ? p : 0];
      const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wa);
      float y[8];
      __nv_bfloat16 rob[8], xbb[8];
  #pragma unroll
      for (int j = 0; j < 8; j++) {
        rob[j] = __float2bfloat16_rn(v[p][j]);
        const float w1 = __fadd_rn(__bfloat162float(wh[j]), 1.0f);
        xbb[j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(rstd, v[p][j]), w1));
        y[j] = valid ? __bfloat162float(xbb[j]) : 0.0f;
      }
      if (valid) {
        *reinterpret_cast<int4*>(ne.ro + (size_t)row * ne.ldro + c) = *reinterpret_cast<const int4*>(rob);
        if (ne.xb != nullptr)
          *reinterpret_cast<int4*>(ne.xb + (size_t)row * ne.ldxb + c) = *reinterpret_cast<const int4*>(xbb);
      }
      ne_pack8(y, t & 15, c / 128, row, ne.m_pad, ne.tmb_c, gs, ne.apk, ne.as);
    }
  }
}
#endif
#endif

#if A4_SILU
#ifndef A4_SILU_UNR  // C32h: 1 unroll the publish loop, 2 the quant loop (default: compact loops)
  #define A4_SILU_UNR 0
#endif
#ifndef A4_SILU_PROF  // C32h: per-CTA globaltimer stamps into A4Silu::dbg
  #define A4_SILU_PROF 0
#endif
#ifndef A4_SILU_ABL  // C32h timing ablations (wrong results): 1 no group wait, 2 no quant, 4 no epilogue at all, 8 epilogue code present but skipped,
  // 16 no SiLU math, 32 no release fence, 64 no done ticket
  #define A4_SILU_ABL 0
#endif
// same fp32 op sequence as c32_prod.cu act_a4_kernel<0> / fused_quant._silu_a4_kernel
__device__ __forceinline__ float sl_div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float sl_div_rn_ftz(float a, float b) {
  float r;
  asm("div.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float sl_ex2(float a) {
  // .ftz: only flushes results < 2^-126, where 1 + e rounds to 1 either way -> same sigmoid
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
// y = bf16(silu(g) * u) of the GEMM's bf16-rounded outputs g, u (fp32 accumulators)
__device__ __forceinline__ float sl_y(float ag, float au) {
  const float g = __bfloat162float(__float2bfloat16(ag)), u = __bfloat162float(__float2bfloat16(au));
  const float e = sl_ex2(__fmul_rn(__fsub_rn(0.0f, g), 1.44269502162933349609375f));
  const float sg = sl_div_full(1.0f, __fadd_rn(e, 1.0f));
  return __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fmul_rn(sg, g), u)));
}
#ifndef A4_SILU_FQ  // 1: quotient y * (1/sc) with an exact div.rn fallback within 1e-4 of a .5 boundary (bit-identical);
  #define A4_SILU_FQ 2  // 2: branch-free correctly rounded quotient (Markstein fma correction; compact epilogue only)
#endif
__device__ __forceinline__ uint32_t sl_nib(float y, float sc, float rc) {
  float q;
  if constexpr (A4_SILU_FQ) {
    // t = y * RN(1/sc) is within ~2 ulp (|t| <= 7.0001 -> < 1e-6) of RN(y / sc): rint differs only if a
    // half-integer lies between them, i.e. only if |t - rint(t)| > 0.5 - 1e-6; those take the exact path
    float t;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(t) : "f"(y), "f"(rc));
    q = rintf(t);
    if (fabsf(__fsub_rn(t, q)) > 0.4999f) q = rintf(sl_div_rn_ftz(y, sc));
  } else {
    q = rintf(sl_div_rn_ftz(y, sc));
  }
  q = fminf(fmaxf(q, -8.0f), 7.0f);
  return (uint32_t)((int)q) & 15u;
}
#endif

template <int threads, int thread_m_blocks, int thread_n_blocks, int stages,
          int MW, int DEFER, int MINB>
__global__ void __launch_bounds__(threads, MINB) MarlinA4(MARLIN_KERNEL_PARAMS A4_XPARAMS) {
  constexpr int thread_k_blocks = 8;        // tk = 128 = one group
  constexpr int tmb = thread_m_blocks;
  constexpr int TMW = tmb / MW;             // m-blocks per warp
  constexpr int tb_n_warps = thread_n_blocks / 2;  // 32 columns per warp
  constexpr int KS = 2;                     // k64 steps per stage
  static_assert(tmb % MW == 0, "MW must divide thread_m_blocks");
  static_assert(threads == 32 * MW * tb_n_warps, "threads = 32*MW*(tn/32)");
  static_assert(stages % 2 == 0, "stages must be even");
  constexpr int m_block_size = 16 * tmb;
  constexpr float kMagic = 12582912.0f;
  // C32b: per-instance selection of the bit-identical A4_PTRINC / A4_S0 variants
  constexpr int kPtrInc = (A4_PTRINC && (A4_PTRINC_TMB == 0 || A4_PTRINC_TMB == thread_m_blocks) &&
                           (A4_PTRINC_ST == 0 || A4_PTRINC_ST == stages)) ? A4_PTRINC : 0;
  constexpr bool kS0 = A4_S0 && (A4_S0_TMB == 0 || A4_S0_TMB == thread_m_blocks) &&
                       (A4_S0_ST == 0 || A4_S0_ST == stages);

  const int4* A0 = A;
  int4* C0 = C;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int nw = warp % tb_n_warps;  // column strip of this warp
  const int mh = warp / tb_n_warps;  // row group of this warp

  extern __shared__ int4 sh[];
#if A4_SILU && A4_SILU_PROF
  int sl_ev = 1;  // stamp slot (thread 0): [0] start, [1 + 3k ..] publish start / publish end / quant start (+ tile),
  auto sl_ts = [&](int slot, unsigned long long extra = 0) {  // [63] end
    if (threadIdx.x == 0 && sl.dbg != nullptr && slot < 64) {
      unsigned long long t;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
      sl.dbg[(size_t)blockIdx.x * 64 + slot] = (t & ((1ull << 48) - 1)) | (extra << 48);
    }
  };
  sl_ts(0);
#endif
#if A4_SILU && A4_SILU_TAG
  // C32h TAG: this CTA's call epoch (every CTA bumps its own slot once per call -> all CTAs agree)
  // (plain uniform load here, the CTA's thread 0 stores it back at kernel end; a kernel-start atomicAdd by
  // thread 0 instead made the kernel +55 us slower, cause not isolated)
  const uint32_t sl_ep = (A4_SILU_ABL & 128) ? 1u : (uint32_t)__ldcg(&sl.cnt[blockIdx.x]) + 1u;
#endif

  int parallel = 1;
  if (prob_m > m_block_size) {
    parallel = prob_m / m_block_size;
    prob_m = m_block_size;
  }
  const int k_tiles = prob_k / 16 / thread_k_blocks;
  const int n_tiles = prob_n / 16 / thread_n_blocks;
  const int global_mn_tiles = parallel * n_tiles;
  int part2_mn_tiles = global_mn_tiles;
  int part1_mn_iters = 0;
  bool in_part2 = false;
  if (global_mn_tiles > gridDim.x) {
    part2_mn_tiles = global_mn_tiles % gridDim.x;
    if (part2_mn_tiles * 3 <= gridDim.x) part2_mn_tiles += gridDim.x;
    part1_mn_iters = (global_mn_tiles - part2_mn_tiles) / gridDim.x;
  }
  const int iters = div_ceil(k_tiles * part2_mn_tiles, gridDim.x);

  int slice_row = 0;
  int slice_col_par = blockIdx.x;
  int slice_col;
  int slice_iters = k_tiles;
  int slice_count = 1;
  int slice_idx = 0;
  int par_id = 0;
  int locks_off = 0;
  if (part2_mn_tiles >= gridDim.x) {
    locks_off = blockIdx.x;
  } else {
    locks_off = (iters * blockIdx.x) / k_tiles - 1;
  }

  // ---- A (s4 fragments): per k-tile 2 k64 blocks x tmb m-blocks x 32 int4 ----
  constexpr int a_sh_stage = KS * tmb * 32;  // int4
  constexpr int a_sh_wr_iters = div_ceil(a_sh_stage, threads);
  const int a_chunk = k_tiles * a_sh_stage;  // int4 per 64-row chunk
  // ---- activation scales: per k-tile m_block_size floats ----
  constexpr int as_sh_stage = m_block_size / 4;  // int4
  const int as_chunk = k_tiles * as_sh_stage;
  const int4* AS = reinterpret_cast<const int4*>(a_scales_ptr);
  // ---- B (uint4b8, Marlin int8-path layout: one k32 row = prob_n int4) ----
  const int b_gl_stride = prob_n;
  constexpr int b_sh_stride = thread_n_blocks * 16;  // int4 per k32 row
  const int b_gl_rd_delta_o = b_gl_stride * 4;       // 4 k32 rows per stage
  constexpr int b_sh_stage = b_sh_stride * 4;
  constexpr int b_ld_iters = div_ceil(b_sh_stage, threads);
  // ---- group scales (int16 in bf16 slots) ----
  const int s_gl_stride = prob_n / 8;
  constexpr int s_sh_stride = 2 * thread_n_blocks;
  constexpr int s_sh_stage = s_sh_stride;

  int a_gl_rd, as_gl_rd, b_gl_rd, s_gl_rd;  // set after init_slice()
  int b_gl_off[b_ld_iters];
  #pragma unroll
  for (int i = 0; i < b_ld_iters; i++) {
    int idx = threads * i + threadIdx.x;
    b_gl_off[i] = b_gl_stride * (idx / b_sh_stride) + (idx % b_sh_stride);
  }
  const int b_sh_rd = 32 * nw + lane;
  const int s_sh_wr = threadIdx.x;
  const bool s_sh_wr_pred = threadIdx.x < s_sh_stage;
  const int s_sh_rd = 4 * nw + (lane % 4);

  constexpr int sh_red_size = (2 * thread_n_blocks + 1) * 16 * tmb;
  constexpr int sh_b_size = stages * b_sh_stage;
  constexpr int sh_br_size = sh_red_size > sh_b_size ? sh_red_size : sh_b_size;
  int4* sh_b = sh;
  int4* sh_red = sh;
  int4* sh_s = sh + sh_br_size;
  int4* sh_as = sh_s + stages * s_sh_stage;
  int4* sh_a = sh_as + stages * as_sh_stage;

  bool first_init = true;
  auto init_part2_slice = [&]() {
    slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= part2_mn_tiles) slice_iters = 0;
    if (slice_iters == 0) return;
    if (slice_row + slice_iters > k_tiles) slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * div_ceil(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = div_ceil(k_tiles - col_off, iters);
      if (col_off > 0) slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0) slice_idx--;
      }
    }
    if (part2_mn_tiles >= gridDim.x) {
      if (slice_count > 1 && slice_idx == slice_count - 1) locks_off++;
    } else {
      locks_off++;
    }
    if (slice_col == n_tiles) {
      A += a_chunk;
      AS += as_chunk;
      C += m_block_size * prob_n / 8;
      slice_col = 0;
      par_id++;
    }
    first_init = false;
  };
  auto init_part1_slice = [&]() {
    if (part1_mn_iters) {
      part1_mn_iters--;
      par_id = slice_col_par / n_tiles;
      slice_col = slice_col_par % n_tiles;
      slice_iters = k_tiles;
      A = A0 + a_chunk * par_id;
      AS = reinterpret_cast<const int4*>(a_scales_ptr) + as_chunk * par_id;
      C = C0 + m_block_size / 8 * par_id * prob_n;
    }
  };
  auto init_slice = [&]() {
    if (!in_part2 && !part1_mn_iters) {
      in_part2 = true;
      slice_col_par = (iters * blockIdx.x) / k_tiles;
      slice_row = (iters * blockIdx.x) % k_tiles;
      slice_col = (slice_col_par + global_mn_tiles - part2_mn_tiles) % n_tiles;
      par_id = (slice_col_par + global_mn_tiles - part2_mn_tiles) / n_tiles;
      A = A0 + a_chunk * par_id;
      AS = reinterpret_cast<const int4*>(a_scales_ptr) + as_chunk * par_id;
      C = C0 + m_block_size / 8 * par_id * prob_n;
    }
    if (!in_part2) init_part1_slice();
    else init_part2_slice();
  };
  init_slice();
  (void)first_init;
  a_gl_rd = a_sh_stage * slice_row;
  as_gl_rd = as_sh_stage * slice_row;
  b_gl_rd = b_sh_stride * slice_col + b_gl_rd_delta_o * slice_row;
  s_gl_rd = s_gl_stride * slice_row + s_sh_stride * slice_col + threadIdx.x;
  // A4_PTRINC: source pointers (a_p / as_p include threadIdx.x; b_p[i] include b_gl_off[i])
  const int4* a_p = nullptr;
  const int4* as_p = nullptr;
  const int4* b_p[b_ld_iters];
  const int4* s_p = nullptr;
  auto set_ptrs = [&]() {
    if constexpr (kPtrInc) {
      a_p = A + a_gl_rd + threadIdx.x;
      as_p = AS + as_gl_rd + threadIdx.x;
  #pragma unroll
      for (int i = 0; i < b_ld_iters; i++) b_p[i] = B + b_gl_rd + b_gl_off[i];
      s_p = scales_ptr + s_gl_rd;
    }
  };
  set_ptrs();

  uint32_t frag_a[2][TMW][4];
  uint32_t frag_b[2][2][4];   // [k64 parity][k32 row][n8 tile]
  int4 frag_s[2];             // 8 x int16 group scales of this thread's 8 columns
  float frag_as[2][TMW][2];   // activation scales of this thread's rows (lane/4, +8)
  float acc[TMW][2][2][4];
  int part[DEFER ? 2 : 1][TMW][2][2][4];
  int4 sv_saved;
  float as_saved[TMW][2];
  bool pending = false;
  int mq[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) mq[i] = 0x4B400000 + ((prob_k >> 31) & (i + 1));
  (void)mq;

  auto zero_accums = [&]() {
  #pragma unroll
    for (int i = 0; i < TMW * 16; i++) reinterpret_cast<float*>(acc)[i] = 0.f;
  };

  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    if constexpr (kPtrInc) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
      if constexpr (!A4_ABL_NOA) {
  #pragma unroll
        for (int i = 0; i < a_sh_wr_iters; i++) {
          int idx = threads * i + threadIdx.x;
          if (a_sh_stage % threads == 0 || idx < a_sh_stage) {
            if constexpr (kPtrInc == 2) cp_async4_pred(&sh_a_stage[idx], a_p + a_sh_stage * a_off + threads * i, pred);
            else if (pred) cp_async4(&sh_a_stage[idx], a_p + a_sh_stage * a_off + threads * i);
          }
        }
        int4* sh_as_stage = sh_as + as_sh_stage * pipe;
        if constexpr (kPtrInc == 2)
          cp_async4_pred(&sh_as_stage[threadIdx.x], as_p + as_sh_stage * a_off, pred && threadIdx.x < as_sh_stage);
        else if (pred && threadIdx.x < as_sh_stage)
          cp_async4(&sh_as_stage[threadIdx.x], as_p + as_sh_stage * a_off);
      }
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < b_ld_iters; i++) {
        if (b_sh_stage % threads == 0 || threads * i + threadIdx.x < b_sh_stage) {
          if constexpr (kPtrInc == 2) cp_async4_pred(&sh_b_stage[threads * i + threadIdx.x], b_p[i], pred);
          else if (pred) cp_async4(&sh_b_stage[threads * i + threadIdx.x], b_p[i]);
        }
        b_p[i] += b_gl_rd_delta_o;  // only consumed again after set_ptrs() when !pred
      }
      int4* sh_s_stage = sh_s + s_sh_stage * pipe;
      if constexpr (kPtrInc == 2) cp_async4_pred(&sh_s_stage[s_sh_wr], s_p, pred && s_sh_wr_pred);
      else if (pred && s_sh_wr_pred) cp_async4(&sh_s_stage[s_sh_wr], s_p);
      s_p += s_gl_stride;
      cp_async_fence();
      return;
    }
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
      if constexpr (!A4_ABL_NOA) {
  #pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        int idx = threads * i + threadIdx.x;
        if (a_sh_stage % threads == 0 || idx < a_sh_stage)
          cp_async4(&sh_a_stage[idx], &A[a_gl_rd + a_sh_stage * a_off + idx]);
      }
      int4* sh_as_stage = sh_as + as_sh_stage * pipe;
      if (threadIdx.x < as_sh_stage)
        cp_async4(&sh_as_stage[threadIdx.x],
                  &AS[as_gl_rd + as_sh_stage * a_off + threadIdx.x]);
      }
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < b_ld_iters; i++) {
        if (A4_ABL_BFRAC == 4 ? (b_sh_stage % threads == 0 || threads * i + threadIdx.x < b_sh_stage)
                              : (threads * i + threadIdx.x < b_sh_stage * A4_ABL_BFRAC / 4))
          cp_async4(&sh_b_stage[threads * i + threadIdx.x], &B[b_gl_rd + b_gl_off[i]]);
      }
      b_gl_rd += b_gl_rd_delta_o;
      int4* sh_s_stage = sh_s + s_sh_stage * pipe;
      if (s_sh_wr_pred) cp_async4(&sh_s_stage[s_sh_wr], &scales_ptr[s_gl_rd]);
      s_gl_rd += s_gl_stride;
    }
    cp_async_fence();
  };

  auto wait_for_stage = [&]() {
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  auto fetch_to_registers = [&](int k, int pipe) {
    const int kk = k % KS;
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
  #pragma unroll
    for (int i = 0; i < TMW; i++)
      *reinterpret_cast<int4*>(frag_a[k % 2][i]) =
          sh_a_stage[(kk * tmb + mh * TMW + i) * 32 + lane];
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;
    *reinterpret_cast<int4*>(frag_b[k % 2][0]) = sh_b_stage[b_sh_stride * (2 * kk) + b_sh_rd];
    *reinterpret_cast<int4*>(frag_b[k % 2][1]) = sh_b_stage[b_sh_stride * (2 * kk + 1) + b_sh_rd];
  };

  auto fetch_scales_to_registers = [&](int k, int full_pipe) {
    const int pipe = full_pipe % stages;
    if (k % KS == 0) {
      frag_s[k % 2] = sh_s[s_sh_stage * pipe + s_sh_rd];
      const float* as_st = reinterpret_cast<const float*>(sh_as + as_sh_stage * pipe);
  #pragma unroll
      for (int i = 0; i < TMW; i++) {
        frag_as[k % 2][i][0] = as_st[(mh * TMW + i) * 16 + lane / 4];
        frag_as[k % 2][i][1] = as_st[(mh * TMW + i) * 16 + 8 + lane / 4];
      }
    } else {
      frag_s[1] = frag_s[0];
  #pragma unroll
      for (int i = 0; i < TMW; i++) {
        frag_as[1][i][0] = frag_as[0][i][0];
        frag_as[1][i][1] = frag_as[0][i][1];
      }
    }
  };

  // acc += (P as float) * s16[col] * a_s[row, group]
  auto fold = [&](int (&p)[TMW][2][2][4], int4 svr, float (&as)[TMW][2], int jsel = -1) {
    const int16_t* s16 = reinterpret_cast<const int16_t*>(&svr);
  #pragma unroll
    for (int j = 0; j < 2; j++) {
      if (jsel >= 0 && j != jsel) continue;
  #pragma unroll
      for (int h = 0; h < 2; h++) {
        const float s0 = (float)s16[(j * 2 + h) * 2 + 0];
        const float s1 = (float)s16[(j * 2 + h) * 2 + 1];
  #if A4_FOLD2
        // (bits(magic + P) * s) - magic * s in ONE fma: the exact value is P * s, rounded once,
        // i.e. the same float as FADD(-magic) (exact, |P| < 2^22) then FMUL(s) (rounded once).
        // -magic * s is exact (1.5 * |s16| < 2^17 fits the 24-bit mantissa). 2 FP32 ops/element
        // instead of 3 (+ 2 FMUL per (j,h), amortised over 4*TMW elements).
        const float n0 = -kMagic * s0, n1 = -kMagic * s1;
  #pragma unroll
        for (int i = 0; i < TMW; i++) {
  #pragma unroll
          for (int g = 0; g < 4; g++) {
            const float ps = fmaf(__int_as_float(p[i][j][h][g]), g % 2 ? s1 : s0, g % 2 ? n1 : n0);
            acc[i][j][h][g] = fmaf(ps, as[i][g / 2], acc[i][j][h][g]);
          }
        }
  #else
  #pragma unroll
        for (int i = 0; i < TMW; i++) {
  #pragma unroll
          for (int g = 0; g < 4; g++) {
            const float t = __int_as_float(p[i][j][h][g]) - kMagic;
            acc[i][j][h][g] = fmaf(t * (g % 2 ? s1 : s0), as[i][g / 2], acc[i][j][h][g]);
          }
        }
  #endif
      }
    }
  };

  auto mma_step = [&](int (&P)[TMW][2][2][4], int k2, bool first, int jsel = -1) {
    if constexpr (A4_ABL_NOMMA) {
      if (first) {
  #pragma unroll
        for (int i = 0; i < TMW * 16; i++)
          reinterpret_cast<int*>(P)[i] = 0x4B400000 ^ (int)(frag_a[k2][0][0] & frag_b[k2][0][0] & 1u);
      }
      return;
    }
    if constexpr (A4_ABL_FOLDONLY) {
  #pragma unroll
      for (int i = 0; i < TMW; i++)
  #pragma unroll
        for (int j = 0; j < 2; j++)
  #pragma unroll
          for (int h = 0; h < 2; h++) {
            const int v = (int)((frag_a[k2][i][j * 2 + h] ^ frag_b[k2][0][j * 2 + h] ^ frag_b[k2][1][j * 2 + h]) & 0xFu);
  #pragma unroll
            for (int g = 0; g < 4; g++)
              P[i][j][h][g] = first ? (0x4B400000 | (g == 0 ? v : 0)) : (P[i][j][h][g] ^ (g == 0 ? v : 0));
          }
      return;
    }
  #pragma unroll
    for (int j = 0; j < 2; j++) {
      if (jsel >= 0 && j != jsel) continue;
  #pragma unroll
      for (int h = 0; h < 2; h++) {
        const uint32_t b0 = frag_b[k2][0][j * 2 + h] ^ 0x88888888u;
        const uint32_t b1 = frag_b[k2][1][j * 2 + h] ^ 0x88888888u;
  #pragma unroll
        for (int i = 0; i < TMW; i++) {
          if (first) {
            if constexpr (A4_MAGICQ) a4_mma_q(frag_a[k2][i], b0, b1, P[i][j][h], mq);
            else a4_mma_magic(frag_a[k2][i], b0, b1, P[i][j][h]);
          }
          else a4_mma(frag_a[k2][i], b0, b1, P[i][j][h]);
        }
      }
    }
  };
  bool abl_run = false;
  auto matmul_a4 = [&](int k, int tile) {
    const int k2 = k % 2;
    const bool first = k == 0;
    const bool last = k == KS - 1;
    if constexpr (DEFER) {
      const int gp = tile % 2;
      if (gp == 0) mma_step(part[0], k2, first);
      else mma_step(part[DEFER ? 1 : 0], k2, first);
      if (k == 1 && pending) {
        if (gp == 0) fold(part[DEFER ? 1 : 0], sv_saved, as_saved);
        else fold(part[0], sv_saved, as_saved);
        pending = false;
      }
      if (last) {
        sv_saved = frag_s[k2];
  #pragma unroll
        for (int i = 0; i < TMW; i++) {
          as_saved[i][0] = frag_as[k2][i][0];
          as_saved[i][1] = frag_as[k2][i][1];
        }
        pending = true;
      }
    } else if constexpr (A4_SPLITFOLD) {
      // fold the n-half j=1 of group g after the next stage barrier, interleaved with group g+1's
      // first k64 mma on the j=0 half (no extra partial registers; per-element op order unchanged)
      if (first) {
        mma_step(part[0], k2, true, 0);
        if (pending) {
          fold(part[0], sv_saved, as_saved, 1);
          pending = false;
        }
        mma_step(part[0], k2, true, 1);
      } else {
        mma_step(part[0], k2, false);
      }
      if (last) {
        fold(part[0], frag_s[k2], frag_as[k2], 0);
        sv_saved = frag_s[k2];
  #pragma unroll
        for (int i = 0; i < TMW; i++) {
          as_saved[i][0] = frag_as[k2][i][0];
          as_saved[i][1] = frag_as[k2][i][1];
        }
        pending = true;
      }
    } else if constexpr (A4_ABL_NOFOLD) {
      mma_step(part[0], k2, first && !abl_run);
      abl_run = true;
    } else {
      mma_step(part[0], k2, first);
      if (last) fold(part[0], frag_s[A4_SLATE ? 0 : k2], frag_as[A4_SLATE ? 0 : k2]);
    }
  };
  int last_gp = 0;
  auto flush_fold = [&]() {
    if constexpr (A4_SPLITFOLD && !DEFER) {
      if (pending) {
        fold(part[0], sv_saved, as_saved, 1);
        pending = false;
      }
    }
    if constexpr (A4_ABL_NOFOLD && !DEFER) {
      fold(part[0], frag_s[0], frag_as[0]);
      abl_run = false;
    }
    if constexpr (DEFER) {
      if (pending) {
        if (last_gp == 0) fold(part[0], sv_saved, as_saved);
        else fold(part[DEFER ? 1 : 0], sv_saved, as_saved);
        pending = false;
      }
    }
  };

  // parallel reduction: partial slot of (column locks_off, slice s)
  const int par_ms = lda;  // slots per column (host)
  __shared__ int sh_last;
  auto par_reduce = [&]() -> bool {  // returns true if this CTA writes the result
    constexpr int tb_m = tmb * 16;
    constexpr int tb_n = thread_n_blocks * 16;
    constexpr int c_size = tb_m * tb_n * sizeof(float) / 16;
    constexpr int th_size = TMW * 16 * sizeof(float) / 16;
    int4* slot = C_tmp + (size_t)(locks_off * par_ms + slice_idx) * c_size;
    const int4* fc4 = reinterpret_cast<const int4*>(&acc);
  #pragma unroll
    for (int k = 0; k < th_size; k++) slot[threads * k + threadIdx.x] = fc4[k];
    if (!(A4_PAR_DBG & 2)) __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      int t = atomicAdd(&locks[locks_off], 1);
      sh_last = (t == slice_count - 1);
    }
    __syncthreads();
    if (!sh_last) return false;
    if (!(A4_PAR_DBG & 2)) __threadfence();
    if (A4_PAR_DBG & 1) {
      if (threadIdx.x == 0) locks[locks_off] = 0;
      return true;
    }
    float* fc = reinterpret_cast<float*>(&acc);
  #pragma unroll
    for (int i = 0; i < TMW * 16; i++) fc[i] = 0.f;
    const int4* base = C_tmp + (size_t)(locks_off * par_ms) * c_size;
    for (int sidx = 0; sidx < slice_count; sidx++) {
  #pragma unroll
      for (int k = 0; k < th_size; k++) {
        int4 v = __ldcg(&base[(size_t)sidx * c_size + threads * k + threadIdx.x]);
        const float* sv = reinterpret_cast<const float*>(&v);
  #pragma unroll
        for (int f = 0; f < 4; f++) fc[k * 4 + f] += sv[f];
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) locks[locks_off] = 0;
    return true;
  };

  auto global_reduce_fp32 = [&](bool first = false, bool last = false) {
    constexpr int tb_m = tmb * 16;
    constexpr int tb_n = thread_n_blocks * 16;
    constexpr int c_size = tb_m * tb_n * sizeof(float) / 16;
    constexpr int th_size = TMW * 16 * sizeof(float) / 16;
    int c_cur_offset = locks_off * (A4_OVLCHAIN ? A4_OVL_MAXS : 1) * c_size;  // disjoint from OVL slots
    if (!first && A4_C32C_RD) {
      // C32c: all partial loads in flight at once (L2-direct), then the adds; same fp32 order (bit-identical)
      float* fc = reinterpret_cast<float*>(&acc);
      int4 v[th_size];
  #pragma unroll
      for (int k = 0; k < th_size; k++) v[k] = __ldcg(&C_tmp[c_cur_offset + threads * k + threadIdx.x]);
  #pragma unroll
      for (int k = 0; k < th_size; k++) {
        fc[k * 4 + 0] += __int_as_float(v[k].x); fc[k * 4 + 1] += __int_as_float(v[k].y);
        fc[k * 4 + 2] += __int_as_float(v[k].z); fc[k * 4 + 3] += __int_as_float(v[k].w);
      }
    } else if (!first) {
      float* fc = reinterpret_cast<float*>(&acc);
  #pragma unroll
      for (int k = 0; k < th_size; k++) {
        sh_red[threadIdx.x] = C_tmp[c_cur_offset + threads * k + threadIdx.x];
        float* sc = reinterpret_cast<float*>(&sh_red[threadIdx.x]);
  #pragma unroll
        for (int f = 0; f < 4; f++) fc[k * 4 + f] += sc[f];
      }
    }
    if (!last) {
  #if A4_SCALAR_CTMP
      const float* fs = reinterpret_cast<const float*>(&acc);
  #pragma unroll
      for (int k = 0; k < th_size; k++) {
        float* dst = reinterpret_cast<float*>(&C_tmp[c_cur_offset + threads * k + threadIdx.x]);
  #pragma unroll
        for (int f = 0; f < 4; f++) dst[f] = fs[k * 4 + f];
      }
  #else
      int4* fc = reinterpret_cast<int4*>(&acc);
  #pragma unroll
      for (int k = 0; k < th_size; k++)
        C_tmp[c_cur_offset + threads * k + threadIdx.x] = fc[k];
  #endif
    }
  };

#if A4_NE == 1
  int ne_t0 = -1, ne_t1 = -1, ne_t2 = -1;
  // C32-NE producer epilogue (see mp_a4_ne.h) for column tile tile_, run at the END of the kernel by the CTA
  // that wrote the tile's final bf16 result to C (an epilogue inside the split-K schedule would delay the
  // next column's slice chain); needs parallel == 1 (prob_m <= 16 tmb) and tn | 128 (host checks).
  auto ne_epilogue = [&](int tile_) {
    constexpr int tn = thread_n_blocks * 16;
    constexpr int cpr = 2 * thread_n_blocks;          // int4 (8 columns) per tile row
    constexpr int rows_it = threads / cpr;
    constexpr int tpg = 128 / tn;                     // column tiles per 128-group
    static_assert(tpg >= 1 && 128 % tn == 0, "C32-NE: tn must divide 128");
    const int tile = tile_;
    const int cseg = threadIdx.x % cpr;
    const int row0 = threadIdx.x / cpr;
    const int col = tile * tn + cseg * 8;
    float w1[8];
    {
      const int4 wv = *reinterpret_cast<const int4*>(ne.w + col);
      const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wv);
  #pragma unroll
      for (int e = 0; e < 8; e++) w1[e] = __fadd_rn(__bfloat162float(wh[e]), 1.0f);
    }
  #pragma unroll
    for (int it = 0; it < div_ceil(16 * tmb, rows_it); it++) {
      const int row = row0 + it * rows_it;
      float ss = 0.0f;
      if (row < prob_m) {
        const int4 hv = *reinterpret_cast<const int4*>(reinterpret_cast<const __nv_bfloat16*>(C0) +
                                                         (size_t)row * prob_n + col);
        const int4 rv = *reinterpret_cast<const int4*>(ne.r + (size_t)row * ne.ldr + col);
        const __nv_bfloat16* hh = reinterpret_cast<const __nv_bfloat16*>(&hv);
        const __nv_bfloat16* rh = reinterpret_cast<const __nv_bfloat16*>(&rv);
        float x[8], a[8];
        __nv_bfloat16 rob[8];
  #pragma unroll
        for (int e = 0; e < 8; e++) {
          x[e] = __fadd_rn(__bfloat162float(hh[e]), __bfloat162float(rh[e]));
          rob[e] = __float2bfloat16_rn(x[e]);
          a[e] = __fmul_rn(x[e], w1[e]);
        }
        ss = __fmul_rn(x[0], x[0]);
  #pragma unroll
        for (int e = 1; e < 8; e++) ss = __fmaf_rn(x[e], x[e], ss);
        if (!(A4_NE_ABL & 2) || a[0] == 1.2345f)
        *reinterpret_cast<int4*>(ne.ro + (size_t)row * ne.ldro + col) = *reinterpret_cast<const int4*>(rob);
        if (ne.xb != nullptr) {
          __nv_bfloat16 xbb[8];
  #pragma unroll
          for (int e = 0; e < 8; e++) xbb[e] = __float2bfloat16_rn(a[e]);
          *reinterpret_cast<int4*>(ne.xb + (size_t)row * ne.ldxb + col) = *reinterpret_cast<const int4*>(xbb);
        }
        float4* sd = reinterpret_cast<float4*>(ne.S + ((size_t)tile * (16 * tmb) + row) * tn + cseg * 8);
        if (!(A4_NE_ABL & 2) || a[1] == 1.2345f) {
          __stcg(sd, make_float4(a[0], a[1], a[2], a[3]));
          __stcg(sd + 1, make_float4(a[4], a[5], a[6], a[7]));
        }
      }
  #pragma unroll
      for (int o = 1; o < cpr; o <<= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
      if (cseg == 0 && row < prob_m) __stcg(ne.P + (size_t)row * n_tiles + tile, ss);
    }
    if constexpr (A4_NE_ABL & 1) {
      __syncthreads();
      return;
    }
    // pair ticket: the second finisher of the 128-group quantises both column tiles
    const int grp = tile / tpg;
    if constexpr (!(A4_NE_ABL & 4)) {
      __threadfence();
      __syncthreads();
      if (threadIdx.x == 0) sh_last = tpg == 1 ? 0 : atomicAdd(&ne.cnt[grp], 1);
    } else {
      if (threadIdx.x == 0) sh_last = tpg - 1;
    }
    __syncthreads();
    if (sh_last == tpg - 1) {
      if (threadIdx.x == 0 && tpg > 1 && !(A4_NE_ABL & 4)) ne.cnt[grp] = 0;
      if constexpr (!(A4_NE_ABL & 4)) __threadfence();
      const float gs = *ne.gs;
      const int s16 = threadIdx.x & 15;
      constexpr int qrows = threads / 16;
      const int ct = grp * tpg + (s16 * 8) / tn, cc = (s16 * 8) % tn;
      // loads of QU row blocks issued before their packs (L2 latency overlapped)
      constexpr int QU = 4;
  #pragma unroll 1
      for (int r0 = 0; r0 < ne.m_pad; r0 += QU * qrows) {
        float y[QU][8];
  #pragma unroll
        for (int u = 0; u < QU; u++) {
          const int row = r0 + u * qrows + threadIdx.x / 16;
          float4 u0 = make_float4(0.f, 0.f, 0.f, 0.f), u1 = u0;
          if (row < prob_m) {
            const float4* src = reinterpret_cast<const float4*>(ne.S + ((size_t)ct * (16 * tmb) + row) * tn + cc);
            u0 = __ldcg(src);
            u1 = __ldcg(src + 1);
          }
          y[u][0] = u0.x; y[u][1] = u0.y; y[u][2] = u0.z; y[u][3] = u0.w;
          y[u][4] = u1.x; y[u][5] = u1.y; y[u][6] = u1.z; y[u][7] = u1.w;
        }
  #pragma unroll
        for (int u = 0; u < QU; u++) {
          const int row = r0 + u * qrows + threadIdx.x / 16;
          if (A4_NE_ABL & 8) {
            if (y[u][0] == 1.2345f) ne.as[row] = y[u][1];
          } else if (r0 + u * qrows < ne.m_pad)
            ne_pack8(y[u], s16, grp, row, ne.m_pad, ne.tmb_c, gs, ne.apk, ne.as);
        }
      }
      // global ticket over the groups: the last one turns P into rstd
      if constexpr (!(A4_NE_ABL & 4)) __threadfence();
      __syncthreads();
      const int ngrp = n_tiles / tpg;
      if (threadIdx.x == 0) sh_last = (A4_NE_ABL & 4) ? (grp == ngrp - 1) : (atomicAdd(&ne.cnt[ngrp], 1) == ngrp - 1 ? 1 : 0);
      __syncthreads();
      if (sh_last == 1) {
        if (threadIdx.x == 0 && !(A4_NE_ABL & 4)) ne.cnt[ngrp] = 0;
        if constexpr (!(A4_NE_ABL & 4)) __threadfence();
        for (int row = threadIdx.x; row < ne.m_pad; row += threads) {
          float r = 0.0f;
          if (row < prob_m) {
            // fixed order over the tiles; 16 float4 loads in flight per chunk
            float sum = 0.0f;
            const float* pr = ne.P + (size_t)row * n_tiles;
            int t0 = 0;
            for (; t0 + 64 <= n_tiles; t0 += 64) {
              float4 v[16];
  #pragma unroll
              for (int q = 0; q < 16; q++) v[q] = __ldcg(reinterpret_cast<const float4*>(pr + t0) + q);
  #pragma unroll
              for (int q = 0; q < 16; q++) {
                sum = __fadd_rn(sum, v[q].x); sum = __fadd_rn(sum, v[q].y);
                sum = __fadd_rn(sum, v[q].z); sum = __fadd_rn(sum, v[q].w);
              }
            }
            float rest[16];
  #pragma unroll
            for (int q = 0; q < 16; q++) rest[q] = t0 + q < n_tiles ? __ldcg(pr + t0 + q) : 0.0f;
  #pragma unroll
            for (int q = 0; q < 16; q++) if (t0 + q < n_tiles) sum = __fadd_rn(sum, rest[q]);
            for (int t = t0 + 16; t < n_tiles; t++) sum = __fadd_rn(sum, __ldcg(pr + t));
            r = rsqrtf(__fadd_rn(__fdiv_rn(sum, (float)prob_n), ne.eps));
          }
          ne.rstd[row] = r;
        }
      }
    }
    __syncthreads();
  };
#endif

#if A4_SILU
  // ---- C32h epilogue (compact loops: a fully unrolled 24-value SiLU / quant body made the kernel ~10 us slower,
  // i-cache pressure next to the co-resident CTA's main loop). Thread <-> s4 word mapping w = it * threads + tid:
  // row = w / 4, c = w % 4 -> y columns {4c..4c+3, 16+4c..16+4c+3} of the tile = the 8 nibbles of one A word.
  constexpr int sl_R = 16 * tmb;
  constexpr int sl_NIT = div_ceil(4 * sl_R, threads);
  static_assert(4 * sl_R % threads == 0, "C32h: 4 * 16 tmb words per tile in whole thread passes");
  static_assert(A4_SILU_TAG, "C32h compact epilogue: TAG protocol only");
  constexpr int sl_GS = 72;  // bf16 row stride of the gate/up tile in the idle pipeline smem (conflict-free stores)
  // y words (uint4 = 8 bf16 per s4 word): DEFER -> private smem after the pipeline, else after the g/u tile
  constexpr int sl_y_off = A4_SILU_DEFER ? sh_br_size + stages * (s_sh_stage + as_sh_stage + a_sh_stage)
                                         : sl_R * sl_GS * 2 / 16;  // int4
  int sl_pend = -1;
  bool sl_pf = false;  // DEFER 2: the pending tile's group PA entries prefetched into pa_sh
  (void)sl_pend;
  (void)sl_pf;
  float sl_gs;  // consumer input_global_scale, loaded at kernel start (not a dependent load in the epilogue)
  asm volatile("ld.global.nc.f32 %0, [%1];\n" : "=f"(sl_gs) : "l"(sl.gs));
  // pa_sh [4][R] u64 after the y words (DEFER 2)
  auto sl_pa_sh = [&]() {
    return reinterpret_cast<unsigned long long*>(sh + sl_y_off + 4 * sl_R);
  };
  auto sl_pf_issue = [&](int tile) {
    if constexpr (A4_SILU_DEFER == 2) {
      const unsigned long long* pt = reinterpret_cast<const unsigned long long*>(sl.PA) +
                                     (size_t)(tile & ~3) * sl.m_pad;
      if (threadIdx.x < sl_R) {
  #pragma unroll
        for (int q = 0; q < 4; q++) {
          const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(sl_pa_sh() + q * sl_R + threadIdx.x));
          asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" : : "r"(dst),
                       "l"(pt + (size_t)q * sl.m_pad + threadIdx.x));
        }
      }
      cp_async_fence();
    }
  };
  (void)sl_pf_issue;

  auto sl_quant = [&](int tile, bool pf = false) {
    if constexpr (A4_SILU_ABL & 256) {  // timing: quant code present, skipped at run time
      if (prob_k > 0) return;
    }
#if A4_SILU_PROF
    sl_ts(sl_ev++, 3);
#endif
    const int grp = tile >> 2, ch = tile & 3;
    const int kb = 2 * grp + (ch >> 1), kr = ch & 1;
    const uint4* ys = reinterpret_cast<const uint4*>(sh + sl_y_off);
    const unsigned long long* pt = reinterpret_cast<const unsigned long long*>(sl.PA) + (size_t)(tile & ~3) * sl.m_pad;
    __syncthreads();  // ys / pa_sh written by other threads
    // the group's 4 per-tile row maxima of every word's row, all loads in flight together
    unsigned long long v[sl_NIT][4];
  #pragma unroll
    for (int it = 0; it < sl_NIT; it++) {
      const int row = (it * threads + threadIdx.x) >> 2;
  #pragma unroll
      for (int q = 0; q < 4; q++) {
        if (A4_SILU_DEFER == 2 && pf) v[it][q] = sl_pa_sh()[q * sl_R + row];
        else asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];\n" : "=l"(v[it][q]) : "l"(pt + (size_t)q * sl.m_pad + row) : "memory");
      }
    }
    // stale entries (a partner had not published when they were read): re-read all of them in one batch
    bool stale = false;
  #pragma unroll
    for (int it = 0; it < sl_NIT; it++)
  #pragma unroll
      for (int q = 0; q < 4; q++) stale |= (uint32_t)(v[it][q] >> 32) != sl_ep;
    while (!(A4_SILU_ABL & 1) && stale) {
      stale = false;
  #pragma unroll
      for (int it = 0; it < sl_NIT; it++) {
        const int row = (it * threads + threadIdx.x) >> 2;
  #pragma unroll
        for (int q = 0; q < 4; q++)
          if ((uint32_t)(v[it][q] >> 32) != sl_ep)
            asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];\n" : "=l"(v[it][q]) : "l"(pt + (size_t)q * sl.m_pad + row) : "memory");
      }
  #pragma unroll
      for (int it = 0; it < sl_NIT; it++)
  #pragma unroll
        for (int q = 0; q < 4; q++) stale |= (uint32_t)(v[it][q] >> 32) != sl_ep;
    }
    float amv[sl_NIT];
  #pragma unroll
    for (int it = 0; it < sl_NIT; it++) {
      amv[it] = 0.0f;
  #pragma unroll
      for (int q = 0; q < 4; q++) amv[it] = fmaxf(amv[it], __uint_as_float((uint32_t)v[it][q]));
    }
#if A4_SILU_PROF
    sl_ts(sl_ev++, 4);
#endif
#if A4_SILU_UNR & 2
  #pragma unroll
#else
  #pragma unroll 1
#endif
    for (int it = 0; it < sl_NIT; it++) {
      const int w = it * threads + threadIdx.x, row = w >> 2, c = w & 3;
      float am = 0.0f;
  #pragma unroll
      for (int k = 0; k < sl_NIT; k++) am = k == it ? amv[k] : am;
      const float sc = __fmul_rn(fmaxf(am, 9.99999993922529029077850282192230224609375e-9f),
                                 0.142857149243354797363281250f);
      const float rc = sl_div_rn_ftz(1.0f, sc);
      const uint4 yw = ys[w];
      const __nv_bfloat16* yh = reinterpret_cast<const __nv_bfloat16*>(&yw);
      float qv[8];
      if constexpr (A4_SILU_FQ == 2) {
        // RN(y / sc) without the div.rn range-check branch: q0 = y * RN(1/sc) is within 1 ulp, and with the
        // reciprocal correctly rounded Markstein's theorem gives RN(q0 + (y - sc q0) RN(1/sc)) = RN(y / sc) for
        // normal operands / results; a quotient that is not normal is < 2^-126 -> rint 0 on both paths (.ftz
        // flushes a denormal y to 0 as div.rn.ftz does). sc >= 1e-8 / 7 is normal.
  #pragma unroll
        for (int k = 0; k < 8; k++) {
          const float y = __bfloat162float(yh[k]);
          float q0, r, q1;
          asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(q0) : "f"(y), "f"(rc));
          asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(-sc), "f"(q0), "f"(y));
          asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(q1) : "f"(r), "f"(rc), "f"(q0));
          qv[k] = rintf(q1);
        }
      } else {
      bool slow = false;
  #pragma unroll
      for (int k = 0; k < 8; k++) {
        float t;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(t) : "f"(__bfloat162float(yh[k])), "f"(rc));
        qv[k] = rintf(t);
        slow |= fabsf(__fsub_rn(t, qv[k])) > 0.4999f;
      }
      if (!A4_SILU_FQ || slow) {
  #pragma unroll 1
        for (int k = 0; k < 8; k++) {
          const float q = rintf(sl_div_rn_ftz(__bfloat162float(yh[k]), sc));
  #pragma unroll
          for (int k2 = 0; k2 < 8; k2++) qv[k2] = k2 == k ? q : qv[k2];
        }
      }
      }
      uint32_t word = 0u;
  #pragma unroll
      for (int k = 0; k < 8; k++)  // k = 2e + (high half): nibble k
        word |= ((uint32_t)((int)fminf(fmaxf(qv[k], -8.0f), 7.0f)) & 15u) << (4 * k);
      const int l = (row & 7) * 4 + c, r = ((row & 15) >> 3) + 2 * kr, mb = row >> 4;
      sl.apk[((size_t)(kb * tmb + mb) * 32 + l) * 4 + r] = word;
      if (ch == 0 && c == 0) sl.as[(size_t)grp * sl.m_pad + row] = __fmul_rn(sc, sl_gs);
    }
    __syncthreads();  // ys / pa_sh reusable
#if A4_SILU_PROF
    sl_ts(sl_ev++, 5);
#endif
  };

  // after the split-K gather: y = bf16(silu(g) u) -> ys, per-(tile, row) max |y| -> PA (tagged with the epoch)
  auto silu_epi = [&]() {
    static_assert(thread_n_blocks == 4, "C32h: tn = 64 (32 y columns per tile)");
    if constexpr (A4_SILU_ABL & 4) return;
    if constexpr (A4_SILU_DEFER == 2) {  // the previous tile (partners long done) before ys is reused
      if (sl_pend >= 0) {
        sl_quant(sl_pend, sl_pf);
        sl_pend = -1;
      }
    }
#if A4_SILU_PROF
    sl_ts(sl_ev++, 1);
#endif
    const int tile = slice_col;
    __nv_bfloat16* gu = reinterpret_cast<__nv_bfloat16*>(sh);
    __syncthreads();  // pipeline smem idle
    {
      const float* fc = reinterpret_cast<const float*>(&acc);
  #pragma unroll
      for (int i = 0; i < TMW; i++)
  #pragma unroll
        for (int j = 0; j < 2; j++)
  #pragma unroll
          for (int h = 0; h < 2; h++)
  #pragma unroll
            for (int hf = 0; hf < 2; hf++) {
              const int row = (mh * TMW + i) * 16 + lane / 4 + 8 * hf;
              const int col = 32 * nw + 8 * (2 * j + h) + 2 * (lane % 4);
              const float* cc = fc + ((i * 2 + j) * 2 + h) * 4 + 2 * hf;
              *reinterpret_cast<__nv_bfloat162*>(gu + row * sl_GS + col) = __floats2bfloat162_rn(cc[0], cc[1]);
            }
    }
    __syncthreads();
    uint4* ys = reinterpret_cast<uint4*>(sh + sl_y_off);
    unsigned long long* pt = reinterpret_cast<unsigned long long*>(sl.PA) + (size_t)tile * sl.m_pad;
#if A4_SILU_UNR & 1
  #pragma unroll
#else
  #pragma unroll 1
#endif
    for (int it = 0; it < sl_NIT; it++) {
      const int w = it * threads + threadIdx.x, row = w >> 2, c = w & 3;
      // y col 4c + e: g at GEMM col 16 (c / 2) + 4 (c % 2) + e, u at + 8; y col 16 + 4c + e: g at + 32, u at + 40
      const __nv_bfloat16* src = gu + row * sl_GS + 16 * (c >> 1) + 4 * (c & 1);
      const uint2 g0 = *reinterpret_cast<const uint2*>(src), u0 = *reinterpret_cast<const uint2*>(src + 8);
      const uint2 g1 = *reinterpret_cast<const uint2*>(src + 32), u1 = *reinterpret_cast<const uint2*>(src + 40);
      const __nv_bfloat16* gl = reinterpret_cast<const __nv_bfloat16*>(&g0);
      const __nv_bfloat16* ul = reinterpret_cast<const __nv_bfloat16*>(&u0);
      const __nv_bfloat16* gh = reinterpret_cast<const __nv_bfloat16*>(&g1);
      const __nv_bfloat16* uh = reinterpret_cast<const __nv_bfloat16*>(&u1);
      __nv_bfloat16 yb[8];
      float am = 0.0f;
  #pragma unroll
      for (int e = 0; e < 4; e++) {
        float y0, y1;
        if constexpr (A4_SILU_ABL & 16) {  // timing: no SiLU math
          y0 = __bfloat162float(gl[e]) * __bfloat162float(ul[e]);
          y1 = __bfloat162float(gh[e]) * __bfloat162float(uh[e]);
        } else {
          y0 = sl_y(__bfloat162float(gl[e]), __bfloat162float(ul[e]));
          y1 = sl_y(__bfloat162float(gh[e]), __bfloat162float(uh[e]));
        }
        if (row >= prob_m) y0 = y1 = 0.0f;
        yb[2 * e] = __float2bfloat16_rn(y0);  // exact (y is bf16-rounded)
        yb[2 * e + 1] = __float2bfloat16_rn(y1);
        am = fmaxf(am, fmaxf(fabsf(y0), fabsf(y1)));
      }
      ys[w] = *reinterpret_cast<const uint4*>(yb);
      am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
      am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
      if (c == 0) {
        const unsigned long long x = (unsigned long long)__float_as_uint(am) | ((unsigned long long)sl_ep << 32);
        asm volatile("st.relaxed.gpu.global.b64 [%0], %1;\n" : : "l"(pt + row), "l"(x) : "memory");
      }
    }
#if A4_SILU_PROF
    sl_ts(sl_ev++, 2);
#endif
    if constexpr (A4_SILU_DEFER) {
      sl_pend = tile;
      sl_pf = false;
    } else {
      sl_quant(tile);
    }
  };
#endif

  auto write_result = [&]() {
#if A4_SILU
    silu_epi();
    return;
#endif
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta = c_sh_stride * (threads / (2 * thread_n_blocks));
    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr = (4 * c_sh_stride) * (lane / 4) + lane % 4 + 16 * nw +
                  (mh * TMW) * 16 * (4 * c_sh_stride);
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));
    int c_gl_wr_end = c_gl_stride * prob_m;
    using c2 = nv_bfloat162;
    auto write = [&](int idx, float c0, float c1) {
      reinterpret_cast<c2*>(sh_red)[idx] =
          __halves2bfloat162(__float2bfloat16(c0), __float2bfloat16(c1));
    };
    __syncthreads();
    const float* fc = reinterpret_cast<const float*>(&acc);
  #pragma unroll
    for (int i = 0; i < TMW; i++) {
  #if A4_NE == 2
      // consumer build: row scale (x * 1.0f is exact: rs = ones leaves the result bit-identical)
      const int rr = (mh * TMW + i) * 16 + lane / 4;
      const float rs0 = ne.rs[rr], rs1 = ne.rs[rr + 8];
  #else
      constexpr float rs0 = 1.0f, rs1 = 1.0f;
  #endif
  #pragma unroll
      for (int j = 0; j < 2; j++) {
        const float* c0 = fc + ((i * 2 + j) * 2 + 0) * 4;
        const float* c1 = fc + ((i * 2 + j) * 2 + 1) * 4;
        int wr = c_sh_wr + 8 * j;
        if constexpr (A4_NE == 2) {
          write(wr + (4 * c_sh_stride) * 0 + 0, c0[0] * rs0, c0[1] * rs0);
          write(wr + (4 * c_sh_stride) * 8 + 0, c0[2] * rs1, c0[3] * rs1);
          write(wr + (4 * c_sh_stride) * 0 + 4, c1[0] * rs0, c1[1] * rs0);
          write(wr + (4 * c_sh_stride) * 8 + 4, c1[2] * rs1, c1[3] * rs1);
        } else {
          write(wr + (4 * c_sh_stride) * 0 + 0, c0[0], c0[1]);
          write(wr + (4 * c_sh_stride) * 8 + 0, c0[2], c0[3]);
          write(wr + (4 * c_sh_stride) * 0 + 4, c1[0], c1[1]);
          write(wr + (4 * c_sh_stride) * 8 + 4, c1[2], c1[3]);
        }
      }
      c_sh_wr += 16 * (4 * c_sh_stride);
    }
    __syncthreads();
  #pragma unroll
    for (int i = 0; i < div_ceil(16 * tmb, threads / (2 * thread_n_blocks)); i++) {
      if (c_gl_wr < c_gl_wr_end) {
        C[c_gl_wr] = sh_red[c_sh_rd];
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
    __syncthreads();
  #if A4_NE == 1
    // producer build: the norm epilogue of this tile runs after the CTA's last slice (kernel end)
    if (ne_t0 < 0) ne_t0 = slice_col;
    else if (ne_t1 < 0) ne_t1 = slice_col;
    else if (ne_t2 < 0) ne_t2 = slice_col;
    else ne_epilogue(slice_col);
  #endif
  };

#if A4_FF
  // FF-port (LEDGER W3-trim-ff): staged first pipeline fill at kernel start only. A4_FF=1: __nanosleep(A4_FF_NS)
  // after stage 0; A4_FF=2: after every prologue stage but the last. Fetch timing only -> bit-identical.
  bool ff_first = true;
  auto start_pipes = [&]() {
  #pragma unroll
    for (int i = 0; i < stages - 1; i++) {
      if ((A4_FF == 1 ? i == 1 : i > 0) && ff_first) __nanosleep(A4_FF_NS);
      fetch_to_shared(i, i, i < slice_iters);
    }
    ff_first = false;
    zero_accums();
#else
  auto start_pipes = [&]() {
  #pragma unroll
    for (int i = 0; i < stages - 1; i++) fetch_to_shared(i, i, i < slice_iters);
    zero_accums();
#endif
    wait_for_stage();
    fetch_to_registers(0, 0);
    fetch_scales_to_registers(0, 0);
    a_gl_rd += a_sh_stage * (stages - 1);
    as_gl_rd += as_sh_stage * (stages - 1);
    if constexpr (kPtrInc) {
      a_p += a_sh_stage * (stages - 1);
      as_p += as_sh_stage * (stages - 1);
    }
  };
  int pend_lock = -1;  // C32c: deferred split-K release (A4_C32C_DEFREL / A4_PAR_REDUCE=6)
  (void)pend_lock;
  if (slice_iters) start_pipes();

  while (slice_iters) {
  #if A4_PIPE_UNROLL1 == 1
  #pragma unroll 1
  #elif A4_PIPE_UNROLL1 == 2
  #pragma unroll 2
  #elif A4_PIPE_UNROLL1 == 3
  #pragma unroll 3
  #else
  #pragma unroll
  #endif
    for (int pipe = 0; pipe < stages;) {
  #pragma unroll
      for (int k = 0; k < KS; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        if constexpr (!(A4_SLATE && !DEFER)) fetch_scales_to_registers(k + 1, pipe);
        if (k == KS - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);
          pipe++;
          wait_for_stage();
        }
        const int tile = pipe - (k >= KS - 2 ? 1 : 0);
        matmul_a4(k, tile);
        if constexpr (A4_SLATE && !DEFER) {
          if (k == KS - 1) fetch_scales_to_registers(0, pipe);  // stage `pipe` = next tile, valid since this k=0 barrier
        }
        if constexpr (DEFER) last_gp = tile % 2;
      }
      slice_iters--;
      if (slice_iters == 0) break;
    }
    a_gl_rd += a_sh_stage * stages;
    as_gl_rd += as_sh_stage * stages;
    if constexpr (kPtrInc) {
      a_p += a_sh_stage * stages;
      as_p += as_sh_stage * stages;
    }

    if (slice_iters == 0) {
#if A4_SILU && A4_SILU_PROF
      sl_ts(sl_ev++, 6 + (slice_idx == slice_count - 1 ? 1 : 0));
#endif
      flush_fold();
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      if constexpr (A4_PAR_REDUCE == 7) {
        // C32c consumer-sum (GEMM side): every slice stores its fp32 partial to P[slice_idx][row][col]
        // (P = C_tmp, slot stride 16 tmb x prob_n); no lock chain, no bf16 write. The consumer sums the
        // column tile's slots 0..cnt-1 in slice order (= the serial chain's fp32 order: bit-identical).
        float* P = reinterpret_cast<float*>(C_tmp) + (size_t)slice_idx * 128 * prob_n;  // slot stride 128 rows
        const float* fc = reinterpret_cast<const float*>(&acc);
  #pragma unroll
        for (int i = 0; i < TMW; i++)
  #pragma unroll
          for (int j = 0; j < 2; j++)
  #pragma unroll
            for (int h = 0; h < 2; h++)
  #pragma unroll
              for (int g = 0; g < 4; g += 2) {
                const int row = (mh * TMW + i) * 16 + lane / 4 + (g >= 2 ? 8 : 0);
                const int col = slice_col * thread_n_blocks * 16 + 32 * nw + 8 * (2 * j + h) + 2 * (lane % 4);
                const int e = ((i * 2 + j) * 2 + h) * 4 + g;
                if (row < prob_m) __stcg(reinterpret_cast<float2*>(P + (size_t)row * prob_n + col), make_float2(fc[e], fc[e + 1]));
              }
        last = false;
      } else
      if (slice_count > 1) {
        if constexpr (A4_ABL_NOREDUCE) {
          // no reduction: nothing (the `last` slice writes its partial as the result)
        } else if constexpr (A4_OVLCHAIN) {
          constexpr int c_sz = tmb * 16 * thread_n_blocks * 16 * sizeof(float) / 16;
          constexpr int th_size = TMW * 16 * sizeof(float) / 16;
          constexpr int MS = A4_OVL_MAXS;
          constexpr int smem_total = sh_br_size + stages * (s_sh_stage + as_sh_stage + a_sh_stage);
          if (slice_count > MS || (slice_count - 1) * c_sz > smem_total) {
            barrier_acquire(&locks[locks_off], slice_idx);
            global_reduce_fp32(slice_idx == 0, last);
            barrier_release(&locks[locks_off], last);
          } else {
            int4* slots = C_tmp + (size_t)locks_off * MS * c_sz;
            if (!last) {
              const int4* fc4 = reinterpret_cast<const int4*>(&acc);
              int4* dst = slots + (size_t)slice_idx * c_sz;
  #pragma unroll
              for (int k = 0; k < th_size; k++) __stcg(&dst[threads * k + threadIdx.x], fc4[k]);
              __syncthreads();
              if (threadIdx.x == 0) {
                asm volatile("fence.acq_rel.gpu;\n");
                asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(&locks[locks_off]), "r"(1));
              }
            } else {
              if (threadIdx.x == 0) {
                int state = -1;
                do
                  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(&locks[locks_off]));
                while (state != slice_count - 1);
              }
              __syncthreads();
              const int nsl = slice_count - 1;
              int4* S = sh;  // whole dynamic smem is idle after the main loop
  #pragma unroll 1
              for (int sl = 0; sl < nsl; sl++) {
  #pragma unroll
                for (int k = 0; k < th_size; k++)
                  cp_async4(&S[(sl * th_size + k) * threads + threadIdx.x],
                            &slots[(size_t)sl * c_sz + threads * k + threadIdx.x]);
              }
              cp_async_fence();
              cp_async_wait<0>();
              float* fc = reinterpret_cast<float*>(&acc);
  #pragma unroll
              for (int k = 0; k < th_size; k++) {
                float4 t = *reinterpret_cast<float4*>(&S[k * threads + threadIdx.x]);
  #pragma unroll 1
                for (int sl = 1; sl < nsl; sl++) {
                  float4 u = *reinterpret_cast<float4*>(&S[(sl * th_size + k) * threads + threadIdx.x]);
                  t.x += u.x; t.y += u.y; t.z += u.z; t.w += u.w;
                }
                fc[k * 4 + 0] += t.x; fc[k * 4 + 1] += t.y; fc[k * 4 + 2] += t.z; fc[k * 4 + 3] += t.w;
              }
              __syncthreads();  // S aliases sh_red (write_result)
              if (threadIdx.x == 0) locks[locks_off] = 0;
            }
          }
        } else if constexpr (A4_PAR_REDUCE == 2) {
          // store the fp32 partial to slot (locks_off, slice_idx); a4_reduce_kernel sums the slots
          constexpr int c_size = tmb * 16 * thread_n_blocks * 16 * sizeof(float) / 16;
          constexpr int th_size = TMW * 16 * sizeof(float) / 16;
          int4* slot = C_tmp + (size_t)(locks_off * par_ms + slice_idx) * c_size;
          const int4* fc4 = reinterpret_cast<const int4*>(&acc);
  #pragma unroll
          for (int k = 0; k < th_size; k++) slot[threads * k + threadIdx.x] = fc4[k];
          last = false;
        } else if constexpr (A4_PAR_REDUCE == 3) {
          constexpr int NF = TMW * 16;
          constexpr int c_sz = tmb * 16 * thread_n_blocks * 16 * sizeof(float) / 16;
          float tmp[NF];
  #pragma unroll
          for (int i = 0; i < NF; i++) tmp[i] = reinterpret_cast<float*>(&acc)[i];
          last = a4_par_reduce_ni(tmp, NF, C_tmp, locks, locks_off, slice_idx, slice_count, par_ms,
                                  c_sz, threads);
          if (last) {
  #pragma unroll
            for (int i = 0; i < NF; i++) reinterpret_cast<float*>(&acc)[i] = tmp[i];
          }
        } else if constexpr (A4_PAR_REDUCE == 4) {
          // C32c: rounding-level split-K combine with fp32 atomics. Every slice adds its partial into the
          // column's zeroed fp32 slot (red.global.add.f32, fire-and-forget), then takes a ticket; the last
          // arriver reads the total (own partial included), re-zeroes the slot and writes the result.
          // Summation order = L2 arrival order (nondeterministic at the fp32 rounding level).
          constexpr int c_size = tmb * 16 * thread_n_blocks * 16 * sizeof(float) / 16;
          constexpr int th_size = TMW * 16 * sizeof(float) / 16;
          float* slot = reinterpret_cast<float*>(C_tmp + (size_t)locks_off * c_size);
          const float* fs = reinterpret_cast<const float*>(&acc);
  #pragma unroll
          for (int k = 0; k < th_size; k++)
  #pragma unroll
            for (int f = 0; f < 4; f++)
              asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;\n"
                           : : "l"(slot + 4 * (threads * k + threadIdx.x) + f), "f"(fs[k * 4 + f]) : "memory");
          __syncthreads();
          if (threadIdx.x == 0) {
            int t;
            asm volatile("fence.acq_rel.gpu;\n");
            asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\n" : "=r"(t) : "l"(&locks[locks_off]) : "memory");
            sh_last = (t == slice_count - 1);
          }
          __syncthreads();
          last = sh_last;
          if (last) {
            float* fc = reinterpret_cast<float*>(&acc);
            int4* s4 = reinterpret_cast<int4*>(slot);
  #pragma unroll
            for (int k = 0; k < th_size; k++) {
              int4 v = __ldcg(&s4[threads * k + threadIdx.x]);
              fc[k * 4 + 0] = __int_as_float(v.x); fc[k * 4 + 1] = __int_as_float(v.y);
              fc[k * 4 + 2] = __int_as_float(v.z); fc[k * 4 + 3] = __int_as_float(v.w);
            }
  #pragma unroll
            for (int k = 0; k < th_size; k++) __stcg(&s4[threads * k + threadIdx.x], make_int4(0, 0, 0, 0));
            if (threadIdx.x == 0) locks[locks_off] = 0;
          }
        } else if constexpr (A4_PAR_REDUCE == 5) {
          // C32c: rounding-level ticket combine. Every slice stores its partial to its own slot and takes a
          // ticket; the last arriver adds the OTHER slots (slot order) to its own registers. No serial chain;
          // the sum order starts at the last arriver (nondeterministic at the fp32 rounding level).
          constexpr int c_size = tmb * 16 * thread_n_blocks * 16 * sizeof(float) / 16;
          constexpr int th_size = TMW * 16 * sizeof(float) / 16;
          int4* base = C_tmp + (size_t)(locks_off * par_ms) * c_size;
          const int4* fc4 = reinterpret_cast<const int4*>(&acc);
  #pragma unroll
          for (int k = 0; k < th_size; k++) __stcg(&base[(size_t)slice_idx * c_size + threads * k + threadIdx.x], fc4[k]);
          __syncthreads();
          if (threadIdx.x == 0) {
            int t;
            asm volatile("fence.acq_rel.gpu;\n");
            asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\n" : "=r"(t) : "l"(&locks[locks_off]) : "memory");
            sh_last = (t == slice_count - 1);
          }
          __syncthreads();
          last = sh_last;
          if (last) {
            float* fc = reinterpret_cast<float*>(&acc);
  #pragma unroll 1
            for (int s = 0; s < slice_count; s++) {
              if (s == slice_idx) continue;
              int4 v[th_size];
  #pragma unroll
              for (int k = 0; k < th_size; k++) v[k] = __ldcg(&base[(size_t)s * c_size + threads * k + threadIdx.x]);
  #pragma unroll
              for (int k = 0; k < th_size; k++) {
                fc[k * 4 + 0] += __int_as_float(v[k].x); fc[k * 4 + 1] += __int_as_float(v[k].y);
                fc[k * 4 + 2] += __int_as_float(v[k].z); fc[k * 4 + 3] += __int_as_float(v[k].w);
              }
            }
            if (threadIdx.x == 0) locks[locks_off] = 0;
          }
        } else if constexpr (A4_PAR_REDUCE == 1) {
          last = par_reduce();
        } else if constexpr (A4_PAR_REDUCE == 8) {
          // C32d end-hop atomics (rounding-level, nondeterministic order): non-last slices red.add.f32 their
          // partial into the column's zeroed fp32 tile (slot 0) and release; the fixed last slice waits for
          // slice_count - 1 arrivals, adds the tile to its registers and re-zeroes it.
          constexpr int c_size = tmb * 16 * thread_n_blocks * 16 * sizeof(float) / 16;
          constexpr int th_size = TMW * 16 * sizeof(float) / 16;
          int4* tile = C_tmp + (size_t)(locks_off * par_ms) * c_size;
          if (!last) {
            const float* fs = reinterpret_cast<const float*>(&acc);
  #pragma unroll
            for (int k = 0; k < th_size; k++)
  #pragma unroll
              for (int f = 0; f < 4; f++)
                asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;\n"
                             : : "l"(reinterpret_cast<float*>(&tile[threads * k + threadIdx.x]) + f), "f"(fs[k * 4 + f])
                             : "memory");
            barrier_release(&locks[locks_off], false);
          } else {
            barrier_acquire(&locks[locks_off], slice_count - 1);
            float* fc = reinterpret_cast<float*>(&acc);
            int4 v[th_size];
  #pragma unroll
            for (int k = 0; k < th_size; k++) v[k] = __ldcg(&tile[threads * k + threadIdx.x]);
  #pragma unroll
            for (int k = 0; k < th_size; k++) {
              fc[k * 4 + 0] += __int_as_float(v[k].x); fc[k * 4 + 1] += __int_as_float(v[k].y);
              fc[k * 4 + 2] += __int_as_float(v[k].z); fc[k * 4 + 3] += __int_as_float(v[k].w);
              __stcg(&tile[threads * k + threadIdx.x], make_int4(0, 0, 0, 0));
            }
            barrier_release(&locks[locks_off], true);
          }
        } else if constexpr (A4_PAR_REDUCE == 6) {
          // C32c gather (rounding-level, deterministic): non-last slices store their partial to their own
          // slot (locks_off, slice_idx) and release (slice 0's release deferred to the CTA's next reduce /
          // kernel end); the last slice waits for all of them and adds the slots in slice order to its own
          // registers. One hop instead of the serial chain.
          constexpr int c_size = tmb * 16 * thread_n_blocks * 16 * sizeof(float) / 16;
          constexpr int th_size = TMW * 16 * sizeof(float) / 16;
          int4* base = C_tmp + (size_t)(locks_off * par_ms) * c_size;
          if (pend_lock >= 0) {
            barrier_release(&locks[pend_lock], false);
            pend_lock = -1;
          }
          if (!last) {
            const float* fs = reinterpret_cast<const float*>(&acc);
            if constexpr (A4_C32C_PB != 0) {
              // C32d: half-width partial (8 values per int4); PB=2 also stores the thread's pow2 descale
              int4* dst = base + (size_t)slice_idx * c_size;
              float sc = 1.f;
              if constexpr (A4_C32C_PB == 2) {
                float am = 0.f;
  #pragma unroll
                for (int i = 0; i < th_size * 4; i++) am = fmaxf(am, fabsf(fs[i]));
                int e;
                frexpf(am, &e);  // am < 2^e
                e = max(-100, min(e, 100));
                sc = ldexpf(1.f, 15 - e);
                __stcg(reinterpret_cast<float*>(dst + (th_size / 2) * threads) + threadIdx.x, ldexpf(1.f, e - 15));
              }
  #pragma unroll
              for (int k2 = 0; k2 < th_size / 2; k2++) {
                uint32_t w[4];
  #pragma unroll
                for (int j = 0; j < 4; j++) {
                  const float a = fs[8 * k2 + 2 * j] * sc, b = fs[8 * k2 + 2 * j + 1] * sc;
                  if constexpr (A4_C32C_PB == 1) {
                    __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
                    w[j] = *reinterpret_cast<uint32_t*>(&h);
                  } else {
                    __half2 h = __floats2half2_rn(a, b);
                    w[j] = *reinterpret_cast<uint32_t*>(&h);
                  }
                }
                __stcg(&dst[threads * k2 + threadIdx.x], make_int4(w[0], w[1], w[2], w[3]));
              }
            } else if (A4_C32C_VST) {
              const int4* fc4 = reinterpret_cast<const int4*>(&acc);
  #pragma unroll
              for (int k = 0; k < th_size; k++) __stcg(&base[(size_t)slice_idx * c_size + threads * k + threadIdx.x], fc4[k]);
            } else if (!(A4_C32C_ABL & 2)) {
  #pragma unroll
            for (int k = 0; k < th_size; k++) {
              float* dst = reinterpret_cast<float*>(&base[(size_t)slice_idx * c_size + threads * k + threadIdx.x]);
  #pragma unroll
              for (int f = 0; f < 4; f++) dst[f] = fs[k * 4 + f];
            }
            }
            if (A4_C32C_DEFREL && slice_idx == 0) pend_lock = locks_off;
            else if (A4_C32C_ABL & 1) {  // timing: release without the fence
              __syncthreads();
              if (threadIdx.x == 0) asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(&locks[locks_off]), "r"(1));
            } else barrier_release(&locks[locks_off], false);
          } else if (A4_C32C_ABL & 4) {  // timing: last slice neither waits nor reads
            if (threadIdx.x == 0) atomicExch(&locks[locks_off], 0);
          } else {
            barrier_acquire(&locks[locks_off], slice_count - 1);
            float* fc = reinterpret_cast<float*>(&acc);
            if constexpr (A4_C32C_PB != 0) {
              constexpr int H = th_size / 2;
              constexpr int NB = A4_C32C_PB2 ? 2 : 1;
              auto addh = [&](const int4& v, int k2, float d) {
                const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
  #pragma unroll
                for (int j = 0; j < 4; j++) {
                  float a, b;
                  if constexpr (A4_C32C_PB == 1) {
                    a = __uint_as_float(w[j] << 16);
                    b = __uint_as_float(w[j] & 0xffff0000u);
                  } else {
                    const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&w[j]));
                    a = f2.x * d;
                    b = f2.y * d;
                  }
                  fc[8 * k2 + 2 * j] += a;
                  fc[8 * k2 + 2 * j + 1] += b;
                }
              };
  #pragma unroll 1
              for (int s = 0; s < slice_count - 1; s += NB) {
                const int nb = min(NB, slice_count - 1 - s);
                int4 v[NB][H];
                float d[NB];
  #pragma unroll
                for (int q = 0; q < NB; q++) {
                  if (q < nb) {
                    const int4* src = base + (size_t)(s + q) * c_size;
  #pragma unroll
                    for (int k2 = 0; k2 < H; k2++) v[q][k2] = __ldcg(&src[threads * k2 + threadIdx.x]);
                    d[q] = A4_C32C_PB == 2 ? __ldcg(reinterpret_cast<const float*>(src + H * threads) + threadIdx.x) : 1.f;
                  }
                }
  #pragma unroll
                for (int q = 0; q < NB; q++)
                  if (q < nb) {
  #pragma unroll
                    for (int k2 = 0; k2 < H; k2++) addh(v[q][k2], k2, d[q]);
                  }
              }
            } else {
  #pragma unroll 1
            for (int s = 0; s < slice_count - 1; s++) {
              int4 v[th_size];
  #pragma unroll
              for (int k = 0; k < th_size; k++) v[k] = __ldcg(&base[(size_t)s * c_size + threads * k + threadIdx.x]);
  #pragma unroll
              for (int k = 0; k < th_size; k++) {
                fc[k * 4 + 0] += __int_as_float(v[k].x); fc[k * 4 + 1] += __int_as_float(v[k].y);
                fc[k * 4 + 2] += __int_as_float(v[k].z); fc[k * 4 + 3] += __int_as_float(v[k].w);
              }
            }
            }
            barrier_release(&locks[locks_off], true);
          }
        } else {
          // C32c DEFREL: slice 0's release deferred (bit-identical: same data, same order; release before
          // this CTA's next acquire or at kernel end, so no wait cycle)
          if (A4_C32C_DEFREL && pend_lock >= 0 && slice_idx > 0) {
            barrier_release(&locks[pend_lock], false);
            pend_lock = -1;
          }
          // FF-port s0: the chain's first slice; its per-column lock is 0 by construction (part-2 locks)
          if (kS0 && slice_idx == 0 && part2_mn_tiles < gridDim.x) __syncthreads(); else
          barrier_acquire(&locks[locks_off], slice_idx);
          global_reduce_fp32(slice_idx == 0, last);
          if (A4_C32C_DEFREL && slice_idx == 0 && !last) {
            if (pend_lock >= 0) barrier_release(&locks[pend_lock], false);
            pend_lock = locks_off;
          } else
          barrier_release(&locks[locks_off], last);
        }
      }
      if (last && !A4_ABL_NOEPI) write_result();
#if A4_SILU
      else if constexpr (A4_SILU_DEFER == 2) {
        if (sl_pend >= 0) {
          sl_quant(sl_pend, sl_pf);
          sl_pend = -1;
        }
      }
#endif
      slice_row = 0;
      if (!in_part2) {
        slice_col_par += gridDim.x;
      } else {
        slice_col_par++;
        slice_col++;
      }
      init_slice();
      if (slice_iters) {
        a_gl_rd = a_sh_stage * slice_row;
        as_gl_rd = as_sh_stage * slice_row;
        b_gl_rd = b_sh_stride * slice_col + b_gl_rd_delta_o * slice_row;
        s_gl_rd = s_gl_stride * slice_row + s_sh_stride * slice_col + threadIdx.x;
        set_ptrs();
#if A4_SILU
        if constexpr (A4_SILU_DEFER == 2) {
          if (sl_pend >= 0 && !sl_pf) {  // completes inside start_pipes' first stage wait
            sl_pf_issue(sl_pend);
            sl_pf = true;
          }
        }
#endif
        start_pipes();
      }
#if A4_SILU
      if constexpr (A4_SILU_DEFER == 1) {
        if (sl_pend >= 0) {  // the next slice's first stages are in flight
          sl_quant(sl_pend);
          sl_pend = -1;
        }
      }
#endif
    }
  }
  if (pend_lock >= 0) barrier_release(&locks[pend_lock], false);
#if A4_SILU
  if constexpr (A4_SILU_DEFER == 2) {
    if (sl_pend >= 0) sl_quant(sl_pend, sl_pf);
  }
#endif
#if A4_SILU && A4_SILU_TAG
  if (threadIdx.x == 0) sl.cnt[blockIdx.x] = (int)sl_ep;
#endif
#if A4_SILU && A4_SILU_PROF
  sl_ts(63);
#endif
  #if A4_NE == 3
  // grid barrier (every CTA is resident: grid <= occupancy * SMs, host-checked), then the row norm
  if ((A4_NE_ABL & 16) && ne.eps < 1e30f) return;  // timing: GB code present, never run
  NePre ne_pre;
  ne_gb_prefetch(ne, prob_m, prob_n, ne_pre);
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(&ne.cnt[0], 1);
    int v;
    do {
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(v) : "l"(&ne.cnt[0]));
    } while (v < (int)gridDim.x);
    if (atomicAdd(&ne.cnt[1], 1) == (int)gridDim.x - 1) {  // last to leave resets both counters
      ne.cnt[1] = 0;
      ne.cnt[0] = 0;
    }
  }
  __syncthreads();
  if (A4_NE_ABL & 1) return;  // timing: barrier only
  ne_gb_rownorm<threads>(ne, reinterpret_cast<const __nv_bfloat16*>(C0), prob_m, prob_n,
                         reinterpret_cast<float*>(sh), ne_pre);
  #endif
  #if A4_NE == 1
  if (ne_t0 >= 0) ne_epilogue(ne_t0);
  if (ne_t1 >= 0) ne_epilogue(ne_t1);
  if (ne_t2 >= 0) ne_epilogue(ne_t2);
  #endif
}

// A4_PAR_REDUCE=2 second pass: one CTA per split column; sums its slice_count partial slots in
// slice order (same fp32 order as the serial chain) and writes bf16. Column list from the host.
// Slot layout: thread t of the GEMM CTA wrote int4 k at [threads*k + t]; the mapping from (t, k) to
// (row, col) is the GEMM's register layout, so the reducer re-derives it the same way:
// acc[i][j][h][g]: i = m-block within the warp's TMW, (j,h) = n8 tile, g: row lane/4 (+8 for g>=2),
// col 8*(2j+h) + 2*(lane%4) + (g%2).
template <int threads, int thread_m_blocks, int thread_n_blocks, int MW>
__global__ void A4Reduce(const int4* __restrict__ C_tmp, int4* __restrict__ C, int prob_m, int prob_n,
                         const int* __restrict__ cols, int par_ms) {
  constexpr int tmb = thread_m_blocks, TMW = tmb / MW, tb_n_warps = thread_n_blocks / 2;
  constexpr int c_size = tmb * 16 * thread_n_blocks * 16 * sizeof(float) / 16;
  constexpr int th_size = TMW * 16 * sizeof(float) / 16;
  const int lock = cols[3 * blockIdx.x], slice_col = cols[3 * blockIdx.x + 1], cnt = cols[3 * blockIdx.x + 2];
  const int t = threadIdx.x, warp = t / 32, lane = t % 32, nw = warp % tb_n_warps, mh = warp / tb_n_warps;
  float acc[th_size * 4];
  #pragma unroll
  for (int i = 0; i < th_size * 4; i++) acc[i] = 0.f;
  const int4* base = C_tmp + (size_t)(lock * par_ms) * c_size;
  for (int sidx = 0; sidx < cnt; sidx++) {
  #pragma unroll
    for (int k = 0; k < th_size; k++) {
      int4 v = __ldcg(&base[(size_t)sidx * c_size + threads * k + t]);
      const float* f = reinterpret_cast<const float*>(&v);
  #pragma unroll
      for (int q = 0; q < 4; q++) acc[k * 4 + q] += f[q];
    }
  }
  __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(C);
  #pragma unroll
  for (int i = 0; i < TMW; i++) {
  #pragma unroll
    for (int j = 0; j < 2; j++) {
  #pragma unroll
      for (int h = 0; h < 2; h++) {
  #pragma unroll
        for (int g = 0; g < 4; g += 2) {
          const int row = (mh * TMW + i) * 16 + lane / 4 + (g >= 2 ? 8 : 0);
          const int col = slice_col * thread_n_blocks * 16 + 32 * nw + 8 * (2 * j + h) + 2 * (lane % 4);
          const int e = ((i * 2 + j) * 2 + h) * 4 + g;
          if (row < prob_m)
            *reinterpret_cast<__nv_bfloat162*>(out + (size_t)row * prob_n + col) =
                __floats2bfloat162_rn(acc[e], acc[e + 1]);
        }
      }
    }
  }
}

#else
template <int threads, int thread_m_blocks, int thread_n_blocks, int stages,
          int MW, int DEFER, int MINB>
__global__ void MarlinA4(MARLIN_KERNEL_PARAMS A4_XPARAMS) {}
template <int threads, int thread_m_blocks, int thread_n_blocks, int MW>
__global__ void A4Reduce(const int4*, int4*, int, int, const int*, int) {}
#endif

}  // namespace MARLIN_NAMESPACE_NAME
