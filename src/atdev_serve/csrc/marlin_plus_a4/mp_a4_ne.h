// C32-NE (A4_NE=1): residual add + Gemma RMSNorm + s4 quant of the NEXT linear's input folded into the
// W4A4 GEMM epilogue (producer: out_proj / o_proj / down_proj, N = 5120), and the row scale rstd applied
// to the fp32 accumulator of the consumer (qkvz / qkv / gate_up). The consumer is linear in its rows:
//   (diag(rstd) X diag(1+w)) W = diag(rstd) ((X diag(1+w)) W)
// so the producer quantises a = x (1+w) per (row, 128-group) without knowing rstd (one 128-group = two
// tn=64 column tiles: the second finisher of a pair quantises both halves from the fp32 scratch S), and
// writes the per-tile partial sums of x^2 to P; the last pair finisher turns P into rstd[row].
// Passed by value as a trailing kernel parameter (A4_XPARAMS) only in A4_NE builds.
#pragma once
#include <cstdint>
#include <cuda_bf16.h>

namespace MARLIN_NAMESPACE_NAME {
struct A4Ne {
  // consumer: fp32 accumulator rows scaled by rs[row] before the bf16 rounding (nullptr: none)
  const float* rs;
  // producer (S != nullptr): x = bf16(acc) + r  ->  ro = bf16(x), a = x (1 + w)
  float* S;                  // [n_tiles][16 tmb][tn] fp32 a (pair scratch)
  float* P;                  // [16 tmb][n_tiles] per-tile sum of x^2
  int* cnt;                  // [n_tiles / 2 + 1] pair tickets + one global ticket, zero between calls
  const __nv_bfloat16* r;    // residual in  [M][ldr]
  __nv_bfloat16* ro;         // residual out [M][ldro]
  __nv_bfloat16* xb;         // optional bf16(a) out [M][ldxb] (GDN in_proj_ba input), nullptr: none
  const __nv_bfloat16* w;    // Gemma norm weight (scale = 1 + w)
  uint32_t* apk;             // consumer s4 fragments ([K/64][tmb_c][32][4] words)
  float* as;                 // consumer per-(group, row) scales [K/128][m_pad]
  const float* gs;           // consumer input_global_scale
  float* rstd;               // out [m_pad]: rsqrt(sum x^2 / N + eps), 0 for rows >= M
  int ldr, ldro, ldxb;
  int m_pad;                 // 16 * tmb_c
  int tmb_c;
  float eps;
};
}  // namespace MARLIN_NAMESPACE_NAME

namespace MARLIN_NAMESPACE_NAME {
// C32h (A4_SILU=1 build): the gate_up GEMM (weights interleaved per n8 tile at load: GEMM n8 tile 2p = gate
// columns 8p..8p+7, 2p+1 = up columns 8p..8p+7) writes y = bf16(silu(g) * u) quantised to s4 per (row, 128-group)
// straight into the down GEMM's A buffers. A 64-column tile holds 32 y columns = one k32 chunk of the down
// input; the 4 tiles of a 128-group publish per-(row, warp half) |y| maxima to PA and a ticket (cnt[grp]),
// wait for the group, then quantise their own chunk. cnt[ngrp + grp] counts finished tiles (last one resets).
struct A4Silu {
  uint32_t* apk;   // down s4 fragments [K_c/64][tmb][32][4] words
  float* as;       // down per-(group, row) scales [K_c/128][m_pad]
  const float* gs; // down input_global_scale
  float* PA;       // [n_tiles * 2][m_pad] per-(tile, warp half, row) max |y|
  int* cnt;        // [2 * n_tiles / 4], zero between calls
  int m_pad;       // 16 tmb (== the down GEMM's 16 tmb)
  unsigned long long* dbg;  // A4_SILU_PROF: per-CTA globaltimer stamps [grid][64], nullptr otherwise
};
}  // namespace MARLIN_NAMESPACE_NAME

#ifndef A4_NE
  #define A4_NE 0
#endif
#ifndef A4_SILU
  #define A4_SILU 0
#endif
#ifndef A4_SILU_TAG  // C32h: PA entries = (max |y|, call epoch) 64-bit pairs, no tickets (cnt = per-CTA epochs)
  #define A4_SILU_TAG 1
#endif
#ifndef A4_SILU_DEFER  // C32h: quantise a tile after the next slice's pipeline prologue (y kept in extra smem)
  #define A4_SILU_DEFER 0
#endif
#if A4_NE && A4_SILU
  #error "A4_NE and A4_SILU are separate builds"
#endif
#if A4_NE
  #define A4_XPARAMS , const MARLIN_NAMESPACE_NAME::A4Ne ne
  #define A4_XARGS , ne
#elif A4_SILU
  #define A4_XPARAMS , const MARLIN_NAMESPACE_NAME::A4Silu sl
  #define A4_XARGS , sl
#else
  #define A4_XPARAMS
  #define A4_XARGS
#endif
