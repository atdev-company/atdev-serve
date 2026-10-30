"""E6 x bet #2: marlin_c1 (E6 fused norm / SiLU epilogues, csrc/marlin_c1) with the W3 3-bit planar weight
stream of marlin_plus_w3 (docs/research/bet2-w3.md), so c1_fuse can run on w3_mixed layers.

The kernel template is *generated* into csrc/marlin_c1/gen_w3/m<mode>/ from csrc/marlin_c1/marlin_template.h
by the same anchored patches marlin_plus_w3_build applies to the marlin_plus template (A1-A6 B-stream patches,
+ the W3_FASTDQ dequant patches when ATDEV_W3_DEFINES has W3_FASTDQ=1, exactly as the unfused W3 kernel).
The kernel symbols keep their marlin_c1 names (own extension .so, loaded RTLD_LOCAL). The host (mp_host.cu)
is copied with two patches: the smem model counts the 3-bit B ring, and gemm_epi checks the planar B shape.
B = int32 [K/16, 3, N/2] (w3_mixed.pack_w3), scales = the unchanged Marlin-permuted bf16 [K/128, N].

Kernel versions (ATDEV_C1_W3_V, default 2 with W3_FASTDQ=1, else 1): 1 = the B-stream + FASTDQ patches above;
2 = + DQ2_PATCHES (shift-free dequant, bit-identical to 1 at the same tile config); 3 (opt-in) = + DQ3_PATCHES
(offset dequant + fp32 group scale, more accurate than 1 / 2, rounding-level different; + tile 256,128,128). One extension per (E6 mode,
FASTDQ, version); the int4 marlin_c1 extensions and their sources are untouched.
"""
from __future__ import annotations

import hashlib
import os

from atdev_serve import marlin_c1_build as c1b
from atdev_serve import marlin_plus_w3_build as w3b

HERE = os.path.dirname(os.path.abspath(__file__))
C1 = os.path.join(HERE, "csrc", "marlin_c1")
GEN_ROOT = os.path.join(C1, "gen_w3")

# (thread_k, thread_n, threads) x stages, tmb 1 / m16 only (the fused epilogues need 8 < M <= 16)
# (64, 64, 128) is NOT valid with the W3 B stream (wrong results; marlin_plus_w3 never instantiates it either)
TILES = [(64, 128, 128), (128, 128, 256), (256, 128, 256), (128, 64, 128), (256, 64, 128), (128, 128, 128),
         (64, 256, 256), (128, 256, 256)]
STAGES = (4, 6, 8)
# V3 only (own extension): 8 k16 slabs per warp and stage = one whole 128-group per warp (fold once per 8 k-steps)
V3_TILES = [(256, 128, 128)]

HOST_PATCHES = [
    ("  int sh_b = stages * (tb_k * tb_n / 8) * 4;\n",
     "  int sh_b = stages * (tb_k * tb_n * 3 / 32) * 4;  // W3: 3-bit planar B ring\n"),
    ('  TORCH_CHECK(cfg.size() == 6 && cfg[0] > 0, "mp_gemm_epi: explicit config required");\n',
     '  TORCH_CHECK(cfg.size() == 6 && cfg[0] > 0, "mp_gemm_epi: explicit config required");\n'
     '  TORCH_CHECK(b_q.scalar_type() == at::kInt && b_q.is_contiguous() &&\n'
     '              b_q.numel() == size_k / 16 * 3 * (size_n / 2), "mp_gemm_epi (W3): B must be planar [K/16, 3, N/2]");\n'),
]


def _apply(src: str, patches) -> str:
    for old, new in patches:
        n = src.count(old)
        assert n == 1, f"marlin_c1 W3 patch anchor matched {n} times:\n{old[:160]}"
        src = src.replace(old, new)
    return src


def version() -> int:
    """ATDEV_C1_W3_V: 1 = the original W3 B stream (A1-A6 [+ FASTDQ]); 2 = + DQ2_PATCHES (bit-identical to 1).
    2 needs ATDEV_W3_DEFINES=W3_FASTDQ=1; without FASTDQ the default is 1."""
    env = os.environ.get("ATDEV_C1_W3_V", "")
    if env:
        return int(env)
    return DEFAULT_VERSION if w3b.fastdq() else 1


DEFAULT_VERSION = 2


# V2 (bit-identical to V1): the FASTDQ dequant reads the odd nibble pair of a word in place. Bits 4-6 / 20-22 under
# the bf16 exponent 2^3 (0x4100 | u << 4) are exactly 8 + u, so (x - 12) = u - 4 like (128 + u) - 132 for the even
# pair: the two `q >>= 4` shifts per word go away (15 -> 13 instructions per 8 weights), values bit-identical.
DQ2_PATCHES = [
    ("""      static constexpr uint32_t MASK = 0x00070007, EX = 0x43004300, SUB = 0x43044304;
      int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
      q >>= 4;
      int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
      frag_b_ptr[0] = __hsub2(*reinterpret_cast<nv_bfloat162*>(&lo), *reinterpret_cast<const nv_bfloat162*>(&SUB));
      frag_b_ptr[1] = __hsub2(*reinterpret_cast<nv_bfloat162*>(&hi), *reinterpret_cast<const nv_bfloat162*>(&SUB));
""", """      // W3 V2: odd nibble pair in place (bits 4-6 under exponent 2^3 = 8 + u, minus 12), no shift
      static constexpr uint32_t MASK = 0x00070007, EX = 0x43004300, SUB = 0x43044304;
      static constexpr uint32_t MASK_HI = 0x00700070, EX_HI = 0x41004100, SUB_HI = 0x41404140;
      int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
      int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK_HI, EX_HI);
      frag_b_ptr[0] = __hsub2(*reinterpret_cast<nv_bfloat162*>(&lo), *reinterpret_cast<const nv_bfloat162*>(&SUB));
      frag_b_ptr[1] = __hsub2(*reinterpret_cast<nv_bfloat162*>(&hi), *reinterpret_cast<const nv_bfloat162*>(&SUB_HI));
"""),
]

# V3 (opt-in ATDEV_C1_W3_V=3, NOT bit-identical: more accurate): offset dequant + fp32 group scale (ported from
# csrc/w3_sk/w3_sk.cu). The codes go to the mma as the exact bf16 values 128 + u (k rows 0-7 of a k16 slab) /
# 8 + u (k rows 8-15) -- one LOP3 per 2 weights, no HSUB2, no bf16 scale HMUL2. Per 128-group and warp the products
# accumulate in fp32 (frag_c_tmp, C = 0 at the group's first k-step), one extra mma per k-step on the constant B
# (132 | 12) gives Z = the zero-point row sums, and at the group's last k-step frag_c += s (tmp - Z) in fp32 with
# the unrounded bf16 scale s of the C fragment's own columns (shuffled from the lanes that own those B columns).
# (u - 4) s is exact instead of bf16(bf16(u - 4) s) as in V1 / V2.
DQ3_TEMPLATE_HELPERS = [
    ("// Multiply dequantized values by the corresponding quantization scale; used\n",
     "// W3 V3: m16n8k16 bf16 mma, fp32 accumulate, C = 0 when zc (starts a group accumulator)\n"
     "__device__ inline void w3v3_mma(const uint32_t* a, uint32_t b0, uint32_t b1, float* c, bool zc) {\n"
     "  if (zc) {\n"
     "    const float z = 0.f;\n"
     "    MP_ASM_PURE(\n"
     "        \"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, \"\n"
     "        \"{%10,%10,%10,%10};\\n\"\n"
     "        : \"=f\"(c[0]), \"=f\"(c[1]), \"=f\"(c[2]), \"=f\"(c[3])\n"
     "        : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"r\"(b0), \"r\"(b1), \"f\"(z));\n"
     "  } else {\n"
     "    MP_ASM_PURE(\n"
     "        \"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, \"\n"
     "        \"{%10,%11,%12,%13};\\n\"\n"
     "        : \"=f\"(c[0]), \"=f\"(c[1]), \"=f\"(c[2]), \"=f\"(c[3])\n"
     "        : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"r\"(b0), \"r\"(b1),\n"
     "          \"f\"(c[0]), \"f\"(c[1]), \"f\"(c[2]), \"f\"(c[3]));\n"
     "  }\n"
     "}\n\n"
     "// Multiply dequantized values by the corresponding quantization scale; used\n"),
]
DQ3_PATCHES = [
    # dequant: raw offset values, no subtract
    ("""      frag_b_ptr[0] = __hsub2(*reinterpret_cast<nv_bfloat162*>(&lo), *reinterpret_cast<const nv_bfloat162*>(&SUB));
      frag_b_ptr[1] = __hsub2(*reinterpret_cast<nv_bfloat162*>(&hi), *reinterpret_cast<const nv_bfloat162*>(&SUB_HI));
""", """      // W3 V3: offset values 128 + u / 8 + u (exact), the zero point is removed per group in fp32
      (void)SUB; (void)SUB_HI;
      frag_b_ptr[0] = *reinterpret_cast<nv_bfloat162*>(&lo);
      frag_b_ptr[1] = *reinterpret_cast<nv_bfloat162*>(&hi);
"""),
    # matmul: no bf16 scale; group accumulators + Z mma + fp32 fold at the group's last k-step
    ("""      } else if constexpr (group_blocks != -1 && !is_a_8bit) {
        scale<a_type_id>(frag_b0, frag_s[k2][j], 0);
        scale<a_type_id>(frag_b1, frag_s[k2][j], 1);
      }

  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        if constexpr (m_block_size_8) {
          mma_trans<a_type_id, use_fp16_accum>(frag_a[k2][i], frag_b0, frag_b1,
                                               frag_c[i][j][0]);
        } else {
          mma<a_type_id, use_fp16_accum>(frag_a[k2][i], frag_b0,
                                         frag_c[i][j][0]);
          mma<a_type_id, use_fp16_accum>(frag_a[k2][i], frag_b1,
                                         frag_c[i][j][1]);
        }
      }
    }
  };
""", """      }
      // W3 V3: group accumulators (C = 0 at the group's first k-step of this warp)
      static_assert(!m_block_size_8 && !use_fp16_accum && group_blocks == 8, "W3 V3: m16, fp32 accum, g128");
  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        const uint32_t* a_ = reinterpret_cast<const uint32_t*>(&frag_a[k2][i]);
        const uint32_t* b0_ = reinterpret_cast<const uint32_t*>(&frag_b0);
        const uint32_t* b1_ = reinterpret_cast<const uint32_t*>(&frag_b1);
        w3v3_mma(a_, b0_[0], b0_[1], reinterpret_cast<float*>(&frag_c_tmp[i][j][0]), v3_first);
        w3v3_mma(a_, b1_[0], b1_[1], reinterpret_cast<float*>(&frag_c_tmp[i][j][1]), v3_first);
      }
    }
    // Z = sum_k a * (132 | 12): the offset row sums of this k-step (constant B, same for every column)
  #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      w3v3_mma(reinterpret_cast<const uint32_t*>(&frag_a[k2][i]), 0x43044304u, 0x41404140u, v3_z[i], v3_first);
    if (v3_last) {
      // C columns of this lane: 2 (lane % 4) + {0, 1} of every n8 tile; their scales sit in the B-column owners
      // lanes 8 (lane % 4) and 8 (lane % 4) + 4 (frag_s word j, half h = tile (j, h))
      const int src0 = (threadIdx.x % 4) * 8;
      uint32_t s0w[4], s1w[4];
  #pragma unroll
      for (int w = 0; w < 4; w++) {
        const uint32_t v = reinterpret_cast<const uint32_t*>(&frag_s[k2][0])[w];
        s0w[w] = __shfl_sync(0xffffffffu, v, src0);
        s1w[w] = __shfl_sync(0xffffffffu, v, src0 + 4);
      }
  #pragma unroll
      for (int j = 0; j < 4; j++) {
  #pragma unroll
        for (int h = 0; h < 2; h++) {
          const float sc0 = __uint_as_float(h ? (s0w[j] & 0xffff0000u) : (s0w[j] << 16));
          const float sc1 = __uint_as_float(h ? (s1w[j] & 0xffff0000u) : (s1w[j] << 16));
  #pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
            float* c_ = reinterpret_cast<float*>(&frag_c[i][j][h]);
            const float* t_ = reinterpret_cast<const float*>(&frag_c_tmp[i][j][h]);
            c_[0] = fmaf(sc0, t_[0] - v3_z[i][0], c_[0]);
            c_[1] = fmaf(sc1, t_[1] - v3_z[i][1], c_[1]);
            c_[2] = fmaf(sc0, t_[2] - v3_z[i][2], c_[2]);
            c_[3] = fmaf(sc1, t_[3] - v3_z[i][3], c_[3]);
          }
        }
      }
    }
  };
"""),
    # group position flags of the k-step (compile-time after unrolling)
    ("""  // We have the m dimension as the inner loop in order to encourage overlapping
  // dequantization and matmul operations.
""", """    // W3 V3: first / last k-step of this warp inside a 128-group (slices start and end group-aligned)
    constexpr int v3_g = div_ceil(group_blocks, thread_k_blocks);
    static_assert(thread_k_blocks < group_blocks || group_blocks % b_sh_wr_iters == 0, "W3 V3: warp k-range");
    const bool v3_first = k == 0 && pipe % v3_g == 0;
    const bool v3_last = k == b_sh_wr_iters - 1 && pipe % v3_g == v3_g - 1;
  // We have the m dimension as the inner loop in order to encourage overlapping
  // dequantization and matmul operations.
"""),
    ("  FragC frag_c_tmp[thread_m_blocks][is_a_8bit ? 2 : 4][2];\n",
     "  FragC frag_c_tmp[thread_m_blocks][is_a_8bit ? 2 : 4][2];\n"
     "  float v3_z[thread_m_blocks][4];  // W3 V3: zero-point row sums of the current group\n"),
]


def xfeat() -> tuple:
    """ATDEV_C1W3_X: comma list of opt-in kernel variant features (own gen dir / extension; default: none, the
    default template text and extension hash are unchanged). prof = per-CTA timeline (dev only, writes into the
    unused tail of c_tmp)."""
    fs = tuple(sorted({f for f in os.environ.get("ATDEV_C1W3_X", "").split(",") if f}))
    bad = [f for f in fs if f not in XFEATS]
    assert not bad, f"ATDEV_C1W3_X: unknown features {bad} (known: {sorted(XFEATS)})"
    return fs


# per-CTA timeline: u64 slots at C_tmp + 16 MB, 16 per CTA: 0 globaltimer start, 1 globaltimer end, 2 clock start,
# 3 clock after the first pipeline fill, then per segment s < 2: 4+3s main-loop end, 5+3s reduction/write done,
# 6+3s next fill done; 10 clock at body entry, 11 clock before the first fill, 12 first stage landed; 13 smid, 14 #segments, 15 clock end
TPROF_OFF_INT4 = 1 << 20
PROF_PATCHES = [
    ("  extern __shared__ int4 sh[];\n", "  extern __shared__ int4 sh[];\n  const long long tp_entry = clock64();\n"),
    ("  if constexpr (E6_MODE >= 2 && E6_RSTD_AT == 1) {\n",
     "  if (threadIdx.x == 0) tp_[11] = clock64();\n  if constexpr (E6_MODE >= 2 && E6_RSTD_AT == 1) {\n"),
    ("\n  init_slice();\n\n",
     "\n  init_slice();\n"
     "  unsigned long long* tp_ = reinterpret_cast<unsigned long long*>(C_tmp + (1 << 20)) + blockIdx.x * 16;\n"
     "  int tp_seg = 0;\n"
     "  if (threadIdx.x == 0) {\n"
     "    unsigned long long g_; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(g_)); tp_[0] = g_;\n"
     "    tp_[2] = clock64();\n"
     "    for (int q_ = 3; q_ < 16; q_++) tp_[q_] = 0;\n"
     "    tp_[10] = tp_entry;\n"
     "    unsigned s_; asm volatile(\"mov.u32 %0, %%smid;\" : \"=r\"(s_)); tp_[13] = s_;\n"
     "  }\n\n"),
    ("    zero_accums();\n    wait_for_stage();\n    fetch_to_registers(0, 0);\n",
     "    zero_accums();\n    wait_for_stage();\n    if (threadIdx.x == 0 && tp_[12] == 0) tp_[12] = clock64();\n"
     "    fetch_to_registers(0, 0);\n"),
    ("  // Main loop.\n  while (slice_iters) {\n",
     "  if (threadIdx.x == 0) tp_[3] = clock64();\n"
     "  // Main loop.\n  while (slice_iters) {\n"),
    ("    if (slice_iters == 0) {\n      // convert fp16 accum to fp32 for reduction\n",
     "    if (slice_iters == 0) {\n"
     "      if (threadIdx.x == 0 && tp_seg < 2) tp_[4 + 3 * tp_seg] = clock64();\n"
     "      // convert fp16 accum to fp32 for reduction\n"),
    ("      slice_row = 0;\n      if (!in_part2) {\n",
     "      if (threadIdx.x == 0 && tp_seg < 2) tp_[5 + 3 * tp_seg] = clock64();\n"
     "      slice_row = 0;\n      if (!in_part2) {\n"),
    ("        start_pipes();\n      }\n    }\n  }\n}\n",
     "        start_pipes();\n"
     "        if (threadIdx.x == 0 && tp_seg < 2) tp_[6 + 3 * tp_seg] = clock64();\n"
     "      }\n"
     "      tp_seg++;\n"
     "    }\n  }\n"
     "  if (threadIdx.x == 0) {\n"
     "    tp_[15] = clock64(); tp_[14] = tp_seg;\n"
     "    unsigned long long g_; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(g_)); tp_[1] = g_;\n"
     "  }\n}\n"),
]

# xs = cross-slice prefetch from the slice-end phase: the split-K reduction / output staging region (sh_red) gets
# its own smem after the A ring (instead of aliasing the B ring), so a CTA that continues with a next column slice
# issues that slice's first stages-1 cp.async stages right after its main loop ends, BEFORE the intra-CTA reduce,
# the split-K lock wait, the global partial round trip and the fused epilogue; the refill latency overlaps them.
# Main loop untouched (no new live registers there); arithmetic and reduction order unchanged -> bit-identical.
XS_PATCHES = [
    ("  int4* sh_a = sh_s + sh_s_size;\n",
     "  int4* sh_a = sh_s + sh_s_size;\n"
     "  // XS: own reduction / staging region after the A ring (the B / A / s rings survive a slice end)\n"
     "  constexpr int xs_red_size = sh_red_size > 4 * threads ? sh_red_size : 4 * threads;\n"
     "  const bool xs_ok = (int)((sh_a + stages * a_sh_stage + xs_red_size - sh) * 16) <= max_shared_mem;\n"
     "  if (xs_ok) sh_red = sh_a + stages * a_sh_stage;\n"),
    ("      cp_async_wait<0>();\n      bool last = slice_idx == slice_count - 1;\n",
     "      cp_async_wait<0>();\n      bool last = slice_idx == slice_count - 1;\n"
     "      // XS: start the next column slice's pipeline fill now (part 2 slices: slice_row 0, slice_col + 1)\n"
     "      static_assert(!has_zp, \"XS: no zero points\");\n"
     "      int xs_iters = 0;\n"
     "      if (xs_ok && in_part2 && slice_col + 1 < n_tiles && slice_col_par + 1 < part2_mn_tiles) {\n"
     "        xs_iters = iters * (blockIdx.x + 1) - k_tiles * (slice_col_par + 1);\n"
     "        if (xs_iters > k_tiles) xs_iters = k_tiles;\n"
     "        if (xs_iters < 0) xs_iters = 0;\n"
     "      }\n"
     "      if (xs_iters > 0) {\n"
     "        __syncthreads();  // every warp is done reading the finished slice's ring\n"
     "        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);\n"
     "        b_gl_rd = (b3_words / 4) * (slice_col + 1);\n"
     "        if constexpr (group_blocks == -1 || group_blocks >= thread_k_blocks) {\n"
     "          s_gl_rd = s_sh_stride * (slice_col + 1) + threadIdx.x;\n"
     "        } else {\n"
     "          s_gl_rd = s_gl_stride * (threadIdx.x / s_sh_stride) + s_sh_stride * (slice_col + 1) +\n"
     "                    threadIdx.x % s_sh_stride;\n"
     "        }\n"
     "  #pragma unroll\n"
     "        for (int i = 0; i < stages - 1; i++) fetch_to_shared(i, i, i < xs_iters);\n"
     "      }\n"),
    ("      if (slice_iters) {\n        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +\n",
     "      if (slice_iters && xs_iters > 0) {\n"
     "        if (xs_iters != slice_iters) __trap();  // XS slice prediction must match init_slice\n"
     "        zero_accums();\n"
     "        wait_for_stage();\n"
     "        fetch_to_registers(0, 0);\n"
     "        fetch_scales_to_registers(0, 0);\n"
     "        fetch_zp_to_registers(0, 0);\n"
     "        a_gl_rd += a_gl_rd_delta_o * (stages - 1);\n"
     "        /*XS_FILL_DONE*/\n"
     "      } else if (slice_iters) {\n        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +\n"),
]
# prof + pp: per-CTA count of PP hits in the high half of slot 13 (smid = slot 13 & 0xffff)
PROF_PP = [("  int tp_seg = 0;\n", "  int tp_seg = 0, tp_pp = 0;\n"),
           ("        pp_hit = false;\n", "        if (threadIdx.x == 0 && pp_hit) tp_pp++;\n        pp_hit = false;\n"),
           ("    tp_[15] = clock64(); tp_[14] = tp_seg;\n",
            "    tp_[15] = clock64(); tp_[14] = tp_seg; tp_[13] |= (unsigned long long)tp_pp << 16;\n")]
PROF_XS = [("        /*XS_FILL_DONE*/\n", "        if (threadIdx.x == 0 && tp_seg < 2) tp_[6 + 3 * tp_seg] = clock64();\n")]

# grid = host only: an optional 7th tile-config entry = launch grid (< sms * blocks_per_sm, all CTAs stay
# co-resident). The kernel derives its split-K partition from gridDim, so a smaller grid = longer per-CTA k-ranges
# = other slice ends (e.g. none on the CTAs that would own two slices): NOT bit-identical to the 6-entry config
# (other fp32 partial-sum split points; rounding-level), tuned per shape by scripts/bench_w3_fixed.py --tune-grid.
GRID_HOST_PATCHES = [
    ('  TORCH_CHECK(cfg.size() == 6 && cfg[0] > 0, "mp_gemm_epi: explicit config required");\n',
     '  TORCH_CHECK((cfg.size() == 6 || cfg.size() == 7) && cfg[0] > 0, "mp_gemm_epi: explicit config required");\n'),
    ("  TORCH_CHECK(cfg.size() == 6);\n", "  TORCH_CHECK(cfg.size() == 6 || cfg.size() == 7);\n"),
    ("    int grid = sms * ch_bps;\n",
     "    int grid = sms * ch_bps;\n"
     "    if (cfg.size() == 7 && cfg[6] > 0) {  // W3 grid override (all CTAs co-resident)\n"
     "      TORCH_CHECK(cfg[6] <= grid, \"mp_gemm_epi: grid \", cfg[6], \" > sms * blocks_per_sm \", grid);\n"
     "      grid = cfg[6];\n"
     "    }\n"),
]

# l2pf = next-linear L2 prefetch: every CTA, when it is done with its own work (tail of the kernel, where the SM
# idles anyway), issues prefetch.global.L2::evict_last for the first nx_pf k-stages (W3 B planes + scales) that the
# CTA with the same index will load first in the NEXT linear (part-2 partition of its (N, K, tile, grid) recomputed
# here), so the next kernel's pipeline fill hits L2 instead of paying the loaded-DRAM latency at its start. The
# next linear is announced on the host right before the launch (ext.set_next(...), one-shot) and baked into the
# kernel params -> CUDA-graph capturable. Pure hint: no arithmetic / ordering change -> bit-identical.
RO_LINE = "  float* Rout;  // consumer: optional rstd[16] out (written by CTA 0) for a second consumer of the same A\n"
L2PF_KERNEL_H = [
    (RO_LINE, RO_LINE +
     "  const char* nx_B;  // L2PF: next linear's W3 planes [K/16, 3, N/2] int32 (nullptr = no prefetch)\n"
     "  const char* nx_S;  // L2PF: next linear's scales [K/128, N] bf16\n"
     "  int nx_n, nx_k, nx_tk, nx_tn, nx_grid, nx_pf;\n"),
]
L2PF_PATCHES = [
    ("}\n\n#define MP_TEMPLATE_HEAD",
     "  // L2PF: warm L2 with the next linear's first stages for the CTA of the same index (part-2 partitions)\n"
     "  if (epi.nx_B != nullptr) {\n"
     "    const int nkt = epi.nx_k / epi.nx_tk, nnt = epi.nx_n / epi.nx_tn, ng = epi.nx_grid;\n"
     "    const int tkb = epi.nx_tk / 16;\n"
     "    int it = div_ceil(nkt * nnt, ng);\n"
     "    if (8 >= tkb) { const int q = 8 / tkb; it = q * div_ceil(it, q); }  // g128 rounding (group_blocks 8)\n"
     "    const int lpp = epi.nx_tn * 2 / 128;  // 128-B lines per (k16 slab, plane) of a tile column\n"
     "    const size_t row_b = (size_t)epi.nx_n * 2;  // bytes per plane row / per scale row\n"
     "    if (nnt <= ng) {\n"
     "      for (int b = blockIdx.x; b < ng; b += gridDim.x) {\n"
     "        const int lin = it * b;\n"
     "        if (lin >= nkt * nnt) break;\n"
     "        const int col = lin / nkt, row = lin % nkt;\n"
     "        int nst = nkt - row;\n"
     "        if (nst > it) nst = it;\n"
     "        if (nst > epi.nx_pf) nst = epi.nx_pf;\n"
     "        const int s0 = row * tkb, ns = nst * tkb, nl = ns * 3 * lpp;\n"
     "        const char* bb = epi.nx_B + (size_t)col * epi.nx_tn * 2;\n"
     "        for (int i = threadIdx.x; i < nl; i += blockDim.x) {\n"
     "          const int l = i % lpp, sp = i / lpp;\n"
     "          const char* ad = bb + ((size_t)(s0 + sp / 3) * 3 + sp % 3) * row_b + l * 128;\n"
     "          asm volatile(\"prefetch.global.L2::evict_last [%0];\\n\" ::\"l\"(ad));\n"
     "        }\n"
     "        const int g0 = s0 / 8, ngr = (s0 + ns - 1) / 8 - g0 + 1;\n"
     "        const char* sb = epi.nx_S + (size_t)col * epi.nx_tn * 2;\n"
     "        for (int i = threadIdx.x; i < ngr * lpp; i += blockDim.x) {\n"
     "          const char* ad = sb + (size_t)(g0 + i / lpp) * row_b + (i % lpp) * 128;\n"
     "          asm volatile(\"prefetch.global.L2::evict_last [%0];\\n\" ::\"l\"(ad));\n"
     "        }\n"
     "      }\n"
     "    }\n"
     "  }\n"
     "}\n\n#define MP_TEMPLATE_HEAD"),
]
L2PF_HOST_PATCHES = [
    ("  MpEpi epi{};\n  epi.mode = (int)mode;\n",
     "  MpEpi epi{};\n  epi.mode = (int)mode;\n"
     "  epi.nx_B = g_nx.nx_B; epi.nx_S = g_nx.nx_S; epi.nx_n = g_nx.nx_n; epi.nx_k = g_nx.nx_k;\n"
     "  epi.nx_tk = g_nx.nx_tk; epi.nx_tn = g_nx.nx_tn; epi.nx_grid = g_nx.nx_grid; epi.nx_pf = g_nx.nx_pf;\n"
     "  g_nx = MpEpi{};  // one-shot\n"),
    ("torch::Tensor mp_gemm_epi(torch::Tensor a,",
     "// L2PF: the next linear (one-shot, consumed by the next gemm_epi launch)\n"
     "static MpEpi g_nx{};\n"
     "void mp_set_next(torch::Tensor b_q, torch::Tensor b_s, int64_t size_n, int64_t size_k, std::vector<int64_t> cfg,\n"
     "                 int64_t pf_stages) {\n"
     "  TORCH_CHECK(cfg.size() >= 6 && cfg[0] > 0 && b_q.is_contiguous() && b_s.is_contiguous());\n"
     "  int sms = 0;\n"
     "  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, b_q.get_device());\n"
     "  g_nx = MpEpi{};\n"
     "  g_nx.nx_B = (const char*)b_q.data_ptr(); g_nx.nx_S = (const char*)b_s.data_ptr();\n"
     "  g_nx.nx_n = (int)size_n; g_nx.nx_k = (int)size_k; g_nx.nx_tk = (int)cfg[0]; g_nx.nx_tn = (int)cfg[1];\n"
     "  g_nx.nx_grid = (cfg.size() > 6 && cfg[6] > 0) ? (int)cfg[6] : sms * (int)(cfg[4] > 0 ? cfg[4] : 1);\n"
     "  g_nx.nx_pf = (int)pf_stages;\n"
     "}\n\n"
     "torch::Tensor mp_gemm_epi(torch::Tensor a,"),
    ('  m.def("instantiated"',
     '  m.def("set_next", &mp_set_next, "L2PF: announce the next linear", py::arg("b_q"), py::arg("b_s"),\n'
     '        py::arg("size_n"), py::arg("size_k"), py::arg("cfg"), py::arg("pf_stages") = 3);\n'
     '  m.def("instantiated"'),
]

# pc = host-precomputed split-K partition: the launch-uniform partition values (iters, part-2 / part-1 tiles) and
# 40-bit magic multipliers for the per-CTA divisions by k_tiles / iters / n_tiles come in through MpEpi, replacing
# the ~8 dependent runtime integer divisions (I2F/MUFU.RCP/F2I chains) of the kernel prologue (~1.5K cycles at
# every kernel start, measured with prof). Same integers (exact for the asserted ranges) -> bit-identical.
PC_KERNEL_H = [
    (RO_LINE, RO_LINE +
     "  int pc_iters, pc_p2, pc_p1;  // PC: 0 = compute in-kernel\n"
     "  unsigned long long pc_mk, pc_mi, pc_mn;  // PC: ceil(2^40 / d) for d = k_tiles, iters, n_tiles\n"),
]
PC_PATCHES = [
    ("  if (global_mn_tiles > gridDim.x) {\n    part2_mn_tiles = global_mn_tiles % gridDim.x;\n",
     "  // PC: host-precomputed partition (same integers)\n"
     "  const bool pc_on = epi.pc_iters > 0 && parallel == 1;\n"
     "  auto pc_div = [&](int x, unsigned long long mg, int d) -> int {\n"
     "    return pc_on ? (int)(((unsigned long long)(unsigned)x * mg) >> 40) : x / d;\n"
     "  };\n"
     "  int iters;\n"
     "  if (pc_on) {\n"
     "    part2_mn_tiles = epi.pc_p2;\n"
     "    part1_mn_iters = epi.pc_p1;\n"
     "    iters = epi.pc_iters;\n"
     "  } else {\n"
     "  if (global_mn_tiles > gridDim.x) {\n    part2_mn_tiles = global_mn_tiles % gridDim.x;\n"),
    ("  int iters = div_ceil(k_tiles * part2_mn_tiles, gridDim.x);\n",
     "  iters = div_ceil(k_tiles * part2_mn_tiles, gridDim.x);\n"),
    ("              div_ceil(iters, (group_blocks / thread_k_blocks));\n    }\n  }\n",
     "              div_ceil(iters, (group_blocks / thread_k_blocks));\n    }\n  }\n  }  // PC\n"),
    ("    locks_off = (iters * blockIdx.x) / k_tiles - 1;\n",
     "    locks_off = pc_div(iters * blockIdx.x, epi.pc_mk, k_tiles) - 1;\n"),
    ("    int col_first = iters * div_ceil(k_tiles * slice_col_par, iters);\n",
     "    int col_first = iters * pc_div(k_tiles * slice_col_par + iters - 1, epi.pc_mi, iters);\n"),
    ("      slice_count = div_ceil(k_tiles - col_off, iters);\n",
     "      slice_count = pc_div(k_tiles - col_off + iters - 1, epi.pc_mi, iters);\n"),
    ("        slice_idx = slice_count - 1 - delta_first / iters;\n",
     "        slice_idx = slice_count - 1 - pc_div(delta_first, epi.pc_mi, iters);\n"),
    ("      slice_col_par = (iters * blockIdx.x) / k_tiles;\n"
     "      slice_row = (iters * blockIdx.x) % k_tiles;\n"
     "      slice_col = (slice_col_par + global_mn_tiles - part2_mn_tiles) % n_tiles;\n"
     "      par_id = (slice_col_par + global_mn_tiles - part2_mn_tiles) / n_tiles;\n",
     "      slice_col_par = pc_div(iters * blockIdx.x, epi.pc_mk, k_tiles);\n"
     "      slice_row = iters * blockIdx.x - slice_col_par * k_tiles;\n"
     "      {\n"
     "        const int t_ = slice_col_par + global_mn_tiles - part2_mn_tiles;\n"
     "        par_id = pc_div(t_, epi.pc_mn, n_tiles);\n"
     "        slice_col = t_ - par_id * n_tiles;\n"
     "      }\n"),
]
PC_HOST_PATCHES = [
    ("    kern<<<grid, c.threads, smem, stream>>>(",
     "    MpEpi epi_l = epi;\n"
     "    {  // PC: host-precomputed partition (must mirror the kernel's part-1 / part-2 math exactly)\n"
     "      const int kt = prob_k / c.thread_k, nt = prob_n / c.thread_n, tkb = c.thread_k / 16;\n"
     "      const int gm = nt;  // parallel == 1 (m_split <= 16 * tmb)\n"
     "      int p2 = gm, p1 = 0;\n"
     "      if (gm > grid) {\n"
     "        p2 = gm % grid;\n"
     "        if (p2 * 3 <= grid) p2 += grid;\n"
     "        p1 = (gm - p2) / grid;\n"
     "      }\n"
     "      int it = (kt * p2 + grid - 1) / grid;\n"
     "      if (group_blocks >= tkb) {\n"
     "        const int q = group_blocks / tkb;\n"
     "        it = q * ((it + q - 1) / q);\n"
     "      }\n"
     "      auto mg = [](long long d) { return (unsigned long long)(((1ll << 40) + d - 1) / d); };\n"
     "      auto exact = [](long long xmax, long long d) { return xmax * d * d < (1ll << 40); };\n"
     "      const long long xk = (long long)it * (grid + 1), xi = (long long)kt * (p2 + 1) + it, xn = 2ll * gm + grid;\n"
     "      if (m_split <= 16 * tmb && !a8 && it > 0 && exact(xk, kt) && exact(xi, it) && exact(xn, nt)) {\n"
     "        epi_l.pc_iters = it; epi_l.pc_p2 = p2; epi_l.pc_p1 = p1;\n"
     "        epi_l.pc_mk = mg(kt); epi_l.pc_mi = mg(it); epi_l.pc_mn = mg(nt);\n"
     "      }\n"
     "    }\n"
     "    kern<<<grid, c.threads, smem, stream>>>("),
    ("bar_ptr, gs_ptr, epi);", "bar_ptr, gs_ptr, epi_l);"),
]

# LL4 ta = tail-aware split points (opt-in, needs pc; rounding-level, NOT bit-identical when enabled by a cfg): the
# part-2 linear tile space gets a virtual gap of e quarter k-tiles after every column (cfg[7] = e > 0; cfg[6] = grid
# override as with `grid`, 0 = sms * bps). CTA b owns the virtual range [V b, V (b + 1)) in quarter-tile units
# (Lv = 4 k_tiles + e per column), mapped to real k-tile boundaries by rounding to the nearest tile (positions in the
# gap -> start of the next column). A CTA whose range crosses a column end (2 segments = an extra slice end, block
# reduce, split-K round trip, epilogue and pipeline refill: the 1.3-2.3 us idle tail of LL3-ovl) therefore gets
# about e / 4 fewer real k-tiles, and the fractional V uses every CTA of the grid. Column ownership stays closed-form:
# owner(col c, row r) = (c Lv + 4 r + 1) / V -> slice_count / slice_idx (bottom-to-top, same serial lock chain).
# The host simulates the partition (monotone, no empty CTA before the end) and leaves ta off otherwise; with e = 0 /
# no 8th entry the kernel takes the original pc path (same integers -> bit-identical to pc).
TA_KERNEL_H = [
    (RO_LINE, RO_LINE +
     "  int ta_V, ta_Lv;  // LL4 TA: quarter-tile units per CTA / per column (0 = off)\n"
     "  unsigned long long ta_mv, ta_ml;  // LL4 TA: ceil(2^40 / d) for d = ta_V, ta_Lv\n"),
]
TA_PATCHES = [
    ("  }  // PC\n",
     "  }  // PC\n"
     "  // LL4 TA: tail-aware split points (host-validated partition; ta_V = 0 -> original iters * blockIdx.x)\n"
     "  const bool ta_on = pc_on && epi.ta_V > 0;\n"
     "  auto ta_map = [&](int xv) -> int {\n"
     "    const int c_ = pc_div(xv, epi.ta_ml, epi.ta_Lv);\n"
     "    const int row_ = (xv - c_ * epi.ta_Lv + 2) >> 2;\n"
     "    return row_ >= k_tiles ? (c_ + 1) * k_tiles : c_ * k_tiles + row_;\n"
     "  };\n"
     "  const int ta_s0 = ta_on ? ta_map(epi.ta_V * blockIdx.x) : iters * blockIdx.x;\n"
     "  const int ta_s1 = ta_on ? ta_map(epi.ta_V * (blockIdx.x + 1)) : iters * (blockIdx.x + 1);\n"),
    ("    locks_off = pc_div(iters * blockIdx.x, epi.pc_mk, k_tiles) - 1;\n",
     "    locks_off = pc_div(ta_s0, epi.pc_mk, k_tiles) - 1;  // TA\n"),
    ("        iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);\n",
     "        ta_s1 - (k_tiles * slice_col_par + slice_row);  // TA\n"),
    ("    int col_first = iters * pc_div(k_tiles * slice_col_par + iters - 1, epi.pc_mi, iters);\n"
     "    if (col_first <= k_tiles * (slice_col_par + 1)) {\n",
     "    if (ta_on) {  // TA: closed-form column owners\n"
     "      const int c0_ = epi.ta_Lv * slice_col_par;\n"
     "      const int o_first = pc_div(c0_ + 1, epi.ta_mv, epi.ta_V);\n"
     "      const int o_last = pc_div(c0_ + 4 * (k_tiles - 1) + 1, epi.ta_mv, epi.ta_V);\n"
     "      slice_count = o_last - o_first + 1;\n"
     "      slice_idx = o_last - (int)blockIdx.x;\n"
     "    } else {\n"
     "    int col_first = iters * pc_div(k_tiles * slice_col_par + iters - 1, epi.pc_mi, iters);\n"
     "    if (col_first <= k_tiles * (slice_col_par + 1)) {\n"),
    ("        if (col_off > 0) slice_idx--;\n      }\n    }\n",
     "        if (col_off > 0) slice_idx--;\n      }\n    }\n    }  // TA\n"),
    ("      slice_col_par = pc_div(iters * blockIdx.x, epi.pc_mk, k_tiles);\n"
     "      slice_row = iters * blockIdx.x - slice_col_par * k_tiles;\n",
     "      slice_col_par = pc_div(ta_s0, epi.pc_mk, k_tiles);  // TA\n"
     "      slice_row = ta_s0 - slice_col_par * k_tiles;\n"),
]
TA_HOST_PATCHES = [
    ('  TORCH_CHECK(cfg.size() == 6 && cfg[0] > 0, "mp_gemm_epi: explicit config required");\n',
     '  TORCH_CHECK(cfg.size() >= 6 && cfg.size() <= 8 && cfg[0] > 0, "mp_gemm_epi: explicit config required");\n'),
    ("  TORCH_CHECK(cfg.size() == 6);\n", "  TORCH_CHECK(cfg.size() >= 6 && cfg.size() <= 8);\n"),
    ("    int grid = sms * ch_bps;\n",
     "    int grid = sms * ch_bps;\n"
     "    if (cfg.size() >= 7 && cfg[6] > 0) {  // grid override (all CTAs co-resident)\n"
     "      TORCH_CHECK(cfg[6] <= grid, \"mp_gemm_epi: grid \", cfg[6], \" > sms * blocks_per_sm \", grid);\n"
     "      grid = cfg[6];\n"
     "    }\n"),
    ("        epi_l.pc_mk = mg(kt); epi_l.pc_mi = mg(it); epi_l.pc_mn = mg(nt);\n      }\n",
     "        epi_l.pc_mk = mg(kt); epi_l.pc_mi = mg(it); epi_l.pc_mn = mg(nt);\n      }\n"
     "      // LL4 TA: virtual gap of e quarter k-tiles per column (cfg[7]); validated by simulation, else off\n"
     "      const bool ta_q1 = !(group_blocks >= tkb && group_blocks / tkb > 1);  // slices need no group rounding\n"
     "      if (cfg.size() > 7 && cfg[7] != 0 && epi_l.pc_iters > 0 && p1 == 0 && p2 < grid && ta_q1) {\n"
     "        // cfg[7] < 0 (diagnostic): no gap, V = -cfg[7] quarter tiles (V = 4 iters = the pc partition, same bits)\n"
     "        const long long e = cfg[7] > 0 ? cfg[7] : 0, Lv = 4ll * kt + e, need = (long long)(p2 - 1) * Lv + 4ll * kt;\n"
     "        const long long V = cfg[7] < 0 ? -cfg[7] : (need + grid - 1) / grid;\n"
     "        auto mapv = [&](long long xv) -> long long {\n"
     "          const long long c_ = xv / Lv, row = (xv - c_ * Lv + 2) / 4;\n"
     "          return row >= kt ? (c_ + 1) * kt : c_ * kt + row;\n"
     "        };\n"
     "        const long long end = (long long)p2 * kt;\n"
     "        bool ok = V > e + 6;\n"
     "        long long prev = 0;\n"
     "        for (int b = 1; ok && b <= grid; b++) {\n"
     "          long long s = mapv(V * b);\n"
     "          if (s > end) s = end;\n"
     "          if (prev < end && s <= prev) ok = false;  // empty CTA before the end: owners would skip it\n"
     "          prev = s;\n"
     "        }\n"
     "        ok = ok && prev >= end && exact(V * (grid + 1), Lv) && exact((long long)p2 * Lv + 4ll * kt, V) &&\n"
     "             exact((long long)V * (grid + 1), kt);\n"
     "        static const bool ta_dbg = getenv(\"ATDEV_C1W3_TA_DEBUG\") != nullptr;\n"
     "        if (ta_dbg) printf(\"TA n=%d k=%d grid=%d e=%lld V=%lld Lv=%lld ok=%d\\n\", prob_n, prob_k, grid, e, V, Lv, (int)ok);\n"
     "        if (ok) {\n"
     "          epi_l.ta_V = (int)V; epi_l.ta_Lv = (int)Lv; epi_l.ta_mv = mg(V); epi_l.ta_ml = mg(Lv);\n"
     "        }\n"
     "      }\n"),
]

# ww = wide warp tile for M=16: every warp computes 16 n8 tiles (2 x 64 columns = two "virtual" V2 warps 2 wn and
# 2 wn + 1) instead of 8, so one A ldmatrix feeds 16 mma, and a CTA of `threads` threads has the k-row / partial /
# in-block-reduction structure of the V2 CTA with 2 * threads threads at the same (thread_k, thread_n): same k16
# slabs per warp row, same reduction order -> bit-identical to V2 at [tk, tn, 2 * threads] with the same grid.
# Needs thread_n >= 128 and an even number (>= 2) of k16 slabs per warp and stage; allows 64-thread CTAs.
WW_TILES = [(64, 128, 64), (128, 128, 64), (256, 128, 64), (128, 128, 128), (256, 128, 128),
            (64, 256, 128), (128, 256, 128), (256, 256, 128)]
WW_STAGES = (3, 4, 6)


def ww_ok(tk: int, tn: int, thr: int) -> bool:
    """WW tile validity (mirrors the kernel static_asserts): >= 1 real n-warp of 128 columns, whole warp rows,
    an even number >= 2 of k16 slabs per warp and stage."""
    if tn < 128 or tn % 128:
        return False
    nw = tn // 128
    if thr % (32 * nw):
        return False
    rows = thr // (32 * nw)
    slabs = (tk // 16) // rows if (tk // 16) % rows == 0 else 0
    return slabs >= 2 and slabs % 2 == 0


def _ww_transform(src: str) -> str:
    """ATDEV_C1W3_X=ww kernel text (applied to the V2 template)."""
    def rep(old, new, count=1):
        nonlocal src
        n = src.count(old)
        assert n == count, f"WW anchor matched {n} (want {count}) times:\n{old[:200]}"
        src = src.replace(old, new)

    rep("  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;\n",
        "  // WW: a warp covers two 64-column virtual warps -> twice the warp rows, half the k16 slabs per warp\n"
        "  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta / 2;\n"
        "  static_assert(b_sh_wr_iters >= 2 && b_sh_wr_iters % 2 == 0, \"WW: even k16 slabs (>= 2) per warp\");\n")
    rep("  constexpr int tb_n_warps = thread_n_blocks / (is_a_8bit ? 2 : 4);\n",
        "  constexpr int tb_n_warps = thread_n_blocks / (is_a_8bit ? 2 : 4);  // WW: virtual (64-column) n-warps\n"
        "  static_assert(tb_n_warps % 2 == 0, \"WW: thread_n must be a multiple of 128\");\n"
        "  constexpr int ww_n = tb_n_warps / 2;  // real n-warps\n"
        "  static_assert(threads % (32 * ww_n) == 0, \"WW: whole warp rows\");\n"
        "  const int ww_wn = (threadIdx.x / 32) % ww_n, ww_row = (threadIdx.x / 32) / ww_n;\n")
    rep("  a_sh_rd += 2 * ((threadIdx.x / 32) / tb_n_warps) * b_sh_wr_iters;\n",
        "  a_sh_rd += 2 * ww_row * b_sh_wr_iters;  // WW\n")
    rep("  const int b3_warp_row = threadIdx.x / b_sh_stride;\n  const int b3_grp = threadIdx.x % b_sh_stride;\n",
        "  const int b3_warp_row = ww_row;  // WW: virtual warp 2 wn, its partner 2 wn + 1 at +32 words\n"
        "  const int b3_grp = 64 * ww_wn + threadIdx.x % 32;\n")
    rep("    s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 4;\n",
        "    s_sh_rd = 8 * (2 * ww_wn) + (threadIdx.x % 32) / 4;  // WW: + 8 = virtual warp 2 wn + 1\n")
    rep("  I4 frag_b_quant[2][b_thread_vecs];\n  FragC frag_c[thread_m_blocks][is_a_8bit ? 2 : 4][2];\n",
        "  I4 frag_b_quant[2][b_thread_vecs];\n"
        "  I4 ww_bq[2];  // WW: B words of virtual warp 2 wn + 1\n"
        "  FragC frag_c[thread_m_blocks][8][2];  // WW: j 0-3 virtual warp 2 wn, j 4-7 virtual warp 2 wn + 1\n")
    rep("  FragC frag_c_tmp[thread_m_blocks][is_a_8bit ? 2 : 4][2];\n",
        "  FragC frag_c_tmp[thread_m_blocks][8][2];  // WW: same shape as frag_c (int8 path only, unused here)\n")
    rep("  FragS frag_s[2][4];\n", "  FragS frag_s[2][4];\n  FragS ww_s[2][4];  // WW: scales of virtual warp 2 wn + 1\n")
    rep("    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)\n",
        "    for (int i = 0; i < thread_m_blocks * 8 * 2 * 4; i++)\n")
    rep("    fq[3] = (int)(((w0 >> 3) & 0x11111111u) | ((w1 >> 2) & 0x22222222u) | ((w2 >> 1) & 0x44444444u));\n  };\n",
        "    fq[3] = (int)(((w0 >> 3) & 0x11111111u) | ((w1 >> 2) & 0x22222222u) | ((w2 >> 1) & 0x44444444u));\n"
        "    {  // WW: virtual warp 2 wn + 1 = the next 32 words of each plane\n"
        "      const uint32_t u0 = sh_b3[32], u1 = sh_b3[b3_words + 32], u2 = sh_b3[2 * b3_words + 32];\n"
        "      int* fw = reinterpret_cast<int*>(&ww_bq[k % 2]);\n"
        "      fw[0] = (int)u0;\n      fw[1] = (int)u1;\n      fw[2] = (int)u2;\n"
        "      fw[3] = (int)(((u0 >> 3) & 0x11111111u) | ((u1 >> 2) & 0x22222222u) | ((u2 >> 1) & 0x44444444u));\n"
        "    }\n  };\n")
    # scales: the partner's int4 sits 8 int4 further in the same scale row
    rep("          reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];\n"
        "        } else {\n"
        "          reinterpret_cast<int4*>(&frag_s[1])[0] =\n"
        "              reinterpret_cast<int4*>(&frag_s[0])[0];\n"
        "        }\n",
        "          reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];\n"
        "          reinterpret_cast<int4*>(&ww_s[k % 2])[0] = sh_s_stage[s_sh_rd + 8];\n"
        "        } else {\n"
        "          reinterpret_cast<int4*>(&frag_s[1])[0] =\n"
        "              reinterpret_cast<int4*>(&frag_s[0])[0];\n"
        "          reinterpret_cast<int4*>(&ww_s[1])[0] = reinterpret_cast<int4*>(&ww_s[0])[0];\n"
        "        }\n")
    rep("      int warp_row = warp_id / tb_n_warps;\n\n      int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;\n"
        "      int cur_group_id = k_blocks / group_blocks2;\n",
        "      int warp_row = ww_row;  // WW\n      (void)warp_id;\n\n"
        "      int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;\n"
        "      int cur_group_id = k_blocks / group_blocks2;\n")
    rep("        reinterpret_cast<int4*>(&frag_s[k % 2])[0] =\n"
        "            sh_s_stage[s_sh_rd + cur_group_id * s_sh_stride];\n",
        "        reinterpret_cast<int4*>(&frag_s[k % 2])[0] =\n"
        "            sh_s_stage[s_sh_rd + cur_group_id * s_sh_stride];\n"
        "        reinterpret_cast<int4*>(&ww_s[k % 2])[0] = sh_s_stage[s_sh_rd + 8 + cur_group_id * s_sh_stride];\n")
    rep("    } else if (group_blocks >= b_sh_wr_iters) {\n"
        "      if constexpr (!is_8bit_scale) {\n"
        "        reinterpret_cast<int4*>(&frag_s[1])[0] =\n"
        "            reinterpret_cast<int4*>(&frag_s[0])[0];\n",
        "    } else if (group_blocks >= b_sh_wr_iters) {\n"
        "      if constexpr (!is_8bit_scale) {\n"
        "        reinterpret_cast<int4*>(&frag_s[1])[0] =\n"
        "            reinterpret_cast<int4*>(&frag_s[0])[0];\n"
        "        reinterpret_cast<int4*>(&ww_s[1])[0] = reinterpret_cast<int4*>(&ww_s[0])[0];\n")
    # matmul: 8 j-blocks (4 per virtual warp), same per-accumulator k order
    rep("    for (int j = 0; j < 4; j++) {\n      FragB frag_b0;\n",
        "    for (int j = 0; j < 8; j++) {  // WW\n      FragB frag_b0;\n")
    rep("        b_quant_0 = frag_b_quant[k2][0][j];\n        b_quant_1 = b_quant_0 >> 8;\n",
        "        b_quant_0 = j < 4 ? frag_b_quant[k2][0][j] : ww_bq[k2][j - 4];\n        b_quant_1 = b_quant_0 >> 8;\n")
    rep("        scale<a_type_id>(frag_b0, frag_s[k2][j], 0);\n        scale<a_type_id>(frag_b1, frag_s[k2][j], 1);\n",
        "        scale<a_type_id>(frag_b0, j < 4 ? frag_s[k2][j] : ww_s[k2][j - 4], 0);\n"
        "        scale<a_type_id>(frag_b1, j < 4 ? frag_s[k2][j] : ww_s[k2][j - 4], 1);\n")
    # in-block reduction: rows of 32 * ww_n threads, 16 FragC per m-block and thread
    a = src.index("  auto thread_block_reduce = [&]() {\n")
    b = src.index("  // Since multiple threadblocks may process parts of the same column slice, we\n")
    blk = src[a:b]
    assert blk.count("b_sh_stride_threads") == 6 and blk.count("(is_a_8bit ? 2 : 4) * 2") == 6, "WW: reduce block"
    blk = blk.replace("b_sh_stride_threads", "ww_tpr").replace("(is_a_8bit ? 2 : 4) * 2", "16")
    blk = blk.replace("  auto thread_block_reduce = [&]() {\n",
                      "  auto thread_block_reduce = [&]() {\n"
                      "    constexpr int ww_tpr = 32 * ww_n;  // WW: threads per warp row\n"
                      "    static_assert(threads / ww_tpr / 2 * ww_tpr * 16 <= sh_b_red_bias_size + stages * zp_sh_stage +\n"
                      "                  sh_s_size + stages * a_sh_stage, \"WW: in-block reduction exceeds the smem layout\");\n")
    src = src[:a] + blk + src[b:]
    rep("    constexpr int active_threads = 32 * tb_n_warps;\n    bool is_th_active = threadIdx.x < active_threads;\n\n"
        "    constexpr int num_floats = thread_m_blocks * (is_a_8bit ? 2 : 4) * 2 * 4;\n",
        "    constexpr int active_threads = 32 * ww_n;  // WW\n    bool is_th_active = threadIdx.x < active_threads;\n\n"
        "    constexpr int num_floats = thread_m_blocks * 16 * 4;  // WW\n")
    # write_result: row-0 warps, virtual warp 2 wn at column 64 wn; j 4-7 continue at +32 bf16x2 = virtual 2 wn + 1
    rep("      c_sh_wr += (is_a_8bit ? 16 : 32) * (threadIdx.x / 32);\n",
        "      c_sh_wr += 64 * ww_wn;  // WW\n")
    a = src.index("    if (threadIdx.x / 32 < tb_n_warps) {\n  #pragma unroll\n      for (int i = 0; i < thread_m_blocks; i++) {\n")
    b = src.index("        c_sh_wr += 16 * (4 * c_sh_stride);\n", a)
    blk = src[a:b]
    blk = blk.replace("    if (threadIdx.x / 32 < tb_n_warps) {\n", "    if (threadIdx.x / 32 < ww_n) {  // WW\n", 1)
    assert blk.count("        for (int j = 0; j < (is_a_8bit ? 2 : 4); j++) {\n") == 1, "WW: write_result j loop"
    blk = blk.replace("        for (int j = 0; j < (is_a_8bit ? 2 : 4); j++) {\n", "        for (int j = 0; j < 8; j++) {  // WW\n")
    blk = blk.replace("[j / 2]", "[(j % 4) / 2]")
    src = src[:a] + blk + src[b:]
    # E6 consumer: 64-thread CTAs keep the 80 producer partials (T_in <= threads / 16 * E6_MAXP)
    # E6 consumer rstd: emulate the V2 twin (2 * threads) exactly -- virtual sub-lanes r and r + TPR per thread, 10
    # partials each, summed per virtual lane, then the twin's xor tree (its first level = s_r + s_(r+TPR))
    a = src.index("  constexpr int E6_TPR = threads / 16;  // threads per row\n")
    b = src.index("  if constexpr (E6_MODE >= 2 && E6_RSTD_AT == 1) {\n", a)
    src = src[:a] + """  constexpr int E6_TPR = threads / 16;  // threads per row
  constexpr int E6_MAXP = 20;  // WW: 2 virtual sub-lanes x 10 partials (T_in <= 2 * E6_TPR * 10)
  auto e6_rstd_load = [&](float* pv) {
    const int row = threadIdx.x / E6_TPR, sub = threadIdx.x % E6_TPR;
  #pragma unroll
    for (int h = 0; h < 2; h++) {
  #pragma unroll
      for (int q = 0; q < 10; q++) {
        const int t = sub + h * E6_TPR + q * (2 * E6_TPR);
        pv[10 * h + q] = (row < prob_m && t < epi.T_in) ? epi.Pin[row * epi.T_in + t] : 0.f;
      }
    }
  };
  auto e6_rstd_finish = [&](const float* pv) {
    const int row = threadIdx.x / E6_TPR, sub = threadIdx.x % E6_TPR;
    float s0 = 0.f, s1 = 0.f;
  #pragma unroll
    for (int q = 0; q < 10; q++) s0 += pv[q];
  #pragma unroll
    for (int q = 0; q < 10; q++) s1 += pv[10 + q];
    float s = s0 + s1;
  #pragma unroll
    for (int o = E6_TPR / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (sub == 0) {
      const float rr = row < prob_m ? rsqrtf(s / (float)epi.n_norm + epi.eps) : 0.f;
      sh_r[row] = rr;
      if (epi.Rout != nullptr && blockIdx.x == 0 && row < prob_m) epi.Rout[row] = rr;
    }
  };
""" + src[b:]
    return src


# kx = in-CTA k-splitting instead of cross-CTA split-K: tiles with many warp rows (8-16) per CTA for 1 CTA / SM,
# so a column is split over ~1-1.5 CTAs (fewer / no global fp32 partial round trips and slice ends) and the k-split
# is reduced in shared memory (thread_block_reduce). Kernel text unchanged (own extension with the extra tiles);
# combine with grid (7th cfg entry = grid, e.g. n_tiles for zero split-K) and ww.
KX_TILES = [(256, 64, 256), (512, 64, 256), (512, 128, 256)]  # 512-thread CTAs exceed the register file
KX_WW_TILES = [(256, 128, 256), (512, 128, 256), (512, 128, 128), (256, 256, 256)]

WW_HOST_PATCHES = [
    ("  if (c.threads < 128) return false;\n", "  if (c.threads < 64) return false;  // WW: 64-thread CTAs\n"),
    ('    TORCH_CHECK(T_in <= (threads / 16) * 10, "mp_gemm_epi: too many partials ", T_in);\n',
     '    TORCH_CHECK(T_in <= (threads / 16) * 20, "mp_gemm_epi: too many partials ", T_in);  // WW\n'),
]

# W3-gap timing-only ablations (WRONG results; dev only): ldo = loads only (no dequant / mma; the smem->register
# fetches become dead code), cmo = compute only (B cp.async re-reads a 1 MB L2-resident window instead of streaming)
LDO_PATCHES = [("        if constexpr (!is_a_8bit) {\n          matmul(k, pipe - (k >= b_sh_wr_iters - 2 ? 1 : 0));\n",
                "        if constexpr (!is_a_8bit) {\n          if (false) matmul(k, pipe - (k >= b_sh_wr_iters - 2 ? 1 : 0));\n")]
CMO_PATCHES = [("          cp_async4(&sh_b_stage[threads * i + threadIdx.x], &B[b_gl_rd + b3_gl_off[i]]);\n",
                "          cp_async4(&sh_b_stage[threads * i + threadIdx.x], &B[(b_gl_rd + b3_gl_off[i]) & 0xFFFF]);\n")]
NOA_PATCHES = [("""        cp_async4_pred(
            &sh_a_stage[a_sh_wr_trans[i]],
            &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
            a_sh_wr_pred[i]);
""", """        if (false) cp_async4_pred(  // W3-gap noa ablation: no A (activation) loads in the main loop
            &sh_a_stage[a_sh_wr_trans[i]],
            &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
            a_sh_wr_pred[i]);
""")]
# W3-trim (bit-identical tail trims of the split-K slice end; the last slice of a column ends the kernel's critical
# path: last-slice main-loop end -> column done = 1.5-2.5 us, measured W3-trim):
# pp = predecessor partial prefetch: a non-first slice (slice_idx > 0, fp32 serial lock chain) polls its column
# lock with a relaxed load PP_AT k-tiles before its main loop ends (thread 0, result consumed one k-tile later), and
# if the predecessors have already released it, cp.async's their fp32 partial (C_tmp, L2) into an own smem region
# after the A ring while the last k-tiles run. At the slice end the acquire poll and the C_tmp L2 round trip are
# skipped; the partial is added from smem with exactly global_reduce_fp32's per-element adds (same order) ->
# bit-identical. Not yet released at the poll (or no smem room / short slice) -> the unchanged serial path.
# s0 = slice 0 of a column (the first in the serial chain) skips its acquire poll: its lock is 0 by construction
# (per-column locks of the part-2 partition, reset by every chain's last slice) -> one L2 round trip less.
PP_PATCHES = [
    ("  // Register storage for double buffer of shared memory reads.\n",
     "  // PP: predecessor split-K partial prefetch region after the A ring\n"
     "  #ifndef PP_AT\n  #define PP_AT 3\n  #endif\n"
     "  #ifndef PP_FENCE\n  #define PP_FENCE 1\n  #endif\n"
     "  static_assert(PP_AT >= 3, \"PP: poll, publish, issue need 3 k-tiles\");\n"
     "  constexpr int pp_c_size = 16 * thread_m_blocks * 16 * thread_n_blocks * 4 / 16;\n"
     "  constexpr int pp_act = 32 * tb_n_warps;\n"
     "  constexpr int pp_th = pp_c_size / pp_act;\n"
     "  int4* sh_pp = sh_a + stages * a_sh_stage;\n"
     "  const bool pp_ok = use_fp32_reduce && !use_atomic_add && !m_block_size_8 &&\n"
     "                     (int)((sh_pp + pp_c_size - sh) * 16) <= max_shared_mem;\n"
     "  int* sh_ppf = reinterpret_cast<int*>(sh_r) + 31;  // flag (sh_r rows < 16 only)\n"
     "  int pp_lock = -1;\n"
     "  bool pp_hit = false, pp_armed = false;\n\n"
     "  // Register storage for double buffer of shared memory reads.\n"),
    ("        if (k == b_sh_wr_iters - 2) {\n          fetch_to_shared((pipe + stages - 1) % stages, pipe,\n",
     "        if (k == b_sh_wr_iters - 2) {\n"
     "          if (pp_ok && slice_idx > 0 && slice_count > 1) {  // PP\n"
     "            if (slice_iters == PP_AT) {\n"
     "              if (threadIdx.x == 0)\n"
     "                asm volatile(\"ld.relaxed.gpu.global.b32 %0, [%1];\\n\" : \"=r\"(pp_lock) : \"l\"(&locks[locks_off]));\n"
     "            } else if (slice_iters == PP_AT - 1) {\n"
     "              if (threadIdx.x == 0) {\n"
     "                const int h = pp_lock == slice_idx;\n"
     "  #if PP_FENCE\n"
     "                if (h) asm volatile(\"fence.acq_rel.gpu;\\n\" ::: \"memory\");\n"
     "  #endif\n"
     "                *sh_ppf = h;\n"
     "              }\n"
     "              pp_lock = -1;\n"
     "              pp_armed = true;  // this slice published sh_ppf (a slice may start below PP_AT)\n"
     "            } else if (slice_iters == PP_AT - 2) {\n"
     "              pp_hit = pp_armed && *sh_ppf != 0;\n"
     "              pp_armed = false;\n"
     "              if (pp_hit && threadIdx.x < pp_act) {\n"
     "                const int4* pp_src = C_tmp + locks_off * pp_c_size;\n"
     "  #pragma unroll\n"
     "                for (int q = 0; q < pp_th; q++)\n"
     "                  cp_async4(&sh_pp[pp_act * q + threadIdx.x], &pp_src[pp_act * q + threadIdx.x]);\n"
     "              }\n"
     "            }\n"
     "          }\n"
     "          fetch_to_shared((pipe + stages - 1) % stages, pipe,\n"),
    ("#if defined(MP_OVLCHAIN) && MP_OVLCHAIN\n  // P3 overlapped split-K chain",
     "  // PP: global_reduce_fp32(first = false, last) with the predecessor partial already in sh_pp\n"
     "  auto pp_reduce = [&](bool last) {\n"
     "    constexpr int c_size = pp_c_size;\n"
     "    if (threadIdx.x >= pp_act) return;\n"
     "    float* frag_c_ptr = reinterpret_cast<float*>(&frag_c);\n"
     "  #pragma unroll\n"
     "    for (int k = 0; k < pp_th; k++) {\n"
     "      const float* sh_c_ptr = reinterpret_cast<const float*>(&sh_pp[pp_act * k + threadIdx.x]);\n"
     "  #pragma unroll\n"
     "      for (int f = 0; f < 4; f++) frag_c_ptr[k * 4 + f] += sh_c_ptr[f];\n"
     "    }\n"
     "    if (!last) {\n"
     "      int4* fc = reinterpret_cast<int4*>(&frag_c);\n"
     "  #pragma unroll\n"
     "      for (int k = 0; k < pp_th; k++) C_tmp[locks_off * c_size + pp_act * k + threadIdx.x] = fc[k];\n"
     "    }\n"
     "  };\n\n"
     "#if defined(MP_OVLCHAIN) && MP_OVLCHAIN\n  // P3 overlapped split-K chain"),
    ("        barrier_acquire(&locks[locks_off], slice_idx);\n        if (use_fp32_reduce) {\n"
     "          global_reduce_fp32(slice_idx == 0, last);\n",
     "        if (pp_hit /*S0_COND*/) {\n"
     "          __syncthreads();  // PP: predecessors observed released at the poll\n"
     "        } else {\n"
     "          barrier_acquire(&locks[locks_off], slice_idx);\n"
     "        }\n"
     "        if (use_fp32_reduce) {\n"
     "          if (pp_hit) pp_reduce(last); else\n"
     "          global_reduce_fp32(slice_idx == 0, last);\n"),
    ("        barrier_release(&locks[locks_off], last);\n",
     "        barrier_release(&locks[locks_off], last);\n        pp_hit = false;\n"),
]
# pq = pp without the per-k-tile main-loop hook (pp measured +400 us on the 256-set: +10 regs, 9 extra compares per
# k-tile): thread 0 re-polls the column lock (relaxed load, consumed at the next poll / the slice end) only at the
# outer main-loop boundary (once per `stages` k-tiles) while <= PQ_W k-tiles remain; at the slice end it publishes
# the result (one extra __syncthreads), every thread cp.async's the predecessor partial BEFORE thread_block_reduce
# (overlaps it), and the acquire poll is skipped. Same adds as global_reduce_fp32 -> bit-identical.
PQ_PATCHES = [PP_PATCHES[0],
    ("    a_gl_rd += a_gl_rd_delta_o * stages;\n",
     "  #ifndef PQ_W\n  #define PQ_W (2 * stages)\n  #endif\n"
     "    if (threadIdx.x == 0 && pp_ok && slice_idx > 0 && slice_iters <= PQ_W && pp_lock != slice_idx)  // PQ\n"
     "      asm volatile(\"ld.relaxed.gpu.global.b32 %0, [%1];\\n\" : \"=r\"(pp_lock) : \"l\"(&locks[locks_off]));\n"
     "    a_gl_rd += a_gl_rd_delta_o * stages;\n"),
    ("      cp_async_wait<0>();\n      bool last = slice_idx == slice_count - 1;\n",
     "      cp_async_wait<0>();\n      bool last = slice_idx == slice_count - 1;\n"
     "      if (pp_ok && slice_idx > 0 && slice_count > 1) {  // PQ: publish the poll, prefetch the partial\n"
     "        if (threadIdx.x == 0) {\n"
     "          const int h = pp_lock == slice_idx;\n"
     "  #if PP_FENCE\n"
     "          if (h) asm volatile(\"fence.acq_rel.gpu;\\n\" ::: \"memory\");\n"
     "  #endif\n"
     "          *sh_ppf = h;\n"
     "        }\n"
     "        pp_lock = -1;\n"
     "        __syncthreads();\n"
     "        pp_hit = *sh_ppf != 0;\n"
     "        if (pp_hit && threadIdx.x < pp_act) {\n"
     "          const int4* pp_src = C_tmp + locks_off * pp_c_size;\n"
     "  #pragma unroll\n"
     "          for (int q = 0; q < pp_th; q++)\n"
     "            cp_async4(&sh_pp[pp_act * q + threadIdx.x], &pp_src[pp_act * q + threadIdx.x]);\n"
     "        }\n"
     "        cp_async_fence();\n"
     "      }\n"),
    (PP_PATCHES[2][0], PP_PATCHES[2][1].replace("    if (threadIdx.x >= pp_act) return;\n",
                                                "    cp_async_wait<0>();\n    if (threadIdx.x >= pp_act) return;\n")),
    PP_PATCHES[3], PP_PATCHES[4]]
S0_PATCHES = [
    ("        barrier_acquire(&locks[locks_off], slice_idx);\n",
     "        if (slice_idx == 0 && part2_mn_tiles < gridDim.x) {\n"
     "          __syncthreads();  // S0: the chain's first slice, its per-column lock is 0 by construction\n"
     "        } else\n"
     "        barrier_acquire(&locks[locks_off], slice_idx);\n"),
]
# ffN = staged first fill (kernel start only; refills unchanged): the prologue issues stage 0 .. N-1, waits for them
# (cp.async.wait_all of this thread's groups), then issues the remaining stages. At kernel start every CTA of the grid
# requests stages-1 stages at once (gu: 320 CTAs x 3 x 6.3 KB = 6 MB burst at 1.5 TB/s = 4 us) and DRAM serves the
# stages interleaved, so stage 0 lands only at the end of the burst (init -> first stage 3.3-7 K cycles, W3-trim
# s0 prof) while the SM idles; issuing stage 0 alone first lets the first MMA start after a third of the burst.
# ffsN = same split, but a __nanosleep(N * 128 ns) instead of the wait (the stage-0 requests stay ahead in the queues
# without waiting for their round trip). Fetch timing only -> bit-identical.
def _ff_patches(n_first: int, sleep_ns: int, every: bool = False, refill: bool = False):
    """every: the sleep after each of stages 0 .. stages-3 (ffd), not only after stage n_first - 1;
    refill: also at every slice refill (ffa), not only at kernel start"""
    wait = (f"__nanosleep({sleep_ns});" if sleep_ns else "cp_async_wait<0>();")
    if refill:
        return [(a, b.replace(" && ff_first", "")) for a, b in _ff_patches(n_first, sleep_ns, every)[:1]]
    return [
        ("  auto start_pipes = [&]() {\n\n  #pragma unroll\n    for (int i = 0; i < stages - 1; i++) {\n",
         "  bool ff_first = true;  // FF: staged fill at kernel start only\n"
         "  auto start_pipes = [&]() {\n\n  #pragma unroll\n    for (int i = 0; i < stages - 1; i++) {\n" +
         (f"      if (i > 0 && ff_first) {{ {wait} }}  // FF (every stage)\n" if every else
          f"      if (i == {n_first} && ff_first) {{ {wait} }}  // FF\n")),
        ("    zero_accums();\n    wait_for_stage();\n    fetch_to_registers(0, 0);\n    fetch_scales_to_registers(0, 0);\n",
         "    ff_first = false;\n"
         "    zero_accums();\n    wait_for_stage();\n    fetch_to_registers(0, 0);\n    fetch_scales_to_registers(0, 0);\n"),
    ]


# ffm = per E6 mode: the consumer qkvz / qkv kernels (mode 2) measured best with the single sleep (ffs2), the
# producer down / out (mode 1) and SiLU gate_up (mode 3) with a sleep after every prologue stage (ffd2).
FFM_PATCHES = [(_ff_patches(1, 256)[0][0], _ff_patches(1, 256)[0][1].replace(
    "      if (i == 1 && ff_first) { __nanosleep(256); }  // FF\n",
    "      if ((E6_MODE == 2 ? i == 1 : i > 0) && ff_first) { __nanosleep(256); }  // FF (ffm)\n")),
    _ff_patches(1, 256)[1]]
# ffkN = ffd2 + a per-CTA start stagger of N * 128 ns x (blockIdx.x % 3) before the first stage (spreads the
# start burst of co-resident CTAs; consecutive block indices share an SM on this grid order)
def _ffk_patches(n: int):
    a, b = _ff_patches(1, 256, every=True)[0]
    b = b.replace("    for (int i = 0; i < stages - 1; i++) {\n",
                  f"    if (ff_first) __nanosleep({n * 128} * (blockIdx.x % 3));  // FFK\n"
                  "    for (int i = 0; i < stages - 1; i++) {\n", 1)
    return [(a, b), _ff_patches(1, 256)[1]]


# LL3 rso = consumer rstd sum order: ptxas hoisted the first `s += pv[0]` (FADD s, RZ, pv0) of e6_rstd_finish above
# the first pipeline-fill cp.async (SASS m2/m3: FADD at 0x22f0-0x2310 before the first LDGSTS), so every CTA waits for
# the rstd-partial load's round trip before it issues stage 0. An empty asm volatile on each pv after start_pipes
# pins the sum after the fill (same adds, same order -> bit-identical).
RSO_PATCHES = [("    e6_rstd_load(pv);\n    if (slice_iters) {\n      start_pipes();\n    }\n    e6_rstd_finish(pv);\n",
                "    e6_rstd_load(pv);\n    if (slice_iters) {\n      start_pipes();\n    }\n"
                "  #pragma unroll\n    for (int q_ = 0; q_ < E6_MAXP; q_++) asm volatile(\"\" : \"+f\"(pv[q_]));  // LL3 rso\n"
                "    e6_rstd_finish(pv);\n")]


# LL3 rel = lighter split-K lock memory ops (same data, same order -> bit-identical): the acquire poll spins with
# ld.relaxed (no CCTL.IVALL per iteration) and does one ld.acquire after the value matched; the release is a single
# red.release.gpu instead of fence.acq_rel (MEMBAR + ERRBAR + CCTL.IVALL) + red.relaxed.
REL_PATCHES = [
    ("""    do
      // Guarantee that subsequent writes by this threadblock will be visible
      // globally.
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\\n"
                   : "=r"(state)
                   : "l"(lock));
    while (state != count);
""", """    do
      asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];\\n" : "=r"(state) : "l"(lock));  // LL3 rel
    while (state != count);
    asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\\n" : "=r"(state) : "l"(lock));
"""),
    ("""    asm volatile("fence.acq_rel.gpu;\\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\\n"
                 :
                 : "l"(lock), "r"(val));
""", """    asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" : : "l"(lock), "r"(val));  // LL3 rel
"""),
]


# C1gdn2 qkc = GDN q / k / v causal conv in the qkvz consumer epilogue (mode 2 only; off unless the host armed it with
# ext.set_qkc(...) right before the launch: one-shot like l2pf's set_next, baked into the kernel params -> capturable).
# The last slice CTA of a column tile in [0, qc_n) holds the full bf16 tile (all T <= 16 tree rows, = what it stores
# to C) in sh_red after the staging; it convolves its 128 channels with the conv-state window of the request's slot
# s0 (3 tokens at acc - 1) and the tree sources csrc exactly like csrc/gdn_var/gdn_tree_c1g_f3.cu's conv (same fmaf
# chain w3 x_i + b, w2 x_p, w1 x_pp, w0 x_ppp, SiLU y * rcp.approx(1 + __expf(-y)), bf16 RN -> same bits), writes the
# result to qc_out [T, qc_n] and a tag (cnt[hk] / nb) + 1 per 128-column tile = the generation of the GDN tree launch
# that follows. It never writes the conv state (the GDN kernel still rewrites it), so a GEMM whose metadata is stale
# (no tree step) only writes scratch; the GDN kernel uses qc_out only if all tags of its key head match its launch.
# Metadata (cu, idx, acc, csrc: 53 words) -> static smem with 4-byte cp.async in the first pipeline group; window /
# weights / bias / cnt -> registers before the split-K reduce of the last slice (latency under the C_tmp round trip).
QKC_KERNEL_H = [
    (RO_LINE, RO_LINE +
     "  // C1gdn2 QKC (qc_out == nullptr: off): GDN conv in the qkvz epilogue\n"
     "  const int* qc_cu; const int* qc_idx; const int* qc_acc; const int* qc_src;\n"
     "  const unsigned long long* qc_cnt;\n"
     "  const void* qc_cw; const void* qc_cb; const void* qc_cs;\n"
     "  long long qc_cs_slot, qc_cs_dim, qc_cs_tok;\n"
     "  void* qc_out; unsigned* qc_tag;\n"
     "  int qc_n, qc_H, qc_G, qc_nb, qc_maxlog, qc_act;\n"),
]
QKC_PATCHES = [
    ("  __shared__ __align__(128) float sh_r[32];\n",
     "  __shared__ __align__(128) float sh_r[32];\n"
     "  // C1gdn2 QKC metadata: [0..1] cu, [2..3] idx, [4] acc, [5..52] csrc (16 x 3); 256 B keeps the 128-B alignment\n"
     "#if E6_MODE == 2\n"
     "  __shared__ __align__(128) int sh_qc[64];\n"
     "#else\n"
     "  int* const sh_qc = nullptr;  // referenced only under if constexpr (E6_MODE == 2)\n"
     "#endif\n"),
    ("    float pv[E6_MAXP];\n    e6_rstd_load(pv);\n    if (slice_iters) {\n      start_pipes();\n    }\n    e6_rstd_finish(pv);\n",
     "    float pv[E6_MAXP];\n"
     "    if constexpr (E6_MODE == 2) {\n"
     "      if (epi.qc_out != nullptr && slice_iters && threadIdx.x < 53) {  // C1gdn2 QKC: joins the first fill group\n"
     "        const int* src_ = threadIdx.x < 2 ? epi.qc_cu + threadIdx.x : threadIdx.x < 4 ? epi.qc_idx + (threadIdx.x - 2)\n"
     "                        : threadIdx.x == 4 ? epi.qc_acc : epi.qc_src + (threadIdx.x - 5);\n"
     "        const uint32_t sa_ = static_cast<uint32_t>(__cvta_generic_to_shared(&sh_qc[threadIdx.x]));\n"
     "        asm volatile(\"cp.async.ca.shared.global [%0], [%1], 4;\\n\" ::\"r\"(sa_), \"l\"(src_));\n"
     "      }\n"
     "    }\n"
     "    e6_rstd_load(pv);\n    if (slice_iters) {\n      start_pipes();\n    }\n    e6_rstd_finish(pv);\n"),
    ("  int4 e6_wpre;\n",
     "  int4 e6_wpre;\n"
     "  // C1gdn2 QKC: the thread's channel pair (window, weights, bias) + the tile tag counter, loaded before the\n"
     "  // split-K reduce of a column-ending slice; the tile's 16 rows of that pair are stashed in registers by\n"
     "  // write_result and convolved only at the CTA's end (or before the next stash): a conv between two segments of\n"
     "  // a CTA delays the next column's serial split-K chain (measured +3.8 us/call when done in write_result)\n"
     "  uint32_t qc_win[3] = {0u, 0u, 0u};\n"
     "  uint2 qc_w0 = make_uint2(0u, 0u), qc_w1 = make_uint2(0u, 0u);\n"
     "  uint32_t qc_bb = 0u;\n"
     "  unsigned long long qc_cv = 0ull;\n"
     "  bool qc_on = false, qc_pend = false;\n"
     "  int qc_col0 = 0;\n"
     "  uint32_t qc_x[16];\n"
     "#ifndef QKC_ABL\n#define QKC_ABL 0\n#endif\n"
     "  auto qc_conv = [&]() {  // conv of the stashed tile pair (bit-identical to gdn_tree_c1g_f3.cu's conv)\n"
     "    constexpr int NPAIR = 8 * thread_n_blocks, NG = threads / NPAIR > 0 ? threads / NPAIR : 1, NPT = 16 / NG;\n"
     "    const int cl = 2 * (threadIdx.x % NPAIR), grp = threadIdx.x / NPAIR, cg = qc_col0 + cl;\n"
     "    const int T = sh_qc[1] - sh_qc[0];\n"
     "    auto bf2 = [](uint32_t u) { return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u)); };\n"
     "    const float2 wa01 = bf2(qc_w0.x), wa23 = bf2(qc_w0.y), wb01 = bf2(qc_w1.x), wb23 = bf2(qc_w1.y);\n"
     "    const float2 bb = bf2(qc_bb);\n"
     "    auto xrow = [&](int r) -> uint32_t {  // GDN smem row r: 0..15 raw rows (0 for r >= T), 16..18 window\n"
     "      uint32_t v = r == 16 ? qc_win[0] : r == 17 ? qc_win[1] : r == 18 ? qc_win[2] : 0u;\n"
     "  #pragma unroll\n"
     "      for (int q = 0; q < 16; q++) v = r == q ? qc_x[q] : v;\n"
     "      return v;\n"
     "    };\n"
     "    uint32_t* out = reinterpret_cast<uint32_t*>(epi.qc_out);\n"
     "  #pragma unroll\n"
     "    for (int ii = 0; ii < NPT; ii++) {\n"
     "      const int i = grp * NPT + ii;\n"
     "      if (i < T && QKC_ABL != 1) {\n"
     "        int sr[3];\n"
     "  #pragma unroll\n"
     "        for (int j = 0; j < 3; j++) {\n"
     "          const int r = sh_qc[5 + 3 * i + j];\n"
     "          sr[j] = r >= 0 ? r : 19 + r;\n"
     "        }\n"
     "        const float2 f0 = bf2(xrow(i)), f1 = bf2(xrow(sr[0])), f2 = bf2(xrow(sr[1])), f3 = bf2(xrow(sr[2]));\n"
     "        float y0 = fmaf(wa23.y, f0.x, bb.x), y1 = fmaf(wb23.y, f0.y, bb.y);\n"
     "        y0 = fmaf(wa23.x, f1.x, y0); y1 = fmaf(wb23.x, f1.y, y1);\n"
     "        y0 = fmaf(wa01.y, f2.x, y0); y1 = fmaf(wb01.y, f2.y, y1);\n"
     "        y0 = fmaf(wa01.x, f3.x, y0); y1 = fmaf(wb01.x, f3.y, y1);\n"
     "        if (epi.qc_act) {\n"
     "          float r0, r1;\n"
     "          asm(\"rcp.approx.ftz.f32 %0, %1;\" : \"=f\"(r0) : \"f\"(1.0f + __expf(-y0)));\n"
     "          asm(\"rcp.approx.ftz.f32 %0, %1;\" : \"=f\"(r1) : \"f\"(1.0f + __expf(-y1)));\n"
     "          y0 = y0 * r0; y1 = y1 * r1;\n"
     "        }\n"
     "        __nv_bfloat162 yo = __floats2bfloat162_rn(y0, y1);\n"
     "        out[((long long)i * epi.qc_n + cg) / 2] = *reinterpret_cast<uint32_t*>(&yo);\n"
     "      }\n"
     "    }\n"
     "    if (threadIdx.x < div_ceil(16 * thread_n_blocks, 128))\n"
     "      epi.qc_tag[qc_col0 / 128 + threadIdx.x] = (unsigned)(qc_cv / (unsigned long long)epi.qc_nb) + 1u;\n"
     "    qc_pend = false;\n"
     "  };\n"
     "  auto qc_prefetch = [&]() {\n"
     "    if (qc_pend) qc_conv();  // a second column-ending segment: flush the first one's conv now\n"
     "    qc_on = false;\n"
     "    const int col0 = slice_col * 16 * thread_n_blocks;\n"
     "    if (epi.qc_out == nullptr || col0 >= epi.qc_n || threads < 8 * thread_n_blocks) return;\n"
     "    const int T = sh_qc[1] - sh_qc[0], s0 = sh_qc[2], s1 = sh_qc[3], acc = sh_qc[4];\n"
     "    if (sh_qc[0] != 0 || T <= 0 || s0 <= 0 || s1 <= 0 || T > epi.qc_maxlog + 1 || T > 16 || T > prob_m) return;\n"
     "    qc_on = true;\n"
     "    const int c = col0 + 2 * (threadIdx.x % (8 * thread_n_blocks));\n"
     "    const unsigned short* cs = reinterpret_cast<const unsigned short*>(epi.qc_cs) + (long long)s0 * epi.qc_cs_slot +\n"
     "                               (long long)c * epi.qc_cs_dim;\n"
     "  #pragma unroll\n"
     "    for (int j = 0; j < 3; j++) {\n"
     "      const long long o = (long long)(acc - 1 + j) * epi.qc_cs_tok;\n"
     "      qc_win[j] = (uint32_t)cs[o] | ((uint32_t)cs[o + epi.qc_cs_dim] << 16);\n"
     "    }\n"
     "    const uint2* w = reinterpret_cast<const uint2*>(epi.qc_cw) + c;\n"
     "    qc_w0 = w[0]; qc_w1 = w[1];\n"
     "    qc_bb = 0u;\n"
     "    if (epi.qc_cb != nullptr) {\n"
     "      const unsigned short* b = reinterpret_cast<const unsigned short*>(epi.qc_cb) + c;\n"
     "      qc_bb = (uint32_t)b[0] | ((uint32_t)b[1] << 16);\n"
     "    }\n"
     "    if (threadIdx.x < div_ceil(16 * thread_n_blocks, 128)) {\n"
     "      const int t = col0 / 128 + threadIdx.x;\n"
     "      const int hk = t < epi.qc_H ? t : t < 2 * epi.qc_H ? t - epi.qc_H : (t - 2 * epi.qc_H) / epi.qc_G;\n"
     "      qc_cv = __ldcg(epi.qc_cnt + hk);\n"
     "    }\n"
     "  };\n"),
    ("      if constexpr (E6_MODE == 1) {\n        if (last) e6_prefetch();\n      }\n",
     "      if constexpr (E6_MODE == 1) {\n        if (last) e6_prefetch();\n      }\n"
     "      if constexpr (E6_MODE == 2) {\n        if (last) qc_prefetch();\n      }\n"),
    ("        c_gl_wr += c_gl_wr_delta;\n        c_sh_rd += c_sh_rd_delta;\n      }\n    }\n    __syncthreads();\n  };\n",
     "        c_gl_wr += c_gl_wr_delta;\n        c_sh_rd += c_sh_rd_delta;\n      }\n    }\n"
     "    if constexpr (E6_MODE == 2) {\n"
     "      if (qc_on && last) {  // C1gdn2 QKC: stash the pair's 16 bf16x2 rows (0 for rows >= T); conv deferred\n"
     "        const int cl = 2 * (threadIdx.x % (8 * thread_n_blocks)), T = sh_qc[1] - sh_qc[0];\n"
     "        const uint32_t* xs = reinterpret_cast<const uint32_t*>(sh_red);  // row r at 4 * c_sh_stride * r\n"
     "  #pragma unroll\n"
     "        for (int r = 0; r < 16; r++) qc_x[r] = r < T ? xs[4 * c_sh_stride * r + cl / 2] : 0u;\n"
     "        qc_col0 = slice_col * 16 * thread_n_blocks;\n"
     "        qc_pend = true;\n"
     "        qc_on = false;\n"
     "      }\n"
     "    }\n"
     "    __syncthreads();\n  };\n"),
    ("        start_pipes();\n      }\n    }\n  }\n}\n",
     "        start_pipes();\n      }\n    }\n  }\n"
     "  if constexpr (E6_MODE == 2) {\n    if (qc_pend) qc_conv();  // C1gdn2 QKC: the deferred conv at the CTA's end\n  }\n"
     "}\n"),
]
QKC_HOST_PATCHES = [
    ("  MpEpi epi{};\n  epi.mode = (int)mode;\n",
     "  MpEpi epi{};\n  epi.mode = (int)mode;\n"
     "  if (g_qc.qc_out != nullptr) {  // C1gdn2 QKC (one-shot)\n"
     "    TORCH_CHECK(mode == 2 && a.size(0) <= 16, \"set_qkc: mode-2 consumer with M <= 16 only\");\n"
     "    epi.qc_cu = g_qc.qc_cu; epi.qc_idx = g_qc.qc_idx; epi.qc_acc = g_qc.qc_acc; epi.qc_src = g_qc.qc_src;\n"
     "    epi.qc_cnt = g_qc.qc_cnt; epi.qc_cw = g_qc.qc_cw; epi.qc_cb = g_qc.qc_cb; epi.qc_cs = g_qc.qc_cs;\n"
     "    epi.qc_cs_slot = g_qc.qc_cs_slot; epi.qc_cs_dim = g_qc.qc_cs_dim; epi.qc_cs_tok = g_qc.qc_cs_tok;\n"
     "    epi.qc_out = g_qc.qc_out; epi.qc_tag = g_qc.qc_tag; epi.qc_n = g_qc.qc_n; epi.qc_H = g_qc.qc_H;\n"
     "    epi.qc_G = g_qc.qc_G; epi.qc_nb = g_qc.qc_nb; epi.qc_maxlog = g_qc.qc_maxlog; epi.qc_act = g_qc.qc_act;\n"
     "    TORCH_CHECK(epi.qc_n <= size_n && epi.qc_n % cfg[1] == 0, \"set_qkc: qc_n must be whole column tiles\");\n"
     "  }\n"
     "  g_qc = MpEpi{};\n"),
    ("torch::Tensor mp_gemm_epi(torch::Tensor a,",
     "// C1gdn2 QKC: arm the GDN conv epilogue for the next gemm_epi launch (one-shot)\n"
     "static MpEpi g_qc{};\n"
     "void mp_set_qkc(torch::Tensor cu, torch::Tensor idx, torch::Tensor acc, torch::Tensor csrc, torch::Tensor cnt,\n"
     "                torch::Tensor cw, c10::optional<torch::Tensor> cb, torch::Tensor cs, torch::Tensor out,\n"
     "                torch::Tensor tag, int64_t H, int64_t G, int64_t nb, int64_t max_log, bool act) {\n"
     "  TORCH_CHECK(cu.scalar_type() == at::kInt && idx.scalar_type() == at::kInt && acc.scalar_type() == at::kInt &&\n"
     "              csrc.scalar_type() == at::kInt && csrc.is_contiguous() && csrc.numel() >= 48);\n"
     "  TORCH_CHECK(cnt.scalar_type() == at::kLong && cw.scalar_type() == at::kBFloat16 && cw.is_contiguous() &&\n"
     "              cw.size(-1) == 4 && cs.scalar_type() == at::kBFloat16 && cs.dim() == 3 &&\n"
     "              out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(0) == 16 &&\n"
     "              tag.scalar_type() == at::kInt && tag.numel() >= out.size(1) / 128 && out.size(1) % 128 == 0);\n"
     "  TORCH_CHECK(!cb || (cb->scalar_type() == at::kBFloat16 && cb->is_contiguous()));\n"
     "  g_qc = MpEpi{};\n"
     "  g_qc.qc_cu = cu.data_ptr<int>(); g_qc.qc_idx = idx.data_ptr<int>(); g_qc.qc_acc = acc.data_ptr<int>();\n"
     "  g_qc.qc_src = csrc.data_ptr<int>(); g_qc.qc_cnt = (const unsigned long long*)cnt.data_ptr();\n"
     "  g_qc.qc_cw = cw.data_ptr(); g_qc.qc_cb = cb ? cb->data_ptr() : nullptr; g_qc.qc_cs = cs.data_ptr();\n"
     "  g_qc.qc_cs_slot = cs.stride(0); g_qc.qc_cs_dim = cs.stride(1); g_qc.qc_cs_tok = cs.stride(2);\n"
     "  g_qc.qc_out = out.data_ptr(); g_qc.qc_tag = (unsigned*)tag.data_ptr(); g_qc.qc_n = (int)out.size(1);\n"
     "  g_qc.qc_H = (int)H; g_qc.qc_G = (int)G; g_qc.qc_nb = (int)nb; g_qc.qc_maxlog = (int)max_log; g_qc.qc_act = act;\n"
     "}\n\n"
     "torch::Tensor mp_gemm_epi(torch::Tensor a,"),
    ('  m.def("instantiated"',
     '  m.def("set_qkc", &mp_set_qkc, "C1gdn2: arm the GDN conv epilogue (next launch)", py::arg("cu"), py::arg("idx"),\n'
     '        py::arg("acc"), py::arg("csrc"), py::arg("cnt"), py::arg("cw"), py::arg("cb"), py::arg("cs"), py::arg("out"),\n'
     '        py::arg("tag"), py::arg("H"), py::arg("G"), py::arg("nb"), py::arg("max_log"), py::arg("act"));\n'
     '  m.def("instantiated"'),
]


XFEATS = {"noa": NOA_PATCHES, "xs": XS_PATCHES, "prof": PROF_PATCHES, "grid": [], "l2pf": L2PF_PATCHES,
          "pc": PC_PATCHES, "ww": [], "kx": [], "ldo": LDO_PATCHES, "cmo": CMO_PATCHES, "pp": PP_PATCHES, "pq": PQ_PATCHES,
          "s0": S0_PATCHES, "ff1": _ff_patches(1, 0), "ff2": _ff_patches(2, 0), "ffs2": _ff_patches(1, 256),
          "ffs4": _ff_patches(1, 512), "ffs1": _ff_patches(1, 128), "ffd2": _ff_patches(1, 256, every=True),
          "ffa2": _ff_patches(1, 256, refill=True), "ffd1": _ff_patches(1, 128, every=True),
          "ffd4": _ff_patches(1, 512, every=True), "ffda2": _ff_patches(1, 256, every=True, refill=True),
          "ffm": FFM_PATCHES, "ffk2": _ffk_patches(2), "ffk4": _ffk_patches(4), "rso": RSO_PATCHES, "rel": REL_PATCHES,
          "ta": TA_PATCHES, "qkc": QKC_PATCHES}
XHOST = {"grid": GRID_HOST_PATCHES, "l2pf": L2PF_HOST_PATCHES, "pc": PC_HOST_PATCHES, "ww": WW_HOST_PATCHES,
         "ta": TA_HOST_PATCHES, "qkc": QKC_HOST_PATCHES}
XKH = {"l2pf": L2PF_KERNEL_H, "pc": PC_KERNEL_H, "ta": TA_KERNEL_H, "qkc": QKC_KERNEL_H}
XORDER = ["ww", "kx", "pc", "ta", "xs", "pp", "pq", "s0", "ff1", "ff2", "ffs2", "ffs4", "ffs1", "ffd2", "ffa2", "ffd1", "ffd4", "ffda2", "ffm", "ffk2", "ffk4", "rso", "rel", "prof", "grid", "l2pf", "ldo", "cmo", "noa", "qkc"]  # application order (prof anchors survive the xs patches)


def generate_template() -> str:
    src = open(os.path.join(C1, "marlin_template.h")).read()
    # the B-stream patches only: the kernel names stay (separate extension)
    pats = [p for p in w3b.PATCHES if "__global__ void" not in p[0]]
    src = _apply(src, pats + (w3b.PATCHES_FASTDQ if w3b.fastdq() else []))
    assert version() in (1, 2, 3), f"ATDEV_C1_W3_V={version()}: only 1, 2 and 3 exist"
    if version() >= 2:
        assert w3b.fastdq(), f"ATDEV_C1_W3_V={version()} needs ATDEV_W3_DEFINES=W3_FASTDQ=1"
        src = _apply(src, DQ2_PATCHES)
    if version() >= 3:
        src = _apply(src, DQ3_TEMPLATE_HELPERS + DQ3_PATCHES)
    xf = xfeat()
    assert len({"pp", "pq", "xs"} & set(xf)) <= 1, "ATDEV_C1W3_X: pp / pq / xs use the same smem after the A ring"
    assert len({"ff1", "ff2", "ffs2", "ffs4", "ffs1", "ffd2", "ffa2", "ffd1", "ffd4", "ffda2", "ffm", "ffk2", "ffk4"} & set(xf)) <= 1, "ATDEV_C1W3_X: one ff* variant at a time"
    if "ta" in xf:  # LL4: built on the pc partition code; owns the cfg[6] grid override itself
        assert "pc" in xf and "grid" not in xf and not {"xs", "l2pf"} & set(xf), "ATDEV_C1W3_X=ta needs pc, no grid/xs/l2pf"
    if "ww" in xf:
        assert version() == 2, "ATDEV_C1W3_X=ww is built on the V2 kernel (ATDEV_C1_W3_V=2)"
        src = _ww_transform(src)
    for f in sorted(xf, key=XORDER.index):
        src = _apply(src, XFEATS[f])
    if "prof" in xf and "xs" in xf:
        src = _apply(src, PROF_XS)
    if "prof" in xf and ("pp" in xf or "pq" in xf):
        src = _apply(src, PROF_PP)
    return ("// GENERATED by atdev_serve/marlin_c1_w3_build.py from csrc/marlin_c1/marlin_template.h + the\n"
            "// marlin_plus_w3_build B-stream patches" + (f" (V{version()})" if version() > 1 else "") +
            (f" + X {','.join(xf)}" if xf else "") + " -- do not edit.\n" + src)


def generate_host() -> str:
    src = _apply(open(os.path.join(C1, "mp_host.cu")).read(), HOST_PATCHES)
    for f in sorted(xfeat(), key=XORDER.index):
        src = _apply(src, XHOST.get(f, []))
    if {"ww", "kx"} & set(xfeat()):  # big / register-heavy CTAs: check residency for every launch (bps 1 too)
        src = _apply(src, [("    if (ch_bps > 1 || fq) {\n", "    if (true) {  // ww / kx: always check residency\n")])
    return ("// GENERATED by atdev_serve/marlin_c1_w3_build.py from csrc/marlin_c1/mp_host.cu -- do not edit.\n"
            + src)


def instantiations(mode: int) -> list[tuple]:
    only = os.environ.get("ATDEV_C1W3_ONLY", "")  # dev: "tk,tn,thr,st;..." restricts the build
    if only:
        sel = [tuple(int(v) for v in s.split(",")) for s in only.split(";") if s]
        return sorted((0, 1, 0) + s for s in sel if not (mode == 3 and s[1] != 128))
    out = []
    ww = "ww" in xfeat()
    tiles = WW_TILES if ww else TILES + (V3_TILES if version() >= 3 else [])
    if "kx" in xfeat():
        tiles = tiles + (KX_WW_TILES if ww else KX_TILES)
    for tk, tn, thr in tiles:
        if mode == 3 and tn != 128:  # SiLU pair: one gate + one up n64 tile per CTA tile
            continue
        if ww and not ww_ok(tk, tn, thr):
            continue
        for st in (WW_STAGES if ww else STAGES):
            if tk < 128 and st % (128 // tk):  # a 128-group spans 128 / tk stages: ring length must be a multiple
                continue
            i = (1, 0, tk, tn, thr, st)
            if w3b._ok(i):
                out.append((0, 1, 0, tk, tn, thr, st))
    return sorted(out)


_MODS: dict = {}


def load(mode: int, verbose: bool = False):
    key = (mode, w3b.fastdq(), version(), xfeat(), os.environ.get("ATDEV_C1W3_NVCC", ""))
    if key in _MODS:
        return _MODS[key]
    import importlib
    from torch.utils.cpp_extension import load as _load
    mb = importlib.util.module_from_spec(importlib.util.find_spec("atdev_serve.marlin_plus_build"))
    mb.__spec__.loader.exec_module(mb)  # private instance (own GEN) for its generate()
    v = version()
    xf = xfeat()
    gen = os.path.join(GEN_ROOT, f"m{mode}{'_fdq' if w3b.fastdq() else ''}{'_v%d' % v if v > 1 else ''}"
                       + ("_x_" + "_".join(xf) if xf else ""))
    os.makedirs(gen, exist_ok=True)
    mb.CSRC, mb.GEN = C1, gen
    mb.instantiations = lambda: instantiations(mode)
    mb._smem_ok = lambda i: True  # filtered by the W3 smem model in instantiations()
    mb.CHAIN = []
    mb._write_if_changed(os.path.join(gen, "marlin_template.h"), generate_template())
    mb._write_if_changed(os.path.join(gen, "mp_host.cu"), generate_host())
    khx = [f for f in sorted(xf, key=XORDER.index) if f in XKH]
    if khx:  # patched MpEpi next to the generated sources (found before csrc/marlin_c1/kernel.h)
        kh = open(os.path.join(C1, "kernel.h")).read()
        for f in khx:
            kh = _apply(kh, XKH[f])
        mb._write_if_changed(os.path.join(gen, "kernel.h"), kh)
    # the generated kernel files include "marlin_template.h" -> found next to them (gen) first
    srcs = mb.generate() + [os.path.join(gen, "mp_host.cu")]
    rstd_at = os.environ.get("ATDEV_E6_RSTD_AT", "0")
    extra = [f"-D{d}" for d in os.environ.get("ATDEV_E6_DEFS", c1b.DEFAULT_DEFS).split(",") if d]
    defs = mb.defines() + [f"-DE6_MODE={mode}", f"-DE6_RSTD_AT={rstd_at}"] + extra
    # W3-gap (dev): extra nvcc flags, e.g. ATDEV_C1W3_NVCC=-maxrregcount=128 (register cap -> more CTAs/SM; same
    # kernel text -> bit-identical results). Part of the extension hash.
    defs = defs + [f for f in os.environ.get("ATDEV_C1W3_NVCC", "").split(",") if f]
    h = hashlib.sha256()
    for p in sorted(os.listdir(C1)):
        fp = os.path.join(C1, p)
        if os.path.isfile(fp):
            h.update(open(fp, "rb").read())
    for p in srcs + [os.path.join(gen, "marlin_template.h"), __file__, w3b.__file__] + \
            ([os.path.join(gen, "kernel.h")] if khx else []):
        h.update(open(p, "rb").read())
    h.update(" ".join(defs).encode())
    name = f"atdev_marlin_c1w3_m{mode}_" + h.hexdigest()[:10]
    os.environ.setdefault("MAX_JOBS", "20")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    _MODS[key] = _load(
        name=name, sources=srcs, verbose=verbose,
        extra_include_paths=[gen, C1],
        extra_cflags=["-O3", "-std=c++17", "-DMARLIN_NAMESPACE_NAME=marlin_plus"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-DMARLIN_NAMESPACE_NAME=marlin_plus", *defs,
                           "-gencode=arch=compute_80,code=sm_80", "--expt-relaxed-constexpr",
                           "-static-global-template-stub=false",
                           "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_HALF2_OPERATORS__"],
    )
    return _MODS[key]


if __name__ == "__main__":
    import sys
    import time
    for md in (int(v) for v in (sys.argv[1:] or ["1", "3"])):
        t = time.time()
        m = load(md, verbose=False)
        print("built W3 mode", md, m.__name__, "fastdq", w3b.fastdq(), "in %.0fs" % (time.time() - t),
              len(m.instantiated()), "kernels", flush=True)
