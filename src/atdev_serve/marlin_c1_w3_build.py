"""marlin_c1 (fused norm / SiLU epilogues, csrc/marlin_c1) with the 3-bit planar weight stream of
marlin_plus_w3, so c1_fuse can run its fused epilogues on the w3_mixed layers.

The kernel template is *generated* into csrc/marlin_c1/gen_w3/m<mode>/ from csrc/marlin_c1/marlin_template.h by
anchored text patches (each asserted to match exactly once):

  * the marlin_plus_w3_build B-stream patches (3-bit planar B, direct 3-bit dequant),
  * DQ2_PATCHES: shift-free dequant of the odd nibble pair (bit-identical values),
  * PC_*: host-precomputed split-K partition (same integers as the in-kernel divisions),
  * S0_PATCHES: the first slice of a split-K column skips its lock acquire (its lock is 0 by construction),
  * FFM_PATCHES: staged first pipeline fill at kernel start (fetch timing only).

None of them changes arithmetic or reduction order. The kernel symbols keep their marlin_c1 names (own
extension .so). The host (mp_host.cu) is copied with two patches: the smem model counts the 3-bit B ring, and
gemm_epi checks the planar B shape. B = int32 [K/16, 3, N/2] (w3_mixed.pack_w3), scales = the unchanged
Marlin-permuted bf16 [K/128, N]. One extension per epilogue mode.
"""
from __future__ import annotations

import os

from atdev_serve import marlin_c1_build as c1b
from atdev_serve import marlin_plus_w3_build as w3b

HERE = os.path.dirname(os.path.abspath(__file__))
C1 = os.path.join(HERE, "csrc", "marlin_c1")
GEN_ROOT = os.path.join(C1, "gen_w3")

# (thread_k, thread_n, threads) x stages, tmb 1 / m16 only (the fused epilogues need 8 < M <= 16)
# (64, 64, 128) is NOT valid with the W3 B stream (marlin_plus_w3 never instantiates it either)
TILES = [(64, 128, 128), (128, 128, 256), (256, 128, 256), (128, 64, 128), (256, 64, 128), (128, 128, 128),
         (64, 256, 256), (128, 256, 256)]
STAGES = (4, 6, 8)

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


# DQ2: the 3-bit dequant reads the odd nibble pair of a word in place. Bits 4-6 / 20-22 under the bf16 exponent
# 2^3 (0x4100 | u << 4) are exactly 8 + u, so (x - 12) = u - 4 like (128 + u) - 132 for the even pair: the two
# `q >>= 4` shifts per word go away, values bit-identical.
DQ2_PATCHES = [
    ("""      static constexpr uint32_t MASK = 0x00070007, EX = 0x43004300, SUB = 0x43044304;
      int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
      q >>= 4;
      int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
      frag_b_ptr[0] = __hsub2(*reinterpret_cast<nv_bfloat162*>(&lo), *reinterpret_cast<const nv_bfloat162*>(&SUB));
      frag_b_ptr[1] = __hsub2(*reinterpret_cast<nv_bfloat162*>(&hi), *reinterpret_cast<const nv_bfloat162*>(&SUB));
""", """      // W3: odd nibble pair in place (bits 4-6 under exponent 2^3 = 8 + u, minus 12), no shift
      static constexpr uint32_t MASK = 0x00070007, EX = 0x43004300, SUB = 0x43044304;
      static constexpr uint32_t MASK_HI = 0x00700070, EX_HI = 0x41004100, SUB_HI = 0x41404140;
      int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
      int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK_HI, EX_HI);
      frag_b_ptr[0] = __hsub2(*reinterpret_cast<nv_bfloat162*>(&lo), *reinterpret_cast<const nv_bfloat162*>(&SUB));
      frag_b_ptr[1] = __hsub2(*reinterpret_cast<nv_bfloat162*>(&hi), *reinterpret_cast<const nv_bfloat162*>(&SUB_HI));
"""),
]


# PC: the launch-uniform split-K partition values (iters, part-2 / part-1 tiles) and 40-bit magic multipliers for
# the per-CTA divisions by k_tiles / iters / n_tiles come in through MpEpi, replacing the dependent runtime integer
# divisions of the kernel prologue. Same integers (exact for the asserted ranges) -> bit-identical.
RO_LINE = "  float* Rout;  // consumer: optional rstd[16] out (written by CTA 0) for a second consumer of the same A\n"
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

# S0: slice 0 of a column (the first in the serial split-K chain) skips its acquire poll: its lock is 0 by
# construction (per-column locks of the part-2 partition, reset by every chain's last slice).
S0_PATCHES = [
    ("        barrier_acquire(&locks[locks_off], slice_idx);\n",
     "        if (slice_idx == 0 && part2_mn_tiles < gridDim.x) {\n"
     "          __syncthreads();  // S0: the chain's first slice, its per-column lock is 0 by construction\n"
     "        } else\n"
     "        barrier_acquire(&locks[locks_off], slice_idx);\n"),
]
# FFM: staged first fill at kernel start (refills unchanged): the consumer kernels of mode 2 sleep once after
# prologue stage 0, modes 1 / 3 after every prologue stage, so the first stage lands before the rest of the
# start burst. Fetch timing only -> bit-identical.
FFM_PATCHES = [
    ("  auto start_pipes = [&]() {\n\n  #pragma unroll\n    for (int i = 0; i < stages - 1; i++) {\n",
     "  bool ff_first = true;  // FF: staged fill at kernel start only\n"
     "  auto start_pipes = [&]() {\n\n  #pragma unroll\n    for (int i = 0; i < stages - 1; i++) {\n"
     "      if ((E6_MODE == 2 ? i == 1 : i > 0) && ff_first) { __nanosleep(256); }  // FF (ffm)\n"),
    ("    zero_accums();\n    wait_for_stage();\n    fetch_to_registers(0, 0);\n    fetch_scales_to_registers(0, 0);\n",
     "    ff_first = false;\n"
     "    zero_accums();\n    wait_for_stage();\n    fetch_to_registers(0, 0);\n    fetch_scales_to_registers(0, 0);\n"),
]


def generate_template() -> str:
    src = open(os.path.join(C1, "marlin_template.h")).read()
    # the B-stream patches only: the kernel names stay (separate extension)
    pats = [p for p in w3b.PATCHES if "__global__ void" not in p[0]]
    src = _apply(src, pats + w3b.PATCHES_FASTDQ + DQ2_PATCHES + PC_PATCHES + S0_PATCHES + FFM_PATCHES)
    return ("// GENERATED by atdev_serve/marlin_c1_w3_build.py from csrc/marlin_c1/marlin_template.h + the\n"
            "// marlin_plus_w3_build B-stream patches -- do not edit.\n" + src)


def generate_host() -> str:
    src = _apply(open(os.path.join(C1, "mp_host.cu")).read(), HOST_PATCHES + PC_HOST_PATCHES)
    return ("// GENERATED by atdev_serve/marlin_c1_w3_build.py from csrc/marlin_c1/mp_host.cu -- do not edit.\n"
            + src)


def instantiations(mode: int) -> list[tuple]:
    out = []
    for tk, tn, thr in TILES:
        if mode == 3 and tn != 128:  # SiLU pair: one gate + one up n64 tile per CTA tile
            continue
        for st in STAGES:
            if tk < 128 and st % (128 // tk):  # a 128-group spans 128 / tk stages: ring length must be a multiple
                continue
            if w3b._ok((1, 0, tk, tn, thr, st)):
                out.append((0, 1, 0, tk, tn, thr, st))
    return sorted(out)


_MODS: dict = {}


def load(mode: int, verbose: bool = False):
    if mode in _MODS:
        return _MODS[mode]
    gen = os.path.join(GEN_ROOT, f"m{mode}")
    os.makedirs(gen, exist_ok=True)
    c1b.write_if_changed(os.path.join(gen, "marlin_template.h"), generate_template())
    c1b.write_if_changed(os.path.join(gen, "mp_host.cu"), generate_host())
    # patched MpEpi next to the generated sources (found before csrc/marlin_c1/kernel.h)
    c1b.write_if_changed(os.path.join(gen, "kernel.h"), _apply(open(os.path.join(C1, "kernel.h")).read(), PC_KERNEL_H))
    # the generated kernel files include "marlin_template.h" -> found next to them (gen) first
    srcs = c1b.generate(gen, instantiations(mode)) + [os.path.join(gen, "mp_host.cu")]
    hashed = [os.path.join(gen, "marlin_template.h"), os.path.join(gen, "kernel.h"), __file__, w3b.__file__]
    _MODS[mode] = c1b.build(f"atdev_marlin_c1w3_m{mode}_", srcs, [gen, C1], [f"-DE6_MODE={mode}"], hashed, verbose)
    return _MODS[mode]


if __name__ == "__main__":
    import sys
    import time
    for md in (int(v) for v in (sys.argv[1:] or ["1", "2", "3"])):
        t = time.time()
        m = load(md, verbose=False)
        print("built W3 mode", md, m.__name__, "in %.0fs" % (time.time() - t), len(m.instantiated()), "kernels",
              flush=True)
