"""Build the marlin_plus W3 extension: vLLM's Marlin W4A16 kernel with a 3-bit planar weight
stream (used by w3_mixed for the 3-bit layers outside the c1_fuse fused epilogues).

The kernel source `csrc/marlin_plus_w3/gen/marlin_template_w3.h` is *generated* here from
`csrc/marlin_plus/marlin_template.h` by a handful of anchored text patches (asserted to match
exactly once), so the pipeline, tile schedule, split-K reduction, scales and epilogue stay
byte-for-byte vLLM's; only the B stream changes:

  * weights: 3-bit values q in [-4, 3] stored as u = q + 4 (3 bits). Every 32 weights that one
    thread consumes per k16 step (= one int4 of the Marlin int4 layout = 4 words w0..w3, one per
    n8 tile) become 3 words in 3 planes: plane p holds u(w_p) in the low 3 bits of each nibble
    and bit p of u(w3) in bit 3 -> 96 bits per 32 weights (0.758x the bytes of int4 g128).
  * smem/global layout per k16 slab: [3 planes][tn/2 words] (CTA slice) / [3 planes][N/2 words].
  * fetch_to_registers rebuilds the 4 int4-layout words (nibbles q + 8 = uint4b8) with 3 LDS.32 +
    ~12 ALU ops, so the stock lop3 uint4b8 dequant, mma and everything downstream are unchanged.
    With the same (thread_k, thread_n, threads) as stock Marlin the output is bit-identical to
    stock Marlin W4A16 running the same 3-bit-valued weights (same reduction order).
  * PATCHES_FASTDQ: the three plane words feed a 3-bit dequant directly (see below).
"""
from __future__ import annotations

import hashlib
import itertools
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(HERE, "csrc", "marlin_plus_w3")
MP_CSRC = os.path.join(HERE, "csrc", "marlin_plus")
GEN = os.path.join(CSRC, "gen")

PATCHES = [
    # A1: W3 B-stream constants (after the stock B constants)
    ("""  constexpr int b_sh_stage =
      b_sh_stride * thread_k_blocks / (is_a_8bit ? 2 : 1);
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;
""", """  constexpr int b_sh_stage =
      b_sh_stride * thread_k_blocks / (is_a_8bit ? 2 : 1);
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;
  // W3: 3-bit planar B. Per k16 slab: 3 planes x b3_words (= tn/2) words in smem (CTA slice),
  // 3 planes x prob_n/2 words in global. One int4 of the int4 layout (32 weights) <-> 3 words.
  static_assert(!is_a_8bit, "W3 kernel: bf16 activations only");
  static_assert(b_type.size_bits() == 4, "W3 kernel: uint4b8 dequant path");
  constexpr int b3_words = thread_n_blocks * 8;
  constexpr int b3_sh_slab = 3 * b3_words / 4;  // int4 per slab
  constexpr int b3_sh_stage = b3_sh_slab * thread_k_blocks;
  constexpr int b3_ld_iters = div_ceil(b3_sh_stage, threads);
  const int b3_gl_stride = 3 * prob_n / 8;  // int4 per k16 slab in global
  const int b3_gl_rd_delta_o = b3_gl_stride * thread_k_blocks;
"""),
    # A2: per-thread global offsets + smem read coordinates
    ("""  int b_gl_rd;
  if (threads <= b_sh_stride) {
    b_gl_rd = threadIdx.x;
  } else {
    b_gl_rd =
        b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
  }

  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  auto b_sh_rd = threadIdx.x * b_thread_vecs;
  b_sh_rd += b_sh_rd / b_sh_stride * (b_sh_stride * (b_sh_wr_iters - 1));
""", """  // W3: static per-thread chunk offsets (int4) relative to the stage base
  int b3_gl_off[b3_ld_iters];
  #pragma unroll
  for (int i = 0; i < b3_ld_iters; i++) {
    int c = threads * i + threadIdx.x;
    int slab = c / b3_sh_slab, rem = c % b3_sh_slab;
    int p = rem / (b3_words / 4), j = rem % (b3_words / 4);
    b3_gl_off[i] = slab * b3_gl_stride + p * (prob_n / 8) + j;
  }
  int b_gl_rd = b3_gl_rd_delta_o * slice_row + (b3_words / 4) * slice_col;
  // smem read: slab = warp_row * b_sh_wr_iters + k % b_sh_wr_iters, group = t % b_sh_stride
  const int b3_warp_row = threadIdx.x / b_sh_stride;
  const int b3_grp = threadIdx.x % b_sh_stride;
"""),
    # A3: smem size of the B ring
    ("  constexpr int sh_b_size = stages * b_sh_stage;\n",
     "  constexpr int sh_b_size = stages * b3_sh_stage;\n"),
    # A4: global -> smem copy of a stage
    ("""      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < (b_sh_wr_iters * b_thread_vecs); i++) {
        constexpr int count = div_ceil(b_sh_stride, threads);
        int b_gl_idx =
            b_gl_rd + (i % count) * threads +
            b_gl_stride * (i / count) * div_ceil(threads, b_sh_stride);

        cp_async4(&sh_b_stage[threads * i + threadIdx.x], &B[b_gl_idx]);
      }

      b_gl_rd += b_gl_rd_delta_o;
""", """      int4* sh_b_stage = sh_b + b3_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < b3_ld_iters; i++) {
        if (b3_sh_stage % threads == 0 || threads * i + threadIdx.x < b3_sh_stage)
          cp_async4(&sh_b_stage[threads * i + threadIdx.x], &B[b_gl_rd + b3_gl_off[i]]);
      }
      b_gl_rd += b3_gl_rd_delta_o;
"""),
    # A5: smem -> registers: 3 plane words -> 4 int4-layout words (nibbles = q + 8)
    ("""    int4* sh_b_stage = sh_b + b_sh_stage * pipe;

  #pragma unroll
    for (int i = 0; i < b_thread_vecs; i++) {
      frag_b_quant[k % 2][i] = *reinterpret_cast<I4*>(
          &sh_b_stage[b_sh_stride * (k % b_sh_wr_iters) + b_sh_rd + i]);
    }
""", """    const uint32_t* sh_b3 =
        reinterpret_cast<const uint32_t*>(sh_b + b3_sh_stage * pipe) +
        (b3_warp_row * b_sh_wr_iters + (k % b_sh_wr_iters)) * (3 * b3_words) + b3_grp;
    const uint32_t w0 = sh_b3[0], w1 = sh_b3[b3_words], w2 = sh_b3[2 * b3_words];
    int* fq = reinterpret_cast<int*>(&frag_b_quant[k % 2][0]);
    fq[0] = (int)((w0 & 0x77777777u) + 0x44444444u);
    fq[1] = (int)((w1 & 0x77777777u) + 0x44444444u);
    fq[2] = (int)((w2 & 0x77777777u) + 0x44444444u);
    fq[3] = (int)((((w0 >> 3) & 0x11111111u) | ((w1 >> 2) & 0x22222222u) |
                   ((w2 >> 1) & 0x44444444u)) + 0x44444444u);
"""),
    # A6: slice change: reset the stage base
    ("""        b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) +
                  (threadIdx.x % b_sh_stride);
        b_gl_rd += b_sh_stride * slice_col + b_gl_rd_delta_o * slice_row;
""", """        b_gl_rd = b3_gl_rd_delta_o * slice_row + (b3_words / 4) * slice_col;
"""),
    # names: separate symbols from marlin_plus
    ("__global__ void Marlin(MARLIN_KERNEL_PARAMS) {\n  MarlinBody<",
     "__global__ void MarlinW3(MARLIN_KERNEL_PARAMS) {\n  MarlinW3Body<"),
    ("__global__ void MarlinChain(MpJobs jobs, int njobs, int4* __restrict__ C_tmp,\n                            int* locks, int* bar, bool use_fp32_reduce,\n                            int max_shared_mem) {",
     "__global__ void MarlinW3Chain(MpJobs jobs, int njobs, int4* __restrict__ C_tmp,\n                            int* locks, int* bar, bool use_fp32_reduce,\n                            int max_shared_mem) {"),
]


# FASTDQ: the three plane words are consumed *directly* by a 3-bit dequant (mask 0x00070007 drops
# the plane bit, bias 4 instead of 8), so only tile 3 needs reconstruction (3 SHF + 2 LOP3) and the
# per-nibble "+4" adds disappear.
PATCHES_FASTDQ = [
    ("""    int* fq = reinterpret_cast<int*>(&frag_b_quant[k % 2][0]);
    fq[0] = (int)((w0 & 0x77777777u) + 0x44444444u);
    fq[1] = (int)((w1 & 0x77777777u) + 0x44444444u);
    fq[2] = (int)((w2 & 0x77777777u) + 0x44444444u);
    fq[3] = (int)((((w0 >> 3) & 0x11111111u) | ((w1 >> 2) & 0x22222222u) |
                   ((w2 >> 1) & 0x44444444u)) + 0x44444444u);
""", """    int* fq = reinterpret_cast<int*>(&frag_b_quant[k % 2][0]);
    fq[0] = (int)w0;
    fq[1] = (int)w1;
    fq[2] = (int)w2;
    fq[3] = (int)(((w0 >> 3) & 0x11111111u) | ((w1 >> 2) & 0x22222222u) | ((w2 >> 1) & 0x44444444u));
"""),
    ("""  auto dequant_data = [&](int q, scalar_32bit_t* frag_b_ptr, int zp = 0) {
    if constexpr (a_type.size_bits() != b_type.size_bits()) {
""", """  auto dequant_data = [&](int q, scalar_32bit_t* frag_b_ptr, int zp = 0) {
    if constexpr (std::is_same<scalar_32bit_t, nv_bfloat162>::value) {
      // 3-bit nibbles (u = q3 + 4 in bits 0-2, bit 3 = foreign plane bit): mask 7, bias 128 + 4
      static constexpr uint32_t MASK = 0x00070007, EX = 0x43004300, SUB = 0x43044304;
      int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
      q >>= 4;
      int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
      frag_b_ptr[0] = __hsub2(*reinterpret_cast<nv_bfloat162*>(&lo), *reinterpret_cast<const nv_bfloat162*>(&SUB));
      frag_b_ptr[1] = __hsub2(*reinterpret_cast<nv_bfloat162*>(&hi), *reinterpret_cast<const nv_bfloat162*>(&SUB));
    } else if constexpr (a_type.size_bits() != b_type.size_bits()) {
"""),
]


def generate_template() -> str:
    src = open(os.path.join(MP_CSRC, "marlin_template.h")).read()
    for old, new in PATCHES + PATCHES_FASTDQ:
        n = src.count(old)
        assert n == 1, f"W3 patch anchor matched {n} times:\n{old[:120]}"
        src = src.replace(old, new)
    # the body function name (several occurrences: definition + 2 call sites)
    assert src.count("MarlinBody<") == 1 and "MarlinBody(" in src, src.count("MarlinBody")
    src = src.replace("MarlinBody<", "MarlinW3Body<").replace("MarlinBody(", "MarlinW3Body(")
    # the sm<80 stub kernel name
    src = src.replace("__global__ void Marlin(\n", "__global__ void MarlinW3(\n")
    return ('// GENERATED by atdev_serve/marlin_plus_w3_build.py from csrc/marlin_plus/marlin_template.h\n'
            '// (vLLM v0.31.0 Marlin, Apache-2.0) -- do not edit; see PATCHES.\n' + src)


# (thread_m_blocks, m8, thread_k, thread_n, threads, stages)
STOCK_SMALL = [(128, 128, 256), (64, 128, 128), (128, 64, 128)]
STOCK_LARGE = [(64, 256, 256), (64, 128, 128), (128, 64, 128)]
EXTRA = [(128, 256, 256), (256, 64, 128), (256, 128, 256)]


def instantiations() -> list[tuple]:
    out = set()
    for c, st in itertools.product(STOCK_SMALL + EXTRA + [(64, 256, 256)], (4, 6, 8)):
        out.add((1, 1) + c + (st,))
        out.add((1, 0) + c + (st,))
        out.add((2, 0) + c + (st,))
    for tmb in (3, 4):
        for c in STOCK_LARGE + [(128, 128, 256)]:
            out.add((tmb, 0) + c + (4,))
    return sorted(out)


def smem_bytes(i) -> int:
    tmb, m8, tk, tn, thr, st = i
    tb_m = 8 if m8 else tmb * 16
    sh_a = st * tb_m * tk * 2
    sh_b = st * (tk * tn * 3 // 32) * 4
    sh_red = tb_m * (tn + 8) * 2
    sh_s = (tk + 127) // 128 * tn * 2 * st
    return max(sh_b, sh_red) + tn * 2 + sh_a + sh_s


def _ok(i) -> bool:
    tmb, m8, tk, tn, thr, st = i
    # one k16 slab of the CTA slice must be a whole number of 16B chunks per plane
    if (tn // 2) % 4:
        return False
    return smem_bytes(i) <= 163 * 1024


def _tmpl(i) -> str:
    tmb, m8, tk, tn, thr, st = i
    return (f"MarlinW3<vllm::kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), "
            f"{thr}, {tmb}, {tn // 16}, {tk // 16}, {'true' if m8 else 'false'}, {st}, 8, false>")


def _write_if_changed(path, src):
    if os.path.exists(path) and open(path).read() == src:
        return
    with open(path, "w") as f:
        f.write(src)


def generate(per_file: int = 3) -> list[str]:
    os.makedirs(GEN, exist_ok=True)
    _write_if_changed(os.path.join(GEN, "marlin_template_w3.h"), generate_template())
    insts = [i for i in instantiations() if _ok(i)]
    files = []
    for fi in range(0, len(insts), per_file):
        chunk = insts[fi:fi + per_file]
        body = "".join(f"template __global__ void {_tmpl(i)}(MARLIN_KERNEL_PARAMS);\n" for i in chunk)
        src = ('#include "kernel.h"\n#include "marlin_template_w3.h"\n'
               f"namespace MARLIN_NAMESPACE_NAME {{\n{body}}}\n")
        path = os.path.join(GEN, f"w3_kernels_{fi // per_file:03d}.cu")
        _write_if_changed(path, src)
        files.append(path)
    for f in os.listdir(GEN):
        p = os.path.join(GEN, f)
        if f.startswith("w3_kernels_") and p not in files:
            os.remove(p)
    sel = []
    for n, i in enumerate(insts):
        tmb, m8, tk, tn, thr, st = i
        cond = (f"thread_m_blocks == {tmb} && thread_n_blocks == {tn // 16} && thread_k_blocks == {tk // 16} && "
                f"m_block_size_8 == {'true' if m8 else 'false'} && threads == {thr} && stages == {st}")
        sel.append(f"{'if' if n == 0 else 'else if'} ({cond})\n  kernel = {_tmpl(i)};\n")
    _write_if_changed(os.path.join(GEN, "w3_selector.h"), "".join(sel))
    lst = ",\n".join("{%d,%d,%d,%d,%d,%d}" % i for i in insts)
    _write_if_changed(os.path.join(GEN, "w3_list.cu"),
                      "#include <vector>\n#include <cstdint>\n"
                      "std::vector<std::vector<int64_t>> w3_instantiated() {\n"
                      f"  return {{{lst}}};\n}}\n")
    files.append(os.path.join(GEN, "w3_list.cu"))
    return files


_MOD = None


def load(verbose: bool = False):
    global _MOD
    if _MOD is not None:
        return _MOD
    from torch.utils.cpp_extension import load as _load
    srcs = generate() + [os.path.join(CSRC, "w3_host.cu")]
    h = hashlib.sha256()
    for d in (CSRC, MP_CSRC):
        for p in sorted(os.listdir(d)):
            fp = os.path.join(d, p)
            if os.path.isfile(fp):
                h.update(open(fp, "rb").read())
    for p in srcs:
        h.update(open(p, "rb").read())
    h.update(open(__file__, "rb").read())
    h.update(generate_template().encode())
    name = "atdev_marlin_plus_w3_" + h.hexdigest()[:10]
    os.environ.setdefault("MAX_JOBS", "20")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    _MOD = _load(
        name=name, sources=srcs, verbose=verbose,
        extra_include_paths=[MP_CSRC, GEN, CSRC],
        extra_cflags=["-O3", "-std=c++17", "-DMARLIN_NAMESPACE_NAME=marlin_plus_w3"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-DMARLIN_NAMESPACE_NAME=marlin_plus_w3",
                           "-DMP_FOLD=1", "-DMP_PURE_ASM=0",
                           "-gencode=arch=compute_80,code=sm_80", "--expt-relaxed-constexpr",
                           "-static-global-template-stub=false",
                           "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_HALF2_OPERATORS__"],
    )
    return _MOD


if __name__ == "__main__":
    import time
    t = time.time()
    print(len([i for i in instantiations() if _ok(i)]), "instantiations")
    m = load(verbose=True)
    print("built", m.__name__, "in %.0fs" % (time.time() - t), len(m.instantiated()))
