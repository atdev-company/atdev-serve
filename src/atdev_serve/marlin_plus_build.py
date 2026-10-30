"""Build the marlin_plus CUDA extension (vLLM 0.31 Marlin source + exposed launch config).

The instantiation set is generated here; `python -m atdev_serve.marlin_plus_build`
compiles it (ninja, parallel) into ~/.cache/torch_extensions. Kernel sources:
src/atdev_serve/csrc/marlin_plus/ (copied from vLLM v0.31.0 csrc, Apache-2.0).
"""
from __future__ import annotations

import hashlib
import itertools
import os

CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "marlin_plus")
GEN = os.path.join(CSRC, "gen")

# (a8, thread_m_blocks, m8, thread_k, thread_n, threads, stages)
STOCK_LARGE = [(64, 256, 256), (64, 128, 128), (128, 64, 128)]
STOCK_SMALL = [(128, 128, 256), (64, 128, 128), (128, 64, 128)]
EXTRA = [(128, 128, 256), (128, 256, 256), (256, 64, 128), (256, 128, 256), (64, 64, 128)]


def instantiations() -> list[tuple]:
    out = set()
    # W4A8 (int8 activations): stock set (stages 4) for tmb 1..4
    for tmb in (1, 2, 3, 4):
        cfgs = STOCK_LARGE if tmb > 1 else STOCK_SMALL
        for c in cfgs + [(128, 64, 128)]:
            out.add((1, tmb, 0) + c + (4,))
    # W4A8 sweep at M=64 (tmb 4): every config x stages
    for c, st in itertools.product(STOCK_LARGE + EXTRA, (4, 6, 8)):
        out.add((1, 4, 0) + c + (st,))
    for c, st in itertools.product(STOCK_LARGE + [(128, 128, 256)], (6, 8)):
        for tmb in (2, 3):
            out.add((1, tmb, 0) + c + (st,))
    # W4A16 (bf16): m8 (M<=8) and tmb 1 (M<=16) sweep; tmb 2..4 stock
    for c, st in itertools.product(STOCK_SMALL + EXTRA, (4, 6, 8)):
        out.add((0, 1, 1) + c + (st,))
        out.add((0, 1, 0) + c + (st,))
    for tmb in (2, 3, 4):
        for c in STOCK_LARGE:
            out.add((0, tmb, 0) + c + (4,))
    # W4A16 M=17..32 (tree verify N=24/32): tmb 2 sweep
    for c, st in itertools.product(STOCK_LARGE + [(128, 128, 256), (128, 64, 128)], (4, 6, 8)):
        out.add((0, 2, 0) + c + (st,))
    return sorted(out)


# P1 stage 2 chain kernel: (a8, tmb, m8, tk, tn, threads, stages) -- stock small-M
# configs (c=1 M=16 bf16) and the marlin_plus M=64 int8 configs
CHAIN = [(0, 1, 0, 128, 128, 256, 4), (0, 1, 0, 64, 128, 128, 4), (0, 1, 0, 128, 64, 128, 4),
         (0, 1, 1, 128, 128, 256, 4), (0, 1, 1, 128, 64, 128, 4), (0, 2, 0, 64, 256, 256, 4),
         (1, 4, 0, 128, 256, 256, 4), (1, 4, 0, 128, 128, 256, 6), (1, 4, 0, 64, 256, 256, 4)]


def _tmpl_chain(i) -> str:
    return _tmpl(i).replace("Marlin<", "MarlinChain<", 1)


def _smem_ok(i) -> bool:
    a8, tmb, m8, tk, tn, thr, st = i
    tb_m = 8 if m8 else tmb * 16
    sh_a = st * tb_m * tk * (1 if a8 else 2)
    sh_b = st * (tk * tn // 8) * 4
    sh_red = tmb * 16 * (tn + 8) * 2
    sh_s = (tk + 127) // 128 * tn * 2 * st
    return max(sh_b, sh_red) + tn * 2 + sh_a + sh_s <= 163 * 1024


def _tmpl(i) -> str:
    a8, tmb, m8, tk, tn, thr, st = i
    a = "vllm::kS8.id()" if a8 else "vllm::kBFloat16.id()"
    return (f"Marlin<{a}, vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), "
            f"{thr}, {tmb}, {tn // 16}, {tk // 16}, {'true' if m8 else 'false'}, {st}, 8, false>")


def generate(per_file: int = 3) -> list[str]:
    os.makedirs(GEN, exist_ok=True)
    insts = [i for i in instantiations() if _smem_ok(i)]
    files = []
    for fi in range(0, len(insts), per_file):
        chunk = insts[fi:fi + per_file]
        body = "".join(f"template __global__ void {_tmpl(i)}(MARLIN_KERNEL_PARAMS);\n" for i in chunk)
        src = ('#include "kernel.h"\n#include "marlin_template.h"\n'
               f"namespace MARLIN_NAMESPACE_NAME {{\n{body}}}\n")
        path = os.path.join(GEN, f"mp_kernels_{fi // per_file:03d}.cu")
        _write_if_changed(path, src)
        files.append(path)
    sel = []
    for n, i in enumerate(insts):
        a8, tmb, m8, tk, tn, thr, st = i
        cond = (f"a_type == {'vllm::kS8' if a8 else 'vllm::kBFloat16'} && thread_m_blocks == {tmb} && "
                f"thread_n_blocks == {tn // 16} && thread_k_blocks == {tk // 16} && "
                f"m_block_size_8 == {'true' if m8 else 'false'} && threads == {thr} && "
                f"stages == {st} && group_blocks == 8 && b_type == vllm::kU4B8 && "
                f"c_type == vllm::kBFloat16 && !has_zp && !is_zp_float")
        sel.append(f"{'if' if n == 0 else 'else if'} ({cond})\n  kernel = {_tmpl(i)};\n")
    _write_if_changed(os.path.join(GEN, "mp_selector.h"), "".join(sel))
    # chain kernel instantiations + selector
    chain = [i for i in CHAIN if _smem_ok(i)]
    body = "".join(f"template __global__ void {_tmpl_chain(i)}(MpJobs, int, int4*, int*, int*, bool, int);\n"
                   for i in chain)
    _write_if_changed(os.path.join(GEN, "mp_chain_kernels.cu"),
                      '#include "kernel.h"\n#include "marlin_template.h"\n'
                      f"namespace MARLIN_NAMESPACE_NAME {{\n{body}}}\n")
    files.append(os.path.join(GEN, "mp_chain_kernels.cu"))
    sel = []
    for n, i in enumerate(chain):
        a8, tmb, m8, tk, tn, thr, st = i
        cond = (f"a_type == {'vllm::kS8' if a8 else 'vllm::kBFloat16'} && thread_m_blocks == {tmb} && "
                f"thread_n_blocks == {tn // 16} && thread_k_blocks == {tk // 16} && "
                f"m_block_size_8 == {'true' if m8 else 'false'} && threads == {thr} && stages == {st}")
        sel.append(f"{'if' if n == 0 else 'else if'} ({cond})\n  kernel = {_tmpl_chain(i)};\n")
    _write_if_changed(os.path.join(GEN, "mp_chain_selector.h"), "".join(sel))
    lst = ",\n".join("{%d,%d,%d,%d,%d,%d,%d}" % i for i in insts)
    _write_if_changed(os.path.join(GEN, "mp_list.cu"),
                      "#include <vector>\n#include <cstdint>\n"
                      "std::vector<std::vector<int64_t>> mp_instantiated() {\n"
                      f"  return {{{lst}}};\n}}\n")
    files.append(os.path.join(GEN, "mp_list.cu"))
    return files


def _write_if_changed(path, src):
    if os.path.exists(path) and open(path).read() == src:
        return
    with open(path, "w") as f:
        f.write(src)


_MOD = None


DEFAULT_DEFINES = "MP_FOLD=1,MP_PURE_ASM=0"


def defines() -> list[str]:
    """ATDEV_MP_DEFINES (comma list, e.g. "MP_FOLD=1,MP_PURE_ASM=1") -> -D flags."""
    s = os.environ.get("ATDEV_MP_DEFINES", DEFAULT_DEFINES)
    return ["-D" + d.strip() for d in s.split(",") if d.strip()]


def load(verbose: bool = False):
    """Compile (cached) and import the extension."""
    global _MOD
    if _MOD is not None:
        return _MOD
    from torch.utils.cpp_extension import load as _load
    srcs = generate() + [os.path.join(CSRC, "mp_host.cu")]
    h = hashlib.sha256()
    for p in sorted(os.listdir(CSRC)):
        fp = os.path.join(CSRC, p)
        if os.path.isfile(fp):
            h.update(open(fp, "rb").read())
    for p in srcs:
        h.update(open(p, "rb").read())
    h.update(open(__file__, "rb").read())
    h.update(" ".join(defines()).encode())
    name = "atdev_marlin_plus_" + h.hexdigest()[:10]
    os.environ.setdefault("MAX_JOBS", "20")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    _MOD = _load(
        name=name, sources=srcs, verbose=verbose,
        extra_include_paths=[CSRC, GEN],
        extra_cflags=["-O3", "-std=c++17", "-DMARLIN_NAMESPACE_NAME=marlin_plus"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-DMARLIN_NAMESPACE_NAME=marlin_plus", *defines(),
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
    print(len([i for i in instantiations() if _smem_ok(i)]), "instantiations")
    m = load(verbose=True)
    print("built", m.__name__, "in %.0fs" % (time.time() - t), len(m.instantiated()))
