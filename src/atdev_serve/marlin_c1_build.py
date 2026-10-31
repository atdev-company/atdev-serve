"""Generator / build helpers for the marlin_c1 extensions (marlin_c1_w3_build): a private copy of the marlin_plus
Marlin W4A16 kernel (csrc/marlin_c1) with fused epilogues for the c=1 verify batch (8 < M <= 16).

The epilogue mode is a compile-time constant (E6_MODE): 1 = producer (residual add + norm input + per-tile sum of
squares), 2 = consumer (row rstd applied to the accumulator), 3 = consumer + SiLU*mul pair; see
csrc/marlin_c1/kernel.h (MpEpi).
"""
from __future__ import annotations

import hashlib
import os

CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "marlin_c1")
CUDA_FLAGS = ["-gencode=arch=compute_80,code=sm_80", "--expt-relaxed-constexpr",
              "-static-global-template-stub=false",
              "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
              "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
              "-U__CUDA_NO_HALF2_OPERATORS__"]


def _tmpl(i) -> str:
    """i = (a8, thread_m_blocks, m8, thread_k, thread_n, threads, stages)"""
    a8, tmb, m8, tk, tn, thr, st = i
    a = "vllm::kS8.id()" if a8 else "vllm::kBFloat16.id()"
    return (f"Marlin<{a}, vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id(), "
            f"{thr}, {tmb}, {tn // 16}, {tk // 16}, {'true' if m8 else 'false'}, {st}, 8, false>")


def write_if_changed(path, src):
    if os.path.exists(path) and open(path).read() == src:
        return
    with open(path, "w") as f:
        f.write(src)


def generate(gen: str, insts: list[tuple], per_file: int = 3) -> list[str]:
    """kernel instantiation files (including "marlin_template.h", found first in `gen` if present there),
    the host selector mp_selector.h and mp_list.cu (instantiated())."""
    os.makedirs(gen, exist_ok=True)
    files = []
    for fi in range(0, len(insts), per_file):
        body = "".join(f"template __global__ void {_tmpl(i)}(MARLIN_KERNEL_PARAMS);\n"
                       for i in insts[fi:fi + per_file])
        path = os.path.join(gen, f"mp_kernels_{fi // per_file:03d}.cu")
        write_if_changed(path, '#include "kernel.h"\n#include "marlin_template.h"\n'
                               f"namespace MARLIN_NAMESPACE_NAME {{\n{body}}}\n")
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
    write_if_changed(os.path.join(gen, "mp_selector.h"), "".join(sel))
    lst = ",\n".join("{%d,%d,%d,%d,%d,%d,%d}" % i for i in insts)
    write_if_changed(os.path.join(gen, "mp_list.cu"),
                     "#include <vector>\n#include <cstdint>\n"
                     "std::vector<std::vector<int64_t>> mp_instantiated() {\n"
                     f"  return {{{lst}}};\n}}\n")
    files.append(os.path.join(gen, "mp_list.cu"))
    return files


def build(name_prefix: str, srcs: list[str], include: list[str], defs: list[str], hashed: list[str],
          verbose: bool = False):
    """JIT-compile (cached in TORCH_EXTENSIONS_DIR) under a name that hashes every source and define."""
    from torch.utils.cpp_extension import load as _load
    h = hashlib.sha256()
    for p in sorted(os.listdir(CSRC)):
        fp = os.path.join(CSRC, p)
        if os.path.isfile(fp):
            h.update(open(fp, "rb").read())
    for p in srcs + hashed:
        h.update(open(p, "rb").read())
    h.update(" ".join(defs).encode())
    os.environ.setdefault("MAX_JOBS", "20")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    return _load(
        name=name_prefix + h.hexdigest()[:10], sources=srcs, verbose=verbose,
        extra_include_paths=include,
        extra_cflags=["-O3", "-std=c++17", "-DMARLIN_NAMESPACE_NAME=marlin_plus"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-DMARLIN_NAMESPACE_NAME=marlin_plus", *defs, *CUDA_FLAGS])
