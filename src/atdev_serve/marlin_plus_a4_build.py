"""Build the marlin_plus W4A4 extension (csrc/marlin_plus_a4): s4 activations x the int4 Marlin
weights (m16n8k64 s4 mma), used for the large int8-activation GEMMs of the c=32 stack.

Instantiation set: exactly the a4 entries of the ATDEV_MP_CONFIG json. Generated sources go into
csrc/marlin_plus_a4/gen_<hash of the set>.
"""
from __future__ import annotations

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(HERE, "csrc", "marlin_plus_a4")
MP_CSRC = os.path.join(HERE, "csrc", "marlin_plus")


def smem_bytes(tmb, tn, st):
    tnb = tn // 16
    red = (2 * tnb + 1) * 16 * tmb
    b = st * tn * 4
    s = st * 2 * tnb
    a_s = st * (16 * tmb // 4)
    a = st * 2 * tmb * 32
    return 16 * (max(red, b) + s + a_s + a)


def _ok(i) -> bool:
    tmb, tn, thr, st, mw, defer, minb = i
    if tmb % mw or thr != mw * tn or thr > 512 or thr < 64 or st % 2:
        return False
    if minb * thr > 1024:
        return False
    lim = 163 * 1024 // minb - (1024 if minb > 1 else 0) - 64
    return smem_bytes(tmb, tn, st) <= lim


def cfg_instantiations(path: str) -> set:
    """The a4 entries ({"a4": true, "cfg": [tn, threads, stages, bps, tmb, MW, DEFER]}) of a
    marlin_plus config json, as instantiation tuples."""
    out = set()
    for shapes in json.load(open(path)).get("a8", {}).values():
        for ent in shapes.values():
            if ent.get("a4"):
                tn, thr, st, bps, tmb, mw, defer = ent["cfg"][:7]
                out.add((tmb, tn, thr, st, mw, defer, bps))
    return out


def instantiations() -> list[tuple]:
    path = os.environ.get("ATDEV_MP_CONFIG", "")
    if not path or not os.path.exists(path):
        raise RuntimeError(f"marlin_plus_a4: ATDEV_MP_CONFIG json not found ({path!r})")
    return sorted(i for i in cfg_instantiations(path) if _ok(i))


def gen_dir(insts: list[tuple]) -> str:
    h = hashlib.sha256(repr(sorted(insts)).encode()).hexdigest()[:10]
    return os.path.join(CSRC, "gen_" + h)


def _tmpl(i) -> str:
    tmb, tn, thr, st, mw, defer, minb = i
    return f"MarlinA4<{thr}, {tmb}, {tn // 16}, {st}, {mw}, {defer}, {minb}>"


def _write_if_changed(path, src):
    if os.path.exists(path) and open(path).read() == src:
        return
    with open(path, "w") as f:
        f.write(src)


def generate(per_file: int = 3) -> list[str]:
    insts = instantiations()
    GEN = gen_dir(insts)
    os.makedirs(GEN, exist_ok=True)
    files = []
    for fi in range(0, len(insts), per_file):
        chunk = insts[fi:fi + per_file]
        body = "".join(f"template __global__ void {_tmpl(i)}(MARLIN_KERNEL_PARAMS);\n" for i in chunk)
        src = ('#include "kernel.h"\n#include "mp_a4.cuh"\n'
               f"namespace MARLIN_NAMESPACE_NAME {{\n{body}}}\n")
        path = os.path.join(GEN, f"a4_kernels_{fi // per_file:03d}.cu")
        _write_if_changed(path, src)
        files.append(path)
    for f in os.listdir(GEN):
        p = os.path.join(GEN, f)
        if f.startswith("a4_kernels_") and p not in files:
            os.remove(p)
    sel = []
    for n, i in enumerate(insts):
        tmb, tn, thr, st, mw, defer, minb = i
        cond = (f"tmb == {tmb} && tnb == {tn // 16} && threads == {thr} && stages == {st} && "
                f"MW == {mw} && DEFER == {defer} && MINB == {minb}")
        sel.append(f"{'if' if n == 0 else 'else if'} ({cond})\n  kernel = {_tmpl(i)};\n")
    _write_if_changed(os.path.join(GEN, "a4_selector.h"), "".join(sel))
    lst = ",\n".join("{%d,%d,%d,%d,%d,%d,%d}" % i for i in insts)
    _write_if_changed(os.path.join(GEN, "a4_list.cu"),
                      "#include <vector>\n#include <cstdint>\n"
                      "std::vector<std::vector<int64_t>> mp_a4_instantiated() {\n"
                      f"  return {{{lst}}};\n}}\n")
    files.append(os.path.join(GEN, "a4_list.cu"))
    return files


_MOD = None


# split-K reduction and fold variants of the c=32 build (A4_PAR_REDUCE=6: per-slice partials summed
# by the last arriver in slice order; c32c_csum builds the same set with A4_PAR_REDUCE=7)
DEFINES = ("A4_FOLD2=1", "A4_MAGICQ=1", "A4_SCALAR_CTMP=1", "A4_PAR_REDUCE=6", "A4_C32C_PB=1")


def defines() -> list[str]:
    return ["-D" + d for d in DEFINES]


def load(verbose: bool = False):
    global _MOD
    if _MOD is None:
        _MOD = _build(verbose, defines())
    return _MOD


def _build(verbose, defs):
    from torch.utils.cpp_extension import load as _load
    srcs = generate() + [os.path.join(CSRC, "a4_host.cu")]
    gen = os.path.dirname(srcs[0])
    h = hashlib.sha256()
    for d in (MP_CSRC, CSRC):
        for p in sorted(os.listdir(d)):
            fp = os.path.join(d, p)
            if os.path.isfile(fp):
                h.update(open(fp, "rb").read())
    for p in srcs:
        h.update(open(p, "rb").read())
    h.update(open(__file__, "rb").read())
    h.update(" ".join(defs).encode())
    name = "atdev_mp_a4_" + h.hexdigest()[:10]
    os.environ.setdefault("MAX_JOBS", "20")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    return _load(
        name=name, sources=srcs, verbose=verbose,
        extra_include_paths=[CSRC, gen, MP_CSRC],
        extra_cflags=["-O3", "-std=c++17", "-DMARLIN_NAMESPACE_NAME=marlin_plus_a4"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-DMARLIN_NAMESPACE_NAME=marlin_plus_a4",
                           "-DMP_FOLD=1", "-DMP_PURE_ASM=0", *defs,
                           "-gencode=arch=compute_80,code=sm_80", "--expt-relaxed-constexpr",
                           "-static-global-template-stub=false", "-Xptxas=-v",
                           "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_HALF2_OPERATORS__"],
    )


if __name__ == "__main__":
    import time
    t = time.time()
    print(len(instantiations()), "instantiations")
    m = load(verbose=True)
    print("built", m.__name__, "in %.0fs" % (time.time() - t), len(m.instantiated()))
