"""Build the marlin_plus M-split W4A8 extension (REFRAME A2, csrc/marlin_plus_ms).

Separate from marlin_plus_build so that module's build cache stays valid.
Instantiation set: ATDEV_MS_SET = "dev" (a handful, for numerics), "full"
(sweep for the tuner, default) or an explicit ";"-separated list of
"tmb,tk,tn,threads,stages,MW,DEFER,MINB".
"""
from __future__ import annotations

import hashlib
import itertools
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(HERE, "csrc", "marlin_plus_ms")
MP_CSRC = os.path.join(HERE, "csrc", "marlin_plus")
GEN = os.path.join(CSRC, "gen")


def smem_bytes(tmb, tk, tn, st):
    tnb = tn // 16
    red = (2 * tnb + 1) * 16 * tmb
    b = st * tn * (tk // 32)
    s = st * 2 * tnb
    a = st * (tk // 16) * 16 * tmb
    return 16 * (4 * tmb + max(red, b) + s + a)


def _ok(i) -> bool:
    tmb, tk, tn, thr, st, mw, defer, minb = i
    if tmb % mw or thr != mw * tn or thr > 512 or thr < 64:
        return False
    if minb * thr > 1024 or (minb > 1 and thr > 512):
        return False
    g = 128 // tk
    if st % (2 * g):
        return False
    lim = 163 * 1024 // minb - (1024 if minb > 1 else 0) - 64
    return smem_bytes(tmb, tk, tn, st) <= lim


def instantiations() -> list[tuple]:
    sel = os.environ.get("ATDEV_MS_SET", "full")
    out = set()
    if sel == "dev":
        out = {(4, 64, 128, 256, 4, 2, 0, 1), (4, 64, 128, 256, 4, 2, 1, 1),
               (4, 64, 256, 256, 4, 1, 0, 1), (8, 64, 128, 256, 4, 2, 0, 1),
               (4, 128, 64, 256, 6, 4, 1, 1), (8, 64, 128, 512, 4, 4, 1, 1)}
    elif sel == "full":
        layouts = {4: [(64, 2), (64, 4), (128, 1), (128, 2), (128, 4), (256, 1), (256, 2)],
                   8: [(64, 2), (64, 4), (128, 2), (128, 4), (256, 2)],
                   # P3: 96-row tiles (MTP k=2 at c=32: 3 verify rows per request)
                   6: [(64, 2), (64, 3), (128, 2), (128, 3), (256, 2), (128, 1)]}
        for tmb, lay in layouts.items():
            for (tn, mw), (tk, st), defer, minb in itertools.product(
                    lay, [(64, 4), (64, 8), (128, 4), (128, 6), (128, 8)], (0, 1), (1, 2)):
                out.add((tmb, tk, tn, mw * tn, st, mw, defer, minb))
    else:
        for t in sel.split(";"):
            out.add(tuple(int(x) for x in t.split(",")))
    return sorted(i for i in out if _ok(i))


def _tmpl(i, u8=0) -> str:
    tmb, tk, tn, thr, st, mw, defer, minb = i
    return f"MarlinA8MS<{thr}, {tmb}, {tn // 16}, {tk // 16}, {st}, {mw}, {defer}, {minb}, {u8}>"


def _u8() -> bool:
    return os.environ.get("ATDEV_MS_U8", "0") == "1"


def _write_if_changed(path, src):
    if os.path.exists(path) and open(path).read() == src:
        return
    with open(path, "w") as f:
        f.write(src)


def generate(per_file: int = 2) -> list[str]:
    os.makedirs(GEN, exist_ok=True)
    insts = instantiations()
    files = []
    for fi in range(0, len(insts), per_file):
        chunk = insts[fi:fi + per_file]
        body = "".join(f"template __global__ void {_tmpl(i, u)}(MARLIN_KERNEL_PARAMS);\n"
                       for i in chunk for u in ((0, 1) if _u8() else (0,)))
        src = ('#include "kernel.h"\n#include "mp_a8ms.cuh"\n'
               f"namespace MARLIN_NAMESPACE_NAME {{\n{body}}}\n")
        path = os.path.join(GEN, f"ms_kernels_{fi // per_file:03d}.cu")
        _write_if_changed(path, src)
        files.append(path)
    # drop stale generated files from a larger set
    for f in os.listdir(GEN):
        p = os.path.join(GEN, f)
        if f.startswith("ms_kernels_") and p not in files:
            os.remove(p)
    sel = []
    for n, i in enumerate(insts):
        tmb, tk, tn, thr, st, mw, defer, minb = i
        cond = (f"tmb == {tmb} && tkb == {tk // 16} && tnb == {tn // 16} && threads == {thr} && "
                f"stages == {st} && MW == {mw} && DEFER == {defer} && MINB == {minb}")
        if _u8():
            sel.append(f"{'if' if n == 0 else 'else if'} ({cond})\n  kernel = U8 ? {_tmpl(i, 1)} : {_tmpl(i, 0)};\n")
        else:
            sel.append(f"{'if' if n == 0 else 'else if'} ({cond} && U8 == 0)\n  kernel = {_tmpl(i)};\n")
    _write_if_changed(os.path.join(GEN, "ms_selector.h"), "".join(sel))
    lst = ",\n".join("{%d,%d,%d,%d,%d,%d,%d,%d}" % i for i in insts)
    _write_if_changed(os.path.join(GEN, "ms_list.cu"),
                      "#include <vector>\n#include <cstdint>\n"
                      "std::vector<std::vector<int64_t>> mp_ms_instantiated() {\n"
                      f"  return {{{lst}}};\n}}\n")
    files.append(os.path.join(GEN, "ms_list.cu"))
    return files


_MOD = None


def defines() -> list[str]:
    s = os.environ.get("ATDEV_MS_DEFINES", "")
    return ["-D" + d.strip() for d in s.split(",") if d.strip()]


def load(verbose: bool = False):
    global _MOD
    if _MOD is not None:
        return _MOD
    from torch.utils.cpp_extension import load as _load
    srcs = generate() + [os.path.join(CSRC, "ms_host.cu")]
    h = hashlib.sha256()
    for d in (MP_CSRC, CSRC):
        for p in sorted(os.listdir(d)):
            fp = os.path.join(d, p)
            if os.path.isfile(fp):
                h.update(open(fp, "rb").read())
    for p in srcs:
        h.update(open(p, "rb").read())
    h.update(open(__file__, "rb").read())
    h.update(" ".join(defines()).encode())
    h.update(b"u8" if _u8() else b"")
    name = "atdev_mp_ms_" + h.hexdigest()[:10]
    os.environ.setdefault("MAX_JOBS", "20")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    _MOD = _load(
        name=name, sources=srcs, verbose=verbose,
        extra_include_paths=[CSRC, GEN, MP_CSRC],
        extra_cflags=["-O3", "-std=c++17", "-DMARLIN_NAMESPACE_NAME=marlin_plus_ms"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-DMARLIN_NAMESPACE_NAME=marlin_plus_ms",
                           "-DMP_FOLD=1", "-DMP_PURE_ASM=0", *defines(),
                           "-gencode=arch=compute_80,code=sm_80", "--expt-relaxed-constexpr",
                           "-static-global-template-stub=false", "-Xptxas=-v",
                           "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_HALF2_OPERATORS__"],
    )
    return _MOD


if __name__ == "__main__":
    import time
    t = time.time()
    print(len(instantiations()), "instantiations")
    m = load(verbose=True)
    print("built", m.__name__, "in %.0fs" % (time.time() - t), len(m.instantiated()))
