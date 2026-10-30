"""Build a small marlin_plus M-split W4A8 extension for the verify_split_head int4 tail
at c=32 (ATDEV_VSPLIT_TAIL_MS, MTP k=2: M=96 verify rows).

Stock Marlin runs M=96 as two launches (64 + 32 rows), each streaming the whole
182784 x 5120 int4 tail (468 MB). The M-split kernel (csrc/marlin_plus_ms, REFRAME A2)
handles tmb=6 (96 rows) in one pass. Microbench (results/vsplit_mtp/bench_m96_explore.txt):
the only bit-identical one-pass configs (stock tile 64x256, MW=1 / MW=2@512 thr) spill and run
1.3-2.5 ms; the fast one, 128,128,256,6,1,6,2,0 (606 us vs stock 759), changes the split-K
stripes -> ROUNDING-LEVEL (52 / 17.5M tail logits differ by 1 bf16 ulp), vsplit_mtp
ATDEV_VSPLIT_TAIL_MS + ATDEV_VSPLIT_TAIL_MS_EXACT=0.

Separate extension (own generated dir csrc/marlin_plus_ms/gen_vtail, own name) so the
shared marlin_plus_ms build cache and generated sources are untouched. Same kernel
sources. Instantiations: ATDEV_VSPLIT_TAIL_MS_SET (";"-list of
"tmb,tk,tn,threads,stages,MW,DEFER,MINB"), default = the stock-tile MW=1 set below.
"""
from __future__ import annotations

import hashlib
import os

from atdev_serve import marlin_plus_ms_build as msb

GEN = os.path.join(msb.CSRC, "gen_vtail")
# the fastest tmb=6 config of the sweep (rounding-level) + the stock-tile MW=1 one (exact, slow)
DEFAULT_SET = "6,128,128,256,6,2,0,1;6,64,256,256,4,1,0,1"

_MOD = None


def instantiations() -> list[tuple]:
    s = os.environ.get("ATDEV_VSPLIT_TAIL_MS_SET", DEFAULT_SET)
    out = {tuple(int(x) for x in t.split(",")) for t in s.split(";") if t.strip()}
    return sorted(out)


def _generate(insts: list[tuple]) -> list[str]:
    os.makedirs(GEN, exist_ok=True)
    files = []
    for n, i in enumerate(insts):
        src = ('#include "kernel.h"\n#include "mp_a8ms.cuh"\n'
               f"namespace MARLIN_NAMESPACE_NAME {{\n"
               f"template __global__ void {msb._tmpl(i, 0)}(MARLIN_KERNEL_PARAMS);\n}}\n")
        p = os.path.join(GEN, f"ms_kernels_{n:03d}.cu")
        msb._write_if_changed(p, src)
        files.append(p)
    for f in os.listdir(GEN):
        p = os.path.join(GEN, f)
        if f.startswith("ms_kernels_") and p not in files:
            os.remove(p)
    sel = []
    for n, i in enumerate(insts):
        tmb, tk, tn, thr, st, mw, defer, minb = i
        cond = (f"tmb == {tmb} && tkb == {tk // 16} && tnb == {tn // 16} && threads == {thr} && "
                f"stages == {st} && MW == {mw} && DEFER == {defer} && MINB == {minb}")
        sel.append(f"{'if' if n == 0 else 'else if'} ({cond} && U8 == 0)\n  kernel = {msb._tmpl(i)};\n")
    msb._write_if_changed(os.path.join(GEN, "ms_selector.h"), "".join(sel))
    lst = ",\n".join("{%d,%d,%d,%d,%d,%d,%d,%d}" % i for i in insts)
    p = os.path.join(GEN, "ms_list.cu")
    msb._write_if_changed(p, "#include <vector>\n#include <cstdint>\n"
                          "std::vector<std::vector<int64_t>> mp_ms_instantiated() {\n"
                          f"  return {{{lst}}};\n}}\n")
    files.append(p)
    return files


def load(verbose: bool = False):
    global _MOD
    if _MOD is not None:
        return _MOD
    from torch.utils.cpp_extension import load as _load
    insts = instantiations()
    srcs = _generate(insts) + [os.path.join(msb.CSRC, "ms_host.cu")]
    h = hashlib.sha256()
    for d in (msb.MP_CSRC, msb.CSRC):
        for p in sorted(os.listdir(d)):
            fp = os.path.join(d, p)
            if os.path.isfile(fp):
                h.update(open(fp, "rb").read())
    for p in srcs:
        h.update(open(p, "rb").read())
    h.update(open(__file__, "rb").read())
    name = "atdev_vtail_ms_" + h.hexdigest()[:10]
    os.environ.setdefault("MAX_JOBS", "20")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    flags = ["-O3", "-std=c++17", "-DMARLIN_NAMESPACE_NAME=marlin_plus_vtail"]
    # GEN first: its ms_selector.h must win over the shared gen/ dir (not on the path anyway)
    _MOD = _load(
        name=name, sources=srcs, verbose=verbose,
        extra_include_paths=[GEN, msb.CSRC, msb.MP_CSRC],
        extra_cflags=flags,
        extra_cuda_cflags=[*flags, "-DMP_FOLD=1", "-DMP_PURE_ASM=0",
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
    m = load(verbose=False)
    print("built", m.__name__, "in %.0fs" % (time.time() - t), m.instantiated())
