"""Build the marlin_plus W4A4 extension (P3, csrc/marlin_plus_a4).

Separate from marlin_plus_ms_build so that module's build cache stays valid.
Instantiation set: ATDEV_A4_SET = "m64" (tmb=4 sweep, default: what serving needs),
"wide" (tmb 1-3), "full", "dev" (a handful), "m96" (tmb=6 sweep, MTP k=2 at c=32),
"m128" (tmb=8 sweep), "k2" (m64 + m96 + m128), "cfg" (exactly the a4 entries of the
ATDEV_MP_CONFIG json: small serving build), "cfg:<path>" (same, explicit json) or an
explicit ";"-separated list of "tmb,tn,threads,stages,MW,DEFER,MINB".
Every set other than "m64" generates into its own csrc/marlin_plus_a4/gen_<hash> directory,
so builds of different sets never overwrite each other's generated sources.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(HERE, "csrc", "marlin_plus_a4")
MP_CSRC = os.path.join(HERE, "csrc", "marlin_plus")
GEN = os.path.join(CSRC, "gen")  # default ("m64") set; see gen_dir()


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


def _wide_m(tmb: int) -> set:
    """tmb 6 / 8 sweep (M=96 / 128): every (tn, MW) with MW | tmb, at most 4 m-blocks per warp
    (TMW = tmb/MW; more spills the int32 partials), DEFER only for TMW <= 3 (it doubles them)."""
    out = set()
    for tn, mw, st, defer, minb in itertools.product((64, 128, 256), (1, 2, 3, 4), (4, 6, 8), (0, 1), (1, 2)):
        if tmb % mw or tmb // mw > 4 or (defer and tmb // mw > 3):
            continue
        out.add((tmb, tn, mw * tn, st, mw, defer, minb))
    return out


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
    sel = os.environ.get("ATDEV_A4_SET", "m64")
    out = set()
    if sel == "dev":
        out = {(4, 256, 256, 4, 1, 0, 1), (4, 256, 256, 4, 1, 1, 1), (4, 128, 256, 6, 2, 0, 1),
               (4, 128, 256, 6, 2, 1, 1), (4, 128, 128, 4, 1, 0, 2), (1, 128, 128, 4, 1, 0, 2)}
    elif sel in ("full", "m64", "wide", "k2"):
        layouts = [(64, 2), (64, 4), (128, 1), (128, 2), (128, 4), (256, 1), (256, 2)]
        tmbs = {"full": (1, 2, 3, 4, 8), "m64": (4,), "wide": (1, 2, 3), "k2": (4,)}[sel]
        for tmb in tmbs:
            for (tn, mw), st, defer, minb in itertools.product(layouts, (4, 6, 8), (0, 1), (1, 2)):
                if tmb % mw:
                    continue
                out.add((tmb, tn, mw * tn, st, mw, defer, minb))
        if sel == "k2":
            out |= _wide_m(6) | _wide_m(8)
    elif sel in ("m96", "m128"):
        out = _wide_m(6 if sel == "m96" else 8)
    elif sel == "cfg" or sel.startswith("cfg:"):
        path = sel[4:] if sel.startswith("cfg:") else os.environ.get("ATDEV_MP_CONFIG", "")
        if not path or not os.path.exists(path):
            raise RuntimeError(f"ATDEV_A4_SET={sel}: config json not found ({path!r})")
        out = cfg_instantiations(path)
    else:
        for t in sel.split(";"):
            out.add(tuple(int(x) for x in t.split(",")))
    return sorted(i for i in out if _ok(i))


def gen_dir(insts: list[tuple]) -> str:
    if os.environ.get("ATDEV_A4_SET", "m64") == "m64":
        return GEN
    h = hashlib.sha256(repr(sorted(insts)).encode()).hexdigest()[:10]
    return os.path.join(CSRC, "gen_" + h)


def ws_instantiations() -> list[tuple]:
    sel = os.environ.get("ATDEV_A4WS_SET", "")
    out = set()
    if sel:
        for t in sel.split(";"):
            out.add(tuple(int(x) for x in t.split(",")))
    else:
        for (tn, mw), st, minb in itertools.product([(64, 2), (128, 1), (128, 2), (256, 1)], (4, 6, 8), (1, 2)):
            out.add((4, tn, mw * tn, st, mw, minb))
    def ok(i):
        tmb, tn, thr, st, mw, minb = i
        tnb = tn // 16
        smem = 16 * (st * (2 * tmb * 32 + tmb * 4 + tn * 4 + 2 * tnb) + (2 * tnb + 1) * 16 * tmb)
        return tmb % mw == 0 and thr == mw * tn and (thr + 32) * minb <= 1024 and \
            smem <= 163 * 1024 // minb - (1024 if minb > 1 else 0) - 128
    return sorted(i for i in out if ok(i))


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
        body = "".join(f"template __global__ void {_tmpl(i)}(MARLIN_KERNEL_PARAMS A4_XPARAMS);\n" for i in chunk)
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
    reds = sorted({(thr, tmb, tn // 16, mw) for (tmb, tn, thr, st, mw, defer, minb) in insts})
    rsel = []
    for n, (thr, tmb, tnb, mw) in enumerate(reds):
        rsel.append(f"{'if' if n == 0 else 'else if'} (threads == {thr} && tmb == {tmb} && tnb == {tnb} && MW == {mw})\n"
                    f"  r = A4Reduce<{thr}, {tmb}, {tnb}, {mw}>;\n")
    _write_if_changed(os.path.join(GEN, "a4_reduce_selector.h"), "".join(rsel))
    body = "".join(f"template __global__ void A4Reduce<{thr}, {tmb}, {tnb}, {mw}>(const int4*, int4*, int, int, const int*, int);\n"
                   for (thr, tmb, tnb, mw) in reds)
    _write_if_changed(os.path.join(GEN, "a4_reduce.cu"),
                      '#include "kernel.h"\n#include "mp_a4.cuh"\n'
                      f"namespace MARLIN_NAMESPACE_NAME {{\n{body}}}\n")
    files.append(os.path.join(GEN, "a4_reduce.cu"))
    ws = ws_instantiations()
    wsel, wbody = [], []
    for n, (tmb, tn, thr, st, mw, minb) in enumerate(ws):
        t = f"MarlinA4WS<{thr}, {tmb}, {tn // 16}, {st}, {mw}, {minb}>"
        wsel.append(f"{'if' if n == 0 else 'else if'} (tmb == {tmb} && tnb == {tn // 16} && threads_c == {thr} && "
                    f"stages == {st} && MW == {mw} && MINB == {minb})\n  kernel = {t};\n")
        wbody.append(f"template __global__ void {t}(MARLIN_KERNEL_PARAMS);\n")
    _write_if_changed(os.path.join(GEN, "a4ws_selector.h"), "".join(wsel) or "if (false) {}\n")
    _write_if_changed(os.path.join(GEN, "a4ws_kernels.cu"),
                      '#include "kernel.h"\n#include "mp_a4ws.cuh"\n'
                      f"namespace MARLIN_NAMESPACE_NAME {{\n{''.join(wbody)}}}\n")
    files.append(os.path.join(GEN, "a4ws_kernels.cu"))
    lst = ",\n".join("{%d,%d,%d,%d,%d,%d}" % i for i in ws)
    _write_if_changed(os.path.join(GEN, "a4ws_list.cu"),
                      "#include <vector>\n#include <cstdint>\n"
                      "std::vector<std::vector<int64_t>> mp_a4ws_instantiated() {\n"
                      f"  return {{{lst}}};\n}}\n")
    files.append(os.path.join(GEN, "a4ws_list.cu"))
    lst = ",\n".join("{%d,%d,%d,%d,%d,%d,%d}" % i for i in insts)
    _write_if_changed(os.path.join(GEN, "a4_list.cu"),
                      "#include <vector>\n#include <cstdint>\n"
                      "std::vector<std::vector<int64_t>> mp_a4_instantiated() {\n"
                      f"  return {{{lst}}};\n}}\n")
    files.append(os.path.join(GEN, "a4_list.cu"))
    return files


_MOD = None


def defines(ne: int | None = None) -> list[str]:
    """ATDEV_A4_DEFINES; ne overrides A4_NE (C32-NE: 1 = producer build, 2 = consumer build)."""
    s = os.environ.get("ATDEV_A4_DEFINES", "")
    ds = [d.strip() for d in s.split(",") if d.strip()]
    if ne is not None:
        ds = [d for d in ds if d.split("=")[0] != "A4_NE"] + ([f"A4_NE={ne}"] if ne else [])
    return ["-D" + d for d in ds]


_NE_MODS: dict = {}


def load_ne(mode: int, verbose: bool = False):
    """C32-NE variant module (mode 1 producer / 2 consumer), same instantiation set as load()."""
    m = _NE_MODS.get(mode)
    if m is None:
        m = _NE_MODS[mode] = _build(verbose, defines(mode))
    return m


_SILU_MODS: dict = {}


def load_silu(extra: str = "", verbose: bool = False):
    """C32h SiLU-epilogue variant module (A4_SILU=1; A4_NE dropped), same instantiation set as load().
    extra: additional comma-separated defines (timing ablations)."""
    m = _SILU_MODS.get(extra)
    if m is None:
        ds = [d for d in defines(0) if not d.startswith("-DA4_SILU")]
        ds += ["-DA4_SILU=1"] + ["-D" + d.strip() for d in extra.split(",") if d.strip()]
        m = _SILU_MODS[extra] = _build(verbose, ds)
    return m


def load(verbose: bool = False):
    global _MOD
    if _MOD is not None:
        return _MOD
    _MOD = _build(verbose, defines(0 if os.environ.get("ATDEV_C32_NE", "0") == "1" else None))
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
