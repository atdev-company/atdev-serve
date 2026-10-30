"""E6: build marlin_c1 = private copy of csrc/marlin_plus with the E6 fused epilogues (csrc/marlin_c1,
docs/research/E6-c1-norms.md), restricted to the c=1 W4A16 M=16 instantiations. The production marlin_plus
build (and its source hash) is untouched."""
from __future__ import annotations

import importlib
import os

_MODS: dict = {}
# SiLU pair read straight from the staged tile (all threads busy) with __expf / __fdividef: +0.1 us on gate_up vs
# +2.9 us for the shuffle-paired IEEE expf / div version (results/e6/round5.txt)
DEFAULT_DEFS = "E6_SILU_SMEM=1,E6_SILU=1"
M16 = [(64, 128, 128), (128, 64, 128), (128, 128, 256)]


def load(mode: int = 1, verbose: bool = False):
    """one extension per epilogue mode (E6_MODE compile-time; 0 = the unmodified kernel code paths)."""
    if mode in _MODS:
        return _MODS[mode]
    from atdev_serve import marlin_plus_build as base
    mb = importlib.util.module_from_spec(importlib.util.find_spec("atdev_serve.marlin_plus_build"))
    mb.__spec__.loader.exec_module(mb)  # private module instance: own CSRC / GEN / _MOD
    mb.CSRC = os.path.join(os.path.dirname(os.path.abspath(base.__file__)), "csrc", "marlin_c1")
    mb.GEN = os.path.join(mb.CSRC, "gen")
    mb.instantiations = lambda: sorted({(0, 1, 0) + c + (st,) for c in M16 for st in (4, 6)})
    mb.CHAIN = []
    base_defs = mb.defines
    rstd_at = os.environ.get("ATDEV_E6_RSTD_AT", "0")
    extra = [f"-D{d}" for d in os.environ.get("ATDEV_E6_DEFS", DEFAULT_DEFS).split(",") if d]
    mb.defines = lambda: base_defs() + [f"-DE6_MODE={mode}", f"-DE6_RSTD_AT={rstd_at}"] + extra
    orig_load = mb.load

    def _load(verbose=False):
        # the extension name hashes the sources; prefix it so it never collides with marlin_plus
        from torch.utils import cpp_extension
        real = cpp_extension.load

        def named(name, *a, **k):
            return real("atdev_marlin_c1_" + name.split("_")[-1], *a, **k)
        cpp_extension.load = named
        try:
            return orig_load(verbose)
        finally:
            cpp_extension.load = real
    _MODS[mode] = _load(verbose)
    return _MODS[mode]


if __name__ == "__main__":
    import sys
    for md in (int(v) for v in (sys.argv[1:] or ["0", "1", "2", "3"])):
        print("built mode", md, load(md).__name__, flush=True)
