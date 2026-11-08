"""PFX-s9: int8 -> s4 activation re-quantisation kernel (csrc/pf_a4/pf_a4_quant.cu) for the large-M W4A4 prefill GEMM."""
from __future__ import annotations

import hashlib
import os

_MOD = None
CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "pf_a4")


def load():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load as _load
        src = os.path.join(CSRC, "pf_a4_quant.cu")
        name = "atdev_pf_a4_" + hashlib.sha256(open(src, "rb").read()).hexdigest()[:10]
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
        _m = _load(name=name, sources=[src], extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_80,code=sm_80"])
        import functools
        raw = _m.quant_a4_i8
        var = int(os.environ.get("ATDEV_PF_A4_RQV", "0"))  # 1: refined-reciprocal kernel (bit-identical, ~17 % faster; C32MT)
        _m.quant_a4_i8 = functools.partial(raw, variant=var)
        _MOD = _m
    return _MOD
