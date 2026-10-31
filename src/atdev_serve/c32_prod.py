"""c=32 W4A4 producers in CUDA (csrc/c32_prod.cu): SiLU.mul->s4 and gate->s4, bit-identical to
fused_quant's Triton _silu_a4_kernel / _gate_a4_kernel (same fp32 op sequence incl. Triton's
approx sigmoid, same s4 fragment layout, branch-free quotient -DMK=1).
"""
from __future__ import annotations

import os

_EXT: list = [None]
GROUPS_PER_CTA = 8  # 128-column groups per CTA (SiLU and gate)


def ext():
    if _EXT[0] is None:
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "c32_prod.cu")
        _EXT[0] = load(name="atdev_c32_prod_mk", sources=[src], verbose=False,
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80", "-std=c++17", "-DMK=1"])
    return _EXT[0]


def silu_a4(gu, gs, apk, as_, m, n, m_pad, tmb):
    ext().act_a4(0, gu, None, gs, apk, as_, m, n, m_pad, tmb, GROUPS_PER_CTA)


def gate_a4(a, gate, gs, apk, as_, m, n, m_pad, tmb):
    ext().act_a4(1, a, gate, gs, apk, as_, m, n, m_pad, tmb, GROUPS_PER_CTA)
