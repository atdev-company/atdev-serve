"""c=32 W4A4 producers in CUDA (csrc/c32_prod.cu): SiLU.mul->s4 and gate->s4 bit-identical to
fused_quant's Triton _silu_a4_kernel / _gate_a4_kernel (same fp32 op sequence incl. Triton's
approx sigmoid, same s4 fragment layout). Used by fused_quant when ATDEV_FQ_A4_CUDA=1 (default
off); checked by tests/test_c32_prod.py, timed by scripts/bench_c32_a4prod.py (M=96: SiLU 10.15
-> 9.87 us, gate 5.54 -> 4.43 us).

norm_a4 (residual add + Gemma RMSNorm -> s4) reproduces Triton's variance bit for bit (same
reduction tree, rstd checked equal) but NOT the bf16 rounding of (rstd*x)*(w+1) on exact fp32
ties (~1e-5 of the values differ; cause not found), and is only 0.8 us faster -> unused.

  ATDEV_FQ_A4_CUDA=1            SiLU.mul->s4, gate->s4 on these kernels
  ATDEV_FQ_A4_CUDA_GPC=s,g      128-groups per CTA for SiLU / gate (default 8,8)
  ATDEV_C32_MK=1                C32h-mk bit-identical faster quotient in all s4 producers
"""
from __future__ import annotations

import os

_EXTS: dict = {}


def ext(mk: bool | None = None):
    """mk None: the deployed variant (ATDEV_C32_MK); True/False: that variant (c32hmk_check shadow runs)"""
    mk = mk_on() if mk is None else mk
    e = _EXTS.get(mk)
    if e is None:
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "c32_prod.cu")
        # C32h-mk: bit-identical branch-free quotient + ex2.approx.ftz (separate module name)
        e = _EXTS[mk] = load(name="atdev_c32_prod" + ("_mk" if mk else ""), sources=[src], verbose=False,
                             extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80", "-std=c++17"]
                             + (["-DMK=1"] if mk else []))
    return e


def mk_on() -> bool:
    """ATDEV_C32_MK=1 (LEDGER C32hmk-*): every s4 producer (act_a4, csum_norm_a4, Triton _a4_pack_store) uses
    q = rint(fma(fma(-sc, y*rc, y), rc, y*rc)), rc = RN(1/sc) instead of div.rn(y, sc) (+ ex2.approx.ftz in
    the sigmoid) -- nibbles / bits proven equal for every bf16 input (scripts/c32hmk_exhaust.cu)."""
    return os.environ.get("ATDEV_C32_MK", "0") == "1"


def enabled() -> bool:
    return os.environ.get("ATDEV_FQ_A4_CUDA", "0") == "1"


def gpc() -> tuple[int, int]:
    s, g = (int(v) for v in os.environ.get("ATDEV_FQ_A4_CUDA_GPC", "8,8").split(","))
    return s, g


def norm_a4(x, r, w, gs, xb, ro, apk, as_, m, eps, m_pad, tmb, dbg=None):
    """r / xb may be None (no residual / no bf16 output); dbg: fp32 [m_pad, 2] <- (sum x^2, rstd)."""
    ext().norm_a4(x, r, w, gs, xb, ro, apk, as_, m, eps, m_pad, tmb, dbg)


def silu_a4(gu, gs, apk, as_, m, n, m_pad, tmb, g=None, mk=None):
    ext(mk).act_a4(0, gu, None, gs, apk, as_, m, n, m_pad, tmb, g or gpc()[0])


def gate_a4(a, gate, gs, apk, as_, m, n, m_pad, tmb, g=None, mk=None):
    ext(mk).act_a4(1, a, gate, gs, apk, as_, m, n, m_pad, tmb, g or gpc()[1])
