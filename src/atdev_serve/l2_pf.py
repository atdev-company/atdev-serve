"""L2 prefetch of a GDN layer's tree-verify state before its CUDA tree kernel (ATDEV_GDN_L2PF, csrc/l2_pf.cu).

The c=1 tree kernel (csrc/gdn_tree.cu) of each of the 48 GDN layers starts by reading the layer's recurrent state
slot s0 (48 v-heads x 128 x 128 bf16 = 1.5 MB), its A3 log in slot s1 and the conv-state slot s0 -- cold in DRAM,
since 30-90 MB of GEMM weights stream through the 40 MB L2 between two uses. The microbench (L2-hot, 48 calls on
the same state) runs 13.7 us/layer, serving 17.4. ``prefetch_gdn`` issues ``prefetch.global.L2::evict_last`` for
those regions (device-side slot indices, graph-safe) on the ATDEV_C1_BA_STREAM side stream while the qkvz GEMM
(skeleton-bound, DRAM mostly idle) runs. A hint only -> outputs bit-identical.
"""
from __future__ import annotations

import os

import torch

_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "l2_pf.cu")
        _EXT = load(name="atdev_l2_pf", sources=[src], verbose=False,
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80", "-std=c++17"])
    return _EXT


def gdn_regions(state: torch.Tensor, conv_state: torch.Tensor | None, si: torch.Tensor, log_floats: int,
                what: str = "slc") -> tuple[list, list, list, list]:
    """(base, idx, stride, len) for state slot si[0,0] ('s'), log slot si[0,1] ('l', first log_floats fp32)
    and conv-state slot si[0,0] ('c')."""
    assert si.dtype == torch.int32 and si.stride(-1) == 1 and si.dim() == 2 and si.shape[1] >= 2
    _, hv, vd, kd = state.shape
    sb = state.stride(0) * state.element_size()
    base, idx, stride, ln = [], [], [], []
    p0, p1 = si.data_ptr(), si.data_ptr() + 4
    if "s" in what:
        base.append(state.data_ptr()); idx.append(p0); stride.append(sb); ln.append(hv * vd * kd * state.element_size())
    if "l" in what and log_floats > 0:
        base.append(state.data_ptr()); idx.append(p1); stride.append(sb); ln.append(log_floats * 4)
    if "c" in what and conv_state is not None:
        cb = conv_state.stride(0) * conv_state.element_size()
        base.append(conv_state.data_ptr()); idx.append(p0); stride.append(cb); ln.append(cb)
    return base, idx, stride, ln


def tensor_regions(ts) -> tuple[list, list, list, list]:
    """plain (non-indexed) regions covering each tensor's storage span"""
    base, idx, stride, ln = [], [], [], []
    for t in ts:
        if t is None or t.numel() == 0:
            continue
        span = (sum((d - 1) * st for d, st in zip(t.shape, t.stride())) + 1) * t.element_size()
        base.append(t.data_ptr()); idx.append(0); stride.append(0); ln.append(span)
    return base, idx, stride, ln


def merge(*regs):
    out = ([], [], [], [])
    for r in regs:
        for a, b in zip(out, r):
            a.extend(b)
    return out


_SINK: dict = {}
MODE = int(os.environ.get("ATDEV_GDN_L2PF_MODE", "1"))  # 0 prefetch hint, 1 ld.global.cg


def prefetch(regs, threads: int = 256, mode: int | None = None) -> None:
    base, idx, stride, ln = regs
    if base:
        dev = torch.cuda.current_device()
        if dev not in _SINK:
            _SINK[dev] = torch.zeros(1, dtype=torch.int32, device="cuda")
        ext().l2pf(base, idx, stride, ln, threads, MODE if mode is None else mode, _SINK[dev].data_ptr())
