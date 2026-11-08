"""Launch config for vLLM's GDN causal_conv1d_update Triton kernel (stock BLOCK_N=256, 4 warps):
ATDEV_CONV_UPDATE_CFG=block_n,num_warps (default off). Per-feature arithmetic (bias + the
width taps in fixed order, SiLU) does not depend on BLOCK_N / num_warps, so outputs and conv
states are bit-identical (scripts/bench_c32_conv.py checks it). Installed by fused_quant's
register() when the env var is set (no ATDEV_OPT entry needed).
"""
from __future__ import annotations

import os

CFG: list = [None, None]  # [BLOCK_N, num_warps]; None -> stock


class _Launcher:
    def __init__(self, kern):
        self._k = kern

    def __getattr__(self, name):
        return getattr(self._k, name)

    def __getitem__(self, grid):
        launch = self._k[grid]
        bn, nw = CFG

        def run(*args, **kw):
            if bn is not None:
                kw["BLOCK_N"] = bn
            if nw is not None:
                kw["num_warps"] = nw
            return launch(*args, **kw)
        return run


def install() -> bool:
    from vllm.model_executor.layers.mamba.ops import causal_conv1d as cc

    s = os.environ.get("ATDEV_CONV_UPDATE_CFG", "")
    if s:
        bn, nw = (int(v) for v in s.split(","))
        CFG[:] = [bn, nw]
    if not isinstance(cc._causal_conv1d_update_kernel, _Launcher):
        cc._causal_conv1d_update_kernel = _Launcher(cc._causal_conv1d_update_kernel)
    return True
