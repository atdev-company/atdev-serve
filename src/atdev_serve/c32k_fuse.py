"""C32k (LEDGER C32k-*): bit-identical glue cuts around the c=32 MTP drafter passes / verify screen.

  ATDEV_C32K_SCR=1   draft_vocab._screen_chunk_kernel (Triton per-chunk exact top-S of the approx logits, used by
                     the verify-head W4A4 screen (vsplit_a4scr, CH 4096) and the 64k draft-vocab screen of every
                     MTP pass (fused_top1, CH 1024)) -> csrc/c32k_scr.cu (CUDA, warp-shuffle top-S; same cv / ci
                     bits, tests/test_c32k.py). Unsupported shapes / strides fall back to the Triton kernel.
  ATDEV_C32K_NT4096 / ATDEV_C32K_NT1024   threads per CTA for the two chunk widths (default 256 / 128).
Opt-in only; installed by ATDEV_OPT=c32k_fuse before any CUDA-graph capture.
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("atdev_serve")
OPT_NAME = "c32k_fuse"
SUPPORTED_VLLM_VERSION = "0.31.0"

_EXT = None
NT = {4096: int(os.environ.get("ATDEV_C32K_NT4096", "-128")), 1024: int(os.environ.get("ATDEV_C32K_NT1024", "-32"))}  # < 0: threshold fast path


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        _EXT = load(name="atdev_c32k_scr", sources=[os.path.join(os.path.dirname(__file__), "csrc", "c32k_scr.cu")],
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"], verbose=False)
    return _EXT


class _ChunkLauncher:
    """Stands in for draft_vocab._screen_chunk_kernel: kernel[grid](x, sx, cv, ci, N, NC, S=, CH=, num_warps=)."""

    def __init__(self, triton_kernel):
        self.tk = triton_kernel
        self.hits = 0
        self.falls = 0

    def __getitem__(self, grid):
        def run(x, sx, cv, ci, n, nc, S, CH, num_warps=4, **kw):
            if (not kw and S == 4 and CH in NT and x.dtype == torch.bfloat16 and x.dim() == 2 and x.stride(1) == 1
                    and sx == x.stride(0) and sx % 8 == 0 and n % 8 == 0 and x.data_ptr() % 16 == 0
                    and tuple(grid) == (x.shape[0], nc) and cv.is_contiguous() and ci.is_contiguous()
                    and cv.dtype == torch.float32 and ci.dtype == torch.int32 and x.shape[1] >= n):
                self.hits += 1
                ext().scr_chunk(x, cv, ci, n, nc, S, CH, NT[CH])
                return
            self.falls += 1
            return self.tk[grid](x, sx, cv, ci, n, nc, S=S, CH=CH, num_warps=num_warps, **kw)
        return run


def register() -> None:
    if os.environ.get("ATDEV_C32K_SCR", "0") == "1":
        from atdev_serve import draft_vocab as dv
        if not isinstance(dv._screen_chunk_kernel, _ChunkLauncher):
            ext()
            dv._screen_chunk_kernel = _ChunkLauncher(dv._screen_chunk_kernel)
            log.warning("atdev c32k_fuse: CUDA screen-chunk top-4 (NT %s) replaces draft_vocab._screen_chunk_kernel",
                        NT)
