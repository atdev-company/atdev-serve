"""c=32 draft-vocab / verify-screen chunk top-S on CUDA (ATDEV_OPT=c32k_fuse), bit-identical.

draft_vocab._screen_chunk_kernel (Triton per-chunk exact top-S of the approximate logits, used by the
verify-head W4A4 screen (vsplit_a4scr, CH 4096) and the 64k draft-vocab screen of every MTP pass
(fused_top1, CH 1024)) is replaced by csrc/c32k_scr.cu (CUDA, warp-shuffle top-S; same cv / ci bits).
Unsupported shapes / strides fall back to the Triton kernel. Installed before any CUDA-graph capture.
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("atdev_serve")
OPT_NAME = "c32k_fuse"
SUPPORTED_VLLM_VERSION = "0.31.0"

_EXT = None
# launch mode per chunk width, passed to the kernel (< 0: threshold fast path with |value| threads)
NT = {4096: -128, 1024: -32}


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

    def __getitem__(self, grid):
        def run(x, sx, cv, ci, n, nc, S, CH, num_warps=4, **kw):
            if (not kw and S == 4 and CH in NT and x.dtype == torch.bfloat16 and x.dim() == 2 and x.stride(1) == 1
                    and sx == x.stride(0) and sx % 8 == 0 and n % 8 == 0 and x.data_ptr() % 16 == 0
                    and tuple(grid) == (x.shape[0], nc) and cv.is_contiguous() and ci.is_contiguous()
                    and cv.dtype == torch.float32 and ci.dtype == torch.int32 and x.shape[1] >= n):
                ext().scr_chunk(x, cv, ci, n, nc, S, CH, NT[CH])
                return
            return self.tk[grid](x, sx, cv, ci, n, nc, S=S, CH=CH, num_warps=num_warps, **kw)
        return run


def register() -> None:
    from atdev_serve import draft_vocab as dv
    if not isinstance(dv._screen_chunk_kernel, _ChunkLauncher):
        ext()
        dv._screen_chunk_kernel = _ChunkLauncher(dv._screen_chunk_kernel)
        log.warning("atdev c32k_fuse: CUDA screen-chunk top-4 (NT %s) replaces draft_vocab._screen_chunk_kernel",
                    NT)
