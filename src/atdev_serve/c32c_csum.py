"""Consumer-sum W4A4 GEMMs (inside fused_quant): the GEMMs whose only consumer is the next residual-add
+ RMSNorm -> s4 producer (attn o_proj, GDN out_proj) run the MarlinA4 csum build (A4_PAR_REDUCE=7):
every split-K slice stores its fp32 partial to P[slice][row][col] and exits -- no lock chain, no bf16
write. The norm producer (csrc/c32c_csum.cu) rebuilds x = bf16(P[0] + ... + P[cnt-1]) in slice order
(= the serial chain's fp32 order, so the GEMM-output part is bit-identical) and continues as the
residual add + Gemma RMSNorm -> s4 producer. The CUDA norm differs from fused_quant's Triton chunk
kernel only on rare fp32 ties of bf16((rstd x)(w + 1)).

Data flow: the csum GEMM op returns an unwritten bf16 [M, N] placeholder and tags P with its data_ptr; the
next atdev_norm_quant call consumes P when its x is that placeholder (producer and consumer are adjacent on
the stream; the decision is taken in Python at capture time). Any other reader of a tagged placeholder
would see garbage, so only the call sites above use it, and a norm call on a tagged x that cannot take
the a4 path first materialises x (csum_to_bf16).

The s4 quotient uses the branch-free form (-DMK=1), bit-identical to div.rn (see fused_quant).
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("atdev_serve")
_EXT: list = [None]
_ST: dict = {"for": 0, "K": 0, "grid": 0, "n": 0}
_P: dict = {}
SMAX, ROWS, TN, NMAX = 4, 128, 64, 5120
_STATS = {"gemm": 0, "norm": 0, "mat": 0}


def ext():
    if _EXT[0] is None:
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(__file__), "csrc", "c32c_csum.cu")
        _EXT[0] = load(name="atdev_c32c_csum_mk", sources=[src], verbose=False,
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80", "-std=c++17", "-DMK=1"])
    return _EXT[0]


_MOD: list = [None]


def mod_csum():
    """MarlinA4 build with A4_PAR_REDUCE=7 (same defines otherwise, same instantiation set)."""
    if _MOD[0] is None:
        from atdev_serve import marlin_plus_a4_build as b4
        ds = [d for d in b4.defines() if not d.startswith("-DA4_PAR_REDUCE=")] + ["-DA4_PAR_REDUCE=7"]
        _MOD[0] = b4._build(False, ds)
    return _MOD[0]


def _pbuf(dev):
    p = _P.get(dev)
    if p is None:
        p = _P[dev] = torch.empty(SMAX * ROWS * NMAX, dtype=torch.float32, device=dev)
    return p


def _grid(cfg, dev) -> int:
    if len(cfg) == 8 and cfg[7] > 0:
        return int(cfg[7])
    return torch.cuda.get_device_properties(dev).multi_processor_count * int(cfg[3])


def _max_slices(n, k, tn, grid) -> int:
    n_tiles, k_tiles = n // tn, k // 128
    part2 = n_tiles
    if n_tiles > grid:
        part2 = n_tiles % grid
        if part2 * 3 <= grid:
            part2 += grid
    iters = (k_tiles * part2 + grid - 1) // grid
    return (k_tiles + iters - 1) // iters + 1


def eligible(ent, m, n, k, dev) -> bool:
    if ent is None or not ent.get("a4") or m == 0 or m > ROWS or n > NMAX or n % 256:
        return False
    cfg = ent["cfg"]
    return cfg[0] == TN and m <= 16 * cfg[4] and _max_slices(n, k, TN, _grid(cfg, dev)) <= SMAX


def gemm(apk, a_s, w_q, w_s, m, n, k, ent, dev) -> torch.Tensor:
    """csum GEMM on the shared a4 buffers; returns the (unwritten) placeholder output."""
    from atdev_serve import marlin_plus as mp
    b = mp.bufs(dev)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    cfg = list(ent["cfg"])
    mod_csum().gemm_a4(apk, a_s, w_q, w_s, out, b["locks"], _pbuf(dev), m, n, k, cfg)
    _ST.update({"for": out.data_ptr(), "K": k, "grid": _grid(cfg, dev), "n": n})
    if _STATS["gemm"] == 0:
        log.warning("atdev c32c_csum: W4A4 GEMM N=%d K=%d M=%d -> slice partials (consumer-sum)", n, k, m)
    _STATS["gemm"] += 1
    return out


def tagged(x: torch.Tensor) -> bool:
    return _ST["for"] != 0 and x.data_ptr() == _ST["for"] and x.shape[1] == _ST["n"]


def materialise(x: torch.Tensor) -> None:
    """write the GEMM result into the placeholder (fallback for a consumer that is not the csum norm)"""
    ext().csum_to_bf16(_pbuf(x.device), _ST["K"], _ST["grid"], x, x.shape[0])
    _ST["for"] = 0
    _STATS["mat"] += 1


def norm_a4(x, r, w, gs, xb, ro, apk, as_, m, eps, m_pad, tmb) -> None:
    ext().csum_norm_a4(_pbuf(x.device), _ST["K"], _ST["grid"], r, w, gs, xb, ro, apk, as_, m, eps, m_pad, tmb)
    _ST["for"] = 0
    if _STATS["norm"] == 0:
        log.warning("atdev c32c_csum: norm producer sums the GEMM slice partials (M=%d)", m)
    _STATS["norm"] += 1


def clear() -> None:
    _ST["for"] = 0
