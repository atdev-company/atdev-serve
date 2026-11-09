"""C32n2-ba: GDN in_proj_ba as ONE low-CTA launch, bit-identical to the deployed cuBLAS call
(torch.mm(x, W^T) = cutlass s16816 64x64 split-K S + cublasLt splitKreduce; csrc/c32n_ba.cu).

cuBLAS's per-element order is a sequential mma.m16n8k16 chain per k-split, then the S fp32 partials summed
in split order. The split count per M (heuristic-dependent) is CALIBRATED against torch.mm at the first eager
call (calibrate(): every M in 1..128, several random draws incl. cancellation-heavy ones, all must match
bitwise); M without an exact candidate -> torch.mm. Used by c32p_ba.py:

  ATDEV_C32N_BA=side   the side-stream ba of ATDEV_C32P_BA=1 runs this kernel instead of torch.mm
                       (16 splits x 3 n-groups = 48 CTAs x 4 warps, 84 KB smem, vs cuBLAS's 64 CTAs x 96 KB smem
                       + a second splitKreduce launch)
  ATDEV_C32N_BA_NPC    warps per CTA for side mode (4 / 6 / 12, default 4)
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("atdev_serve")
_EXT: list = [None]
_BUF: dict = {}
_TABLE: dict = {}  # (dev, N, K) -> {M: (S, ksz, mode) or None}
MAX_M = 128
_WARN = [False]


def mode() -> str:
    return os.environ.get("ATDEV_C32N_BA", "")


def ext():
    if _EXT[0] is None:
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(__file__), "csrc", "c32n_ba.cu")
        _EXT[0] = load(name="atdev_c32n_ba", sources=[src], verbose=False,
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80", "-std=c++17"])
    return _EXT[0]


def bufs(dev, n):
    b = _BUF.get(dev.index)
    if b is None:
        b = _BUF[dev.index] = dict(ws=torch.empty(64 * 128 * 256, dtype=torch.float32, device=dev),
                                   cnt=torch.zeros(64, dtype=torch.int32, device=dev))
    return b


def _ksz(k, s, align):
    q = -(-k // s)
    return -(-q // align) * align


def candidates(k):
    out = []
    for s in (16, 8, 10, 20, 32, 4, 5, 2, 12, 24, 40, 1, 3, 6, 7, 9, 11, 13, 14, 15):
        for al in (64, 16):
            z = _ksz(k, s, al)
            if s > 1 and z * (s - 1) >= k:
                continue
            c = (s, z if s > 1 else k)
            if c not in out:
                out.append(c)
    return out


def run(x, w, out, s, ksz, md=0, npc=4):
    b = bufs(x.device, w.shape[0])
    ext().ba_gemm(x, w, out, b["ws"], b["cnt"], s, ksz, md, npc)
    return out


def _draws(m, k, n, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    xs = []
    x = torch.randn(m, k, device=dev, generator=g).to(torch.bfloat16)
    xs.append(x)
    # cancellation-heavy: large +/- pairs, small residuals
    y = torch.randn(m, k, device=dev, generator=g) * 30
    y[:, 1::2] = -y[:, 0::2] + torch.randn(m, k // 2, device=dev, generator=g) * 0.01
    xs.append(y.to(torch.bfloat16))
    z = (torch.rand(m, k, device=dev, generator=g) * 2 - 1) * torch.exp2(torch.randint(-8, 8, (m, k), device=dev,
                                                                                        generator=g).float())
    xs.append(z.to(torch.bfloat16))
    return xs


def calibrate(w: torch.Tensor, trials: int = 6, ms=None, verbose=False) -> dict:
    """Per M: first (S, ksz, mode) whose output equals torch.mm(x, w.t()) bitwise on every draw (real w and a
    random w)."""
    n, k = w.shape
    dev = w.device
    key = (dev.index, n, k)
    tab = _TABLE.setdefault(key, {})
    cands = candidates(k)
    wr = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
    last = None
    for m in (ms or range(1, MAX_M + 1)):
        if m in tab:
            continue
        order = ([last] if last else []) + [c for c in cands if c != last]
        hit = None
        for (s, z) in order:
            if (16 * ((m + 15) // 16) + 32) * (z + 8) * 2 > 160 * 1024:
                continue  # chunk does not fit the kernel's smem
            ok = True
            for t in range(trials):
                for ww in (w, wr):
                    for x in _draws(m, k, n, dev, 1000 * m + t):
                        ref = torch.mm(x, ww.t())
                        o = torch.empty_like(ref)
                        run(x, ww, o, s, z)
                        if not torch.equal(ref.view(torch.int16), o.view(torch.int16)):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                hit = (s, z, 0)
                break
        tab[m] = hit
        last = (hit[0], hit[1]) if hit else last
        if verbose:
            print(f"M={m}: {hit}", flush=True)
    return tab


def table(w):
    return _TABLE.get((w.device.index, w.shape[0], w.shape[1]))


def ensure(w) -> dict | None:
    """Calibrated table for this weight shape, or None while capturing a graph before calibration."""
    t = table(w)
    if t is not None and len(t) >= MAX_M:
        return t
    if torch.cuda.is_current_stream_capturing():
        if not _WARN[0]:
            _WARN[0] = True
            log.warning("atdev c32n_ba: not calibrated before graph capture -> torch.mm")
        return None
    ext()
    t = calibrate(w)
    n_ok = sum(1 for v in t.values() if v)
    ss = sorted({v[0] for v in t.values() if v})
    log.warning("atdev c32n_ba: calibrated in_proj_ba %s vs torch.mm: %d / %d M exact (splits %s)",
                tuple(w.shape), n_ok, len(t), ss)
    return t


def cfg_for(w, m):
    if m < 1 or m > MAX_M:
        return None
    t = ensure(w)
    return t.get(m) if t else None


def ok_tensors(x, w) -> bool:
    return (x.dim() == 2 and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and x.stride(1) == 1
            and w.stride(1) == 1 and x.stride(0) % 8 == 0 and w.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
            and w.data_ptr() % 16 == 0 and w.shape[0] % 32 == 0 and x.shape[1] == w.shape[1])
