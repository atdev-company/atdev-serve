"""C32e verify-head screen at c=32 (ATDEV_VSPLIT_A4SCR=1, with vsplit_mtp's argmax-only greedy verify;
default off). ROUNDING-LEVEL vs stock (not vs the deployed split head: more exact than it).

Deployed greedy verify head at M=96 (vsplit_mtp): exact bf16 cuBLAS over the 64k frequent rows (0.47 ms)
+ int4 W4A8 Marlin tail over the other 182784 rows (0.70 ms) -> argmax of the two; the tail logits are
used as final values (int4 weights, int8 activations).

Here: approximate logits of the WHOLE vocabulary from an int4 g128 shadow of the lm_head on the c=32
MarlinA4 W4A4 kernel (s4 g128 activations, tmb 6 cfg, ~0.55 ms at M=96, one weight pass), exact top-S
candidate selection of those approx values (per-4096-chunk top-SC, merged), exact bf16 rescoring of the
S candidates against the original bf16 lm_head rows (fp32 dot -> bf16, draft_vocab's rescoring kernel),
argmax of the rescored values. The approximate values only choose candidates (like shadow_head /
draft_vocab); the output token differs from the exact bf16 argmax only if the true argmax is not among
the S best approx values of its row (or on exact bf16 ties).

  ATDEV_VSPLIT_A4SCR=1          enable (needs ATDEV_OPT verify_split_head,vsplit_mtp + ATDEV_VSPLIT_FAST=1,
                                the c=32 W4A4 stack: marlin_plus a4 build with the gate_up tmb 4/6/8 cfgs)
  ATDEV_VSPLIT_A4SCR_CFG=S,SC   candidates per row / per 4096-chunk (default 16,4)
"""
from __future__ import annotations

import logging
import os

import torch
import triton

log = logging.getLogger("atdev_serve")

CH = 4096
# W4A4 cfg per tmb bucket (= the deployed gate_up 5120 -> 34816 entries of marlin_plus_a4k2q_cfg.json)
_CFG = {4: [64, 128, 6, 2, 4, 2, 0], 6: [64, 128, 6, 2, 6, 2, 0]}
MAX_M = 96  # larger batches -> the deployed split head (caller)


def enabled() -> bool:
    return os.environ.get("ATDEV_VSPLIT_A4SCR", "0") == "1"


class A4Screen:
    def __init__(self, weight: torch.Tensor, n_org: int):
        from atdev_serve.shadow_head import ShadowHead, ShadowHeadConfig

        self.w = weight  # full bf16 lm_head [V_pad, K] (rescoring rows, by vocab id)
        self.n_org = n_org
        self.sh = ShadowHead(weight[:n_org], ShadowHeadConfig(bits=4, group_size=128, screen=1, a8=True))
        assert self.sh.backend == "marlin" and self.sh.a8 and self.sh.padded_n == self.sh.size_n
        s, sc = (int(v) for v in os.environ.get("ATDEV_VSPLIT_A4SCR_CFG", "16,4").split(","))
        self.S, self.SC = s, sc
        self.NC = triton.next_power_of_2(triton.cdiv(n_org, CH))
        assert self.NC * sc <= 16384 and s & (s - 1) == 0 and sc <= s
        dev = weight.device
        k = weight.shape[1]
        self.k = k
        self.ids = torch.arange(n_org, dtype=torch.int64, device=dev)
        self.pk = torch.zeros(k // 64 * 8 * 32 * 4, dtype=torch.int32, device=dev)
        self.ps = torch.zeros(k // 128 * 16 * 8, dtype=torch.float32, device=dev)
        # gather-mode (A4_PAR_REDUCE=6) split-K slots: lock_bound x par_ms x 16 tmb x tn x 4 B, tmb <= 6
        n_tiles, k_tiles = n_org // 64, k // 128
        grid = torch.cuda.get_device_properties(dev).multi_processor_count * 2
        part2 = n_tiles % grid if n_tiles > grid else n_tiles
        if n_tiles > grid and part2 * 3 <= grid:
            part2 += grid
        iters = (k_tiles * part2 + grid - 1) // grid
        par_ms = (k_tiles + iters - 1) // iters + 1
        lock_bound = max(grid + 8, n_tiles + 2)
        self.locks = torch.zeros(lock_bound + 64, dtype=torch.int32, device=dev)
        self.ctmp = torch.empty(lock_bound * par_ms * 16 * 6 * 64 + 1024, dtype=torch.float32, device=dev)
        self.nbytes = self.sh.nbytes + self.ctmp.numel() * 4
        log.warning("atdev vsplit_a4scr: int4 W4A4 screen of %d rows (%.2f GB incl. %.0f MB split-K slots), "
                    "top-%d (per-chunk %d) exact bf16 rescoring", n_org, self.nbytes / 1e9,
                    self.ctmp.numel() * 4 / 1e6, s, sc)

    def approx(self, h: torch.Tensor) -> torch.Tensor:
        from atdev_serve import marlin_plus

        m = h.shape[0]
        tmb = 4 if m <= 64 else 6
        a4 = marlin_plus.mod_a4()
        if h.stride(-1) != 1 or h.stride(0) % 8:
            h = h.contiguous()
        t = self.sh
        a4.quant_a4(h, t.gs, self.pk, self.ps, tmb)
        out = torch.empty((m, t.size_n), dtype=torch.bfloat16, device=h.device)
        return a4.gemm_a4(self.pk, self.ps, t.qweight, t.scales, out, self.locks, self.ctmp, m, t.size_n,
                          self.k, list(_CFG[tmb]))

    def argmax(self, h: torch.Tensor) -> torch.Tensor | None:
        """int64 [M] greedy token per row (None: M > MAX_M, use the deployed head)."""
        from atdev_serve import draft_vocab as dv

        m = h.shape[0]
        out = torch.empty(m, dtype=torch.int64, device=h.device)
        if m == 0:
            return out
        if m > MAX_M:
            return None
        ap = self.approx(h)
        S, SC, NC = self.S, self.SC, self.NC
        ncs = NC * SC
        cv = torch.empty((m, ncs), dtype=torch.float32, device=h.device)
        ci = torch.empty((m, ncs), dtype=torch.int32, device=h.device)
        dv._screen_chunk_kernel[(m, NC)](ap, ap.stride(0), cv, ci, self.n_org, NC, S=SC, CH=CH, num_warps=4)
        hb = h.to(self.w.dtype)
        ex = torch.empty((m, S), dtype=torch.float32, device=h.device)
        fid = torch.empty((m, S), dtype=torch.int64, device=h.device)
        dv._screen_rescore_kernel[(m, S)](cv, ci, self.ids, self.w, self.w.stride(0), hb, hb.stride(0), ex, fid,
                                          self.k, self.n_org, NCS=ncs, S=S, BK=1024, num_warps=4)
        oval = torch.empty((m, 1), dtype=self.w.dtype, device=h.device)
        dv._screen_argmax_kernel[(m,)](ex, fid, out.view(m, 1), oval, S=S, num_warps=1)
        return out
