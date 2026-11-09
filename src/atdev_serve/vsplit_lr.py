"""LR1 low-rank verify-head screen at c=1 (ATDEV_VSPLIT_LR=<factor file>; default off). Accuracy-gated
(full stack): ROUNDING-LEVEL vs stock bf16 when the exact argmax is among the screened candidates.

The c=1 greedy tree verify (verify_split_head FAST argmax, M = 16 rows) normally reads the deployed split
head (64k int8 F rows + int4 tail, ~0.8 GB, ~525 us) and uses those low-bit values as final logits.
Here:
  1. approx logits of the whole vocabulary = (h A) B^T, a data-whitened rank-r factorisation of the
     lm_head (offline fit: A = C^-1/2 Q_r, B = W C^1/2 Q_r, C = E[h h^T] of logged target final
     hidden states; B is built from the bf16 lm_head at load: r=512 -> 254 MB),
  2. exact top-K of the approx values per row (torch.topk),
  3. exact bf16 rescoring of the K candidates against the original lm_head rows (fp32 dot -> bf16),
  4. argmax of the rescored values, ties -> lowest vocab id (torch.argmax rule).
The output differs from the exact bf16 argmax only when the exact argmax is outside the top-K approx
values (LR1 offline: r=512 K=128 misses 0.08 % of rows vs 0.25 % for the deployed split head).

  ATDEV_VSPLIT_LR=<file>        factor file {A, T} from an offline fit (needs verify_split_head +
                                ATDEV_VSPLIT_FAST=1; only the greedy tree argmax path uses it)
  ATDEV_VSPLIT_LR_K=128         candidates per row = exact global top-K of the approx values (power of 2):
                                threshold = K-th largest of the 486 per-512-chunk maxima (>= the K-th value),
                                all values >= threshold collected (cap ATDEV_VSPLIT_LR_CAP=4096 per row; LR1:
                                median 486, max 1943), torch.topk(K) of those, then K rescored.
                                (Per-chunk top-4 pools miss 1.6-2.8 % of rows: the high approx values cluster
                                in vocab-id space.)
  ATDEV_VSPLIT_LR_MODE=thr      thr (above) | topk (torch.topk over the whole row, ~+100 us)
  ATDEV_VSPLIT_LR_MAXM=16       larger batches -> the deployed split head
  ATDEV_VSPLIT_LR_DUMP=<file>   diagnostic (syncs every step): save the first ATDEV_VSPLIT_LR_DUMP_N (default
                                100000) verify rows {H [N, d] bf16, am = exact bf16 argmax} for refitting the factors
                                on the served stack's hidden states
  (LR1b) ATDEV_VSPLIT_LR_DUMP_EVERY=k  dump only every k-th verify call (decorrelates rows; default 1)
         ATDEV_VSPLIT_LR_DUMP_SHARD=n  write shards <dump>.<phase>.<idx>.pt every n rows (default 0 = one file
                                at DUMP_N as before); each shard {H, am_lr = [exact bf16, LR, deployed split
                                head] argmax [3, N] int64, phase}
         ATDEV_VSPLIT_LR_DUMP_PHASE=<file>  phase tag = contents of <file> when it exists (else "a"); a phase
                                change flushes the current shard (separates bench rows from fit rows in one
                                server session)
"""
from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

log = logging.getLogger("atdev_serve")


def enabled() -> bool:
    return bool(os.environ.get("ATDEV_VSPLIT_LR", ""))


@triton.jit
def _lr_rescore_kernel(idx_ptr, w_ptr, sw, h_ptr, sh, ex_ptr, KD, NV, SI, S: tl.constexpr, TB: tl.constexpr,
                       BK: tl.constexpr):
    """program (r, tb): exact bf16 logits of candidates idx[r, tb*TB : tb*TB+TB] (fp32 dot -> bf16)."""
    r = tl.program_id(0)
    tb = tl.program_id(1)
    t = tb * TB + tl.arange(0, TB)
    fid = tl.load(idx_ptr + r * SI + t).to(tl.int64)
    fid = tl.minimum(tl.maximum(fid, 0), NV - 1)
    acc = tl.zeros([TB, BK], dtype=tl.float32)
    for k0 in range(0, KD, BK):
        kk = k0 + tl.arange(0, BK)
        km = kk < KD
        hv = tl.load(h_ptr + r * sh + kk, mask=km, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + fid[:, None] * sw + kk[None, :], mask=km[None, :], other=0.0).to(tl.float32)
        acc += wv * hv[None, :]
    e = tl.sum(acc, axis=1).to(tl.bfloat16).to(tl.float32)
    tl.store(ex_ptr + r * S + t, tl.where(e == e, e, float("-inf")))


@triton.jit
def _lr_argmax_kernel(ex_ptr, idx_ptr, out_ptr, SI, S: tl.constexpr):
    r = tl.program_id(0)
    j = tl.arange(0, S)
    ex = tl.load(ex_ptr + r * S + j)
    fid = tl.load(idx_ptr + r * SI + j, mask=j < SI, other=2147483647).to(tl.int64)
    best = tl.max(ex, axis=0)
    tl.store(out_ptr + r, tl.min(tl.where(ex == best, fid, 9223372036854775807), axis=0))


@triton.jit
def _lr_cmax_kernel(x_ptr, sx, cm_ptr, N, NCP, CH: tl.constexpr):
    """cm[r, c] = max of approx chunk c (NaN -> -inf; chunks past N -> -inf)."""
    r = tl.program_id(0)
    c = tl.program_id(1)
    offs = c * CH + tl.arange(0, CH)
    v = tl.load(x_ptr + r * sx + offs, mask=offs < N, other=float("-inf")).to(tl.float32)
    v = tl.where(v == v, v, float("-inf"))
    tl.store(cm_ptr + r * NCP + c, tl.max(v, axis=0))


@triton.jit
def _lr_thr_kernel(cm_ptr, thr_ptr, cnt_ptr, NCP: tl.constexpr, KT: tl.constexpr):
    """thr[r] = KT-th largest chunk max (>= KT approx values are >= thr, so the global top-KT is too)."""
    r = tl.program_id(0)
    j = tl.arange(0, NCP)
    s = tl.sort(tl.load(cm_ptr + r * NCP + j), descending=True)
    tl.store(thr_ptr + r, tl.max(tl.where(j == KT - 1, s, float("-inf")), axis=0))
    tl.store(cnt_ptr + r, 0)


@triton.jit
def _lr_collect_kernel(x_ptr, sx, thr_ptr, cnt_ptr, bv_ptr, bi_ptr, N, CAP, CH: tl.constexpr):
    """append every approx value >= thr[r] of chunk c (value, vocab id) to row r's buffer (first CAP kept)."""
    r = tl.program_id(0)
    c = tl.program_id(1)
    offs = c * CH + tl.arange(0, CH)
    v = tl.load(x_ptr + r * sx + offs, mask=offs < N, other=float("-inf")).to(tl.float32)
    t = tl.load(thr_ptr + r)
    keep = (v >= t) & (v == v) & (offs < N)
    k = keep.to(tl.int32)
    n = tl.sum(k, axis=0)
    if n > 0:
        base = tl.atomic_add(cnt_ptr + r, n)
        pos = base + tl.cumsum(k, axis=0) - 1
        ok = keep & (pos < CAP)
        tl.store(bv_ptr + r * CAP + pos, v, mask=ok)
        tl.store(bi_ptr + r * CAP + pos, offs, mask=ok)


class LRScreen:
    def __init__(self, weight: torch.Tensor, n_org: int, path: str):
        f = torch.load(path, map_location="cpu")
        dev = weight.device
        self.w = weight  # full bf16 lm_head (rescoring rows, by vocab id; not copied)
        self.n_org = n_org
        self.r = int(f["r"])
        self.A = f["A"].to(dev, torch.bfloat16).contiguous()                  # [d, r]
        T = f["T"].to(dev, torch.float32)
        B = torch.empty((n_org, self.r), dtype=torch.bfloat16, device=dev)
        for s in range(0, n_org, 32768):
            e = min(n_org, s + 32768)
            B[s:e] = (weight[s:e].float() @ T).to(torch.bfloat16)
        del T
        self.B = B                                                            # [V, r]
        self.K = int(os.environ.get("ATDEV_VSPLIT_LR_K", "128"))
        assert self.K & (self.K - 1) == 0 and 16 <= self.K <= 1024
        self.mode = os.environ.get("ATDEV_VSPLIT_LR_MODE", "thr")
        self.cap = int(os.environ.get("ATDEV_VSPLIT_LR_CAP", "4096"))
        self.CH = 512
        self.nc = triton.cdiv(n_org, self.CH)
        self.ncp = triton.next_power_of_2(self.nc)
        assert self.K <= self.nc and self.cap >= self.K
        self.max_m = int(os.environ.get("ATDEV_VSPLIT_LR_MAXM", "16"))
        self.kd = weight.shape[1]
        self.dump = os.environ.get("ATDEV_VSPLIT_LR_DUMP", "")
        self.dump_n = int(os.environ.get("ATDEV_VSPLIT_LR_DUMP_N", "100000"))
        self._dh, self._da, self._dn = [], [], 0
        self.dump_every = int(os.environ.get("ATDEV_VSPLIT_LR_DUMP_EVERY", "1"))
        self.dump_shard = int(os.environ.get("ATDEV_VSPLIT_LR_DUMP_SHARD", "0"))
        self.dump_phase_file = os.environ.get("ATDEV_VSPLIT_LR_DUMP_PHASE", "")
        self._dcall, self._dbuf, self._dsh, self._dphase = 0, 0, 0, "a"
        self.nbytes = B.numel() * 2 + self.A.numel() * 2
        log.warning("atdev vsplit_lr: rank-%d whitened screen (%s, %.0f MB), exact rescoring of %s candidates, "
                    "M <= %d", self.r, os.path.basename(path), self.nbytes / 1e6,
                    f"top-{self.K} ({self.mode})", self.max_m)

    def approx(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h @ self.A, self.B)

    def argmax(self, h: torch.Tensor, out: torch.Tensor, post=None) -> torch.Tensor:
        """greedy token per row into out[:M] (int64)."""
        m = h.shape[0]
        ap = self.approx(h)
        if post is not None:
            post()
        dev = h.device
        if self.mode == "topk":
            idx = torch.topk(ap, self.K, dim=1, sorted=False).indices.to(torch.int32).contiguous()
        else:
            cm = torch.empty((m, self.ncp), dtype=torch.float32, device=dev)
            _lr_cmax_kernel[(m, self.ncp)](ap, ap.stride(0), cm, self.n_org, self.ncp, CH=self.CH, num_warps=4)
            thr = torch.empty(m, dtype=torch.float32, device=dev)
            cnt = torch.empty(m, dtype=torch.int32, device=dev)
            _lr_thr_kernel[(m,)](cm, thr, cnt, NCP=self.ncp, KT=self.K, num_warps=4)
            bv = torch.full((m, self.cap), float("-inf"), dtype=torch.float32, device=dev)
            bi = torch.zeros((m, self.cap), dtype=torch.int32, device=dev)
            _lr_collect_kernel[(m, self.nc)](ap, ap.stride(0), thr, cnt, bv, bi, self.n_org, self.cap, CH=self.CH,
                                            num_warps=4)
            self.last_cnt = cnt
            idx = bi.gather(1, torch.topk(bv, self.K, dim=1, sorted=False).indices).contiguous()
        np_, S = self.K, self.K
        ex = torch.full((m, S), float("-inf"), dtype=torch.float32, device=h.device)
        TB = 4
        _lr_rescore_kernel[(m, np_ // TB)](idx, self.w, self.w.stride(0), h, h.stride(0), ex, self.kd,
                                           self.n_org, SI=np_, S=S, TB=TB, BK=512, num_warps=4)
        _lr_argmax_kernel[(m,)](ex, idx, out, SI=np_, S=S, num_warps=4)
        return out

    def want_dump(self) -> bool:
        if not self.dump or not (0 <= self._dn < self.dump_n) or torch.cuda.is_current_stream_capturing():
            return False
        self._dcall += 1
        return (self._dcall - 1) % self.dump_every == 0

    def _flush(self) -> None:
        if not self._dh:
            return
        f = f"{self.dump}.{self._dphase}.{self._dsh:03d}.pt"
        torch.save({"H": torch.cat(self._dh), "am_lr": torch.cat(self._da, 1), "phase": self._dphase}, f)
        log.warning("atdev vsplit_lr: dumped %d verify rows to %s", self._dbuf, f)
        self._dh, self._da, self._dbuf = [], [], 0
        self._dsh += 1

    def _dump(self, h: torch.Tensor, out: torch.Tensor, split_am: torch.Tensor | None = None) -> None:
        """h: verify rows, out: LR argmax, split_am: the deployed split head's argmax (same rows)."""
        if self.dump_phase_file and os.path.exists(self.dump_phase_file):
            ph = open(self.dump_phase_file).read().strip() or "b"
            if ph != self._dphase:
                if self.dump_shard:
                    self._flush()
                self._dphase = ph
        ex = F.linear(h, self.w[: self.n_org]).argmax(1)
        rows = [ex, out[: h.shape[0]]] + ([split_am[: h.shape[0]]] if split_am is not None else [])
        self._dh.append(h.detach().to("cpu", torch.bfloat16))
        self._da.append(torch.stack(rows).cpu())
        self._dn += h.shape[0]
        self._dbuf += h.shape[0]
        if self.dump_shard:
            if self._dbuf >= self.dump_shard:
                self._flush()
            if self._dn >= self.dump_n:
                self._flush()
                self._dn = -1
        elif self._dn >= self.dump_n:
            torch.save({"H": torch.cat(self._dh), "am_lr": torch.cat(self._da, 1)}, self.dump)
            log.warning("atdev vsplit_lr: dumped %d verify rows to %s", self._dn, self.dump)
            self._dh, self._da, self._dn = [], [], -1
