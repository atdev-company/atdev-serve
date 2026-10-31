"""Reduced draft vocabulary (ATDEV_OPT=draft_vocab), FR-Spec / VocabTrim style.

Draft heads (MTP with ``use_local_argmax_reduction``: ``get_top_tokens``;
DFlash2: ``get_top_k_tokens``) only *propose* tokens; the target verifies
them with its own head, so restricting proposals to the N most frequent
tokens never changes generated text, only the acceptance rate (a target
token outside the draft vocabulary can never be proposed).

Stock reads the whole 248320 x 5120 bf16 lm_head (2.54 GB) per draft pass.
This module keeps the rows of the first 65536 ids of
``atdev_serve/data/token_order.npy`` (token frequency order measured on 22.4M
target-generated response tokens for ultrachat_200k / gsm8k-train prompts;
held-out coverage 99.3% at 64k) as an int4 g128 Marlin shadow (0.17 GB read
per pass), takes the top-16 approximate candidates and rescores only those
rows exactly in bf16 against the full head.

Delivered through ``shadow_head.ScreenedLogitsProcessor`` (its draft-only
top-token methods consult this head first); the target's ``forward`` is
unchanged.

Env knobs:
  ATDEV_DRAFT_VOCAB_FUSED   1: k=1 (MTP local argmax) runs the screen as three Triton kernels
                            (per-chunk top-S, then merge + exact bf16 rescoring, then argmax)
                            instead of torch.topk / index_select / bmm / topk. Same candidate rule
                            (exact top-S of the approximate logits, lowest id on ties) and the same
                            exact bf16 rescoring; the rescoring dot product is fp32 in a different
                            order than cuBLAS -> drafter-only rounding changes.
  ATDEV_DRAFT_VOCAB_TOPKX   1: the unsorted top-S candidate selection runs as two Triton kernels
                            that reproduce torch.topk(sorted=False) bit-exactly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger(__name__)

OPT_NAME = "draft_vocab"
# last approximate draft logits [M, N] computed by DraftVocabHead.top_k (read by di1_varlen)
LAST: dict = {}
_ATTR = "_atdev_draft_vocab"
_DEFAULT_FILE = os.path.join(os.path.dirname(__file__), "data", "token_order.npy")


@dataclass(frozen=True)
class DraftVocabConfig:
    size: int = 65536
    bits: int = 4
    screen: int = 16
    path: str = _DEFAULT_FILE


def load_ids(cfg: DraftVocabConfig, vocab: int) -> torch.Tensor:
    import numpy as np

    order = np.load(cfg.path)
    order = order[order < vocab][: cfg.size]
    # sorted ids: the gather/GEMM walks the head in memory order
    return torch.from_numpy(np.sort(order).astype("int64"))


class DraftVocabHead:
    """Top-k over a fixed vocabulary subset of an [V, K] lm_head weight."""

    def __init__(self, weight: torch.Tensor, ids: torch.Tensor, cfg: DraftVocabConfig):
        self.cfg = cfg
        self.ids = ids.to(weight.device)
        self.weight = weight  # full head (for exact rescoring), not copied
        sub = weight.index_select(0, self.ids).contiguous()
        from atdev_serve.shadow_head import ShadowHead, ShadowHeadConfig

        self.shadow = ShadowHead(sub, ShadowHeadConfig(bits=cfg.bits, group_size=128,
                                                       screen=cfg.screen))
        self.nbytes = self.shadow.nbytes
        del sub

    def top_k(self, hidden: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (full-vocab ids int64 [M, k], exact logits [M, k] in weight dtype)."""
        approx = self.shadow.approx_logits(hidden)
        LAST["approx"] = approx  # di1_varlen: draft confidence from the screen's approx logits
        screen = min(max(self.cfg.screen, k), self.ids.numel())
        if _FUSED and k == 1 and fused_ok(approx, screen) and self.weight.stride(-1) == 1:
            return fused_top1(approx, self.ids, self.weight, hidden, screen)
        if _TOPKX and topkx_ok(approx, screen):
            # == torch.topk(.., sorted=False)[1] and ids[that], bit-exact
            cand, full = topk_unsorted_exact(approx, screen, self.ids)
        else:
            _, cand = torch.topk(approx, screen, dim=-1, sorted=False)
            full = self.ids[cand]  # [M, C] full-vocab ids
        rows = self.weight.index_select(0, full.reshape(-1)).view(hidden.shape[0], screen, -1)
        exact = torch.bmm(rows, hidden.to(self.weight.dtype).unsqueeze(-1)).squeeze(-1)
        vals, idx = torch.topk(exact, k, dim=-1, sorted=True)
        return full.gather(-1, idx), vals


# ---------------------------------------------------------------------------------------------
# ATDEV_DRAFT_VOCAB_FUSED: fused screen for k=1
# ---------------------------------------------------------------------------------------------
_FUSED = os.environ.get("ATDEV_DRAFT_VOCAB_FUSED", "0") == "1"
# chunk width, candidates kept per chunk (SC < screen: the merged set is the exact top-screen unless
# more than SC of the global top-screen fall in one chunk -- drafter-only approximation)
_FCHUNK, _FSC = 1024, 4


@triton.jit
def _screen_chunk_kernel(x_ptr, sx, cv_ptr, ci_ptr, N, NC, S: tl.constexpr, CH: tl.constexpr):
    # S = candidates kept per chunk
    """exact top-S (value desc, lowest index first) of chunk c of row r -> cv/ci[r, c*S : c*S+S]"""
    r = tl.program_id(0)
    c = tl.program_id(1)
    offs = c * CH + tl.arange(0, CH)
    ok = offs < N
    v = tl.load(x_ptr + r * sx + offs, mask=ok, other=float("-inf")).to(tl.float32)
    v = tl.where(v == v, v, float("-inf"))  # NaN (e.g. garbage rows of dummy / padded batches)
    base = (r * NC + c) * S
    for s in tl.static_range(S):
        mx = tl.max(v, axis=0)
        i = tl.min(tl.where((v == mx) & ok, offs, 2147483647), axis=0)
        i = tl.where(i == 2147483647, c * CH, i)  # always a valid in-range index
        tl.store(cv_ptr + base + s, mx)
        tl.store(ci_ptr + base + s, i)
        v = tl.where(offs == i, float("-inf"), v)


@triton.jit
def _screen_rescore_kernel(cv_ptr, ci_ptr, ids_ptr, w_ptr, sw, h_ptr, sh, ex_ptr, fid_ptr, K, NSUB,
                           NCS: tl.constexpr, S: tl.constexpr, BK: tl.constexpr):
    """program (r, t): merge the per-chunk candidates of row r to the global top-S (value desc,
    lowest index first), take the t-th one and rescore it exactly against the full bf16 head
    (fp32 dot -> bf16) -> ex[r, t], fid[r, t]"""
    r = tl.program_id(0)
    t = tl.program_id(1)
    j = tl.arange(0, NCS)
    v = tl.load(cv_ptr + r * NCS + j)
    ix = tl.load(ci_ptr + r * NCS + j)
    i = 0
    for s in tl.static_range(S):
        if s <= t:
            mx = tl.max(v, axis=0)
            i = tl.min(tl.where(v == mx, ix, 2147483647), axis=0)
            v = tl.where((v == mx) & (ix == i), float("-inf"), v)
    i = tl.minimum(tl.maximum(i, 0), NSUB - 1)
    fid = tl.load(ids_ptr + i)
    acc = tl.zeros([BK], dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        km = kk < K
        hv = tl.load(h_ptr + r * sh + kk, mask=km, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + fid * sw + kk, mask=km, other=0.0).to(tl.float32)
        acc += wv * hv
    e = tl.sum(acc, axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(ex_ptr + r * S + t, tl.where(e == e, e, float("-inf")))
    tl.store(fid_ptr + r * S + t, fid)


@triton.jit
def _screen_argmax_kernel(ex_ptr, fid_ptr, oid_ptr, oval_ptr, S: tl.constexpr):
    r = tl.program_id(0)
    sj = tl.arange(0, S)
    ex = tl.load(ex_ptr + r * S + sj)
    fid = tl.load(fid_ptr + r * S + sj)
    best = tl.max(ex, axis=0)
    pos = tl.min(tl.where(ex == best, sj, S), axis=0)
    tl.store(oid_ptr + r, tl.sum(tl.where(sj == pos, fid, 0), axis=0))
    tl.store(oval_ptr + r, best.to(tl.bfloat16))


def fused_top1(approx: torch.Tensor, ids: torch.Tensor, weight: torch.Tensor, hidden: torch.Tensor,
               screen: int, chunk: int = 0, sc: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """(full-vocab ids int64 [M, 1], exact bf16 logits [M, 1]) = DraftVocabHead.top_k(k=1) screen."""
    m, n = approx.shape
    chunk = chunk or _FCHUNK
    sc = min(sc or _FSC, screen)
    nc = triton.cdiv(n, chunk)
    ncs = nc * sc
    dev = approx.device
    cv = torch.empty((m, ncs), dtype=torch.float32, device=dev)
    ci = torch.empty((m, ncs), dtype=torch.int32, device=dev)
    oid = torch.empty((m, 1), dtype=torch.int64, device=dev)
    oval = torch.empty((m, 1), dtype=weight.dtype, device=dev)
    if m:
        _screen_chunk_kernel[(m, nc)](approx, approx.stride(0), cv, ci, n, nc, S=sc, CH=chunk,
                                      num_warps=1)
        h = hidden.to(weight.dtype)
        ex = torch.empty((m, screen), dtype=torch.float32, device=dev)
        fid = torch.empty((m, screen), dtype=torch.int64, device=dev)
        _screen_rescore_kernel[(m, screen)](cv, ci, ids, weight, weight.stride(0), h, h.stride(0),
                                            ex, fid, weight.shape[1], n, NCS=ncs, S=screen, BK=1024,
                                            num_warps=4)
        _screen_argmax_kernel[(m,)](ex, fid, oid, oval, S=screen, num_warps=1)
    return oid, oval


def fused_ok(approx, screen: int, chunk: int = 0, sc: int = 0) -> bool:
    chunk = chunk or _FCHUNK
    ncs = triton.cdiv(approx.shape[1], chunk) * min(sc or _FSC, screen)
    return (approx.dim() == 2 and approx.stride(-1) == 1 and screen & (screen - 1) == 0
            and chunk & (chunk - 1) == 0 and ncs & (ncs - 1) == 0 and ncs <= 16384)


# ---------------------------------------------------------------------------------------------
# ATDEV_DRAFT_VOCAB_TOPKX: torch.topk(approx, S, sorted=False) indices, bit-identical, in 2 Triton launches
# instead of the radix multi-block topk (~13 kernels, ~50 us per draft). torch's rule (TensorTopK.cu, gather
# step): every element whose radix key is > the k-th key, in ascending index order, then the elements equal to
# the k-th key in ascending index order until k are written. Radix key of a 16-bit float = bits ^ (sign ?
# 0xffff : 0x8000), NaN -> 0xffff (so -0 < +0 and NaN is the largest). The set = top-k by (key desc, index asc).
# ---------------------------------------------------------------------------------------------
_TOPKX = int(os.environ.get("ATDEV_DRAFT_VOCAB_TOPKX", "0"))
_XCH = 1024  # chunk width of the first pass


@triton.jit
def _topkx_key(x_ptr, sx, r, offs, N):
    ok = offs < N
    b = tl.load(x_ptr + r * sx + offs, mask=ok, other=0.0)
    u = b.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    key = tl.where((u & 0x8000) != 0, u ^ 0xFFFF, u ^ 0x8000)
    key = tl.where(b != b, 0xFFFF, key)
    # composite: (key desc, index asc) as one unique signed int32 (N <= 65536)
    comp = ((key - 32768) << 16) | (65535 - offs)
    return tl.where(ok, comp, -2147483647 - 1)


@triton.jit
def _topkx_chunk_kernel(x_ptr, sx, cc_ptr, N, NC, S: tl.constexpr, CH: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1)
    offs = c * CH + tl.arange(0, CH)
    v = _topkx_key(x_ptr, sx, r, offs, N)
    sj = tl.arange(0, S)
    out = tl.full([S], -2147483647 - 1, tl.int32)
    for s in tl.static_range(S):
        mx = tl.max(v, axis=0)
        out = tl.where(sj == s, mx, out)
        v = tl.where(v == mx, -2147483647 - 1, v)
    tl.store(cc_ptr + (r * NC + c) * S + sj, out)


@triton.jit
def _topkx_merge_kernel(cc_ptr, out_ptr, so, ids_ptr, fo_ptr, NCS: tl.constexpr, S: tl.constexpr):
    r = tl.program_id(0)
    v = tl.load(cc_ptr + r * NCS + tl.arange(0, NCS))
    sj = tl.arange(0, S)
    sel = tl.full([S], -2147483647 - 1, tl.int32)
    for s in tl.static_range(S):
        mx = tl.max(v, axis=0)
        sel = tl.where(sj == s, mx, sel)
        v = tl.where(v == mx, -2147483647 - 1, v)
    key = sel >> 16                    # key - 32768 (arithmetic shift)
    idx = 65535 - (sel & 0xFFFF)
    kth = tl.min(key, axis=0)          # S-th largest key
    cat = tl.where(key > kth, 0, 1)
    # output position = number of selected elements before this one in (cat asc, idx asc) order
    ordk = cat * 65536 + idx
    pos = tl.sum((ordk[None, :] < ordk[:, None]).to(tl.int32), axis=1)
    tl.store(out_ptr + r * so + pos, idx.to(tl.int64))
    tl.store(fo_ptr + r * so + pos, tl.load(ids_ptr + idx))


def topkx_ok(approx: torch.Tensor, screen: int) -> bool:
    n = approx.shape[-1]
    return (approx.dim() == 2 and approx.is_cuda and approx.dtype in (torch.bfloat16, torch.float16)
            and approx.stride(-1) == 1 and n <= 65536 and screen & (screen - 1) == 0 and screen <= _XCH
            and n >= screen and (triton.cdiv(n, _XCH) * screen) & (triton.cdiv(n, _XCH) * screen - 1) == 0)


def topk_unsorted_exact(approx: torch.Tensor, k: int, ids: torch.Tensor | None = None):
    """int64 [M, k] == torch.topk(approx, k, dim=-1, sorted=False).indices (same order) for 16-bit floats;
    with ids (int64 [N]) also returns ids[indices]."""
    m, n = approx.shape
    nc = triton.cdiv(n, _XCH)
    out = torch.empty((m, k), dtype=torch.int64, device=approx.device)
    fo = torch.empty((m, k), dtype=torch.int64, device=approx.device)
    if m:
        cc = torch.empty((m, nc * k), dtype=torch.int32, device=approx.device)
        _topkx_chunk_kernel[(m, nc)](approx, approx.stride(0), cc, n, nc, S=k, CH=_XCH, num_warps=4)
        _topkx_merge_kernel[(m,)](cc, out, out.stride(0), out if ids is None else ids, fo, NCS=nc * k, S=k,
                                  num_warps=4)
    return out if ids is None else (out, fo)


_CFG: DraftVocabConfig | None = None


def config() -> DraftVocabConfig:
    global _CFG
    if _CFG is None:
        _CFG = DraftVocabConfig()
    return _CFG


def head_for(lm_head: torch.nn.Module, hidden: torch.Tensor) -> DraftVocabHead | None:
    """Cached DraftVocabHead of ``lm_head`` (built outside graph capture)."""
    dv = getattr(lm_head, _ATTR, None)
    if dv is not None:
        return dv
    if hidden.is_cuda and torch.cuda.is_current_stream_capturing():
        logger.warning_once("atdev draft_vocab: head missing during capture; stock path")
        return None
    w = lm_head.weight.data
    cfg = config()
    org_vocab = w.shape[0] - lm_head.shard_indices.num_org_vocab_padding
    dv = DraftVocabHead(w, load_ids(cfg, org_vocab), cfg)
    setattr(lm_head, _ATTR, dv)
    logger.warning(
        "atdev draft_vocab: %d of %d ids, bits=%d (%.3f GB per draft pass vs %.2f GB)",
        dv.ids.numel(), org_vocab, cfg.bits, dv.nbytes / 1e9,
        w.numel() * w.element_size() / 1e9,
    )
    return dv


def register() -> None:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    from atdev_serve import shadow_head
    from atdev_serve.register import enabled

    config()
    shadow_head.ScreenedLogitsProcessor._draft_vocab_enabled = True
    if "shadow_head" not in enabled():
        # Nobody else installs the OOT class: install it with the shadow
        # screening of non-draft-vocab calls disabled.
        shadow_head.ScreenedLogitsProcessor._draft_screen_enabled = False
        LogitsProcessor.register_oot(shadow_head.ScreenedLogitsProcessor)
    logger.info("atdev draft_vocab: registered %s", config())
