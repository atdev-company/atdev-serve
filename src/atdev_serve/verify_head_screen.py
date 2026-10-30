"""Certified greedy verify head (ATDEV_OPT=verify_head_screen).

Every spec-decode verify step applies the target lm_head (248320 x 5120 bf16,
2.54 GB) to the k+1 verify rows and the rejection sampler then needs, for a
greedy (temperature 0) batch, only the per-row argmax. This module replaces
that full bf16 GEMM, for eligible batches, by:

1. approximate logits from a low-bit *two-plane* shadow of the lm_head
   (``int4x2``: an int8-equivalent code split into a hi and a lo int4 plane,
   two Marlin W4A16 GEMMs, 1.31 GB; or one int8 Marlin plane, 1.29 GB),
2. a per-(row, token) **upper bound** ``u_ij`` on the value stock vLLM could
   produce for token j (quantization error + both GEMMs' fp32 accumulation
   error + bf16 output rounding, all bounded rigorously, see below),
3. the top-``C`` tokens by ``u`` as candidates, and ``U_i`` = the (C+1)-th
   largest ``u`` = an upper bound for every excluded token,
   (in fact the top-(C+1) set, U_i being its minimum),
4. exact fp32 dot products for the candidates only (gathered rows, Triton),
5. a certificate per row; failed rows are recomputed in full.

The result is handed to the *stock* rejection sampler as an ordinary logits
tensor: candidate columns carry their bf16 logits, every other column -inf.

Exactness argument (per row i; t_j = exact real dot product h_i . W_j)
---------------------------------------------------------------------
Stock computes ``l_j = bf16(t_j + e_j)`` with a single round-to-nearest of an
fp32-accumulated cuBLAS GEMM, ``|e_j| <= g_s * K * sum_k |h_k W_jk|``
(``g_s`` = ``ATDEV_VERIFY_SCREEN_STOCK_GAMMA``, default 2**-22: worst case
fp32 accumulation on tensor cores, including truncating adders). Rounding is
monotone, so:

* Excluded tokens: ``u_ij >= t_j + e_j`` by construction (the shadow residual
  ``||W_j - What_j||_2`` is measured exactly at build time, Cauchy-Schwarz
  gives ``|h.(W_j - What_j)| <= ||h|| r_j``; the shadow GEMMs' accumulation
  and output rounding are added; ``g_s ||h|| ||W_j||`` covers ``e_j``). Hence
  ``l_j <= bf16(U_i)``.
* Candidates: our fp32 dot ``tt_c`` has ``|tt_c - t_c| <= 2**-24 K a_c``
  (CUDA-core FMA, round-to-nearest) where ``a_c = sum |h_k W_ck|``, so
  ``lo_c = bf16(tt_c - d_c) <= l_c <= bf16(tt_c + d_c) = hi_c`` with
  ``d_c = (2**-24 + g_s) K a_c``.
* Let ``L = max_c lo_c``. If ``bf16(U_i) < L`` no excluded token can be (or
  tie with) stock's argmax ("screen certificate").
* If in addition either exactly one candidate has ``hi_c >= L`` (clear
  winner), or every candidate with ``hi_c >= L`` has ``lo_c == hi_c == L``
  (an exact bf16 tie whose members we know, so any index-based tie-break in
  the stock kernel picks the same token from our tensor), the row's stock
  token is determined ("strict certificate").

Rows that fail the screen certificate always fall back to a full-vocab
computation. Rows that pass the screen but not the strict certificate are
*numerical near-ties*: stock's own choice there depends on cuBLAS summation
order (it changes with batch shape in stock vLLM too).

* ``ATDEV_VERIFY_SCREEN_FALLBACK=device`` (default, no host sync): near-ties
  are resolved by our fp32 candidate values (bf16(tt_c), argmax as stock),
  and screen failures are recomputed on device by a flag-gated full GEMV that
  exits immediately when no row is flagged. Output differs from stock only
  when stock's own top-2 lie inside its fp32 accumulation error. Counted.
* ``ATDEV_VERIFY_SCREEN_FALLBACK=graph`` (E5-(a), docs/research/E5-c1-profile.md): exact like ``sync``
  but without a host sync -- one CUDA graph runs the screen, sets a conditional from the device-side
  "any row without a strict certificate" flag, and an IF node runs the *stock* head (same op, same
  shapes -> bit-identical logits) only then (csrc/cond_head.cu). Draft screening keeps shadow_head's
  own shadow (drafting identical to shadow_head alone); the verify shadow is cached separately.
* ``ATDEV_VERIFY_SCREEN_FALLBACK=sync``: any row without a strict certificate
  makes the whole batch use the stock GEMM (bit-identical to stock vLLM,
  costs one host sync per verify step). Use for the accuracy gate.

Eligibility (host side, checked every step in ``GPUModelRunner.sample``):
spec-decode verify through the stock ``RejectionSampler``; every request
greedy, no logprobs, no logits processing (penalties / bias / bad words /
min-tokens / thinking budget / custom processors), no structured output, no
batch-sharded sampling, no watermark, no NaN counting, no sampling masks;
``LogitsProcessor`` with scale 1, no soft cap, TP 1, bf16 unquantized head;
rows <= ``ATDEV_VERIFY_SCREEN_MAX_ROWS``. Anything else: stock path.

Delivery:
* ``LogitsProcessor`` OOT subclass (of ``shadow_head.ScreenedLogitsProcessor``
  so one shadow serves draft screening and verify when both opts are on; the
  draft uses the hi plane only, i.e. an int4 g128 head).
* A version-asserted wrapper around ``GPUModelRunner.sample`` (MRV2) that
  only sets a context flag for eligible batches; vLLM's code is otherwise
  unchanged.

Env knobs:
  ATDEV_VERIFY_SCREEN_FORMAT      int4x2 (default) | int8
  ATDEV_VERIFY_SCREEN_CANDIDATES  C (default 256)
  ATDEV_VERIFY_SCREEN_MAX_ROWS    max verify rows per batch (default 64)
  ATDEV_VERIFY_SCREEN_FALLBACK    device (default) | sync
  ATDEV_VERIFY_SCREEN_STOCK_GAMMA per-term relative accumulation bound of the
                                  stock GEMM (default 2**-22)
  ATDEV_VERIFY_SCREEN_STATS       log certificate statistics every N calls
                                  (0 = off, default 0; costs a sync per log)
  ATDEV_VERIFY_SCREEN_CHECK       1 = also run the stock GEMM every call and
                                  count argmax mismatches (debug; slow)
"""

from __future__ import annotations

import contextvars
import functools
import os
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import _topk

from atdev_serve.shadow_head import (
    ScreenedLogitsProcessor,
    ShadowHead,
    ShadowHeadConfig,
    _pack_rows,
)

try:  # Triton is only needed on CUDA (or under TRITON_INTERPRET=1 in tests).
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

logger = init_logger(__name__)

OPT_NAME = "verify_head_screen"
SUPPORTED_VLLM_VERSION = "0.31.0"

# Unit roundoff of bf16 (8 significant bits) and fp32 (round-to-nearest).
U_BF16 = 2.0**-8
U_FP32 = 2.0**-24
# Per-term relative error bound for fp32 accumulation on tensor cores (allows
# round-toward-zero adders): used for the shadow GEMMs and, by default, stock.
GAMMA_TC = 2.0**-22


@dataclass(frozen=True)
class VerifyScreenConfig:
    fmt: str = "int4x2"
    group_size: int = 128
    candidates: int = 256
    max_rows: int = 64
    fallback: str = "device"
    stock_gamma: float = GAMMA_TC
    stats_every: int = 0
    check: bool = False

    def __post_init__(self) -> None:
        if self.fmt not in ("int4x2", "int8"):
            raise ValueError(f"verify screen format must be int4x2|int8: {self.fmt}")
        if self.group_size not in (32, 64, 128):
            raise ValueError(f"verify screen group must be 32/64/128: {self.group_size}")
        if self.candidates < 2:
            raise ValueError("verify screen needs >= 2 candidates")
        if self.fallback not in ("device", "sync", "graph"):
            raise ValueError(f"verify screen fallback must be device|sync|graph: {self.fallback}")
        if not (0.0 <= self.stock_gamma < 1e-2):
            raise ValueError(f"implausible stock gamma {self.stock_gamma}")

    @classmethod
    def from_env(cls) -> VerifyScreenConfig:
        return cls(
            fmt=os.environ.get("ATDEV_VERIFY_SCREEN_FORMAT", "int4x2"),
            group_size=int(os.environ.get("ATDEV_VERIFY_SCREEN_GROUP", "128")),
            candidates=int(os.environ.get("ATDEV_VERIFY_SCREEN_CANDIDATES", "256")),
            max_rows=int(os.environ.get("ATDEV_VERIFY_SCREEN_MAX_ROWS", "64")),
            fallback=os.environ.get("ATDEV_VERIFY_SCREEN_FALLBACK", "device"),
            stock_gamma=float(
                os.environ.get("ATDEV_VERIFY_SCREEN_STOCK_GAMMA", str(GAMMA_TC))
            ),
            stats_every=int(os.environ.get("ATDEV_VERIFY_SCREEN_STATS", "0")),
            check=os.environ.get("ATDEV_VERIFY_SCREEN_CHECK", "0") == "1",
        )


# ---------------------------------------------------------------------------
# Two-plane / int8 shadow with exact per-row residual norms
# ---------------------------------------------------------------------------


@dataclass
class PlaneCodes:
    packed: torch.Tensor  # GPTQ row-packed [K * bits / 32, N] int32
    scales: torch.Tensor  # [K / group, N] weight dtype
    bits: int


@torch.no_grad()
def quantize_planes(
    weight: torch.Tensor, fmt: str, group_size: int, chunk_rows: int = 8192
) -> tuple[list[PlaneCodes], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize an [N, K] head into 1 (int8) or 2 (int4x2) Marlin planes.

    int4x2: s = max(amax/119, amin/-120) per group (bf16), q8 = round(w/s) in
    [-120, 119], hi = floor((q8 + 8) / 16) in [-8, 7] with scale 16 s, lo =
    q8 - 16 hi in [-8, 7] with scale s. hi alone is an int4 g128 quantizer
    (scale ~ amax/7.4), so draft screening can reuse it.

    Returns ``(planes, resid_norm, plane_norm_sum, row_norm)``, all norms fp32
    [N]: ``resid_norm[j] = ||W_j - What_j||_2`` maximised over the two ways a
    kernel may materialise What (per-element bf16-rounded products, as Marlin
    does for grouped scales, or exact products), ``plane_norm_sum[j] =
    sum_p ||What_p,j||_2`` (for the accumulation-error term) and ``row_norm[j]
    = ||W_j||_2``.
    """
    assert weight.dim() == 2 and weight.is_floating_point()
    n, k = weight.shape
    if k % group_size:
        raise ValueError(f"K={k} not divisible by group {group_size}")
    g = k // group_size
    dev, dt = weight.device, weight.dtype
    if fmt == "int4x2":
        specs = [(4, 8), (4, 8)]  # (bits, bias) for hi, lo
        qmax, qmin = 119, -120
    elif fmt == "int8":
        specs = [(8, 128)]
        qmax, qmin = 127, -128
    else:
        raise ValueError(fmt)
    planes = [
        PlaneCodes(
            packed=torch.empty((k * bits // 32, n), dtype=torch.int32, device=dev),
            scales=torch.empty((g, n), dtype=dt, device=dev),
            bits=bits,
        )
        for bits, _ in specs
    ]
    resid = torch.empty(n, dtype=torch.float32, device=dev)
    pnorm = torch.empty(n, dtype=torch.float32, device=dev)
    wnorm = torch.empty(n, dtype=torch.float32, device=dev)
    for start in range(0, n, chunk_rows):
        end = min(n, start + chunk_rows)
        rows = end - start
        w = weight[start:end].float().view(rows, g, group_size)
        s = torch.maximum(w.amax(-1) / qmax, w.amin(-1) / qmin).to(dt)  # [R, G]
        s_f = s.float().clamp(min=torch.finfo(torch.float32).tiny)
        q8 = torch.round(w / s_f.unsqueeze(-1)).clamp(qmin, qmax)
        if fmt == "int4x2":
            hi = torch.floor((q8 + 8) / 16)
            codes = [(hi, (s.float() * 16).to(dt)), (q8 - 16 * hi, s)]
        else:
            codes = [(q8, s)]
        deq_round = torch.zeros_like(w)
        deq_exact = torch.zeros_like(w)
        pn = torch.zeros(rows, dtype=torch.float32, device=dev)
        for plane, (bits, bias), (q, sc) in zip(planes, specs, codes):
            prod = q * sc.float().unsqueeze(-1)  # exact in fp32
            rnd = prod.to(dt).float()
            deq_round += rnd
            deq_exact += prod
            pn += torch.maximum(
                rnd.flatten(1).norm(dim=1), prod.flatten(1).norm(dim=1)
            )
            qb = (q.to(torch.int32) + bias).view(rows, k).t().contiguous()
            plane.packed[:, start:end] = _pack_rows(qb, bits)
            plane.scales[:, start:end] = sc.t()
        wf = w.flatten(1)
        resid[start:end] = torch.maximum(
            (wf - deq_round.flatten(1)).double().norm(dim=1),
            (wf - deq_exact.flatten(1)).double().norm(dim=1),
        ).float()
        pnorm[start:end] = pn
        wnorm[start:end] = wf.double().norm(dim=1).float()
    return planes, resid, pnorm, wnorm


class VerifyShadow:
    """Shadow lm_head for certified verify screening (and draft screening).

    Duck-types the bits of ``ShadowHead`` that ``shadow_head.screened_top_k``
    uses (``config.screen``, ``approx_logits``, ``backend``, ``nbytes``), with
    plane 0 (the hi int4 plane, or the int8 plane) as the draft head.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        cfg: VerifyScreenConfig,
        draft_screen: int = 128,
        backend: str | None = None,
    ) -> None:
        n, k = weight.shape
        self.cfg = cfg
        self.size_n, self.size_k = n, k
        planes, resid, pnorm, wnorm = quantize_planes(weight, cfg.fmt, cfg.group_size)
        self.planes: list[ShadowHead] = []
        for p in planes:
            pcfg = ShadowHeadConfig(bits=p.bits, group_size=cfg.group_size,
                                    screen=draft_screen)
            self.planes.append(ShadowHead.from_packed(
                p.packed, p.scales, pcfg, n, k, weight.dtype, backend=backend))
        del planes
        self.backend = self.planes[0].backend
        self.config = self.planes[0].config  # draft screening uses plane 0
        # Per-token bound coefficient, multiplied by ||h_i||_2 at run time:
        #   quantization residual (Cauchy-Schwarz)
        # + shadow GEMMs' fp32 accumulation   (GAMMA_TC * K * ||What_p||)
        # + stock GEMM's fp32 accumulation    (stock_gamma * K * ||W_j||)
        # The shadow outputs' bf16 rounding is added per element at run time.
        self.row_bound = (
            resid * (1 + 1e-4)
            + (GAMMA_TC * k) * pnorm * (1 + 1e-4)
            + (cfg.stock_gamma * k) * wnorm * (1 + 1e-4)
        ).contiguous()
        self.resid_norm = resid
        self.nbytes = sum(p.nbytes for p in self.planes) + self.row_bound.numel() * 4

    def describe(self) -> str:
        return f"{self.cfg.fmt} g{self.cfg.group_size} (verify C={self.cfg.candidates})"

    def approx_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.planes[0].approx_logits(hidden_states)

    def approx_planes(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        return [p.approx_logits(hidden_states) for p in self.planes]


# ---------------------------------------------------------------------------
# Kernels (Triton on CUDA; torch references used on CPU and in tests)
# ---------------------------------------------------------------------------

if triton is not None:

    @triton.jit
    def _upper_bound_kernel(
        a0_ptr, a1_ptr, stride_a0, stride_a1,
        row_bound_ptr, hnorm_ptr, u_ptr, stride_u,
        n_valid, n_cols, c_rel,
        TWO_PLANES: tl.constexpr, BLOCK: tl.constexpr,
    ):
        m = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        in_range = cols < n_cols
        valid = cols < n_valid
        a0 = tl.load(a0_ptr + m * stride_a0 + cols, mask=valid, other=0.0).to(tl.float32)
        approx = a0
        mag = tl.abs(a0)
        if TWO_PLANES:
            a1 = tl.load(a1_ptr + m * stride_a1 + cols, mask=valid, other=0.0).to(tl.float32)
            approx = approx + a1
            mag = mag + tl.abs(a1)
        hn = tl.load(hnorm_ptr + m)
        rb = tl.load(row_bound_ptr + cols, mask=valid, other=0.0)
        u = approx + c_rel * mag + hn * rb
        u = tl.where(valid, u, float("-inf"))
        tl.store(u_ptr + m * stride_u + cols, u, mask=in_range)

    @triton.jit
    def _candidate_dot_kernel(
        h_ptr, stride_h, w_ptr, stride_w, cand_ptr, stride_c,
        dot_ptr, absdot_ptr, stride_o,
        num_cand, K,
        BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < num_cand
        rows = tl.load(cand_ptr + m * stride_c + offs_c, mask=mask_c, other=0).to(tl.int64)
        acc = tl.zeros([BLOCK_C], dtype=tl.float32)
        aacc = tl.zeros([BLOCK_C], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            hk = tl.load(h_ptr + m * stride_h + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            w = tl.load(
                w_ptr + rows[:, None] * stride_w + offs_k[None, :],
                mask=mask_c[:, None] & mask_k[None, :], other=0.0,
            ).to(tl.float32)
            prod = w * hk[None, :]
            acc += tl.sum(prod, axis=1)
            aacc += tl.sum(tl.abs(prod), axis=1)
        tl.store(dot_ptr + m * stride_o + offs_c, acc, mask=mask_c)
        tl.store(absdot_ptr + m * stride_o + offs_c, aacc, mask=mask_c)

    @triton.jit
    def _flagged_gemv_kernel(
        h_ptr, stride_h, w_ptr, stride_w, flags_ptr, out_ptr, stride_out,
        M, n_cols, K,
        M_POW2: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        # Full-vocab bf16 logits (fp32 accumulate, one rounding) for the rows
        # whose flag is set. Exits immediately when no row is flagged, so it
        # costs one tiny launch on the common path and needs no host sync.
        offs_m = tl.arange(0, M_POW2)
        flags = tl.load(flags_ptr + offs_m, mask=offs_m < M, other=0)
        if tl.max(flags, axis=0) == 0:
            return
        offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_cols
        for m in range(0, M):
            if tl.load(flags_ptr + m) != 0:
                acc = tl.zeros([BLOCK_N], dtype=tl.float32)
                for k0 in range(0, K, BLOCK_K):
                    offs_k = k0 + tl.arange(0, BLOCK_K)
                    mask_k = offs_k < K
                    hk = tl.load(h_ptr + m * stride_h + offs_k, mask=mask_k,
                                 other=0.0).to(tl.float32)
                    w = tl.load(
                        w_ptr + offs_n[:, None].to(tl.int64) * stride_w + offs_k[None, :],
                        mask=mask_n[:, None] & mask_k[None, :], other=0.0,
                    ).to(tl.float32)
                    acc += tl.sum(w * hk[None, :], axis=1)
                tl.store(out_ptr + m * stride_out + offs_n,
                         acc.to(out_ptr.dtype.element_ty), mask=mask_n)


def _use_triton(t: torch.Tensor) -> bool:
    if triton is None:
        return False
    return t.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"


def upper_bounds(
    planes: list[torch.Tensor],
    hnorm: torch.Tensor,
    row_bound: torch.Tensor,
    n_valid: int,
    force_torch: bool = False,
) -> torch.Tensor:
    """u[i, j] >= any value stock could compute for token j (see module doc)."""
    m, n = planes[0].shape
    c_rel = U_BF16 / (1 - U_BF16) * 1.001
    if not force_torch and _use_triton(planes[0]):
        u = torch.empty((m, n), dtype=torch.float32, device=planes[0].device)
        two = len(planes) == 2
        a1 = planes[1] if two else planes[0]
        BLOCK = 2048
        _upper_bound_kernel[(m, triton.cdiv(n, BLOCK))](
            planes[0], a1, planes[0].stride(0), a1.stride(0),
            row_bound, hnorm, u, u.stride(0),
            n_valid, n, c_rel,
            TWO_PLANES=two, BLOCK=BLOCK,
        )
        return u
    approx = torch.zeros((m, n), dtype=torch.float32, device=planes[0].device)
    mag = torch.zeros_like(approx)
    for p in planes:
        pf = p.float()
        approx += pf
        mag += pf.abs()
    u = approx + c_rel * mag + hnorm[:, None] * row_bound[None, :]
    u[:, n_valid:] = float("-inf")
    return u


def candidate_dots(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    cand: torch.Tensor,
    force_torch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 dot and |.|-dot of hidden[i] with weight[cand[i, c]]: [M, C] each."""
    m, c = cand.shape
    k = hidden.shape[1]
    if not force_torch and _use_triton(hidden):
        dot = torch.empty((m, c), dtype=torch.float32, device=hidden.device)
        absdot = torch.empty_like(dot)
        cand_c = cand.contiguous()
        BLOCK_C, BLOCK_K = 16, 256
        _candidate_dot_kernel[(m, triton.cdiv(c, BLOCK_C))](
            hidden, hidden.stride(0), weight, weight.stride(0),
            cand_c, cand_c.stride(0), dot, absdot, dot.stride(0),
            c, k, BLOCK_C=BLOCK_C, BLOCK_K=BLOCK_K,
        )
        return dot, absdot
    rows = weight.index_select(0, cand.reshape(-1)).view(m, c, k).float()
    prod = rows * hidden.float().unsqueeze(1)
    return prod.sum(-1), prod.abs().sum(-1)


def flagged_full_logits_(
    out: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    flags: torch.Tensor,
    force_torch: bool = False,
) -> None:
    """Overwrite rows of ``out`` whose flag is set with full logits."""
    m, n_cols = out.shape
    k = hidden.shape[1]
    if not force_torch and _use_triton(hidden):
        BLOCK_N, BLOCK_K = 64, 128
        _flagged_gemv_kernel[(triton.cdiv(n_cols, BLOCK_N),)](
            hidden, hidden.stride(0), weight, weight.stride(0),
            flags, out, out.stride(0), m, n_cols, k,
            M_POW2=triton.next_power_of_2(max(m, 1)),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return
    rows = flags.nonzero().flatten()
    if rows.numel():
        full = (hidden[rows].float() @ weight[:n_cols].float().t()).to(out.dtype)
        out[rows] = full


# ---------------------------------------------------------------------------
# Certified screening
# ---------------------------------------------------------------------------


@dataclass
class ScreenResult:
    logits: torch.Tensor  # [M, n_out] bf16 (valid only if not need_stock)
    fallback_rows: torch.Tensor  # [M] int32: 1 = screen certificate failed
    near_tie_rows: torch.Tensor  # [M] bool: screen ok, strict certificate not
    cand: torch.Tensor  # [M, C]
    dot: torch.Tensor  # [M, C] fp32


def screen_logits(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    shadow: VerifyShadow,
    n_out: int,
    candidates: int,
    stock_gamma: float,
    fill_fallback: bool = True,
    force_torch: bool = False,
) -> ScreenResult:
    """Certified sparse logits for greedy rows (see module docstring).

    ``n_out`` = number of logits columns stock returns (org vocab); weight
    rows beyond it (vocab padding) are never candidates.
    """
    m, k = hidden.shape
    n_valid = min(n_out, weight.shape[0])
    c = min(candidates, n_valid - 1)
    planes = shadow.approx_planes(hidden)
    hnorm = hidden.float().norm(dim=-1) * (1 + 1e-5)
    u = upper_bounds(planes, hnorm, shadow.row_bound, n_valid, force_torch)
    # Candidates = the top-(c+1) tokens by upper bound (order irrelevant);
    # every other token's u is <= the smallest u in that set.
    vals, cand = _topk(u, c + 1)
    cand = cand.to(torch.int64)
    u_excl = vals.amin(dim=-1)
    dot, absdot = candidate_dots(hidden, weight, cand, force_torch)

    d = (U_FP32 + stock_gamma) * k * absdot * 1.001 + 1e-30
    lo = (dot - d).to(torch.bfloat16).float()
    hi = (dot + d).to(torch.bfloat16).float()
    best_lo = lo.amax(dim=-1)
    screen_ok = u_excl.to(torch.bfloat16).float() < best_lo
    contenders = hi >= best_lo[:, None]
    clear_win = contenders.sum(-1) == 1
    exact_tie = ((lo == hi) | ~contenders).all(-1)
    strict_ok = screen_ok & (clear_win | exact_tie)

    logits = torch.full((m, n_out), float("-inf"), dtype=weight.dtype,
                        device=hidden.device)
    logits.scatter_(1, cand, dot.to(weight.dtype))
    fallback = (~screen_ok).to(torch.int32)
    if fill_fallback:
        flagged_full_logits_(logits, hidden, weight, fallback, force_torch)
    return ScreenResult(logits, fallback, screen_ok & ~strict_ok, cand, dot)


# ---------------------------------------------------------------------------
# LogitsProcessor override + runner hook
# ---------------------------------------------------------------------------

# Set (True) by the GPUModelRunner.sample wrapper for eligible greedy verify
# batches only; LogitsProcessor.forward reads it.
_VERIFY_CTX: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "atdev_verify_screen", default=False
)


_COND_EXT = None


def _cond_ext():
    global _COND_EXT
    if _COND_EXT is None:
        from torch.utils.cpp_extension import load

        src = os.path.join(os.path.dirname(__file__), "csrc", "cond_head.cu")
        _COND_EXT = load(name="atdev_cond_head", sources=[src], verbose=False,
                         extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80", "-std=c++17"])
    return _COND_EXT


class _CondHead:
    """Static-buffer conditional graph for one row count M (see ATDEV_VERIFY_SCREEN_FALLBACK=graph).

    screen_fn(h) -> (logits, fallback_rows, near_tie_rows); stock_fn(h) -> logits (the stock op).
    """

    def __init__(self, hidden_states, screen_fn, stock_fn, desc=""):
        dev = hidden_states.device
        m = hidden_states.shape[0]
        self.m = m
        self.h_in = hidden_states.detach().clone()
        self.need = torch.zeros(1, dtype=torch.int32, device=dev)
        self.stats = torch.zeros(4, dtype=torch.int64, device=dev)  # rows, fallback steps, fb rows, near ties
        ref = stock_fn(self.h_in)
        self.logits = torch.empty_like(ref)

        def g1_body():
            lg, fb, nt = screen_fn(self.h_in)
            self.logits.copy_(lg)
            fb = fb.bool()
            nd = (fb | nt).any()
            self.need.copy_(nd.to(torch.int32).view(1))
            self.stats.add_(torch.stack([torch.full((), m, dtype=torch.int64, device=dev), nd.to(torch.int64),
                                         fb.sum().to(torch.int64), nt.sum().to(torch.int64)]))

        def g2_body():
            self.logits.copy_(stock_fn(self.h_in))

        g1_body(); g2_body()  # warm up (Triton compile, cuBLAS handles) outside capture
        torch.cuda.synchronize()
        # vLLM's workspace manager keys persistent resources (e.g. Marlin locks) by stream and is locked
        # after warmup; the capture stream needs its own -> build once with the manager unlocked.
        relock = False
        try:
            from vllm.v1.worker import workspace as _ws
            mgr = _ws.current_workspace_manager()
            if getattr(mgr, "_locked", False):
                _ws.unlock_workspace()
                relock = True
        except Exception:  # noqa: BLE001  (no manager: tests)
            _ws = None
        try:
            pool = torch.cuda.graph_pool_handle()
            self.g1 = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(self.g1, pool=pool, capture_error_mode="thread_local"):
                g1_body()
            self.g2 = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(self.g2, pool=pool, capture_error_mode="thread_local"):
                g2_body()
        finally:
            if relock:
                _ws.lock_workspace()
        self.stats.zero_()
        self.id = _cond_ext().build(self.g1.raw_cuda_graph(), self.g2.raw_cuda_graph(), self.need)
        torch.cuda.synchronize()
        logger.warning("atdev %s: conditional exact head graph built for M=%d %s", OPT_NAME, m, desc)

    def __call__(self, hidden_states):
        self.h_in.copy_(hidden_states)
        _cond_ext().launch(self.id)
        return self.logits


class VerifyScreenLogitsProcessor(ScreenedLogitsProcessor):
    """Certified greedy verify logits; draft screening as in shadow_head."""

    _verify_config: VerifyScreenConfig | None = None
    _calls = 0
    _stats: torch.Tensor | None = None  # [rows, fallback rows, near ties, mism.]

    @classmethod
    def verify_config(cls) -> VerifyScreenConfig:
        if cls._verify_config is None:
            cls._verify_config = VerifyScreenConfig.from_env()
        return cls._verify_config

    def _build_shadow(self, weight: torch.Tensor):
        if self.verify_config().fallback == "graph":
            # draft screening keeps shadow_head's own shadow (drafting identical to shadow_head alone)
            from atdev_serve.shadow_head import ShadowHead as _SH
            return _SH(weight, self.shadow_config())
        return VerifyShadow(weight, self.verify_config(),
                            draft_screen=self.shadow_config().screen)

    def _vshadow_for(self, lm_head):
        vs = getattr(lm_head, "_atdev_verify_shadow", None)
        if vs is None:
            vs = VerifyShadow(lm_head.weight.data, self.verify_config(),
                              draft_screen=self.shadow_config().screen)
            setattr(lm_head, "_atdev_verify_shadow", vs)
            logger.warning("atdev %s: built verify shadow %s (%.2f GB)", OPT_NAME, vs.describe(),
                           vs.nbytes / 1e9)
        return vs

    def _verify_eligible(self, lm_head, hidden_states, embedding_bias, skip_gather):
        return (
            not self.logits_as_input
            and not skip_gather
            and self.soft_cap is None
            and self.scale == 1.0
            and 0 < hidden_states.shape[0] <= self.verify_config().max_rows
            and self._can_screen(lm_head, hidden_states, embedding_bias)
        )

    def forward(
        self,
        lm_head,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
        skip_gather: bool = False,
    ) -> torch.Tensor | None:
        # Build the shadow on the first structurally eligible forward (the
        # profile run's dummy sampler call, whatever its row count), i.e.
        # before KV-cache sizing and CUDA-graph capture, so the profiler
        # accounts for its memory.
        shadow = None
        if hidden_states.shape[0] > 0 and not skip_gather:
            shadow = self._shadow_for(lm_head, hidden_states, embedding_bias)
        if self.verify_config().fallback == "graph":
            return self._forward_graph(lm_head, hidden_states, embedding_bias, skip_gather, shadow)
        if (
            not _VERIFY_CTX.get()
            or not isinstance(shadow, VerifyShadow)
            or not self._verify_eligible(
                lm_head, hidden_states, embedding_bias, skip_gather
            )
        ):
            return super().forward(lm_head, hidden_states, embedding_bias, skip_gather)
        if hidden_states.is_cuda and torch.cuda.is_current_stream_capturing():
            return super().forward(lm_head, hidden_states, embedding_bias, skip_gather)
        cfg = self.verify_config()
        res = screen_logits(
            hidden_states,
            lm_head.weight,
            shadow,
            n_out=self.org_vocab_size,
            candidates=cfg.candidates,
            stock_gamma=cfg.stock_gamma,
            fill_fallback=cfg.fallback == "device",
        )
        stock = None
        if cfg.fallback == "sync":
            if bool((res.fallback_rows.bool() | res.near_tie_rows).any()):
                stock = super().forward(lm_head, hidden_states, embedding_bias,
                                        skip_gather)
        if cfg.check or cfg.stats_every:
            self._account(res, lm_head, hidden_states, embedding_bias, stock, cfg)
        return stock if stock is not None else res.logits

    _cond: dict = {}
    _gcalls = 0
    _gmism = 0

    def _forward_graph(self, lm_head, hidden_states, embedding_bias, skip_gather, shadow):
        stock = lambda: super(VerifyScreenLogitsProcessor, self).forward(  # noqa: E731
            lm_head, hidden_states, embedding_bias, skip_gather)
        if shadow is None or hidden_states.shape[0] == 0 or skip_gather:
            return stock()
        capturing = hidden_states.is_cuda and torch.cuda.is_current_stream_capturing()
        if not capturing:
            vs = self._vshadow_for(lm_head)  # built at the first (profile) forward, before KV sizing
        if (capturing or not _VERIFY_CTX.get()
                or not self._verify_eligible(lm_head, hidden_states, embedding_bias, skip_gather)):
            return stock()
        cfg = self.verify_config()
        m = hidden_states.shape[0]
        key = (m, id(lm_head))
        ch = type(self)._cond.get(key)
        if ch is None:
            n_out = self.org_vocab_size

            def screen_fn(h, vs=vs, w=lm_head.weight, cfg=cfg, n_out=n_out):
                res = screen_logits(h, w, vs, n_out=n_out, candidates=cfg.candidates,
                                    stock_gamma=cfg.stock_gamma, fill_fallback=False)
                return res.logits, res.fallback_rows, res.near_tie_rows

            def stock_fn(h):
                return ScreenedLogitsProcessor.forward(self, lm_head, h, embedding_bias, skip_gather)

            ch = type(self)._cond[key] = _CondHead(hidden_states, screen_fn, stock_fn,
                                                   f"(screen {vs.describe()})")
        out = ch(hidden_states)
        cls = type(self)
        cls._gcalls += 1
        if cfg.check:
            ref = stock()
            cls._gmism += int((out.float().argmax(-1) != ref.float().argmax(-1)).sum())
        every = cfg.stats_every or 0
        if every and cls._gcalls % every == 0:
            rows, fs, fr, nt = ch.stats.tolist()
            logger.warning("atdev %s graph: calls=%d rows=%d stock-fallback steps=%d (%.4f) fb rows=%d near-tie "
                           "rows=%d%s", OPT_NAME, cls._gcalls, rows, fs, fs / max(1, rows // m), fr, nt,
                           f" argmax-mismatch-vs-stock={cls._gmism}" if cfg.check else "")
        return out

    def _account(self, res, lm_head, hidden_states, embedding_bias, stock, cfg):
        cls = type(self)
        if cls._stats is None or cls._stats.device != hidden_states.device:
            cls._stats = torch.zeros(4, dtype=torch.int64, device=hidden_states.device)
        st = cls._stats
        st[0] += hidden_states.shape[0]
        st[1] += res.fallback_rows.sum()
        st[2] += res.near_tie_rows.sum()
        if cfg.check:
            ref = stock if stock is not None else super().forward(
                lm_head, hidden_states, embedding_bias)
            ours = stock if stock is not None else res.logits
            st[3] += (ours.float().argmax(-1) != ref.float().argmax(-1)).sum()
        cls._calls += 1
        every = cfg.stats_every or 256
        if cls._calls % every == 0:
            rows, fb, nt, mm = st.tolist()
            logger.info(
                "atdev verify_head_screen: rows=%d full-fallback=%.4f "
                "near-tie=%.4f%s", rows, fb / max(rows, 1), nt / max(rows, 1),
                f" argmax-mismatch-vs-stock={mm}" if cfg.check else "",
            )


def greedy_verify_batch_eligible(runner, input_batch, grammar_output) -> bool:
    """Host-side check that stock sampling of this batch is a pure argmax."""
    import numpy as np

    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    if grammar_output is not None or getattr(runner, "batch_sharder", None) is not None:
        return False
    rs = getattr(runner, "rejection_sampler", None)
    s = getattr(runner, "sampler", None)
    if type(rs) is not RejectionSampler or type(s) is not Sampler:
        return False
    if input_batch.num_draft_tokens == 0 or input_batch.num_reqs == 0:
        return False
    if rs.watermark_key is not None or rs.synthetic_conditional_rates is not None:
        return False
    spec = getattr(runner, "speculator", None)
    if spec is not None and getattr(spec, "draft_logits", None) is not None:
        return False
    if s.compute_nans or s.return_sampling_mask or s.trace_replay_state is not None:
        return False
    idx = input_batch.idx_mapping_np
    if np.any(s.needs_logits_processing[idx]):
        return False
    if not np.all(s.sampling_states.temperature.np[idx] == 0.0):
        return False
    if s.sampling_states.max_num_logprobs(idx) != NO_LOGPROBS:
        return False
    return True


def _install_runner_hook() -> None:
    import vllm
    from vllm.v1.worker.gpu import model_runner as mr

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"atdev {OPT_NAME}: runner hook written for vLLM "
            f"{SUPPORTED_VLLM_VERSION}, found {vllm.__version__}"
        )
    cls = mr.GPUModelRunner
    if getattr(cls.sample, "_atdev_verify_screen", False):
        return
    orig = cls.sample

    @functools.wraps(orig)
    def sample(self, hidden_states, input_batch, grammar_output):
        if not greedy_verify_batch_eligible(self, input_batch, grammar_output):
            return orig(self, hidden_states, input_batch, grammar_output)
        token = _VERIFY_CTX.set(True)
        try:
            return orig(self, hidden_states, input_batch, grammar_output)
        finally:
            _VERIFY_CTX.reset(token)

    sample._atdev_verify_screen = True
    cls.sample = sample


def register() -> None:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    from atdev_serve.register import enabled

    VerifyScreenLogitsProcessor._draft_screen_enabled = "shadow_head" in enabled()
    LogitsProcessor.register_oot(VerifyScreenLogitsProcessor)
    _install_runner_hook()
    cfg = VerifyScreenLogitsProcessor.verify_config()
    logger.info(
        "atdev %s: registered (format=%s C=%d fallback=%s max_rows=%d, "
        "draft screening %s)", OPT_NAME, cfg.fmt, cfg.candidates, cfg.fallback,
        cfg.max_rows,
        "on" if VerifyScreenLogitsProcessor._draft_screen_enabled else "off",
    )
