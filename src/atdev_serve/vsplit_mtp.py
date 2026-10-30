"""verify_split_head FAST path for the MTP / chain greedy rejection sampler at c=32
(ATDEV_OPT=...,verify_split_head,vsplit_mtp + ATDEV_VSPLIT_FAST=1). Output bit-identical.

At c=32, MTP k=2 every decode step verifies 96 rows (32 req x 3). The stock chain after
the transformer (LEDGER C32-prof, eager, 1.58 ms) is

  F = bf16 cuBLAS head part (65536 rows)             0.47 ms
  tail = int4 Marlin W4A8 (182784 rows) at M=96      0.76 ms  (stock Marlin = 2 launches,
                                                               64 + 32 rows, each streams
                                                               the full 468 MB tail)
  2x index_copy_ into [96, 248320] logits            0.18 ms
  RejectionSampler: gather_draft + local stats (96 x 31 blocks) + rejection + resample
  (32 x 243 blocks) + insert                         ~0.12 ms

This module (each piece default off, all bit-identical to the above):

* ATDEV_VSPLIT_MTP_ARGMAX=1 (default 1 when the opt is on): on all-greedy spec-decode
  sample() calls (no logprobs / grammar / logits processing / penalties / draft logits /
  watermark / NaN counting / batch sharding), _apply_head returns an unused placeholder
  and computes the per-row argmax with verify_split_head.split_argmax over the two GEMM
  outputs (torch.argmax semantics = leftmost max of the same bf16 values); the patched
  RejectionSampler.__call__ then runs one accept kernel (stock greedy rule incl. the -1
  prefill-placeholder drafts) and the stock get_num_sampled_and_rejected. Same tokens and
  counts as stock whenever the logits contain no NaN (stock maps NaN block maxima to -inf).
* ATDEV_VSPLIT_TAIL_MP=64,256,256,6,1,4: the A8 tail GEMM (M >= ATDEV_VSPLIT_TAIL_MIN,
  default 17) on marlin_plus with the stock tile (thread_k 64, thread_n 256, 256 threads,
  same 108-CTA grid -> same split-K stripes) and a 6-stage pipeline: bit-identical, checked
  torch.equal against the stock tail the first time each M is seen (else stock for that M).
  M=96: 759 -> 715 us. Used by both the argmax path and the stock logits path.
* ATDEV_VSPLIT_TAIL_MS=128,128,256,6,1,6,2,0 + ATDEV_VSPLIT_TAIL_MS_EXACT=0 (ROUNDING-LEVEL,
  not bit-identical): one-pass M-split kernel (vsplit_tail_build, tmb=6 = 96 rows, MW=2);
  thread_n 128 changes the split-K stripes -> fp32 cross-CTA order: 52 of 17.5M tail logits
  differ by 1 bf16 ulp at M=96 (random head). M=96: 606 us. With EXACT=1 (default) it is
  bit-checked like TAIL_MP and therefore falls back to stock.
* ATDEV_VSPLIT_STREAMS=1 (verify_split_head): tail on a side stream next to the bf16 part.

Eager only (the sampling step is not CUDA-graphed at c=32); capture-time calls fall back.
"""

from __future__ import annotations

import functools
import os

import numpy as np
import torch

import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger(__name__)

OPT_NAME = "vsplit_mtp"
SUPPORTED_VLLM_VERSION = "0.31.0"


class _State:
    rows: int | None = None          # expected verify rows of the current eligible sample()
    fast: tuple | None = None        # (placeholder data_ptr, rows, int64 argmax [rows])
    hits: int = 0
    batch = None                     # input_batch of the current eligible sample() (dump only)


S = _State()
# di1_varlen: greedy verify on adaptive-verification (varlen, device cu_num_logits) batches is the same
# stock rule (single logits chunk), so the argmax fast path stays valid there
ADAPTIVE_OK = False

# ATDEV_VSPLIT_DUMP=<file.pt>[,N[,skip]]: save the verify hidden states + draft tokens of N eligible
# steps (default 64, after `skip` steps, default 200) for offline real-logit tests (tests/
# test_vsplit_f8_c32.py). Instrumentation only (host syncs); outputs unchanged.
_DUMP = None
if os.environ.get("ATDEV_VSPLIT_DUMP"):
    _p = os.environ["ATDEV_VSPLIT_DUMP"].split(",")
    _DUMP = dict(path=_p[0], n=int(_p[1]) if len(_p) > 1 else 64,
                 skip=int(_p[2]) if len(_p) > 2 else 200, seen=0, rows=[], saved=False)


def _dump(h, ib) -> None:
    d = _DUMP
    d["seen"] += 1
    if d["saved"] or d["seen"] <= d["skip"]:
        return
    li = ib.logits_indices
    d["rows"].append(dict(h=h.detach().cpu(), tok=ib.input_ids[li].detach().cpu(),
                          cu=ib.cu_num_logits.detach().cpu()))
    if len(d["rows"]) >= d["n"]:
        torch.save(d["rows"], d["path"])
        d["saved"] = True
        logger.warning("atdev vsplit_mtp: dumped %d verify steps -> %s", len(d["rows"]), d["path"])


# ---------------------------------------------------------------------------------------
# accept kernel (stock greedy rejection on precomputed per-row argmax)
# ---------------------------------------------------------------------------------------

@triton.jit
def _greedy_accept_kernel(
    am_ptr, cu_ptr, input_ids_ptr, positions_ptr, logits_indices_ptr,
    exp_idx_ptr, exp_lpos_ptr, prefill_len_ptr,
    sampled_ptr, sampled_stride, num_sampled_ptr,
):
    req = tl.program_id(0)
    start = tl.load(cu_ptr + req).to(tl.int64)
    end = tl.load(cu_ptr + req + 1).to(tl.int64)
    n_draft = end - start - 1
    accepted = tl.zeros((), tl.int64)
    verifying = tl.full((), 1, tl.int1)
    for d in range(n_draft):
        lrow = start + d
        tgt = tl.load(am_ptr + lrow)
        # draft token = input token of the next logits row (stock gather_draft_sampled,
        # incl. the prefill placeholder rule -> -1, which greedy rejects via `accepted`)
        nrow = lrow + 1
        tok_idx = tl.load(logits_indices_ptr + nrow)
        draft = tl.load(input_ids_ptr + tok_idx).to(tl.int64)
        pos = tl.load(positions_ptr + tok_idx)
        rsi = tl.load(exp_idx_ptr + nrow)
        lpos = tl.load(exp_lpos_ptr + nrow)
        plen = tl.load(prefill_len_ptr + rsi)
        draft = tl.where((lpos > 0) & (pos - lpos < plen), -1, draft)
        ok = (tgt == draft) & (draft >= 0)
        tl.store(sampled_ptr + req * sampled_stride + d, tl.where(ok, draft, tgt), mask=verifying)
        accepted += (ok & verifying).to(tl.int64)
        verifying = verifying & ok
    bonus = tl.load(am_ptr + end - 1)
    tl.store(sampled_ptr + req * sampled_stride + accepted, bonus, mask=verifying)
    tl.store(num_sampled_ptr + req, (accepted + 1).to(tl.int32))


def greedy_accept(am, cu_num_logits, num_reqs, width, input_ids, positions, logits_indices,
                  expanded_idx_mapping, expanded_local_pos, prefill_len):
    """(sampled [num_reqs, width] int64, num_sampled [num_reqs] int32) in the stock
    rejection_sample format (entries past num_sampled are unspecified, as in stock)."""
    dev = am.device
    sampled = torch.empty((num_reqs, width), dtype=torch.int64, device=dev)
    num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=dev)
    if num_reqs:
        _greedy_accept_kernel[(num_reqs,)](
            am, cu_num_logits, input_ids, positions, logits_indices,
            expanded_idx_mapping, expanded_local_pos, prefill_len,
            sampled, sampled.stride(0), num_sampled, num_warps=1)
    return sampled, num_sampled


# ---------------------------------------------------------------------------------------
# A8 tail on the M-split kernel (one weight pass for 96 rows)
# ---------------------------------------------------------------------------------------

_MS = None
_BUFS: dict = {}


def _ms_mod():
    global _MS
    if _MS is None:
        from atdev_serve import vsplit_tail_build
        _MS = vsplit_tail_build.load()
    return _MS


def _bufs(dev):
    b = _BUFS.get(dev)
    if b is None:
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        # private lock / fp32-reduce scratch (the tail may run on a side stream)
        b = _BUFS[dev] = dict(locks=torch.zeros(sms * 4, dtype=torch.int32, device=dev),
                              c_tmp=torch.empty(sms * 4 * 128 * 256, dtype=torch.float32,
                                                device=dev))
    return b


def _quant(t, h):
    """Stock A8 tail prologue (apply_gptq_marlin_linear): per_token_quant_int8, a_s * gs."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_quant_input

    a_q, a_s = marlin_quant_input(h, torch.int8)
    return a_q, (a_s * t.gs).reshape(-1).contiguous()


def _quant_plain(h):
    """per_token_quant_int8(h) -> (a_q int8 [M, K], a_s fp32 [M, 1]); shared by the W8A8 F part
    (ATDEV_VSPLIT_FA8) and the A8 tail (tail_mp_q scales a_s by the tail's gs)."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_quant_input

    return marlin_quant_input(h, torch.int8)


def tail_mp_q(t, a_q, a_s, m: int, cfg) -> torch.Tensor:
    """tail_mp on an already quantised activation (same values as tail_mp(t, h, cfg))."""
    from atdev_serve import marlin_plus

    out = torch.empty((m, t.size_n), dtype=torch.bfloat16, device=a_q.device)
    b = _bufs(a_q.device)
    return marlin_plus.mod().gemm(a_q, t.qweight, t.scales, (a_s * t.gs).reshape(-1).contiguous(),
                                  out, b["locks"], b["c_tmp"], t.size_n, t.size_k, list(cfg),
                                  True, False)


def tail_ms(t, h: torch.Tensor, cfg) -> torch.Tensor:
    """A8 tail with the GEMM on the M-split kernel (cfg = tk,tn,thr,stages,bps,tmb,MW,DEFER)."""
    a_q, a_s = _quant(t, h)
    out = torch.empty((h.shape[0], t.size_n), dtype=h.dtype, device=h.device)
    b = _bufs(h.device)
    return _ms_mod().gemm_ms(a_q, t.qweight, t.scales, a_s, out, b["locks"], b["c_tmp"],
                             t.size_n, t.size_k, list(cfg))


def tail_mp(t, h: torch.Tensor, cfg) -> torch.Tensor:
    """A8 tail with the GEMM on marlin_plus (cfg = tk,tn,thr,stages,bps,max_tmb; chunks whose
    tile is not instantiated fall back to the stock heuristic inside the op)."""
    from atdev_serve import marlin_plus

    a_q, a_s = _quant(t, h)
    out = torch.empty((h.shape[0], t.size_n), dtype=h.dtype, device=h.device)
    b = _bufs(h.device)
    return marlin_plus.mod().gemm(a_q, t.qweight, t.scales, a_s, out, b["locks"], b["c_tmp"],
                                  t.size_n, t.size_k, list(cfg), True, False)


@torch.no_grad()
def check_tail(t, fn, m: int, dev) -> bool:
    """fn(h) == stock tail (torch.equal) at M=m on random activations with outlier channels."""
    if t.padded_n != t.size_n or t.size_k % 128 or not getattr(t, "a8", False):
        return False
    g = torch.Generator(device=dev).manual_seed(4321 + m)
    for sc in (0.05, 1.0, 8.0):
        h = (torch.randn(m, t.size_k, device=dev, generator=g) * sc).to(torch.bfloat16)
        h[:, ::97] *= 20.0
        try:
            if not torch.equal(fn(h), t.approx_logits(h)):
                return False
        except RuntimeError as e:
            logger.warning("atdev vsplit_mtp: tail kernel failed at M=%d: %s", m, e)
            return False
    return True


def _setup_tail(sh) -> None:
    """ATDEV_VSPLIT_TAIL_MP (bit-identical: stock tile, other pipeline depth) or
    ATDEV_VSPLIT_TAIL_MS (+ ATDEV_VSPLIT_TAIL_MS_EXACT=0: rounding-level, not checked)."""
    if getattr(sh, "_mtp_tail_ready", False):
        return
    sh._mtp_tail_ready = True
    sh.tail_fn, sh.tail_ok, sh.tail_exact = None, {}, True
    sh.tail_min = int(os.environ.get("ATDEV_VSPLIT_TAIL_MIN", "17"))
    mp_cfg = os.environ.get("ATDEV_VSPLIT_TAIL_MP", "")
    ms_cfg = os.environ.get("ATDEV_VSPLIT_TAIL_MS", "")
    if not (mp_cfg or ms_cfg):
        return
    if sh.tail.backend != "marlin" or not getattr(sh.tail, "a8", False):
        logger.warning("atdev vsplit_mtp: tail cfg needs the A8 Marlin tail (ATDEV_VSPLIT_A8=1); "
                       "ignored")
        return
    t = sh.tail
    # C32e: ATDEV_VSPLIT_TAIL_MS_MIN=R with both MP and MS set (and MS_EXACT=0): one-pass M-split
    # tail for M >= R (M=96: 606 vs 714 us), marlin_plus MP below (M=64: 398 vs 597 us)
    ms_min = int(os.environ.get("ATDEV_VSPLIT_TAIL_MS_MIN", "0"))
    if ms_cfg and mp_cfg and ms_min > 0:
        c = [int(x) for x in ms_cfg.split(",")]
        c2 = [int(x) for x in mp_cfg.split(",")]
        sh.tail_fn = lambda h: tail_ms(t, h, c) if h.shape[0] >= ms_min else tail_mp(t, h, c2)
        sh.tail_exact = os.environ.get("ATDEV_VSPLIT_TAIL_MS_EXACT", "1") == "1"
    elif ms_cfg:
        c = [int(x) for x in ms_cfg.split(",")]
        sh.tail_fn = lambda h: tail_ms(t, h, c)
        sh.tail_exact = os.environ.get("ATDEV_VSPLIT_TAIL_MS_EXACT", "1") == "1"
    else:
        c = [int(x) for x in mp_cfg.split(",")]
        sh.tail_fn = lambda h: tail_mp(t, h, c)
    logger.warning("atdev vsplit_mtp: tail on %s cfg %s (%s)", "ms" if ms_cfg else "marlin_plus",
                   c, "bit-checked per M" if sh.tail_exact else "ROUNDING-LEVEL, unchecked")


def _tail(sh, h: torch.Tensor):
    """Tuned tail logits, or None (-> stock). Exact mode: each new M is checked bit-identical
    against the stock tail first (eager only; never during graph capture)."""
    fn = sh.tail_fn
    m = h.shape[0]
    if fn is None or m < sh.tail_min:
        return None
    if not sh.tail_exact:
        return fn(h)
    ok = sh.tail_ok.get(m)
    if ok is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        ok = sh.tail_ok[m] = check_tail(sh.tail, fn, m, h.device)
        if not ok:
            logger.warning("atdev vsplit_mtp: tuned tail differs from stock at M=%d -> stock", m)
    return fn(h) if ok else None


# ---------------------------------------------------------------------------------------
# vLLM hooks
# ---------------------------------------------------------------------------------------

def _eligible(runner, input_batch, grammar_output) -> bool:
    from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    rs = runner.rejection_sampler
    if (grammar_output is not None or runner.batch_sharder is not None or rs is None
            or type(rs) is not RejectionSampler or input_batch.num_reqs == 0
            or input_batch.num_draft_tokens == 0):
        return False
    spec = getattr(runner, "speculator", None)
    if spec is None or getattr(spec, "draft_logits", None) is not None:
        return False
    if (rs.watermark_key is not None or rs.synthetic_conditional_rates is not None
            or rs.use_block_verification or (rs.enable_adaptive_verification and not ADAPTIVE_OK)):
        return False
    s = rs.sampler
    if s.compute_nans:
        return False
    idx = input_batch.idx_mapping_np
    if np.any(s.needs_logits_processing[idx]):
        return False
    if not np.all(s.sampling_states.temperature.np[idx] == 0.0):
        return False
    return s.sampling_states.max_num_logprobs(idx) == NO_LOGPROBS


def _install() -> None:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    from atdev_serve import verify_split_head as vsh

    use_argmax = os.environ.get("ATDEV_VSPLIT_MTP_ARGMAX", "1") == "1"

    # 1) tail GEMM on the M-split kernel (both the stock logits path and the argmax path)
    if not getattr(vsh.SplitHead, "_atdev_mtp_tail", False):
        o_logits, o_tail = vsh.SplitHead.logits, vsh.SplitHead.tail_logits

        def logits(self, h, vocab_cols):
            _setup_tail(self)
            tl_ = _tail(self, h)
            if tl_ is None:
                return o_logits(self, h, vocab_cols)
            out = torch.empty(h.shape[0], vocab_cols, dtype=self.fw.dtype, device=h.device)
            out.index_copy_(1, self.f_ids, self.f_logits(h))  # bf16 / FBITS=8 F part
            out.index_copy_(1, self.t_ids, tl_)
            return out

        def tail_logits(self, h):
            _setup_tail(self)
            tl_ = _tail(self, h)
            return o_tail(self, h) if tl_ is None else tl_

        vsh.SplitHead.logits, vsh.SplitHead.tail_logits = logits, tail_logits
        vsh.SplitHead._atdev_mtp_tail = True

    if not use_argmax or getattr(GPUModelRunner.sample, "_atdev_vsplit_mtp", False):
        return

    # 2) sample(): mark all-greedy spec-decode verify calls
    o_sample = GPUModelRunner.sample

    @functools.wraps(o_sample)
    def sample(self, hidden_states, input_batch, grammar_output):
        if _eligible(self, input_batch, grammar_output):
            S.rows = int(input_batch.logits_indices.shape[0])
            S.batch = input_batch
        try:
            return o_sample(self, hidden_states, input_batch, grammar_output)
        finally:
            S.rows = None
            S.fast = None
            S.batch = None

    sample._atdev_vsplit_mtp = True
    GPUModelRunner.sample = sample

    # 3) _apply_head: argmax only (placeholder logits) on marked calls
    o_head = LogitsProcessor._apply_head

    def _apply_head(self, lm_head, hidden_states, embedding_bias):
        if (S.rows is None or hidden_states.dim() != 2 or hidden_states.shape[0] != S.rows
                or embedding_bias is not None or self.scale != 1.0 or self.soft_cap is not None
                or self.logits_as_input or torch.cuda.is_current_stream_capturing()):
            return o_head(self, lm_head, hidden_states, embedding_bias)
        sh = getattr(lm_head, vsh._ATTR, None)
        if sh is None:  # first call builds the split head (stock-format logits this once)
            return o_head(self, lm_head, hidden_states, embedding_bias)
        sh._fast_init()
        if not sh.fast_ok or hidden_states.dtype != sh.fw.dtype:
            return o_head(self, lm_head, hidden_states, embedding_bias)
        scr = getattr(sh, "a4scr", None)  # C32e ATDEV_VSPLIT_A4SCR: None for M > 96 or when off
        am = scr.argmax(hidden_states) if scr is not None else None
        if am is None:
            am = sh.argmax(hidden_states)
        if _DUMP is not None and S.batch is not None:
            _dump(hidden_states, S.batch)
        w = lm_head.weight
        logits = torch.empty(hidden_states.shape[0], w.shape[0], dtype=hidden_states.dtype,
                             device=hidden_states.device)  # never read
        S.fast = (logits.data_ptr(), logits.shape[0], am)
        return logits

    LogitsProcessor._apply_head = _apply_head

    # 4) RejectionSampler: accept on the precomputed argmax
    o_call = RejectionSampler.__call__

    @functools.wraps(o_call)
    def __call__(self, logits, input_batch, draft_logits=None):
        fa = S.fast
        if fa is None or logits.data_ptr() != fa[0] or logits.shape[0] != fa[1]:
            return o_call(self, logits, input_batch, draft_logits)
        S.fast = None
        S.hits += 1
        prefill_len = self.sampler.req_states.prefill_len.gpu
        sampled, num_sampled = greedy_accept(
            fa[2], input_batch.cu_num_logits, input_batch.num_reqs,
            self.num_speculative_steps + 1, input_batch.input_ids, input_batch.positions,
            input_batch.logits_indices, input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos, prefill_len)
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled, input_batch.seq_lens, input_batch.cu_num_logits,
            input_batch.idx_mapping, prefill_len)
        if S.hits == 1:
            logger.warning("atdev vsplit_mtp: argmax-only greedy verify active (%d rows)", fa[1])
        return SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None, num_nans=None,
                             num_sampled=num_sampled, num_rejected=num_rejected)

    RejectionSampler.__call__ = __call__


def register() -> None:
    import vllm

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written for vLLM {SUPPORTED_VLLM_VERSION}, "
                           f"found {vllm.__version__}")
    from atdev_serve import register as reg

    if "verify_split_head" not in reg.enabled():
        raise RuntimeError("atdev vsplit_mtp: needs ATDEV_OPT verify_split_head")
    if os.environ.get("ATDEV_VSPLIT_MTP_ARGMAX", "1") == "1" and \
            os.environ.get("ATDEV_VSPLIT_FAST", "0") != "1":
        raise RuntimeError("atdev vsplit_mtp: the argmax path needs ATDEV_VSPLIT_FAST=1 "
                           "(ascending F ids for the split_argmax tie rule)")
    _install()
    logger.info("atdev vsplit_mtp: registered (argmax=%s, tail_ms=%s)",
                os.environ.get("ATDEV_VSPLIT_MTP_ARGMAX", "1"),
                os.environ.get("ATDEV_VSPLIT_TAIL_MS", ""))
