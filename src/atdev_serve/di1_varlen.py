"""Confidence-scheduled variable-length MTP verification at c=32 (ATDEV_OPT+=di1_varlen).

Draft MTP up to k = kmax (``num_speculative_tokens``), keep a fixed verify row budget (96 rows at c=32,
the M bucket of fixed k=2: 3 rows per request) and give each request 1..kmax+1 verify rows by ranking
the cumulative draft confidence prod_{j<=d} q_j over all (request, depth) and admitting the top
(budget - non-draft rows) drafts, at least 2 per verified request. Greedy verification itself is
unchanged; only which drafts get verified changes.

vLLM 0.31's V2 model runner already implements the GPU side ("adaptive verification":
AdaptiveVerificationManager, device-side re-allocation of drafts into query_start_loc / cu_num_logits,
varlen FULL decode graphs, an online logistic acceptance estimator per draft position), but only
enables it for drafters that materialise draft logits and for non-SSM attention backends. This module:

* forces ``speculative_config.enable_adaptive_verification`` on inside the worker (the config validator
  rejects it together with use_local_argmax_reduction, which draft_vocab needs);
* confidence: draft_vocab's int4 screen already computes approximate logits over the 64k draft vocab; the
  local-argmax branch of ``DraftModelSpeculator.sample_draft`` feeds them to the stock online estimator
  (p = sigmoid(w * logit(max q) + b_depth), refit online);
* GDN: declares the GDN backend safe for device/CPU query-length mismatch and its FULL graphs safe for
  varlen decode up to kmax+1 tokens per request. Pure spec-decode GDN metadata (spec_query_start_loc,
  num_accepted_tokens) is device-derived, and gdn_single_state takes per-request T from it
  (T <= kmax+1 <= 4, log capacity kmax); conv1d update takes query_start_loc;
* the varlen FULL graphs are captured with at most 32 dummy requests (stock pads every varlen graph to
  min(num_tokens, max_num_seqs) requests -> 96 GDN request slots at M=96, 64 of them empty CTAs).

Run with ``--speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}'``.
"""
from __future__ import annotations

import dataclasses
import functools
import logging

import numpy as np
import triton
import triton.language as tl

OPT_NAME = "di1_varlen"
SUPPORTED_VLLM_VERSION = "0.31.0"
log = logging.getLogger("atdev_serve")


# admitted drafts per step = min(scheduled, max(_PER_REQ * n_verify, _ROWS - non-draft rows)): small
# batches (tail of a run) verify every draft while the step stays within the 96-row GEMM bucket
_PER_REQ, _ROWS = 2, 96
# cap on the dummy-request padding of the varlen FULL graphs
_CG_MAXREQS = 32


class _VL:
    max_q: int | None = None  # max_query_len of the verify batch whose drafts are being proposed


@triton.jit
def _alloc_kernel(conf_ptr, conf_stride, idx_ptr, caps_ptr, n, budget,
                  K: tl.constexpr, NRP: tl.constexpr, KP: tl.constexpr):
    """caps[r] in: scheduled drafts of request r; out: admitted drafts = number of request r's slots among
    the global top-`budget` survival probabilities prod_{j<=d} conf[idx[r], j] (ties: lower (r, d) first)."""
    r = tl.arange(0, NRP)
    d = tl.arange(0, KP)
    rm = r < n
    idx = tl.load(idx_ptr + r, mask=rm, other=0).to(tl.int64)
    cap = tl.load(caps_ptr + r, mask=rm, other=0)
    ok = rm[:, None] & (d[None, :] < K)
    c = tl.load(conf_ptr + idx[:, None] * conf_stride + d[None, :], mask=ok, other=1.0).to(tl.float32)
    s = tl.cumprod(c, axis=1)
    valid = ok & (d[None, :] < cap[:, None])
    s = tl.where(valid, s, float("-inf"))
    flat = tl.reshape(s, (NRP * KP,))
    pos = tl.arange(0, NRP * KP)
    # rank = number of slots ordered before this one
    before = (flat[None, :] > flat[:, None]) | ((flat[None, :] == flat[:, None]) & (pos[None, :] < pos[:, None]))
    rank = tl.sum(before.to(tl.int32), axis=1)
    adm = ((rank < budget) & (flat > float("-inf"))).to(tl.int32)
    tot = tl.sum(tl.reshape(adm, (NRP, KP)), axis=1)
    tl.store(caps_ptr + r, tot.to(caps_ptr.dtype.element_ty), mask=rm)


def assign_draft_token_budget(confidence_probs, idx_mapping, capacities, draft_budget, num_steps):
    """Drop-in for vLLM's _assign_draft_token_budget (same admitted set up to tie order), one launch."""
    n = idx_mapping.shape[0]
    if n == 0:
        return
    nrp = max(2, triton.next_power_of_2(n))
    kp = max(2, triton.next_power_of_2(num_steps))
    _alloc_kernel[(1,)](confidence_probs, confidence_probs.stride(0), idx_mapping, capacities, n,
                        int(draft_budget), K=num_steps, NRP=nrp, KP=kp, num_warps=4)


def _install():
    from vllm.v1.attention.backends import gdn_attn
    from vllm.v1.worker.gpu import cudagraph_utils as cgu
    from vllm.v1.worker.gpu import model_runner as mr
    from vllm.v1.worker.gpu.spec_decode import adaptive_verification as av
    from vllm.v1.worker.gpu.spec_decode import speculator as spm
    from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import OnlineAcceptanceEstimator

    cg_cap = _CG_MAXREQS

    # 1) adaptive verification on inside the worker -------------------------------------------
    o_init = mr.GPUModelRunner.__init__

    @functools.wraps(o_init)
    def runner_init(self, vllm_config, device):
        sc = vllm_config.speculative_config
        if sc is not None and sc.method == "mtp" and not sc.enable_adaptive_verification:
            object.__setattr__(sc, "enable_adaptive_verification", True)
            log.warning("atdev di1_varlen: adaptive verification forced on (MTP k=%d, budget rows %d / "
                        "%d per request, cg_maxreqs=%d)", sc.num_speculative_tokens, _ROWS, _PER_REQ, cg_cap)
        o_init(self, vllm_config, device)

    mr.GPUModelRunner.__init__ = runner_init

    # 2) estimator despite use_local_argmax_reduction -----------------------------------------
    o_load = spm.DraftModelSpeculator.load_model

    @functools.wraps(o_load)
    def load_model(self, target_model):
        want = self.use_acceptance_estimator and self.use_local_argmax_reduction
        if want:
            self.use_acceptance_estimator = False
        o_load(self, target_model)
        if want:
            self.use_acceptance_estimator = True
            self.acceptance_estimator = OnlineAcceptanceEstimator(
                self.max_num_reqs, self.num_speculative_steps, self.device)
            log.warning("atdev di1_varlen: draft confidence = online estimator over the draft-vocab "
                        "approx logits")

    spm.DraftModelSpeculator.load_model = load_model

    o_sample = spm.DraftModelSpeculator.sample_draft

    @functools.wraps(o_sample)
    def sample_draft(self, hidden_states, sample_src_positions, idx_mapping, temperature, seeds,
                     draft_step, draft_logits):
        if not (draft_logits is None and self.use_local_argmax_reduction
                and self.acceptance_estimator is not None):
            return o_sample(self, hidden_states, sample_src_positions, idx_mapping, temperature, seeds,
                            draft_step, draft_logits)
        from atdev_serve import draft_vocab
        draft_vocab.LAST.pop("approx", None)
        tokens = self.model.get_top_tokens(hidden_states)
        approx = draft_vocab.LAST.pop("approx", None)
        if approx is None or approx.shape[0] != hidden_states.shape[0]:
            raise RuntimeError("atdev di1_varlen: draft_vocab screen (approx logits) required")
        self.acceptance_estimator.predict(approx, idx_mapping, draft_step,
                                          self.draft_token_confidence_probs, temperature)
        return tokens

    spm.DraftModelSpeculator.sample_draft = sample_draft

    # 3) GDN: varlen-safe ---------------------------------------------------------------------
    gdn_attn.GDNAttentionBackend.supports_device_cpu_query_lens_mismatch = classmethod(lambda cls: True)

    def _gdn_varlen_max_q(cls, vllm_config, kv_cache_spec):
        sc = vllm_config.speculative_config
        return None if sc is None else sc.num_speculative_tokens + 1

    gdn_attn.GDNAttentionMetadataBuilder.get_varlen_cudagraph_max_query_len = classmethod(_gdn_varlen_max_q)

    # 4) budget -------------------------------------------------------------------------------
    # the cudagraph-memory profiling capture (profile_only) also runs the cost profiling but yields no
    # graph-replay samples -> stock raises; keep the tables unset then (the real capture sets them)
    o_set = av.AdaptiveVerificationManager.set_cost_curves

    def set_cost_curves(self, draft_curve, verify_curve):
        if not draft_curve or not verify_curve:
            log.warning("atdev di1_varlen: no cost samples in this capture pass (draft %d, verify %d); "
                        "cost tables left unset", len(draft_curve), len(verify_curve))
            return
        o_set(self, draft_curve, verify_curve)
        log.warning("atdev di1_varlen: cost curves draft %s verify %s",
                    [(n, round(t, 3)) for n, t in draft_curve], [(n, round(t, 3)) for n, t in verify_curve])

    av.AdaptiveVerificationManager.set_cost_curves = set_cost_curves

    o_get = av.AdaptiveVerificationManager.get_num_tokens

    def get_num_tokens(self, num_tokens_per_req, draft_tokens):
        req_ids = list(num_tokens_per_req)
        n = len(req_ids)
        sched = np.fromiter(num_tokens_per_req.values(), dtype=np.int32, count=n)
        drafts = np.fromiter((len(draft_tokens.get(r, ())) for r in req_ids), dtype=np.int32, count=n)
        non_draft = sched - drafts
        n_verify = int((drafts > 0).sum())
        per_req, rows_cap = _PER_REQ, _ROWS
        draft_budget = min(int(drafts.sum()), max(per_req * n_verify, rows_cap - int(non_draft.sum())),
                           max(0, self._max_total_logits - n * self.num_bonus_tokens))
        self._batch_budget = (
            {r: int(x) for r, x in zip(req_ids, drafts, strict=True)},
            {r: int(x) for r, x in zip(req_ids, non_draft, strict=True)},
            draft_budget,
        )
        return int(non_draft.sum()) + draft_budget

    get_num_tokens._atdev_orig = o_get
    av.AdaptiveVerificationManager.get_num_tokens = get_num_tokens

    # This budget never reads the stale CPU confidences, so skip their D2H copy and, above all,
    # the host wait on the previous step's copy event (it serialises host launch and GPU time,
    # worst on eager / piecewise prefill-mixed steps); only the device ranking buffer is updated.
    def record_confidences(self, confidence_probs, input_batch):
        self._pending_resets.clear()
        self._confidence_probs[input_batch.idx_mapping] = confidence_probs[: input_batch.num_reqs]

    av.AdaptiveVerificationManager.record_confidences = record_confidences

    # one-CTA Triton ranking instead of the torch.compile(dynamic) cumprod/topk/index_fill/sum chain
    av._assign_draft_token_budget_compiled = assign_draft_token_budget

    # 5) vsplit_mtp argmax-only greedy verify also on adaptive (varlen) batches ----------------
    from atdev_serve import register as reg
    if "vsplit_mtp" in reg.enabled():
        from atdev_serve import vsplit_mtp
        vsplit_mtp.ADAPTIVE_OK = True

    # 6) draft prefill (MTP pass 1) as a varlen FULL graph -------------------------------------
    # stock builds the drafter's prefill graph manager uniform-only (decode_query_len = k+1) and
    # dispatches it without max_query_len, so every varlen verify batch ran MTP pass 1 eagerly
    from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as ars
    from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import SpeculatorCudaGraphManager

    o_icg = ars.AutoRegressiveSpeculator.init_cudagraph_manager

    def init_cudagraph_manager(self, cudagraph_mode):
        o_icg(self, cudagraph_mode)
        if not self.enable_adaptive_verification:
            return
        mgr = SpeculatorCudaGraphManager(self.vllm_config, self.device, cudagraph_mode,
                                         self.num_speculative_steps + 1, varlen_decode=True)
        o_dispatch = mgr.dispatch

        def dispatch(num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len=None,
                     num_ubatches=1):
            if max_query_len is None:
                max_query_len = _VL.max_q
            return o_dispatch(num_reqs, num_tokens, uniform_token_count, num_active_loras,
                              max_query_len=max_query_len, num_ubatches=num_ubatches)

        mgr.dispatch = dispatch
        self.prefill_cudagraph_manager = mgr
        log.warning("atdev di1_varlen: drafter prefill graphs varlen (decode_query_len %d)",
                    self.num_speculative_steps + 1)

    ars.AutoRegressiveSpeculator.init_cudagraph_manager = init_cudagraph_manager

    o_propose = ars.AutoRegressiveSpeculator.propose

    @functools.wraps(o_propose)
    def propose(self, input_batch, *args, **kwargs):
        # varlen graphs only for pure decode batches (the runner passes None for any prefill)
        _VL.max_q = input_batch.max_query_len if input_batch.decode_graph_eligible else None
        try:
            return o_propose(self, input_batch, *args, **kwargs)
        finally:
            _VL.max_q = None

    ars.AutoRegressiveSpeculator.propose = propose

    # 7) varlen graph request padding ------------------------------------------------------------
    # Stock captures each varlen FULL graph with num_reqs = min(num_tokens, max_num_seqs) dummy requests,
    # i.e. 1 token per request for every size <= max_num_seqs: the GDN builder then sees no drafts and the
    # graph bakes in the NON-spec GDN decode kernel, which garbles every real (multi-token) verify batch it
    # replays. Capture with >= 2 tokens per dummy request (num_reqs <= num_tokens // 2) so the spec path is
    # recorded; the request padding is additionally capped at _CG_MAXREQS.
    o_cand = cgu.CudaGraphManager._init_candidates

    def _init_candidates(self):
        o_cand(self)
        if not self.varlen_decode:
            return

        def f(d):
            if d.max_query_len is not None and d.uniform_token_count is None and d.num_reqs is not None:
                # the capture dummy spreads num_tokens over num_reqs, <= max_query_len each
                lo = -(-d.num_tokens // d.max_query_len)
                hi = max(1, d.num_tokens // 2)
                if cg_cap > 0:
                    hi = min(hi, cg_cap)
                return dataclasses.replace(d, num_reqs=min(d.num_reqs, max(lo, hi)))
            return d

        def dedup(lst):
            out = []
            for d in map(f, lst):
                if d.max_query_len is not None and d.uniform_token_count is None and d.num_tokens < 2:
                    continue  # a 1-token dummy cannot record the spec GDN path
                if d not in out:
                    out.append(d)
            return out

        self._capture_descs = {m: dedup(v) for m, v in self._capture_descs.items()}
        self._candidates = {k: dedup(v) for k, v in self._candidates.items()}

    cgu.CudaGraphManager._init_candidates = _init_candidates


def register() -> None:
    import vllm

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written for vLLM {SUPPORTED_VLLM_VERSION}, "
                           f"found {vllm.__version__}")
    from atdev_serve import register as reg

    if "draft_vocab" not in reg.enabled():
        raise RuntimeError("atdev di1_varlen: needs ATDEV_OPT draft_vocab for the confidences")
    _install()
    log.info("atdev di1_varlen: registered")
