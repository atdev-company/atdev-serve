"""Tree verification for DFlash2 (ATDEV_OPT=dflash_tree), vLLM 0.31.0 MRV2.

The DFlash2 drafter (block 16 -> depth D = 15, top-16 candidates per position,
CRF edge scores) proposes one greedy path in stock vLLM. Here it proposes a
token *tree* of N = 16 nodes (root + 15 drafts) built best-first over the CRF
lattice (node value = cumulative log-softmax of the conditional edge scores,
temperature 1), and the target verifies all N nodes in one forward with an
ancestor mask:

* full attention (16 layers): ``tree_attn`` (CUDA, bit-identical to TRITON_ATTN's
  ``unified_attention`` with ``qq_bias`` [N, N], 0 ancestor-or-self, -inf otherwise)
  + RoPE position = pos0 + depth; K/V of node i lands in the slot of pos0 + i.
* GDN (48 layers): the CUDA tree kernel (``gdn_tree_cuda``: conv + ancestor-masked
  WY form + gated norm in one launch, on top of gdn_single_state's slots).
* acceptance: greedy walk of the target argmax down the tree: bit-exact greedy
  semantics -- every committed token is the target's argmax after the committed prefix.
* compaction after acceptance: K/V of path node d -> slot of pos0 + d; GDN log
  entries and conv-state columns into the sequential order; drafter context rows
  (hidden states) gathered along the path. The post-verify chain (tree accept,
  num_sampled / num_rejected, position shift back, compaction, drafter gather index)
  is one fused Triton launch (tree_post) plus the compaction kernels, replayed as one
  CUDA graph (ring of 3 graphs, so the sampled-token outputs AsyncOutput copies to
  the host are not overwritten by the next steps; conv / KV compaction run on two
  forked branches concurrent with the GDN log compaction). Same kernels on the same
  inputs -> bit-identical to running them eagerly.

``num_speculative_tokens`` in ``--speculative-config`` must be N-1 (it sizes
the scheduler, KV lookahead, GDN/conv slots and the target's uniform CUDA
graph); the drafter itself is built with D = block_size - 1 steps.

Tree mode is used only when the step's batch is exactly one pure spec-decode
request (c=1), greedy, no logprobs / logits processors / structured output,
and its drafts were built as a tree. Any other batch runs in *chain* mode:
same N-token verify with causal semantics, stock rejection sampler; the
drafter then emits vLLM's greedy chain (D tokens) padded with dead tokens (so
c > 1 is correct but pays N-token verifies: this is a c=1 optimization).
Mixed prefill + decode batches use the stock conv and gdn_single_state's
dispatcher, which are valid because compaction restores their layouts after
every tree step.

Requires: gdn_single_state, --attention-backend TRITON_ATTN,
--no-enable-prefix-caching, DFlash2 drafter, draft_sample_method greedy.
"""

from __future__ import annotations

import copy
import functools
import logging

import numpy as np
import torch

from atdev_serve import gdn_tree as gt

OPT_NAME = "dflash_tree"
SUPPORTED_VLLM_VERSION = "0.31.0"
log = logging.getLogger("atdev_serve")
N_NODES = 16  # tree nodes per verify (root + 15 drafts)
ACCEPT_GRAPH_RING = 3


class _State:
    """Process-global (worker) state shared by the hooks."""

    def __init__(self) -> None:
        self.N: int | None = None
        self.D: int | None = None
        self.spec = None
        self.meta: dict | None = None  # per request-state tree metadata
        self.act: dict | None = None  # batch-row metadata read by the kernels
        self.qq: torch.Tensor | None = None
        self.tree_reqs: set[str] = set()
        self.mode: str | None = None  # "tree" for the step in flight
        self.batch = None
        self.row0_tree = False
        self.capturing = False
        self.attn_on = False
        self.accepting = False
        # (logits data_ptr, rows, int64 argmax) from verify_split_head's fused argmax
        self.fast_am: tuple | None = None
        self.block_tables = None
        self.slot_mappings = None
        self.pending_gather: torch.Tensor | None = None
        self.tables: dict | None = None
        self.checked_backend = False
        self.n_tree = 0
        self.n_chain = 0
        self.ag: _AcceptGraph | None = None
        self.ag_done = False  # this step's finish part already ran inside the accept graph
        self.last_pre = None  # (state_idx, gather) from the fused post kernel of this step
        self.ctx_gather = None  # gather index handed to the wrapped combine_hidden_states
        self.idx_stashed = None  # input batch whose idx_mapping[0] tree_activate copied to ag.idx
        self.ag_streams = None  # forked capture streams of the accept graph's conv / KV compaction


class _AcceptGraph:
    """CUDA-graph replay of the c=1 tree accept + finish chain."""

    def __init__(self, n: int, device, ring: int, warm: int = 2) -> None:
        self.am = torch.zeros(n, dtype=torch.int64, device=device)  # verify-head argmax (FAST)
        self.idx = torch.zeros(1, dtype=torch.int32, device=device)  # idx_mapping (fresh per step)
        self.cu = torch.tensor([0, n], dtype=torch.int32, device=device)  # cu_num_logits (c=1 tree)
        self.ring = max(1, ring)
        self.slots: list[dict | None] = [None] * self.ring
        self.next = 0
        self.warm = warm
        self.n_eager = 0
        self.n_replay = 0
        self.n_capture = 0
        self.failed = False


def _nullctx():
    import contextlib

    return contextlib.nullcontext()


def am_out(m: int) -> torch.Tensor | None:
    """Persistent int64 buffer for the verify head's tree argmax (input of the accept graph),
    None -> the head allocates a fresh one."""
    ag = S.ag
    if ag is None or ag.failed or m > ag.am.numel():
        return None
    return ag.am


S = _State()


# ---------------------------------------------------------------------------
# Speculator: DFlash2 drafter (D steps) + tree builder (N-1 drafts)
# ---------------------------------------------------------------------------


def _make_speculator_class():
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

    class TreeDFlash2Speculator(DFlash2Speculator):
        _speculator_name = "DFlash2Tree"

        def __init__(self, vllm_config, device):
            sc = vllm_config.speculative_config
            hf = sc.draft_model_config.hf_config
            depth = int(hf.dflash_config["block_size"]) - 1
            if hf.dflash_config.get("segment_causal"):
                raise ValueError(f"atdev {OPT_NAME}: segment-causal drafters are not supported")
            n = N_NODES
            if sc.num_speculative_tokens != n - 1:
                raise ValueError(
                    f"atdev {OPT_NAME}: --speculative-config num_speculative_tokens must "
                    f"be {n - 1} (verify width), got "
                    f"{sc.num_speculative_tokens}")
            if n - 1 < depth:
                raise ValueError(f"atdev {OPT_NAME}: N-1={n - 1} < drafter depth {depth}")
            if sc.draft_sample_method != "greedy":
                raise ValueError(f"atdev {OPT_NAME}: draft_sample_method must be greedy")
            if sc.enable_adaptive_verification:
                raise ValueError(f"atdev {OPT_NAME}: adaptive verification unsupported")
            # The drafter runs its trained block (D steps); only the verify is N wide.
            cfg = copy.copy(vllm_config)
            sc_d = copy.copy(sc)
            object.__setattr__(sc_d, "num_speculative_tokens", depth)
            cfg.speculative_config = sc_d
            super().__init__(cfg, device)
            assert self.num_speculative_steps == depth
            mr = self.max_num_reqs
            self.tree_nodes = n
            self.tree_depth = depth
            i32, i64 = torch.int32, torch.int64
            self.tree_tokens = torch.zeros(mr, n - 1, dtype=i64, device=device)
            meta = {
                "parent": torch.zeros(mr, n, dtype=i32, device=device),
                "depth": torch.zeros(mr, n, dtype=i32, device=device),
                "anc": torch.zeros(mr, n, dtype=i64, device=device),
                "conv": torch.zeros(mr, n, 3, dtype=i32, device=device),
            }
            ar = torch.arange(n, device=device)
            act = {
                "parent": (ar - 1).to(i32).repeat(mr, 1).contiguous(),
                "depth": ar.to(i32).repeat(mr, 1).contiguous(),
                "anc": ((2 << ar) - 1).to(i64).repeat(mr, 1).contiguous(),
                "conv": torch.stack([ar - 1, ar - 2, ar - 3], -1).to(i32)
                .repeat(mr, 1, 1).contiguous(),
            }
            # extra rows: unified_attention's last q block reads one row past N
            qq = torch.zeros(n + 16, n, dtype=torch.float32, device=device)
            S.N, S.D, S.spec, S.meta, S.act, S.qq = n, depth, self, meta, act, qq
            S.ar = ar.to(i64)
            S.ag = _AcceptGraph(n, device, ACCEPT_GRAPH_RING)
            log.warning("atdev %s: tree verify N=%d (drafter depth %d, top-%d), accept chain as CUDA graph "
                        "(ring %d)", OPT_NAME, n, depth, self.top_k, S.ag.ring)

        def _num_graph_context_tokens(self, num_reqs: int) -> int:
            # the target verify is N rows per request (not D+1)
            return min(num_reqs * self.tree_nodes, self.max_num_tokens)

        def _generate_draft(self, num_reqs, num_tokens_padded, attn_metadata,
                            slot_mappings, num_tokens_across_dp,
                            cudagraph_runtime_mode=CUDAGraphMode.NONE):
            last_hidden_states = self._run_model(
                num_tokens_padded, attn_metadata, slot_mappings, num_tokens_across_dp,
                cudagraph_runtime_mode)
            d = self.num_speculative_steps
            num_sample = num_reqs * d
            hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(
                num_reqs, d, -1)
            candidate_ids, unary_logits = self.model.compute_candidates(
                hidden_states.flatten(0, 1))
            candidate_ids = candidate_ids.view(num_reqs, d, self.top_k)
            unary_logits = unary_logits.view_as(candidate_ids)
            scores = self._score_candidates(candidate_ids, unary_logits, hidden_states)
            # num_reqs is fixed per captured graph: 1 -> tree, else vLLM's chain
            gt.build_tree_gpu(
                scores.float().contiguous(), candidate_ids.contiguous(),
                self.sample_idx_mapping, d, self.tree_tokens,
                S.meta["parent"], S.meta["depth"], S.meta["anc"], S.meta["conv"],
                self.tree_nodes, chain=num_reqs != 1)

        def prepare_context_anchor(self, input_batch, num_rejected) -> None:
            super().prepare_context_anchor(input_batch, num_rejected)
            gidx = S.pending_gather
            if gidx is not None:
                # accepted path nodes -> context rows 0..a-1 (row order the
                # DFlash input kernel assumes)
                S.pending_gather = None
                n = gidx.numel()
                self.hidden_states[:n] = self.hidden_states[gidx]

        def _atdev_wrap_combine(self) -> None:
            """model.combine_hidden_states -> fc output rows gathered along the accepted path
            (tree_post.gather_rows, one launch), written into self.hidden_states[:n]; the stock
            copy that follows is then a same-view no-op (no launch) and prepare_context_anchor
            finds nothing left to gather."""
            orig = self.model.combine_hidden_states
            spec = self

            def combine_hidden_states(hidden_states):
                out = orig(hidden_states)
                gidx = S.ctx_gather
                if (gidx is None or out.dim() != 2 or out.shape[0] != gidx.numel()
                        or out.stride(1) != 1 or out.dtype != spec.hidden_states.dtype
                        or torch.cuda.is_current_stream_capturing()):
                    return out
                S.ctx_gather = None
                S.pending_gather = None
                from atdev_serve.tree_post import gather_rows

                return gather_rows(out, gidx, spec.hidden_states[: out.shape[0]])

            self.model.combine_hidden_states = combine_hidden_states
            self._atdev_combine_wrapped = True

        def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states,
                    aux_hidden_states, num_sampled, num_rejected, last_sampled,
                    next_prefill_tokens, temperature, seeds, dp_sync=None,
                    dummy_run=False, skip_attn_for_dummy_run=False, mm_inputs=None,
                    is_profile=False):
            gidx = S.pending_gather
            if (gidx is not None and not dummy_run and aux_hidden_states
                    and input_batch.num_tokens == gidx.numel()):
                # gather the accepted path rows straight out of the fc output into
                # self.hidden_states (stock: copy rows, gather, copy back)
                if not getattr(self, "_atdev_combine_wrapped", False):
                    self._atdev_wrap_combine()
                S.ctx_gather = gidx
            try:
                super().propose(
                    input_batch, attn_metadata, slot_mappings, last_hidden_states,
                    aux_hidden_states, num_sampled, num_rejected, last_sampled,
                    next_prefill_tokens, temperature, seeds, dp_sync=dp_sync,
                    dummy_run=dummy_run, skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                    mm_inputs=mm_inputs, is_profile=is_profile)
            finally:
                S.ctx_gather = None
            nr = input_batch.num_reqs
            if not dummy_run:
                S.tree_reqs = {input_batch.req_ids[0]} if nr == 1 else set()
            S.pending_gather = None
            return self.tree_tokens[:nr]

    return TreeDFlash2Speculator


# ---------------------------------------------------------------------------
# GDN layer: tree conv + tree scan on pure spec-decode batches
# ---------------------------------------------------------------------------


def _make_layer_class(base):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn_mod

    from atdev_serve.gdn_single_state import _block_v

    dim_first = gdn_mod.is_conv_state_dim_first

    from atdev_serve import gdn_tree_cuda as gtc

    gtc.cuda_ext()  # build / load the CUDA tree kernel before any graph capture

    class TreeGDNAttention(base):
        """Single-state GDN layer whose pure spec-decode verify runs the tree kernels (with
        a causal ancestor mask in chain mode, so they cover every pure batch)."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            # exchange buffers must exist before CUDA-graph capture (a captured zero-init
            # would be replayed every step and break the kernel's generation tags)
            gtc.prealloc(torch.device("cuda", torch.cuda.current_device()), 1,
                         self.num_k_heads // self.tp_size, self.num_v_heads // self.tp_size)

        def _atdev_ba_defer_ok(self, qkvz) -> bool:
            """c1_fuse: will this call reach the CUDA tree path, where the kernel rounds b / a from the fp32
            in_proj_ba accumulator itself? (False -> ba is materialised up front.) Mirrors the dispatch of
            _forward_core_fused_norm_packed / _forward_core_fused_norm / _forward_core_decode_spec_fused_norm;
            anything that fails later materialises ba before use (_atdev_ba_materialize)."""
            if S.act is None:
                return False
            from vllm.forward_context import get_forward_context
            from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
            md = get_forward_context().attn_metadata
            md = md.get(self.prefix) if isinstance(md, dict) else None
            if not isinstance(md, GDNAttentionMetadata) or md.num_prefills != 0 \
                    or not self._can_use_fused_gdn_mtp_decode(md):
                return False
            si = md.spec_state_indices_tensor
            return si is not None and si.size(1) == S.N

        @staticmethod
        def _atdev_ba_materialize():
            c = gtc.BA_CTX
            if c and not c.get("done"):
                # bf16(tot * rstd) == the serial in_proj_ba GEMV's output
                from atdev_serve import c1_norm
                c1_norm.ba_scale(c["tot"], c["r"], c["ba"].shape[0], out=c["ba"])
                c["done"] = True

        def _atdev_tree_cuda(self, mixed_qkv, b, a, output_gate, core_attn_out, attn_metadata,
                             si, r, n_tok, cu, acc, conv_state, w):
            """Fused conv + tree verify + gated norm in one CUDA launch; False -> not
            applicable (caller runs the Triton kernels)."""
            hk = self.num_k_heads // self.tp_size
            x = mixed_qkv[:n_tok]
            og = output_gate[:n_tok]
            st = self.kv_cache[1]
            if not (gtc.supported(state=st, state_indices=si, q_dtype=x.dtype,
                                  block_v=_block_v(), num_k_heads=hk, num_requests=r)
                    and self.head_k_dim == 128 and self.head_v_dim == 128
                    and x.stride(1) == 1 and x.data_ptr() % 16 == 0 and (x.stride(0) * 2) % 16 == 0
                    and og.dim() == 3 and og.stride(2) == 1 and og.stride(1) == self.head_v_dim
                    and og.data_ptr() % 16 == 0 and (og.stride(0) * 2) % 16 == 0
                    and self.norm.norm_before_gate and self.norm.group_size is None
                    and self.norm.bias is None and conv_state.dtype == torch.bfloat16
                    and w.dtype == torch.bfloat16 and w.is_contiguous()
                    and a.stride(-1) == 1 and b.stride(-1) == 1 and a.dtype == torch.bfloat16
                    and (self.conv1d.bias is None or self.conv1d.bias.dtype == torch.bfloat16)):
                return False
            bap = None
            c = gtc.BA_CTX
            if c and not c.get("done"):
                tot, rr = c["tot"], c["r"]
                hv = self.num_v_heads // self.tp_size
                if (tot.dtype == torch.float32 and rr.dtype == torch.float32 and tot.shape[0] >= n_tok
                        and tot.shape[1] == 2 * hv and tot.stride(-1) == 1 and rr.shape[0] >= n_tok):
                    bap = dict(b=tot[:n_tok, :hv], a=tot[:n_tok, hv:], r=rr[:n_tok].contiguous())
                    c["done"] = True  # b / a never materialised: only this kernel reads them
                else:
                    self._atdev_ba_materialize()
            if not getattr(type(self), "_atdev_tree_cuda_logged", False):
                type(self)._atdev_tree_cuda_logged = True
                log.warning("atdev %s: CUDA tree verify (conv + tree + gated norm fused) ENGAGED "
                            "(fp32 ba + rstd %s)", OPT_NAME, bap is not None)
            gtc.gdn_tree_verify_cuda(bap=bap,
                a=a[:n_tok], b=b[:n_tok], A_log=self.A_log, dt_bias=self.dt_bias, state=st,
                state_indices=si, cu_seqlens=cu, num_accepted=acc, anc_bits=S.act["anc"],
                out=core_attn_out[:n_tok], num_k_heads=hk, scale=self.head_k_dim ** -0.5,
                num_requests=r, output_gate=og, norm_weight=self.norm.weight,
                norm_eps=self.norm.eps, sigmoid_gate=self.norm.activation == "sigmoid",
                conv=dict(x=x, weight=w, bias=self.conv1d.bias, conv_state=conv_state,
                          conv_src=S.act["conv"],
                          activation=self.activation in ("silu", "swish", True)))
            return True

        def _forward_core_decode_spec_fused_norm(self, mixed_qkv, b, a, output_gate,
                                                 core_attn_out, attn_metadata):
            si = attn_metadata.spec_state_indices_tensor
            if S.act is None or si is None or si.size(1) != S.N:
                self._atdev_ba_materialize()
                return super()._forward_core_decode_spec_fused_norm(
                    mixed_qkv, b, a, output_gate, core_attn_out, attn_metadata)
            if not getattr(type(self), "_atdev_tree_logged", False):
                type(self)._atdev_tree_logged = True
                log.warning("atdev %s: GDN tree verify path ENGAGED (state %s, BV=%d)",
                            OPT_NAME, self.kv_cache[1].dtype, _block_v())
            r = attn_metadata.num_spec_decodes
            n_tok = attn_metadata.num_actual_tokens
            cu = attn_metadata.spec_query_start_loc[: r + 1]
            acc = attn_metadata.num_accepted_tokens[:r]
            si = si[:r]
            conv_state = self.kv_cache[0] if dim_first() else self.kv_cache[0].transpose(-1, -2)
            w = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            if self._atdev_tree_cuda(mixed_qkv, b, a, output_gate, core_attn_out,
                                                  attn_metadata, si, r, n_tok, cu, acc,
                                                  conv_state, w):
                return
            self._atdev_ba_materialize()
            x = mixed_qkv[:n_tok]
            xc = torch.empty((n_tok, x.shape[1]), dtype=x.dtype, device=x.device)
            gt.conv_tree_update(
                x=x, weight=w, bias=self.conv1d.bias, conv_state=conv_state,
                state_indices=si, cu_seqlens=cu, num_accepted=acc, conv_src=S.act["conv"],
                out=xc, activation=self.activation in ("silu", "swish", True),
                num_requests=r)
            hk = self.num_k_heads // self.tp_size
            key_w = hk * self.head_k_dim
            out = core_attn_out[:n_tok]
            og = output_gate[:n_tok]
            # gated norm inside the tree kernel when the layout allows it, else the stock FLA launch
            fuse = (og.dim() == 3 and og.stride(2) == 1 and og.stride(1) == self.head_v_dim
                    and self.norm.norm_before_gate and self.norm.group_size is None
                    and self.norm.bias is None)
            if not getattr(type(self), "_atdev_fuse_logged", False):
                type(self)._atdev_fuse_logged = True
                log.warning("atdev %s: Triton tree kernel, gated norm fused=%s", OPT_NAME, fuse)
            gt.gdn_tree_verify(
                q=xc[:, :key_w], k=xc[:, key_w: 2 * key_w], v=xc[:, 2 * key_w:],
                a=a[:n_tok], b=b[:n_tok], A_log=self.A_log, dt_bias=self.dt_bias,
                state=self.kv_cache[1], state_indices=si, cu_seqlens=cu, num_accepted=acc,
                anc_bits=S.act["anc"], out=out, num_k_heads=hk,
                scale=self.head_k_dim ** -0.5, block_v=_block_v(), num_requests=r,
                output_gate=og if fuse else None, norm_weight=self.norm.weight,
                norm_eps=self.norm.eps, sigmoid_gate=self.norm.activation == "sigmoid")
            if not fuse:
                self._rms_norm_gated_cuda(out, og, out)

    return TreeGDNAttention


# ---------------------------------------------------------------------------
# Runner hooks
# ---------------------------------------------------------------------------


def _decide(runner, input_batch) -> bool:
    """Host-side: may this step verify a tree?"""
    if input_batch.num_reqs != 1 or input_batch.has_prefill:
        return False
    if input_batch.num_draft_tokens != S.N - 1 or input_batch.num_tokens != S.N:
        return False
    if input_batch.req_ids[0] not in S.tree_reqs:
        return False
    if input_batch.has_structured_output_reqs or runner.pcp_manager is not None:
        return False
    return _greedy_batch(runner, input_batch)


def _greedy_batch(runner, input_batch) -> bool:
    """Host-side check that stock sampling of this batch is a pure argmax."""
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    if getattr(runner, "batch_sharder", None) is not None:
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


def _check_backend(runner) -> None:
    from vllm.model_executor.layers.attention import Attention

    fc = runner.compilation_config.static_forward_context
    bad = [n for n, l in fc.items()
           if isinstance(l, Attention) and not getattr(l, "is_draft_layer", False)
           and type(l.impl).__name__ != "TritonAttentionImpl"]
    if bad:
        raise RuntimeError(f"atdev {OPT_NAME}: tree attention needs TRITON_ATTN for the "
                           f"target; found {type(fc[bad[0]].impl).__name__}")
    S.checked_backend = True


def _begin_step(runner, input_batch, attn_out) -> None:
    if not S.checked_backend:
        _check_backend(runner)
        _wrap_model_state(runner)
    S.block_tables, S.slot_mappings = attn_out
    tree = _decide(runner, input_batch)
    if tree:
        ag = S.ag
        stash = (ag is not None and not ag.failed and input_batch.idx_mapping.dtype == ag.idx.dtype)
        # the activation kernel also copies idx_mapping[0] into the accept
        # graph's persistent input (no extra launch before the replay)
        gt.tree_activate(input_batch.idx_mapping, S.meta, S.act, S.qq, S.N, True,
                         idx_out=ag.idx if stash else None)
        S.idx_stashed = input_batch if stash else None
        S.row0_tree = True
        # RoPE position of node i = pos0 + depth(i) (slot mappings already built)
        gt.tree_shift_positions(input_batch.positions, input_batch.query_start_loc,
                                S.act["depth"], S.N, 1)
        S.mode, S.batch = "tree", input_batch
        S.n_tree += 1
    else:
        if S.row0_tree:
            gt.tree_activate(input_batch.idx_mapping, S.meta, S.act, S.qq, S.N, False)
            S.row0_tree = False
        S.mode, S.batch = None, None
        if input_batch.num_draft_tokens:
            S.n_chain += 1


def _wrap_model_state(runner) -> None:
    """M-RoPE models compute their positions in model_state.prepare_inputs."""
    ms = runner.model_state
    if getattr(ms, "_atdev_tree_wrapped", False):
        return
    orig = ms.prepare_inputs

    def prepare_inputs(input_batch, req_states):
        out = orig(input_batch, req_states)
        pos = out.get("positions") if isinstance(out, dict) else None
        if (S.mode == "tree" and S.batch is input_batch and pos is not None
                and pos is not input_batch.positions):
            gt.tree_shift_positions(pos, input_batch.query_start_loc, S.act["depth"], S.N, 1)
        return out

    ms.prepare_inputs = prepare_inputs
    ms._atdev_tree_wrapped = True


def _build_tables(runner) -> dict:
    from vllm.model_executor.layers.attention import Attention

    fc = runner.compilation_config.static_forward_context
    gid = {ln: g for g, grp in enumerate(runner.kv_cache_config.kv_cache_groups)
           for ln in grp.layer_names}
    dev = runner.device
    gdn = [(n, l) for n, l in fc.items() if isinstance(l, _LAYER_CLS)]
    attn = [(n, l) for n, l in fc.items() if isinstance(l, Attention)
            and not getattr(l, "is_draft_layer", False)]
    t: dict = {}
    # GDN: ssm log + conv state
    ssm = [l.kv_cache[1] for _, l in gdn]
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn_mod

    conv = [l.kv_cache[0] if gdn_mod.is_conv_state_dim_first()
            else l.kv_cache[0].transpose(-1, -2) for _, l in gdn]
    l0 = gdn[0][1]
    hk = l0.num_k_heads // l0.tp_size
    groups = sorted({gid[n] for n, _ in gdn})
    t["gdn_groups"] = groups
    t["gdn_grp"] = torch.tensor([groups.index(gid[n]) for n, _ in gdn], dtype=torch.int32,
                                device=dev)
    t["ssm_ptr"] = torch.tensor([s.data_ptr() for s in ssm], dtype=torch.int64, device=dev)
    t["ssm_slot"] = torch.tensor([s.stride(0) * s.element_size() // 4 for s in ssm],
                                 dtype=torch.int64, device=dev)
    from atdev_serve.gdn_single_state import _block_v

    t["lay"] = gt.tree_log_layout(ssm[0], hk, S.N, _block_v())
    t["dims"] = (hk, l0.head_k_dim, ssm[0].shape[1], ssm[0].shape[2])
    for s in ssm:
        assert s.shape == ssm[0].shape and s.dtype == ssm[0].dtype
    t["conv_ptr"] = torch.tensor([c.data_ptr() for c in conv], dtype=torch.int64, device=dev)
    cst = conv[0].stride()
    for c in conv:
        assert c.stride() == cst and c.dtype == conv[0].dtype and c.shape[1:] == conv[0].shape[1:]
    t["conv_strides"] = cst
    t["conv_dim"] = conv[0].shape[1]
    t["conv_elem"] = conv[0].element_size()
    assert t["conv_elem"] in (2, 4)
    # full attention K/V: kv[blk, head, off, 2*head_size]
    kvs = [l.kv_cache for _, l in attn]
    kvs = [k[0] if isinstance(k, (list, tuple)) else k for k in kvs]
    agids = sorted({gid[n] for n, _ in attn})
    k0 = kvs[0]
    for k in kvs:
        assert k.shape == k0.shape and k.stride() == k0.stride() and k.dtype == k0.dtype
    assert k0.dim() == 4 and k0.stride(3) == 1 and k0.element_size() in (1, 2), (
        k0.shape, k0.stride(), k0.dtype)
    t["attn_grp"] = torch.tensor([gid[n] for n, _ in attn], dtype=torch.int32, device=dev)
    t["kv_ptr"] = torch.tensor([k.data_ptr() for k in kvs], dtype=torch.int64, device=dev)
    t["kv_strides"] = (k0.stride(0), k0.stride(1), k0.stride(2))
    t["kv_heads"], t["kv_block"], t["kv_entry"] = k0.shape[1], k0.shape[2], k0.shape[3]
    t["kv_elem"] = k0.element_size()
    t["n_gdn"], t["n_attn"] = len(gdn), len(attn)
    log.warning("atdev %s: compaction tables: %d GDN layers (groups %s), %d attention "
                "layers (groups %s, kv %s %s %s)", OPT_NAME, len(gdn), groups, len(attn),
                agids, tuple(k0.shape), tuple(k0.stride()), k0.dtype)
    return t


def _finish_gpu(input_batch, path, ns, pre=None) -> torch.Tensor:
    """GPU part of a tree step's finish (also the tail of the accept graph): positions back to
    the chain layout, GDN log / conv / KV compaction; returns the drafter context gather index.
    pre = (state_idx, gather) when the fused post kernel already did the shift / stack / gather."""
    n, d = S.N, S.D
    r = input_batch.num_reqs
    t = S.tables
    hk, kd, hv, vd = t["dims"]
    if pre is None:
        # positions back to the chain layout (the drafter reads them)
        gt.tree_shift_positions(input_batch.positions, input_batch.query_start_loc,
                                S.act["depth"], n, -1)
        si = torch.stack([S.block_tables[g][:r, :2] for g in t["gdn_groups"]])
    else:
        si = pre[0]
    fork = torch.cuda.is_current_stream_capturing()
    if fork:  # inside the accept graph capture: conv / KV compaction on 2 forked branches
        cur = torch.cuda.current_stream(path.device)
        if S.ag_streams is None:
            S.ag_streams = (torch.cuda.Stream(device=path.device), torch.cuda.Stream(device=path.device))
        s_conv, s_kv = S.ag_streams
        s_conv.wait_stream(cur)
        s_kv.wait_stream(cur)
    with torch.cuda.stream(s_conv) if fork else _nullctx():
        gt.compact_conv_state(ptr_tab=t["conv_ptr"], grp_tab=t["gdn_grp"], state_idx=si,
                              path=path, num_sampled=ns, strides=t["conv_strides"],
                              dim=t["conv_dim"], elem_size=t["conv_elem"], max_depth=d,
                              num_layers=t["n_gdn"], num_reqs=r)
    with torch.cuda.stream(s_kv) if fork else _nullctx():
        gt.compact_kv(ptr_tab=t["kv_ptr"], grp_tab=t["attn_grp"], slot_mappings=S.slot_mappings,
                      query_start_loc=input_batch.query_start_loc, path=path, num_sampled=ns,
                      block_size=t["kv_block"], strides=t["kv_strides"], heads=t["kv_heads"],
                      entry=t["kv_entry"], elem_size=t["kv_elem"], max_depth=d,
                      num_layers=t["n_attn"], num_reqs=r)
    gt.compact_gdn_log(ptr_tab=t["ssm_ptr"], slot_tab=t["ssm_slot"], grp_tab=t["gdn_grp"],
                       state_idx=si, path=path, num_sampled=ns, lay=t["lay"],
                       num_k_heads=hk, head_k_dim=kd, num_v_heads=hv, head_v_dim=vd,
                       max_depth=d, num_layers=t["n_gdn"], num_reqs=r)
    if fork:
        cur.wait_stream(s_conv)
        cur.wait_stream(s_kv)
    if pre is not None:
        return pre[1]
    # drafter context rows: path nodes first (r == 1, query start 0)
    ar = S.ar
    return torch.where(ar < ns[0].to(torch.int64), path[0, :n].to(torch.int64), ar)


def _finish_tree_step(runner, input_batch) -> None:
    if S.tables is None:
        S.tables = _build_tables(runner)
    if S.ag_done:  # the accept graph already ran the finish part (S.pending_gather set)
        S.ag_done = False
    else:
        S.pending_gather = _finish_gpu(input_batch, S.last_path, S.last_ns, S.last_pre)
    S.last_pre = None


def _accept_gpu(sampler, am, input_batch, cu_num_logits, idx_mapping):
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected

    n, r = S.N, input_batch.num_reqs
    if (r == 1 and input_batch.positions.dim() == 1
            and S.tables is not None and am.is_contiguous()):
        from atdev_serve.tree_post import tree_post_c1

        path, sampled, ns, nrej, si, gather = tree_post_c1(
            parent=S.act["parent"][0], tokens=input_batch.input_ids[:n], argmax=am.view(-1)[:n],
            max_depth=S.D, seq_lens=input_batch.seq_lens, cu_num_logits=cu_num_logits,
            idx_mapping=idx_mapping, prefill_len=sampler.sampler.req_states.prefill_len.gpu,
            positions=input_batch.positions, query_start_loc=input_batch.query_start_loc,
            depth=S.act["depth"][0], block_tables=S.block_tables, groups=S.tables["gdn_groups"])
        return path, sampled, ns, nrej, (si, gather)
    tokens = input_batch.input_ids[: r * n].view(r, n)
    # fresh outputs every step: AsyncOutput copies sampled / num_sampled to
    # the host on a side stream while the next step runs
    path, sampled, ns = gt.tree_accept(S.act["parent"][:r], tokens, am, S.D, out_cols=n)
    num_sampled, num_rejected = get_num_sampled_and_rejected(
        ns, input_batch.seq_lens, cu_num_logits, idx_mapping,
        sampler.sampler.req_states.prefill_len.gpu)
    return path, sampled, num_sampled, num_rejected, None


def _ag_key(sampler, input_batch) -> tuple:
    def k(t):
        return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)

    ts = [input_batch.input_ids, input_batch.positions, input_batch.query_start_loc,
          input_batch.seq_lens, sampler.sampler.req_states.prefill_len.gpu, S.slot_mappings,
          S.act["parent"], S.act["depth"], S.ar]
    ts += [S.block_tables[g] for g in S.tables["gdn_groups"]]
    return tuple(k(t) for t in ts) + (id(S.tables), input_batch.num_reqs, S.N, S.D)


def _accept_graph_step(sampler, input_batch):
    """Replay (capturing on first use of a ring slot) the accept + finish chain; None -> eager."""
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    ag, n = S.ag, S.N
    cu_np = getattr(input_batch, "cu_num_logits_np", None)
    if (S.tables is None or input_batch.num_reqs != 1 or cu_np is None or len(cu_np) != 2
            or int(cu_np[0]) != 0 or int(cu_np[1]) != n or torch.cuda.is_current_stream_capturing()):
        return None
    key = _ag_key(sampler, input_batch)
    i = ag.next
    slot = ag.slots[i]
    if slot is None or slot["key"] != key:
        if ag.n_eager < ag.warm:  # warm-up: Triton compiles / tables on eager steps
            ag.n_eager += 1
            return None
        if S.idx_stashed is not input_batch:
            ag.idx.copy_(input_batch.idx_mapping[:1])
        from atdev_serve.tree_post import bt_ptr_table

        bt_ptr_table(S.block_tables, S.tables["gdn_groups"], ag.am.device)  # pointer table before capture (HtoD)
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                path, sampled, num_sampled, num_rejected, pre = _accept_gpu(
                    sampler, ag.am[:n].view(1, n), input_batch, ag.cu, ag.idx)
                gather = _finish_gpu(input_batch, path, num_sampled, pre)
        except Exception as e:  # noqa: BLE001 -- stay correct: eager from now on
            ag.failed = True
            log.warning("atdev %s: accept graph capture FAILED (%s: %s) -> eager", OPT_NAME,
                        type(e).__name__, e)
            return None
        slot = ag.slots[i] = dict(key=key, graph=g, path=path, sampled=sampled,
                                  num_sampled=num_sampled, num_rejected=num_rejected,
                                  gather=gather)
        ag.n_capture += 1
        if ag.n_capture == 1:
            log.warning("atdev %s: accept graph ENGAGED (tree accept + compaction as one "
                        "CUDA graph, ring %d)", OPT_NAME, ag.ring)
    elif S.idx_stashed is not input_batch:  # else _begin_step's tree_activate already wrote it
        ag.idx.copy_(input_batch.idx_mapping[:1])
    slot["graph"].replay()
    ag.next = (i + 1) % ag.ring
    ag.n_replay += 1
    S.last_path, S.last_ns = slot["path"], slot["num_sampled"]
    S.pending_gather = slot["gather"]
    S.ag_done = True
    return SamplerOutput(sampled_token_ids=slot["sampled"], logprobs_tensors=None, num_nans=None,
                         num_sampled=slot["num_sampled"], num_rejected=slot["num_rejected"])


def _tree_accept(sampler, logits, input_batch):
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    n, r = S.N, input_batch.num_reqs
    fa, S.fast_am = S.fast_am, None
    S.ag_done = False
    if fa is not None and fa[0] == logits.data_ptr() and fa[1] == logits.shape[0]:
        ag = S.ag
        if (ag is not None and not ag.failed and r == 1
                and fa[2].data_ptr() == ag.am.data_ptr() and fa[2].numel() == n):
            out = _accept_graph_step(sampler, input_batch)
            if out is not None:
                return out
        am = fa[2][: r * n].view(r, n)  # verify_split_head fused argmax (bit-identical)
    else:
        am = logits[: r * n].argmax(dim=-1).view(r, n)
    path, sampled, num_sampled, num_rejected, S.last_pre = _accept_gpu(
        sampler, am, input_batch, input_batch.cu_num_logits, input_batch.idx_mapping)
    S.last_path, S.last_ns = path, num_sampled
    return SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None, num_nans=None,
                         num_sampled=num_sampled, num_rejected=num_rejected)


def _install_hooks() -> None:
    from vllm.v1.attention.backends import triton_attn as ta
    from vllm.v1.worker.gpu import model_runner as mr
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    spec_cls = _make_speculator_class()
    if not getattr(mr.init_speculator, "_atdev_tree", False):
        orig_init = mr.init_speculator

        def init_speculator(vllm_config, device):
            sc = vllm_config.speculative_config
            if (sc.method == "dflash"
                    and "DFlash2DraftModel" in sc.draft_model_config.architectures):
                return spec_cls(vllm_config, device)
            raise ValueError(f"atdev {OPT_NAME}: needs a DFlash2 drafter "
                             f"(method={sc.method})")

        init_speculator._atdev_tree = True
        init_speculator._atdev_orig = orig_init
        mr.init_speculator = init_speculator

    cls = mr.GPUModelRunner
    if getattr(cls.prepare_attn, "_atdev_tree", False):
        return
    orig_pa = cls.prepare_attn

    @functools.wraps(orig_pa)
    def prepare_attn(self, input_batch):
        out = orig_pa(self, input_batch)
        if S.act is not None:
            _begin_step(self, input_batch, out)
        return out

    prepare_attn._atdev_tree = True
    cls.prepare_attn = prepare_attn

    orig_cap = cls.capture_model

    @functools.wraps(orig_cap)
    def capture_model(self, *args, **kwargs):
        S.capturing = True
        try:
            return orig_cap(self, *args, **kwargs)
        finally:
            S.capturing = False

    cls.capture_model = capture_model

    orig_sample = cls.sample

    @functools.wraps(orig_sample)
    def sample(self, hidden_states, input_batch, grammar_output):
        if S.mode != "tree" or S.batch is not input_batch:
            S.mode, S.batch = None, None
            return orig_sample(self, hidden_states, input_batch, grammar_output)
        S.accepting = True
        try:
            res = orig_sample(self, hidden_states, input_batch, grammar_output)
        finally:
            S.accepting = False
            S.mode, S.batch = None, None
        _finish_tree_step(self, input_batch)
        return res

    cls.sample = sample

    orig_call = RejectionSampler.__call__

    @functools.wraps(orig_call)
    def __call__(self, logits, input_batch, draft_logits=None):
        if not S.accepting:
            return orig_call(self, logits, input_batch, draft_logits)
        return _tree_accept(self, logits, input_batch)

    RejectionSampler.__call__ = __call__

    # full attention: ancestor mask through unified_attention's qq_bias
    orig_ua = ta.unified_attention

    @functools.wraps(orig_ua)
    def unified_attention(*args, **kwargs):
        if S.attn_on:
            kwargs["qq_bias"] = S.qq
        return orig_ua(*args, **kwargs)

    ta.unified_attention = unified_attention
    orig_fwd = ta.TritonAttentionImpl.forward

    @functools.wraps(orig_fwd)
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        use = (S.qq is not None and attn_metadata is not None
               and not getattr(layer, "is_draft_layer", False)
               and attn_metadata.max_query_len == S.N
               and (S.capturing or S.mode == "tree"))
        if not use:
            return orig_fwd(self, layer, query, key, value, kv_cache, attn_metadata,
                            output, output_scale, output_block_scale)
        S.attn_on = True
        try:
            return orig_fwd(self, layer, query, key, value, kv_cache, attn_metadata,
                            output, output_scale, output_block_scale)
        finally:
            S.attn_on = False

    ta.TritonAttentionImpl.forward = forward


_LAYER_CLS = None


def register() -> None:
    global _LAYER_CLS
    import vllm
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn_mod

    from atdev_serve import gdn_single_state as a3
    from atdev_serve.register import enabled

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written against vLLM "
                           f"{SUPPORTED_VLLM_VERSION}, found {vllm.__version__}")
    if "gdn_single_state" not in enabled():
        raise RuntimeError(f"atdev {OPT_NAME}: requires ATDEV_OPT gdn_single_state")
    a3.register()  # builds the single-state class and swaps the mixed-batch dispatcher
    if _LAYER_CLS is None:
        _LAYER_CLS = _make_layer_class(a3._LAYER_CLS)
        gdn_mod.QwenGatedDeltaNetAttention.register_oot(_LAYER_CLS)
    _install_hooks()
    log.info("atdev %s: registered (N=%d)", OPT_NAME, N_NODES)
