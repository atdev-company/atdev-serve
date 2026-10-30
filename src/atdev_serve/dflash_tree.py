"""Tree verification for DFlash2 (ATDEV_OPT=dflash_tree), vLLM 0.31.0 MRV2.

The DFlash2 drafter (block 8 -> depth D = 7, top-16 candidates per position,
CRF edge scores) proposes one greedy path in stock vLLM. Here it proposes a
token *tree* of N nodes (default 16: root + 15 drafts) built best-first over
the CRF lattice (``tree-crf``: node value = cumulative log-softmax of the
conditional edge scores, temperature 1, no fitted statistics), and the target
verifies all N nodes in one forward with an ancestor mask:

* full attention (16 layers): TRITON_ATTN's ``unified_attention`` ``qq_bias``
  [N, N] (0 ancestor-or-self, -inf otherwise) + RoPE position = pos0 + depth;
  K/V of node i lands in the slot of pos0 + i (slot mapping unchanged).
* GDN (48 layers): ``gdn_tree.gdn_tree_verify`` (ancestor-masked WY form on
  top of A3's single-state slots) and ``gdn_tree.conv_tree_update``.
* acceptance: greedy walk of the target argmax down the tree
  (``tree_accept``): bit-exact greedy semantics -- every committed token is
  the target's argmax after the committed prefix.
* compaction after acceptance (one launch each): K/V of path node d -> slot of
  pos0 + d; GDN log entries and conv-state columns into A3/stock sequential
  order; drafter context rows (hidden states) gathered along the path.

``num_speculative_tokens`` in ``--speculative-config`` must be N-1 (it sizes
the scheduler, KV lookahead, GDN/conv slots and the target's uniform CUDA
graph); the drafter itself is built with D = block_size - 1 steps.

Tree mode is used only when the step's batch is exactly one pure spec-decode
request (c=1), greedy, no logprobs / logits processors / structured output,
and its drafts were built as a tree. Any other batch runs in *chain* mode:
same N-token verify with causal semantics, stock rejection sampler; the
drafter then emits vLLM's greedy chain (D tokens) padded with dead tokens (so
c > 1 is correct but pays N-token verifies: this is a c=1 optimization).
Mixed prefill + decode batches use the stock conv and A3's dispatcher, which
are valid because compaction restores their layouts after every tree step.

Requires: gdn_single_state (A3), --attention-backend TRITON_ATTN,
--no-enable-prefix-caching, DFlash2 drafter, draft_sample_method greedy.

Env:
  ATDEV_TREE_NODES   N (default 16; 4..32, need not be a power of two)
  ATDEV_TREE_TEMP    softmax temperature of the CRF edge scores (default 1)
  ATDEV_TREE_MIX     E7: weight of log_softmax(unary) mixed into the edge log-probs (default 0)
  ATDEV_TREE_PW      E7: scale of the pairwise (selector) term, scores = u + pw (s - u) (default 1)
  ATDEV_TREE_STATS   log acceptance statistics every this many tree steps
                     (0 = off; each log costs one host sync)
  ATDEV_TREE_ACCEPT_GRAPH  1 = replay the c=1 post-verify chain (tree accept, num_sampled /
                     num_rejected, position shift back, GDN log / conv / KV compaction, drafter
                     context gather index: ~11 tiny eager kernels) as one CUDA graph (ring of
                     ATDEV_TREE_ACCEPT_GRAPH_RING graphs, default 3, so the sampled-token outputs
                     AsyncOutput copies to the host are not overwritten by the next steps). Same
                     kernels on the same inputs -> bit-identical; needs ATDEV_VSPLIT_FAST (the
                     argmax lands in a persistent buffer). First 2 tree steps eager (warm-up),
                     any pointer/shape change of an input -> eager. Default 0.
  ATDEV_TREE_AG_FORK  1 = inside the accept graph capture, the conv-state and KV compaction kernels
                     run on 2 forked branches concurrent with the GDN log compaction (independent
                     buffers; same kernels / inputs -> bit-identical). Default 0. (C1-agf)
  ATDEV_TREE_POST_FUSED  1 = one Triton launch (tree_post.tree_post_c1) for the c=1 accept walk,
                     num_sampled / num_rejected, position shift back, the GDN state-index stack and
                     the drafter gather index (8 tiny kernels -> 1; integer work, identical values),
                     and the drafter context rows gathered straight out of the fc output
                     (tree_post.gather_rows; stock copy + gather + copy -> 1 launch).
                     Combines with ATDEV_TREE_ACCEPT_GRAPH. Default 0.
  ATDEV_FC_EARLY     1 = on a c=1 tree step, launch the drafter's context projection (torch.cat of
                     the target's aux hidden states + model.combine_hidden_states = fc, W8A16 Marlin,
                     ~84 us) on a third stream right before the verify head instead of after the
                     tree accept (it does not depend on the accept), so it overlaps the head's two
                     GEMMs. propose() waits on its event and gathers from its output. Same kernels
                     on the same inputs (same workspace lane) -> bit-identical. Default 0.
                     2 = same side-stream fc, but launched right AFTER the verify head GEMMs
                     (verify_split_head FAST argmax hook, C1-fcl): it overlaps the latency-bound
                     split argmax / accept graph / post-update chain (~55 us) instead of the
                     bandwidth-bound head (NW-fce: no gain there).
  ATDEV_TREE_CHAIN  1 = degenerate test mode: the drafter emits vLLM's greedy
                     chain (parent = i-1, dead leaves past depth D) but every
                     step still runs the tree path (tree GDN/conv kernels,
                     qq_bias, tree accept, compaction). With N-1 = D it must
                     reproduce the chain run's tokens and acceptance.
"""

from __future__ import annotations

import copy
import functools
import logging
import os

import numpy as np
import torch

from atdev_serve import gdn_tree as gt

OPT_NAME = "dflash_tree"
SUPPORTED_VLLM_VERSION = "0.31.0"
log = logging.getLogger("atdev_serve")


def num_nodes() -> int:
    n = int(os.environ.get("ATDEV_TREE_NODES", "16"))
    if not 4 <= n <= 48:
        # the A3 log must also fit one spec slot: <= 24 nodes with a bf16 SSM
        # state, <= 48 with fp32 (checked by gdn_tree.tree_log_layout); 64
        # would need a 2-slot log (P3 wide-tree note)
        raise ValueError(f"ATDEV_TREE_NODES must be in [4, 48], got {n}")
    return n


def tree_temperature() -> float:
    return float(os.environ.get("ATDEV_TREE_TEMP", "1.0"))


def force_chain() -> bool:
    return os.environ.get("ATDEV_TREE_CHAIN", "0") == "1"


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
        # ATDEV_VSPLIT_FAST: (logits data_ptr, rows, int64 argmax) from verify_split_head
        self.fast_am: tuple | None = None
        self.block_tables = None
        self.slot_mappings = None
        self.pending_gather: torch.Tensor | None = None
        self.tables: dict | None = None
        self.checked_backend = False
        self.n_tree = 0
        self.n_chain = 0
        self.stats_every = int(os.environ.get("ATDEV_TREE_STATS", "0"))
        self.acc_sum: torch.Tensor | None = None
        self.ag: _AcceptGraph | None = None  # ATDEV_TREE_ACCEPT_GRAPH
        self.ag_done = False  # this step's finish part already ran inside the accept graph
        self.post_fused = os.environ.get("ATDEV_TREE_POST_FUSED", "0") == "1"
        self.last_pre = None  # (state_idx, gather) from the fused post kernel of this step
        self.ctx_gather = None  # gather index handed to the wrapped combine_hidden_states
        self.idx_stashed = None  # input batch whose idx_mapping[0] tree_activate copied to ag.idx
        # ATDEV_FC_EARLY: (aux data_ptrs, rows, fc output, done event) of this step's early fc
        self.fc_mode = os.environ.get("ATDEV_FC_EARLY", "0")
        self.fc_early_on = self.fc_mode in ("1", "2")
        self.fc_early: tuple | None = None
        # ATDEV_FC_EARLY=2: (runner, hidden_states) waiting for verify_split_head's post-head hook
        self.fc_late: tuple | None = None
        self.fc_stream = None
        self.n_fc_early = 0
        # ATDEV_TREE_AG_FORK: forked capture streams of the accept graph's conv / KV compaction
        self.ag_fork = os.environ.get("ATDEV_TREE_AG_FORK", "0") == "1"
        self.ag_streams = None


class _AcceptGraph:
    """ATDEV_TREE_ACCEPT_GRAPH: CUDA-graph replay of the c=1 tree accept + finish chain."""

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


def accept_graph_on() -> bool:
    return os.environ.get("ATDEV_TREE_ACCEPT_GRAPH", "0") == "1"


def am_out(m: int) -> torch.Tensor | None:
    """Persistent int64 buffer for the verify head's tree argmax (ATDEV_TREE_ACCEPT_GRAPH),
    None -> the head allocates a fresh one (stock behaviour)."""
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
            seg = int(hf.dflash_config.get("segment_causal") or 0)
            if seg and seg < depth + 1:  # D16-seg drafter: rows < seg blind to rows >= seg (drafter_seg.py)
                from atdev_serve import drafter_seg
                drafter_seg.enable(seg)
            n = num_nodes()
            if sc.num_speculative_tokens != n - 1:
                raise ValueError(
                    f"atdev {OPT_NAME}: --speculative-config num_speculative_tokens must "
                    f"be ATDEV_TREE_NODES-1 = {n - 1} (verify width), got "
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
            self.tree_temp = tree_temperature()
            # E7 (docs/research/E7-tree-policy.md): marginal mix of the unary softmax into the edge log-probs
            self.tree_mix = float(os.environ.get("ATDEV_TREE_MIX", "0"))
            self.tree_pw = float(os.environ.get("ATDEV_TREE_PW", "1"))
            self.tree_mix_on = self.tree_mix != 0.0 or self.tree_pw != 1.0
            if self.tree_mix_on:
                log.warning("atdev dflash_tree: E7 marginal mix %.3g, pairwise scale %.3g",
                            self.tree_mix, self.tree_pw)
            self.tree_chain = force_chain()
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
            if accept_graph_on():
                S.ag = _AcceptGraph(n, device, int(os.environ.get("ATDEV_TREE_ACCEPT_GRAPH_RING", "3")))
                log.warning("atdev %s: post-verify accept chain as CUDA graph (ring %d)", OPT_NAME,
                            S.ag.ring)
            log.warning("atdev %s: tree verify N=%d (drafter depth %d, top-%d, T=%.2f%s)",
                        OPT_NAME, n, depth, self.top_k, self.tree_temp,
                        ", FORCED CHAIN" if self.tree_chain else "")

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
                self.tree_nodes, chain=num_reqs != 1 or self.tree_chain,
                temperature=self.tree_temp,
                unary=unary_logits.float().contiguous() if self.tree_mix_on else None,
                mix=self.tree_mix, pw=self.tree_pw)

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
            fe = _fc_early_take(aux_hidden_states, dummy_run)
            if fe is not None:
                # ATDEV_FC_EARLY: the fc already ran on the side stream (same kernels / inputs);
                # hand its (path-gathered) output in as last_hidden_states with no aux list, so
                # the stock propose skips the cat + combine and copies the same values
                gidx = S.pending_gather
                if (S.post_fused and gidx is not None and fe.dim() == 2
                        and input_batch.num_tokens == gidx.numel() == fe.shape[0]
                        and fe.dtype == self.hidden_states.dtype):
                    from atdev_serve.tree_post import gather_rows

                    fe = gather_rows(fe, gidx, self.hidden_states[: fe.shape[0]])
                    S.pending_gather = None
                aux_hidden_states = None
                last_hidden_states = fe
            gidx = S.pending_gather
            if (S.post_fused and gidx is not None and not dummy_run and aux_hidden_states
                    and input_batch.num_tokens == gidx.numel()):
                # ATDEV_TREE_POST_FUSED: gather the accepted path rows straight out of the fc
                # output into self.hidden_states (stock: copy rows, gather, copy back)
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

    use_cuda = os.environ.get("ATDEV_TREE_CUDA", "0") == "1"
    if use_cuda:  # E2: build / load the CUDA tree kernel before any graph capture
        from atdev_serve import gdn_tree_cuda as gtc

        gtc.cuda_ext()

    class TreeGDNAttention(base):
        """A3 layer whose pure spec-decode verify runs the tree kernels (with
        a causal ancestor mask in chain mode, so they cover every pure batch)."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            if use_cuda:
                # exchange buffers must exist before CUDA-graph capture (a captured zero-init
                # would be replayed every step and break the kernel's generation tags)
                gtc.prealloc(torch.device("cuda", torch.cuda.current_device()), 1,
                             self.num_k_heads // self.tp_size, self.num_v_heads // self.tp_size)

        def _atdev_ba_defer_ok(self, qkvz) -> bool:
            """c1_fuse.gdn_core_ba (ATDEV_TREE_BA_FOLD): will this call reach the CUDA tree path, where the
            kernel computes b / a itself? (False -> ba is materialised up front.) Mirrors the dispatch of
            _forward_core_fused_norm_packed / _forward_core_fused_norm / _forward_core_decode_spec_fused_norm;
            anything that fails later materialises ba before use (_atdev_ba_materialize)."""
            if not (use_cuda and (gtc.BA_FOLD or gtc.BA_PRE)) or S.act is None:
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
            if use_cuda and gtc.BA_CTX and not gtc.BA_CTX.get("done"):
                from atdev_serve import c1_fuse
                c = gtc.BA_CTX
                if c.get("kind") == "tot":  # ATDEV_C1_BA_STREAM=2: bf16(tot * rstd) == ba_gemv_r
                    from atdev_serve import c1_norm
                    c1_norm.ba_scale(c["tot"], c["r"], c["ba"].shape[0], out=c["ba"])
                else:
                    c1_fuse.materialize_ba(c["x"], c["w"], c["r"], out=c["ba"])
                c["done"] = True

        def _atdev_tree_cuda(self, mixed_qkv, b, a, output_gate, core_attn_out, attn_metadata,
                             si, r, n_tok, cu, acc, conv_state, w):
            """E2: fused conv + tree verify + gated norm in one CUDA launch; False -> not
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
            ba_fold = bap = None
            c = gtc.BA_CTX
            if c and not c.get("done") and c.get("kind") == "tot":
                tot, rr = c["tot"], c["r"]
                hv = self.num_v_heads // self.tp_size
                if (tot.dtype == torch.float32 and rr.dtype == torch.float32 and tot.shape[0] >= n_tok
                        and tot.shape[1] == 2 * hv and tot.stride(-1) == 1 and rr.shape[0] >= n_tok):
                    bap = dict(b=tot[:n_tok, :hv], a=tot[:n_tok, hv:], r=rr[:n_tok].contiguous())
                    c["done"] = True  # b / a never materialised: only this kernel reads them
                else:
                    self._atdev_ba_materialize()
            elif c and not c.get("done"):
                bx, bw, br = c["x"], c["w"], c["r"]
                hv = self.num_v_heads // self.tp_size
                if (bx.shape[0] >= n_tok and bx.dtype == torch.bfloat16 and bw.dtype == torch.bfloat16
                        and br.dtype == torch.float32 and bx.stride(-1) == 1 and bw.stride(-1) == 1
                        and bw.shape[0] == 2 * hv and bx.shape[1] == bw.shape[1] and bx.shape[1] % 128 == 0
                        and bx.shape[1] // 8 <= 640 and 2 * (hv // hk) <= 8 and hv % hk == 0
                        and bx.data_ptr() % 16 == 0 and (bx.stride(0) * 2) % 16 == 0
                        and bw.data_ptr() % 16 == 0 and (bw.stride(0) * 2) % 16 == 0):
                    ba_fold = dict(x=bx[:n_tok], w=bw, r=br[:n_tok].contiguous())
                    c["done"] = True  # b / a never materialised: only this kernel reads them
                else:
                    self._atdev_ba_materialize()
            if not getattr(type(self), "_atdev_tree_cuda_logged", False):
                type(self)._atdev_tree_cuda_logged = True
                log.warning("atdev %s: E2 CUDA tree verify (conv + tree + gated norm fused) ENGAGED (in_proj_ba "
                            "fold %s, fp32 ba + rstd %s)", OPT_NAME, ba_fold is not None, bap is not None)
            qkc = None
            if gtc.QKC:
                qkc = gtc.qkc_bufs(st.device, hk, self.num_v_heads // self.tp_size)
            gtc.gdn_tree_verify_cuda(ba_fold=ba_fold, bap=bap, qkc=qkc,
                a=a[:n_tok], b=b[:n_tok], A_log=self.A_log, dt_bias=self.dt_bias, state=st,
                state_indices=si, cu_seqlens=cu, num_accepted=acc, anc_bits=S.act["anc"],
                out=core_attn_out[:n_tok], num_k_heads=hk, scale=self.head_k_dim ** -0.5,
                num_requests=r, output_gate=og, norm_weight=self.norm.weight,
                norm_eps=self.norm.eps, sigmoid_gate=self.norm.activation == "sigmoid",
                conv=dict(x=x, weight=w, bias=self.conv1d.bias, conv_state=conv_state,
                          conv_src=S.act["conv"],
                          activation=self.activation in ("silu", "swish", True)))
            return True

        def _atdev_qkc_args(self, md):
            """C1gdn2 (ATDEV_C1GDN2_QKC=1): set_qkc kwargs for this layer's qkvz W3 GEMM (GDN conv in its epilogue,
            into the shared gtc.qkc_bufs scratch), or None when this step's metadata is not a c=1 tree verify. The
            pointers are the GDN metadata builder's persistent spec buffers (captured graphs replay them); the
            GEMM never writes the conv state, and the tree kernel checks the tags -> a stale capture is only slower."""
            si = getattr(md, "spec_state_indices_tensor", None)
            if (not use_cuda or not gtc.QKC or S.act is None or si is None or si.size(1) != S.N
                    or getattr(md, "num_prefills", 1) != 0 or getattr(md, "num_spec_decodes", 0) != 1
                    or si.dtype != torch.int32 or si.stride(-1) != 1 or md.num_accepted_tokens is None
                    or md.spec_query_start_loc is None or md.num_accepted_tokens.dtype != torch.int32
                    or md.spec_query_start_loc.dtype != torch.int32
                    or self.head_k_dim != 128 or self.head_v_dim != 128):
                return None
            hk = self.num_k_heads // self.tp_size
            hv = self.num_v_heads // self.tp_size
            cs = self.kv_cache[0] if dim_first() else self.kv_cache[0].transpose(-1, -2)
            w = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            if (cs.dtype != torch.bfloat16 or cs.dim() != 3 or w.dtype != torch.bfloat16 or not w.is_contiguous()
                    or w.size(1) != 4 or (self.conv1d.bias is not None and self.conv1d.bias.dtype != torch.bfloat16)
                    or not S.act["conv"].is_contiguous()):
                return None
            b = gtc.qkc_bufs(cs.device, hk, hv)
            if b["out"].shape[1] != w.size(0):
                return None
            cnt = gtc._bufs(cs.device, 1, hk, hv)[0]
            return dict(cu=md.spec_query_start_loc, idx=si[0], acc=md.num_accepted_tokens, csrc=S.act["conv"],
                        cnt=cnt, cw=w, cb=self.conv1d.bias, cs=cs, out=b["out"], tag=b["tag"], H=hk, G=hv // hk,
                        nb=(hv // hk) * 4, max_log=S.N - 1, act=self.activation in ("silu", "swish", True))

        def _atdev_pf_regions(self, md):
            """ATDEV_GDN_L2PF: L2-warm regions of everything the CUDA tree kernel of this layer reads that the
            preceding qkvz GEMM does not produce (state / log / conv-state slots by device slot index, layer
            params, tree metadata, exchange buffers); None when this step will not take the CUDA tree path."""
            si = getattr(md, "spec_state_indices_tensor", None)
            if (not use_cuda or S.act is None or si is None or si.size(1) != S.N
                    or getattr(md, "num_prefills", 1) != 0 or getattr(md, "num_spec_decodes", 0) != 1
                    or si.dtype != torch.int32 or si.stride(-1) != 1):
                return None
            st = self.kv_cache[1]
            hk = self.num_k_heads // self.tp_size
            hv = self.num_v_heads // self.tp_size
            key = (si.data_ptr(), st.data_ptr(), md.num_accepted_tokens.data_ptr(),
                   md.spec_query_start_loc.data_ptr(), S.act["anc"].data_ptr())
            c = getattr(self, "_atdev_pf_cache", None)
            if c is not None and c[0] == key:
                return c[1]
            from atdev_serve import l2_pf
            lay = gt.tree_log_layout(st, hk, S.N, _block_v())
            bufs = list(gtc._bufs(st.device, 1, hk, hv))
            small = l2_pf.tensor_regions(
                [self.A_log, self.dt_bias, self.norm.weight, self.conv1d.weight, self.conv1d.bias,
                 md.num_accepted_tokens[:1], md.spec_query_start_loc[:2], si[:1], S.act["anc"][:1],
                 S.act["conv"][:1]] + bufs)
            regs = l2_pf.merge(l2_pf.gdn_regions(st, self.kv_cache[0], si[:1], lay["total"]), small)
            self._atdev_pf_cache = (key, regs)
            return regs

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
            if use_cuda and self._atdev_tree_cuda(mixed_qkv, b, a, output_gate, core_attn_out,
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
            # P1-F1: gated norm inside the tree kernel (ATDEV_TREE_FUSE_NORM=0 -> stock FLA launch)
            fuse = (os.environ.get("ATDEV_TREE_FUSE_NORM", "1") == "1" and og.dim() == 3
                    and og.stride(2) == 1 and og.stride(1) == self.head_v_dim
                    and self.norm.norm_before_gate and self.norm.group_size is None
                    and self.norm.bias is None)
            if not getattr(type(self), "_atdev_fuse_logged", False):
                type(self)._atdev_fuse_logged = True
                log.warning("atdev %s: tree kernel gated norm fused=%s (P1-F1)", OPT_NAME, fuse)
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


def _fc_early_launch(runner, hidden_states) -> None:
    """ATDEV_FC_EARLY: start the drafter's fc on the aux hidden states of this tree step's verify
    graph on a side stream (called right before the verify head). Only when the target ran as the
    FULL graph whose persistent output buffers propose() will receive."""
    S.fc_early = None
    cg = getattr(runner, "cudagraph_manager", None)
    aux = getattr(cg, "aux_hidden_states", None) if cg is not None else None
    hs = getattr(cg, "hidden_states", None) if cg is not None else None
    spec = getattr(runner, "speculator", None)
    if (not aux or hs is None or spec is None or hidden_states.data_ptr() != hs.data_ptr()
            or hidden_states.dim() != 2 or torch.cuda.is_current_stream_capturing()):
        return
    from vllm.v1.worker.workspace import use_workspace_lane

    n = hidden_states.shape[0]
    views = [x[:n] for x in aux]
    cur = torch.cuda.current_stream(hidden_states.device)
    if S.fc_stream is None:
        S.fc_stream = torch.cuda.Stream(device=hidden_states.device)
    st = S.fc_stream
    st.wait_stream(cur)
    with torch.cuda.stream(st), use_workspace_lane(runner._draft_workspace_lane):
        out = spec.model.combine_hidden_states(torch.cat(views, dim=-1))
    ev = torch.cuda.Event()
    ev.record(st)
    S.fc_early = (tuple(v.data_ptr() for v in views), n, out, ev)


def fc_post_head() -> None:
    """ATDEV_FC_EARLY=2 hook, called by verify_split_head right after the verify head GEMMs are
    enqueued (before the split argmax): the fc side stream then waits for the head and overlaps the
    latency-bound argmax / tree accept / compaction chain instead of the head's own GEMMs."""
    fl, S.fc_late = S.fc_late, None
    if fl is not None:
        _fc_early_launch(*fl)


def _fc_early_warm(runner) -> None:
    """Run the drafter fc once on the side stream before capture_model locks vLLM's workspace: the Marlin
    lock buffer is a per-(stream, lane) persistent resource (marlin_utils: ("marlin_locks", stream))."""
    spec = getattr(runner, "speculator", None)
    fc = getattr(getattr(getattr(spec, "model", None), "model", None), "fc", None)
    k = getattr(fc, "input_size", None)
    if spec is None or k is None:
        log.warning("atdev %s: ATDEV_FC_EARLY warm-up skipped (no drafter fc)", OPT_NAME)
        return
    from vllm.v1.worker.workspace import use_workspace_lane

    dev = runner.device
    if S.fc_stream is None:
        S.fc_stream = torch.cuda.Stream(device=dev)
    dt = getattr(getattr(spec, "hidden_states", None), "dtype", torch.bfloat16)
    x = torch.zeros(S.N or 16, k, dtype=dt, device=dev)
    S.fc_stream.wait_stream(torch.cuda.current_stream(dev))
    with torch.inference_mode(), torch.cuda.stream(S.fc_stream), \
            use_workspace_lane(runner._draft_workspace_lane):
        spec.model.combine_hidden_states(x)
    torch.cuda.current_stream(dev).wait_stream(S.fc_stream)
    torch.cuda.synchronize(dev)


def _fc_early_take(aux_hidden_states, dummy_run) -> torch.Tensor | None:
    """The early fc output for exactly these aux tensors (main stream made to wait), else None."""
    fe, S.fc_early = S.fc_early, None
    if fe is None or dummy_run or not aux_hidden_states:
        return None
    ptrs, n, out, ev = fe
    if (len(ptrs) != len(aux_hidden_states)
            or any(a.data_ptr() != p or a.shape[0] != n for a, p in zip(aux_hidden_states, ptrs))):
        return None
    cur = torch.cuda.current_stream(out.device)
    cur.wait_event(ev)
    out.record_stream(cur)
    S.n_fc_early += 1
    if S.n_fc_early == 1:
        log.warning("atdev %s: early fc (drafter context projection overlapped with the verify "
                    "head) ENGAGED", OPT_NAME)
    return out


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
    from atdev_serve.verify_head_screen import greedy_verify_batch_eligible

    return greedy_verify_batch_eligible(runner, input_batch, None)


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
        # ATDEV_TREE_ACCEPT_GRAPH: the activation kernel also copies idx_mapping[0] into the
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
    fork = S.ag_fork and torch.cuda.is_current_stream_capturing()
    if fork:  # ATDEV_TREE_AG_FORK: conv / KV compaction on 2 forked branches of the accept graph
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
    ns = S.last_ns
    if S.stats_every:
        if S.acc_sum is None:
            S.acc_sum = torch.zeros(2, dtype=torch.int64, device=ns.device)
        S.acc_sum[0] += ns.sum()
        S.acc_sum[1] += 1
        if S.n_tree % S.stats_every == 0:
            tot, steps = S.acc_sum.tolist()
            log.warning("atdev %s: tree steps=%d chain steps=%d mean tokens/step %.3f",
                        OPT_NAME, S.n_tree, S.n_chain, tot / max(steps, 1))


def _accept_gpu(sampler, am, input_batch, cu_num_logits, idx_mapping):
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected

    n, r = S.N, input_batch.num_reqs
    if (S.post_fused and r == 1 and input_batch.positions.dim() == 1
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
        if S.post_fused:  # pointer table must exist before capture (HtoD)
            from atdev_serve.tree_post import bt_ptr_table

            bt_ptr_table(S.block_tables, S.tables["gdn_groups"], ag.am.device)
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
        if S.fc_early_on:
            _fc_early_warm(self)
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
        if S.fc_early_on:
            if S.fc_mode == "2":  # launched by fc_post_head() once the head GEMMs are enqueued
                S.fc_early, S.fc_late = None, (self, hidden_states)
            else:
                _fc_early_launch(self, hidden_states)
        S.accepting = True
        try:
            res = orig_sample(self, hidden_states, input_batch, grammar_output)
        finally:
            S.accepting = False
            S.fc_late = None
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
        raise RuntimeError(f"atdev {OPT_NAME}: requires ATDEV_OPT gdn_single_state (A3)")
    num_nodes()
    a3.register()  # builds A3's class and swaps the mixed-batch dispatcher
    if _LAYER_CLS is None:
        _LAYER_CLS = _make_layer_class(a3._LAYER_CLS)
        gdn_mod.QwenGatedDeltaNetAttention.register_oot(_LAYER_CLS)
    _install_hooks()
    log.info("atdev %s: registered (N=%d)", OPT_NAME, num_nodes())
