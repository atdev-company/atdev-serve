"""Fast GDN attention metadata for pure spec-decode batches (ATDEV_OPT=gdn_meta_fast).

At c=1 the phase between the drafter graph and the next target graph (~60 eager GPU ops) is dominated
by the stock GDN metadata builder. Qwen3.8's 48 GDN layers sit in 10 KV-cache groups; group 0 runs
``build()`` (12 GPU ops: a 1-byte HtoD of the spec mask, an unused ``query_lens`` sub, an ``arange``,
two CPU-mask gathers = 8-byte HtoD + index kernel each, five staging copies into the FULL-graph buffers)
and the other 9 run ``update_block_table()`` (CPU-mask gather + staging copy = 3 ops each): 39 of the
60 ops, >100 us busy plus the launch gaps between them.

When every row of the (padded) batch is a spec-decode row (every c=1 verify step), all of that metadata is
a fixed function of (n rows, k+1, the gathered block tables, query_start_loc, num_accepted_tokens):

  * group 0's ``build()`` launches ONE Triton kernel that writes every group's staged state indices
    straight from the runner's gathered block tables (fixed ``input_block_tables`` buffers) plus the four
    batch-level staging buffers (spec mask, token index arange, query_start_loc, num_accepted);
    ``update_block_table`` then costs zero GPU ops. The GDN builders are discovered during the first two
    steps (stock path), so the fused path starts at step 3.
  * otherwise, when the batch is still all spec rows, ``update_block_table``'s CPU-mask gather
    ``blk_table[masks, :k+1]`` is the contiguous slice ``blk_table[:n, :k+1]`` (a view): only the staging
    copy remains.

The block tables are stashed from ``build_attn_metadata(block_tables=...)`` on every call (real, dummy and
capture builds); the fused path checks that each group's block table is the tabled buffer (pointer,
stride, rows >= bs, cols >= k+1) and ``update_block_table`` re-checks the group's pointer before trusting
the fused staging.

Metadata only: the kernels read the same staged tensors with the same values, outputs are bit-identical.
Everything else (prefill in batch, mixed decode rows, mamba_cache_mode=align, eager/non-FULL paths, capture)
takes the stock methods.
"""
from __future__ import annotations

import functools
import logging
from dataclasses import replace

import torch

OPT_NAME = "gdn_meta_fast"
SUPPORTED_VLLM_VERSION = "0.31.0"
log = logging.getLogger("atdev_serve")


class _State:
    def __init__(self) -> None:
        self.block_tables: tuple | None = None  # runner's gathered per-group block tables (fixed buffers)
        self.builders: dict[int, object] = {}  # kv-cache group id -> GDN builder (discovered)
        self.seen_step: dict[int, object] = {}
        self.step_seen = 0  # steps in which group-0 build ran the stock path (discovery)
        self.ready = False
        self.tabs: dict | None = None  # pointer / stride tables for the fused kernel
        self.staged_step = -1  # id of the metadata object whose staging the fused kernel wrote
        self.n_fast1 = self.n_fast2 = self.n_stock = self.n_build = 0
        self.logged = set()


S = _State()
_NULL = None  # NULL_BLOCK_ID, bound at register


# ---------------------------------------------------------------------------
# fused staging kernel
# ---------------------------------------------------------------------------


def _make_kernel():
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _load_ptr(ptr_to_ptr):
        p = tl.load(ptr_to_ptr)
        return tl.multiple_of(tl.cast(p, tl.pointer_type(tl.int32)), 16)

    @triton.jit
    def _gdn_meta_stage_kernel(
        stage_ptrs, stage_strides, bt_ptrs, bt_strides,  # [G] int64 tables
        mask_stage_ptr, tok_stage_ptr, qsl_stage_ptr, acc_stage_ptr,  # group-0 batch-level staging
        qsl_ptr, acc_ptr,  # inputs: query_start_loc [>= n+1] int32, num_accepted [>= bs] int32
        n, bs, S, G, null_id,
        K1: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        if pid < G:
            # state indices of group pid: rows < n copied from the gathered block table, rows n..bs-1 NULL
            dst = _load_ptr(stage_ptrs + pid)
            src = _load_ptr(bt_ptrs + pid)
            ds = tl.load(stage_strides + pid)
            ss = tl.load(bt_strides + pid)
            cm = cols < K1
            for r in range(0, bs):
                if r < n:
                    v = tl.load(src + r * ss + cols, mask=cm, other=0)
                else:
                    v = tl.zeros([BLOCK], dtype=tl.int32) + null_id
                tl.store(dst + r * ds + cols, v, mask=cm)
        else:
            rows = cols
            rm = rows < bs
            # spec_sequence_masks[:bs] = row < n
            tl.store(mask_stage_ptr + rows, (rows < n).to(tl.int8), mask=rm)
            # num_accepted_tokens[:bs] = input (already padded with 1 by the model state); rows >= n -> 1
            acc = tl.load(acc_ptr + rows, mask=rows < n, other=1)
            tl.store(acc_stage_ptr + rows, acc, mask=rm)
            # spec_query_start_loc[:bs+1] = qsl[:n+1], then qsl[n] for the padded rows
            qm = rows < bs + 1
            q = tl.load(qsl_ptr + tl.minimum(rows, n), mask=qm, other=0)
            tl.store(qsl_stage_ptr + rows, q, mask=qm)
            # spec_token_indx[:S] = arange(S)
            for i in range(0, S, BLOCK):
                t = i + cols
                tl.store(tok_stage_ptr + t, t, mask=t < S)

    return _gdn_meta_stage_kernel


_KERNEL = None


def _tables(dev):
    """Pointer/stride tables over the discovered builders (group order) and their block tables."""
    gids = sorted(S.builders)
    bts = [S.block_tables[g] for g in gids]
    stages = [S.builders[g].spec_state_indices_tensor for g in gids]
    for t in stages + bts:
        assert t.dtype == torch.int32 and t.stride(1) == 1, (t.dtype, t.stride())
    i64 = torch.int64
    return {
        "gids": gids,
        "stage_ptrs": torch.tensor([t.data_ptr() for t in stages], dtype=i64, device=dev),
        "stage_strides": torch.tensor([t.stride(0) for t in stages], dtype=i64, device=dev),
        "bt_ptrs": torch.tensor([t.data_ptr() for t in bts], dtype=i64, device=dev),
        "bt_strides": torch.tensor([t.stride(0) for t in bts], dtype=i64, device=dev),
        "bt_data_ptrs": tuple(t.data_ptr() for t in bts),
        "bt_strides_h": tuple(t.stride(0) for t in bts),
        "bt_ptr_of": {g: t.data_ptr() for g, t in zip(gids, bts)},
    }


def _group_of(block_table) -> int | None:
    if S.block_tables is None:
        return None
    p = block_table.data_ptr()
    for g, bt in enumerate(S.block_tables):
        if bt.data_ptr() == p:
            return g
    return None


def _all_spec(self, m, num_accepted_tokens, num_decode_draft_tokens_cpu):
    """(n, total_tokens, uniform_len) when the whole padded batch is spec-decode rows on the FULL path."""
    if (not self.use_spec_decode or num_decode_draft_tokens_cpu is None or num_accepted_tokens is None
            or self.vllm_config.cache_config.mamba_cache_mode == "align"
            or getattr(self, "mamba_aligned_state_indices", None) is not None):
        return None
    masks = num_decode_draft_tokens_cpu >= 0
    n = int(masks.sum())
    if n == 0 or n != m.num_reqs or int(num_decode_draft_tokens_cpu.sum()) == 0:
        return None
    qsl_cpu = m.query_start_loc_cpu
    total = int(qsl_cpu[-1])
    if not self._stage_spec_decode(0, 0, n, total):
        return None
    ql = qsl_cpu[1:] - qsl_cpu[:-1]
    uniform = int(ql[0]) if int(ql[0]) > 0 and bool(torch.all(ql == ql[0])) else None
    return n, total, uniform, masks


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def register() -> None:
    global _NULL, _KERNEL
    import vllm
    from vllm.v1.attention.backends import gdn_attn as ga
    from vllm.v1.worker.gpu import model_runner as mr

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written against vLLM {SUPPORTED_VLLM_VERSION}, "
                           f"found {vllm.__version__}")
    cls = ga.GDNAttentionMetadataBuilder
    if getattr(cls.update_block_table, "_atdev_meta_fast", False):
        return
    _NULL = ga.NULL_BLOCK_ID
    _KERNEL = _make_kernel()
    orig_ubt = cls.update_block_table
    orig_build = cls.build

    # --- runner: stash the gathered per-group block tables (fixed buffers) -------------------------------
    orig_pa = mr.GPUModelRunner.prepare_attn

    @functools.wraps(orig_pa)
    def prepare_attn(self, input_batch):
        out = orig_pa(self, input_batch)
        S.block_tables = out[0]
        return out

    prepare_attn._atdev_meta_fast = True
    mr.GPUModelRunner.prepare_attn = prepare_attn

    # --- discovery: the set of GDN builders seen in a step becomes the registry once it repeats -----------
    from vllm.v1.worker.gpu.model_states import mamba_hybrid as mh

    orig_bam = mh.build_attn_metadata

    @functools.wraps(orig_bam)
    def build_attn_metadata(*a, **kw):
        # the block tables of THIS call (real steps, dummy runs and FULL-graph capture alike; the
        # prepare_attn stash alone misses dummy / capture builds)
        bts = kw.get("block_tables")
        if bts is not None:
            S.block_tables = tuple(bts)
        S.seen_step = {}
        out = orig_bam(*a, **kw)
        if S.seen_step:
            if S.seen_step.keys() == S.builders.keys() and not S.ready and S.block_tables is not None:
                S.ready = True
                S.tabs = _tables(S.block_tables[0].device)
                log.warning("atdev %s: %d GDN builders registered (groups %s)", OPT_NAME,
                            len(S.builders), sorted(S.builders))
            S.builders = dict(S.seen_step)
        return out

    mh.build_attn_metadata = build_attn_metadata

    # --- build (group 0) ---------------------------------------------------------------------------------
    @functools.wraps(orig_build)
    def build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens=None,
              num_decode_draft_tokens_cpu=None, fast_build=False):
        m = common_attn_metadata
        g = _group_of(m.block_table_tensor)
        if g is not None:
            S.seen_step[g] = self
        spec = _all_spec(self, m, num_accepted_tokens, num_decode_draft_tokens_cpu)
        if spec is None or g is None or not S.ready or S.tabs is None or g not in S.builders:
            # stock path; discovery of the GDN builders happens in update_block_table / here
            S.n_stock += 1
            md = orig_build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens,
                            num_decode_draft_tokens_cpu, fast_build)
            S.staged_step = -1
            return md
        n, total, uniform, masks = spec
        bs = m.num_reqs
        k1 = self.num_spec + 1
        S_tok = min(n * k1, total)
        qsl = m.query_start_loc
        dev = qsl.device
        t = S.tabs
        if (t["bt_data_ptrs"] != tuple(S.block_tables[gg].data_ptr() for gg in t["gids"])
                or t["bt_strides_h"] != tuple(S.block_tables[gg].stride(0) for gg in t["gids"])):
            S.tabs = t = _tables(dev)  # block tables re-gathered into other buffers (should not happen)
        if (m.block_table_tensor.data_ptr() != S.block_tables[g].data_ptr()
                or any(S.block_tables[gg].shape[0] < bs or S.block_tables[gg].shape[1] < k1
                       for gg in t["gids"])):
            S.n_stock += 1
            S.staged_step = -1
            return orig_build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens,
                              num_decode_draft_tokens_cpu, fast_build)
        G = len(t["gids"])
        block = max(16, 1 << (max(k1, bs + 1) - 1).bit_length())
        _KERNEL[(G + 1,)](
            t["stage_ptrs"], t["stage_strides"], t["bt_ptrs"], t["bt_strides"],
            self.spec_sequence_masks.view(torch.int8), self.spec_token_indx, self.spec_query_start_loc,
            self.num_accepted_tokens,
            qsl, num_accepted_tokens, n, bs, S_tok, G, _NULL, K1=k1, BLOCK=block, num_warps=1)
        S.n_build += 1
        S.n_fast2 += 1
        if "fused" not in S.logged:
            S.logged.add("fused")
            log.warning("atdev %s: fused all-spec metadata ENGAGED (n=%d, bs=%d, k+1=%d, %d GDN groups)",
                        OPT_NAME, n, bs, k1, G)
        md = ga.GDNAttentionMetadata(
            num_prefills=0, num_prefill_tokens=0, num_decodes=0, num_decode_tokens=0,
            num_spec_decodes=n, num_spec_decode_tokens=total, num_actual_tokens=m.num_actual_tokens,
            has_initial_state=None,
            spec_query_start_loc=self.spec_query_start_loc[: bs + 1],
            non_spec_query_start_loc=None,
            spec_state_indices_tensor=self.spec_state_indices_tensor[:bs],
            non_spec_state_indices_tensor=None,
            spec_sequence_masks=self.spec_sequence_masks[:bs],
            spec_sequence_masks_cpu=masks,
            spec_token_indx=self.spec_token_indx[:S_tok],
            non_spec_token_indx=self.non_spec_token_indx[:0],
            num_accepted_tokens=self.num_accepted_tokens[:bs],
            uniform_spec_sequence_length=uniform,
        )
        S.staged_step = id(md)
        return md

    build._atdev_meta_fast = True
    cls.build = build

    # --- update_block_table (groups 1..) ------------------------------------------------------------------
    @functools.wraps(orig_ubt)
    def update_block_table(self, metadata, blk_table, slot_mapping):
        m = metadata
        g = _group_of(blk_table)
        if g is not None:
            S.seen_step[g] = self
        if (S.staged_step == id(m) and m.spec_state_indices_tensor is not None
                and S.tabs is not None and g is not None
                and S.tabs["bt_ptr_of"].get(g) == blk_table.data_ptr()):
            # the fused kernel already wrote this group's staged state indices: zero GPU ops
            bs = m.spec_state_indices_tensor.shape[0]
            return replace(m, spec_state_indices_tensor=self.spec_state_indices_tensor[:bs],
                           non_spec_state_indices_tensor=None, prefill_state_indices=None)
        masks = m.spec_sequence_masks_cpu
        if (masks is None or m.num_prefills > 0 or m.num_decodes > 0
                or m.spec_state_indices_tensor is None
                or self.vllm_config.cache_config.mamba_cache_mode == "align"
                or masks.numel() != m.num_spec_decodes or not bool(masks.all())
                or not self._stage_spec_decode(m.num_prefills, m.num_decodes, m.num_spec_decodes,
                                               m.num_spec_decode_tokens)):
            return orig_ubt(self, metadata, blk_table, slot_mapping)
        # all rows are spec rows -> the CPU-mask gather is the contiguous slice (no HtoD/index)
        n = m.num_spec_decodes
        bs = m.spec_state_indices_tensor.shape[0]
        stage = self.spec_state_indices_tensor
        stage[:n].copy_(blk_table[:n, : self.num_spec + 1], non_blocking=True)
        spec_indices = stage[:bs]
        spec_indices[n:].fill_(_NULL)
        S.n_fast1 += 1
        if "l1" not in S.logged:
            S.logged.add("l1")
            log.warning("atdev %s: all-spec-decode slice path ENGAGED (n=%d rows, k+1=%d, staged bs=%d)",
                        OPT_NAME, n, self.num_spec + 1, bs)
        return replace(m, spec_state_indices_tensor=spec_indices, non_spec_state_indices_tensor=None,
                       prefill_state_indices=None)

    update_block_table._atdev_meta_fast = True
    cls.update_block_table = update_block_table
    log.info("atdev %s: registered", OPT_NAME)
