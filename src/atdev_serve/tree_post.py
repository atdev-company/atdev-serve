"""Fused c=1 tree post-verify prologue (dflash_tree ATDEV_TREE_POST_FUSED=1).

One single-program Triton launch replaces the 8 tiny eager kernels between the verify head's argmax
and the compaction kernels of a c=1 tree step (results/prof-DG-stack e1: _tree_accept_kernel,
_get_num_sampled_and_rejected_kernel, _tree_pos_kernel, the torch.stack of the 10 GDN groups' state
indices, and the 4 elementwise ops of the drafter context gather index):

  1. greedy tree accept           == gdn_tree._tree_accept_kernel (same walk, same -1 padding)
  2. num_sampled / num_rejected   == vLLM 0.31 input_batch._get_num_sampled_and_rejected_kernel
  3. positions back to chain      == gdn_tree._tree_pos_kernel (SIGN = -1)
  4. state_idx [G, 1, 2]          == torch.stack([block_tables[g][:1, :2] for g in gdn_groups])
  5. drafter gather index [N]     == torch.where(ar < ns[0], path[0, :N].long(), ar)

All integer work, so "identical" is exact (tests/test_tree_accept_graph.py compares every output and
every cache / position tensor against the unfused path, eager and graphed).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _tree_post_c1_kernel(
    parent_ptr, tokens_ptr, argmax_ptr,
    path_ptr, sampled_ptr, nsamp_ptr, nrej_ptr,
    seq_lens_ptr, cu_ptr, idx_ptr, prefill_len_ptr,
    pos_ptr, start_ptr, dep_ptr,
    bt_ptrs, si_ptr, gather_ptr,
    N, G,
    NP: tl.constexpr, MAX_DEPTH: tl.constexpr, OUT_COLS: tl.constexpr, GP: tl.constexpr,
):
    rows = tl.arange(0, NP)
    valid = rows < N
    # ---- 1. tree accept (gdn_tree._tree_accept_kernel, r = 0) --------------------------------------
    par = tl.load(parent_ptr + rows, mask=valid, other=-2)
    toks = tl.load(tokens_ptr + rows, mask=valid, other=-1).to(tl.int64)
    cols = tl.arange(0, OUT_COLS)
    tl.store(path_ptr + cols, -1)
    tl.store(sampled_ptr + cols, -1)
    node = tl.full([], 0, tl.int32)
    alive = tl.full([], 1, tl.int32)
    cnt = tl.full([], 1, tl.int32)
    tl.store(path_ptr, 0)
    for d in range(0, MAX_DEPTH):
        want = tl.load(argmax_ptr + node).to(tl.int64)
        hit = (par == node) & (toks == want) & (rows > 0) & valid
        nxt = tl.max(tl.where(hit, rows, -1), axis=0).to(tl.int32)
        go = (alive != 0) & (nxt >= 0)
        tl.store(path_ptr + cnt, nxt, mask=go)
        tl.store(sampled_ptr + cnt - 1, want, mask=go)
        cnt = tl.where(go, cnt + 1, cnt)
        node = tl.where(go, nxt, node)
        alive = go.to(tl.int32)
    bonus = tl.load(argmax_ptr + node).to(tl.int64)
    tl.store(sampled_ptr + cnt - 1, bonus)
    # ---- 2. num_sampled / num_rejected (vLLM _get_num_sampled_and_rejected_kernel, batch_idx 0) ------
    req_state_idx = tl.load(idx_ptr)
    seq_len = tl.load(seq_lens_ptr)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    is_chunked_prefilling = seq_len < prefill_len
    num_sampled = tl.where(is_chunked_prefilling, 0, cnt)
    tl.store(nsamp_ptr, num_sampled)
    num_logits = tl.load(cu_ptr + 1) - tl.load(cu_ptr)
    num_rejected = num_logits - num_sampled
    num_rejected = tl.where(is_chunked_prefilling, 0, num_rejected)
    tl.store(nrej_ptr, num_rejected)
    # ---- 3. positions[start + i] -= depth[i] - i (gdn_tree._tree_pos_kernel, SIGN -1, 1 row) --------
    st = tl.load(start_ptr).to(tl.int64)
    dlt = (tl.load(dep_ptr + rows, mask=valid, other=0) - rows).to(tl.int64) * -1
    pp = pos_ptr + st + rows
    x = tl.load(pp, mask=valid, other=0)
    tl.store(pp, (x + dlt).to(pos_ptr.dtype.element_ty), mask=valid)
    # ---- 4. state_idx[g, 0, :2] = block_tables[g][0, :2] ---------------------------------------------
    gg = tl.arange(0, GP)
    gm = gg < G
    bp = tl.load(bt_ptrs + gg, mask=gm, other=0)
    for j in tl.static_range(2):
        v = tl.load(bp.to(tl.pointer_type(tl.int32)) + j, mask=gm, other=0)
        tl.store(si_ptr + gg * 2 + j, v, mask=gm)
    # ---- 5. gather[i] = path[i] if i < num_sampled else i ---------------------------------------------
    tl.debug_barrier()  # path written by scattered scalar stores above
    pv = tl.load(path_ptr + rows, mask=valid, other=0).to(tl.int64)
    gat = tl.where(rows.to(tl.int64) < num_sampled.to(tl.int64), pv, rows.to(tl.int64))
    tl.store(gather_ptr + rows, gat, mask=valid)


_BT_CACHE: dict = {}


def bt_ptr_table(block_tables, groups, device) -> torch.Tensor:
    """int64 [G] data pointers of block_tables[g] (cached by pointer tuple: the runner's
    input_block_tables are persistent). Must be called outside graph capture on first use."""
    key = tuple(block_tables[g].data_ptr() for g in groups)
    t = _BT_CACHE.get(key)
    if t is None:
        for g in groups:
            assert block_tables[g].dtype == torch.int32 and block_tables[g].stride(-1) == 1
            assert block_tables[g].shape[-1] >= 2
        t = _BT_CACHE[key] = torch.tensor(key, dtype=torch.int64, device=device)
    return t


def tree_post_c1(*, parent, tokens, argmax, max_depth, seq_lens, cu_num_logits, idx_mapping, prefill_len,
                 positions, query_start_loc, depth, block_tables, groups):
    """r = 1 tree step. Returns (path [1, N] int32, sampled [1, N] int64, num_sampled [1] int32,
    num_rejected [1] int32, state_idx [G, 1, 2] int32, gather [N] int64); positions updated in place.
    Same values as gdn_tree.tree_accept + get_num_sampled_and_rejected + tree_shift_positions(-1) +
    torch.stack(state idx) + the gather where()."""
    n = parent.shape[-1]
    dev = parent.device
    assert parent.dim() == 1 and tokens.dim() == 1 and argmax.dim() == 1
    assert parent.stride(0) == 1 and tokens.stride(0) == 1 and argmax.stride(0) == 1
    assert positions.dim() == 1 and positions.stride(0) == 1 and depth.stride(0) == 1
    cp = triton.next_power_of_2(n)
    assert n >= max_depth + 1
    G = len(groups)
    path = torch.empty((1, cp), dtype=torch.int32, device=dev)
    sampled = torch.empty((1, cp), dtype=torch.int64, device=dev)
    nsamp = torch.empty(1, dtype=torch.int32, device=dev)
    nrej = torch.empty(1, dtype=torch.int32, device=dev)
    si = torch.empty((G, 1, 2), dtype=torch.int32, device=dev)
    gather = torch.empty(n, dtype=torch.int64, device=dev)
    _tree_post_c1_kernel[(1,)](
        parent, tokens, argmax, path, sampled, nsamp, nrej,
        seq_lens, cu_num_logits, idx_mapping, prefill_len,
        positions, query_start_loc, depth,
        bt_ptr_table(block_tables, groups, dev), si, gather, n, G,
        NP=triton.next_power_of_2(n), MAX_DEPTH=max_depth, OUT_COLS=cp,
        GP=triton.next_power_of_2(max(G, 2)), num_warps=1)
    return path[:, :n], sampled[:, :n], nsamp, nrej, si, gather


@triton.jit
def _gather_rows_kernel(src_ptr, src_stride, dst_ptr, dst_stride, idx_ptr, H, BLK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLK + tl.arange(0, BLK)
    m = c < H
    i = tl.load(idx_ptr + r).to(tl.int64)
    tl.store(dst_ptr + r * dst_stride + c, tl.load(src_ptr + i * src_stride + c, mask=m), mask=m)


def gather_rows(src: torch.Tensor, idx: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """out[r] = src[idx[r]] (pure copy, == torch.index_select / src[idx]); one launch (~3 us for
    16 x 5120 bf16 vs ~9 us for the stock copy + gather + copy of the drafter context rows)."""
    n, h = out.shape
    assert src.dim() == 2 and src.shape[1] == h and src.stride(1) == 1 and out.stride(1) == 1
    assert src.dtype == out.dtype and idx.numel() == n
    blk = 1024
    _gather_rows_kernel[(n, triton.cdiv(h, blk))](src, src.stride(0), out, out.stride(0), idx, h,
                                                  BLK=blk, num_warps=4)
    return out
