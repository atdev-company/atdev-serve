"""gdn_single_state under prefix caching (``--mamba-cache-mode align``), opt-in ATDEV_GDN_SS_ALIGN=1.

Why single-state breaks under align
-----------------------------------
vLLM's V2 runner (``MambaSpecDecodeGPUContext``) moves GDN states between
block-table columns with two fused copy kernels:

* pre-copy (before the forward, when a request's running block column advances):
  ``bt[dst_col] <- bt[src_col + token_bias]`` with ``token_bias = num_accepted - 1``;
* post-process (after the step, when a block boundary lies inside the accepted
  tokens): ``bt[dest_col] <- bt[state_idx + bias]``, ``bias`` = index of the verify
  token whose state sits exactly on the boundary.

Both read spec slot ``bias`` = "state after verify token ``bias``". Single-state
(gdn_single_state.py) never materialises that slot: slot 0 holds the state after
token 0 and slot 1 holds the token log (k, v, decay, beta of tokens 1..T-1); the
state after token ``j`` is ``slot0 + replay(log[0:j])``.

What this module does
---------------------
It wraps the two ``MambaSpecDecodeGPUContext`` methods:

    A  materialise  state(slot0 + replay of `bias` log entries) -> bt[col + 2]  (an unused spec slot;
                    must not be the destination: dst column col+1 *is* the log block)
    -  stock copy   (conv states are correct as is; the temporal copy writes garbage into dst)
    B  copy         bt[col + 2] -> bt[dst]  (temporal state of every GDN layer)

A and B decide on the GPU exactly as the stock kernels do (same inputs), so no extra host sync
and no change on steps that do not cross a boundary (one tiny kernel pair with early exit).
Replay arithmetic is the Triton single-state kernel's rank-1 step in fp32, the result is rounded
to the state dtype (bf16) once more than the no-cache path, which keeps the state fp32 across the
replay: cache-on vs cache-off outputs are therefore equal only up to that extra rounding at a
block boundary (every 416 tokens / request), not bitwise.

Requirements: V2 model runner, bf16 state, num_speculative_tokens >= 2 (slot 2 free), not
ATDEV_GDN_STATE=int8.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

logger = init_logger(__name__)

GEOM: dict = {}  # filled by SingleStateGDNAttention.__init__: H, HV, K, V, max_log


def enabled() -> bool:
    return os.environ.get("ATDEV_GDN_SS_ALIGN", "0") == "1"


def note_geometry(h: int, hv: int, k: int, v: int, max_log: int) -> None:
    g = dict(H=h, HV=hv, K=k, V=v, max_log=max_log)
    if GEOM and GEOM != g:
        raise RuntimeError(f"gdn_ss_align: layers with different GDN geometry {GEOM} vs {g}")
    GEOM.update(g)


if triton is not None:

    @triton.jit
    def _rank1(h, k, v, decay, beta):
        # identical op order to gdn_single_state._gdn_rank1_step
        h = h * decay
        delta = (v - tl.sum(h * k[None, :], axis=1)) * beta
        return h + delta[:, None] * k[None, :]

    @triton.jit
    def _decide(
        MODE: tl.constexpr,
        row, idx_map_ptr, x0_ptr, x1_ptr, x2_ptr, BS: tl.constexpr,
    ):
        """-> (active, src_col, dst_col, bias). PRE: x0=src_col, x1=state_idx(dst), x2=bias.
        POST: x0=num_accepted snapshot, x1=state_idx(src), x2=new_num_computed."""
        req = tl.load(idx_map_ptr + row)
        ok = req >= 0
        req = tl.maximum(req, 0)
        if MODE == 0:
            src = tl.load(x0_ptr + req)
            dst = tl.load(x1_ptr + req)
            bias = tl.load(x2_ptr + req)
            act = ok & (src >= 0) & (src != dst) & (bias > 0)
        else:
            a = tl.load(x0_ptr + req)
            src = tl.load(x1_ptr + req)
            nc = tl.load(x2_ptr + req)
            running = nc - a + 1
            aligned = (nc // BS) * BS
            bias = aligned - running
            dst = aligned // BS - 1
            act = ok & (aligned >= running) & (bias > 0)
        return act, src, dst, bias

    @triton.jit
    def _materialize_kernel(
        base_ptr, bstride_ptr, grp_ptr, bt_ptrs_ptr, bt_stride,
        idx_map_ptr, x0_ptr, x1_ptr, x2_ptr, num_reqs,
        MODE: tl.constexpr, BS: tl.constexpr,
        H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BV: tl.constexpr, LOG_BASE: tl.constexpr, BUF: tl.constexpr,
        V_OFF: tl.constexpr, D_OFF: tl.constexpr, B_OFF: tl.constexpr,
    ):
        row = tl.program_id(0)
        layer = tl.program_id(1)
        if row >= num_reqs:
            return
        act, src, dst, bias = _decide(MODE, row, idx_map_ptr, x0_ptr, x1_ptr, x2_ptr, BS)
        if not act:
            return
        base = tl.load(base_ptr + layer)
        bstride = tl.load(bstride_ptr + layer)
        g = tl.load(grp_ptr + layer).to(tl.int64)
        bt = tl.load(bt_ptrs_ptr + g).to(tl.pointer_type(tl.int32)) + row * bt_stride
        s0 = tl.load(bt + src).to(tl.int64)
        s1 = tl.load(bt + src + 1).to(tl.int64)
        st = tl.load(bt + src + 2).to(tl.int64)
        p0 = (base + s0 * bstride).to(tl.pointer_type(tl.bfloat16))
        a1 = base + s1 * bstride
        lg = a1.to(tl.pointer_type(tl.float32))
        par = a1.to(tl.pointer_type(tl.int32))
        pt = (base + st * bstride).to(tl.pointer_type(tl.bfloat16))
        NV: tl.constexpr = V // BV
        G: tl.constexpr = HV // H
        offs_k = tl.arange(0, K)
        for hv in range(0, HV):
            hk = hv // G
            for iv in range(0, NV):
                offs_v = iv * BV + tl.arange(0, BV)
                ptrs = (hv * V + offs_v[:, None]) * K + offs_k[None, :]
                h = tl.load(p0 + ptrs).to(tl.float32)
                p = tl.load(par + hv * NV + iv) & 1
                rbuf = lg + LOG_BASE + p * BUF
                for t in range(0, bias):
                    kk = tl.load(rbuf + (t * H + hk) * K + offs_k)
                    vv = tl.load(rbuf + V_OFF + (t * HV + hv) * V + offs_v)
                    dd = tl.load(rbuf + D_OFF + t * HV + hv)
                    bb = tl.load(rbuf + B_OFF + t * HV + hv)
                    h = _rank1(h, kk, vv, dd, bb)
                tl.store(pt + ptrs, h.to(tl.bfloat16))

    @triton.jit
    def _copy_kernel(
        base_ptr, bstride_ptr, grp_ptr, bt_ptrs_ptr, bt_stride,
        idx_map_ptr, x0_ptr, x1_ptr, x2_ptr, num_reqs,
        MODE: tl.constexpr, BS: tl.constexpr, N: tl.constexpr, BLK: tl.constexpr,
    ):
        row = tl.program_id(0)
        layer = tl.program_id(1)
        if row >= num_reqs:
            return
        act, src, dst, bias = _decide(MODE, row, idx_map_ptr, x0_ptr, x1_ptr, x2_ptr, BS)
        if not act:
            return
        base = tl.load(base_ptr + layer)
        bstride = tl.load(bstride_ptr + layer)
        g = tl.load(grp_ptr + layer).to(tl.int64)
        bt = tl.load(bt_ptrs_ptr + g).to(tl.pointer_type(tl.int32)) + row * bt_stride
        st = tl.load(bt + src + 2).to(tl.int64)
        d = tl.load(bt + dst).to(tl.int64)
        ps = (base + st * bstride).to(tl.pointer_type(tl.bfloat16))
        pd = (base + d * bstride).to(tl.pointer_type(tl.bfloat16))
        offs = tl.arange(0, BLK)
        for i in range(0, N, BLK):
            m = (i + offs) < N
            tl.store(pd + i + offs, tl.load(ps + i + offs, mask=m), mask=m)


def _tables(ctx):
    tb = getattr(ctx, "_atdev_ssa_tb", None)
    if tb is not None:
        return tb
    from vllm.utils.gpu_sync_debug import gpu_sync_allowed

    g = GEOM
    if not g:
        raise RuntimeError("gdn_ss_align: no SingleStateGDNAttention layer was constructed")
    n_state = g["HV"] * g["V"] * g["K"]
    with gpu_sync_allowed():
        cw = ctx.state_conv_widths.cpu()
        sel = torch.nonzero(cw == 0).flatten().to(ctx.state_base_addrs.device)
        inner = ctx.state_inner_sizes[sel].cpu()
        esz = ctx.state_elem_sizes[sel].cpu()
        if not bool((inner == n_state).all()) or not bool((esz == 2).all()):
            raise RuntimeError(
                "gdn_ss_align: temporal states are not all bf16 [HV, V, K] "
                f"(inner {inner.unique().tolist()}, elem {esz.unique().tolist()}); "
                "needs --mamba-ssm-cache-dtype bfloat16")
    from atdev_serve.gdn_single_state import log_layout

    lay = log_layout(g["H"], g["HV"], g["K"], g["V"], 32, g["max_log"])
    tb = dict(
        base=ctx.state_base_addrs[sel].contiguous(),
        bstride=ctx.state_block_strides[sel].contiguous(),
        grp=ctx.state_group_indices[sel].to(torch.int32).contiguous(),
        layers=int(sel.numel()), lay=lay, n_state=n_state,
    )
    ctx._atdev_ssa_tb = tb
    logger.warning("atdev gdn_ss_align: %d GDN layers, geometry %s, log %d fp32/slot", tb["layers"], g, lay["total"])
    return tb


def _launch(ctx, mode: int, phase_copy: bool, num_reqs, idx_map, x0, x1, x2):
    tb = _tables(ctx)
    g, lay = GEOM, tb["lay"]
    grid = (num_reqs, tb["layers"])
    common = (tb["base"], tb["bstride"], tb["grp"], ctx.block_table_ptrs, ctx.block_table_stride_req,
              idx_map, x0, x1, x2, num_reqs)
    if phase_copy:
        _copy_kernel[grid](*common, MODE=mode, BS=ctx.block_size, N=tb["n_state"], BLK=4096, num_warps=4)
    else:
        _materialize_kernel[grid](
            *common, MODE=mode, BS=ctx.block_size, H=g["H"], HV=g["HV"], K=g["K"], V=g["V"], BV=32,
            LOG_BASE=lay["log_base"], BUF=lay["buf"], V_OFF=lay["v_off"], D_OFF=lay["d_off"],
            B_OFF=lay["b_off"], num_warps=4)


def install() -> None:
    from vllm.v1.worker import mamba_utils as mu

    cls = mu.MambaSpecDecodeGPUContext
    if getattr(cls, "_atdev_ssa", False):
        return
    if os.environ.get("ATDEV_GDN_STATE", "native") != "native":
        raise ValueError("ATDEV_GDN_SS_ALIGN does not support ATDEV_GDN_STATE=int8")
    orig_pre = cls.run_fused_precopy
    orig_post = cls.run_fused_postprocess_align

    def run_fused_precopy(self, num_reqs, state_idx_gpu, src_col_gpu, token_bias_gpu, idx_mapping):
        if num_reqs == 0 or not self.is_initialized:
            return orig_pre(self, num_reqs, state_idx_gpu, src_col_gpu, token_bias_gpu, idx_mapping)
        assert idx_mapping is not None, "gdn_ss_align needs the V2 model runner"
        args = (num_reqs, idx_mapping, src_col_gpu, state_idx_gpu, token_bias_gpu)
        _launch(self, 0, False, *args)
        orig_pre(self, num_reqs, state_idx_gpu, src_col_gpu, token_bias_gpu, idx_mapping)
        _launch(self, 0, True, *args)

    def run_fused_postprocess_align(self, num_reqs, num_accepted_tokens_gpu, state_idx_gpu,
                                    new_num_computed_tokens_gpu, idx_mapping):
        if num_reqs == 0 or not self.is_initialized:
            return orig_post(self, num_reqs, num_accepted_tokens_gpu, state_idx_gpu,
                             new_num_computed_tokens_gpu, idx_mapping)
        snap = self.num_accepted_tokens_out
        snap.copy_(num_accepted_tokens_gpu)  # the stock method repeats this copy (same values)
        args = (num_reqs, idx_mapping, snap, state_idx_gpu, new_num_computed_tokens_gpu)
        _launch(self, 1, False, *args)
        orig_post(self, num_reqs, num_accepted_tokens_gpu, state_idx_gpu,
                  new_num_computed_tokens_gpu, idx_mapping)
        _launch(self, 1, True, *args)

    cls.run_fused_precopy = run_fused_precopy
    cls.run_fused_postprocess_align = run_fused_postprocess_align
    cls._atdev_ssa = True
    logger.warning("atdev gdn_ss_align: MambaSpecDecodeGPUContext pre-copy / post-process wrapped")
