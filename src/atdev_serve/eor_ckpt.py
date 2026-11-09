"""ATDEV_EOR_CKPT=1: end-of-reply GDN/conv state checkpoint for prefix caching (align mode).

Problem: vLLM 0.31 hybrid prefix caching writes recurrent (GDN) state only at block-aligned positions
inside *prefill*, so the reply a request generates is never reusable as prefix by the next turn.

Design:
* Hash unit = 16 tokens (``--prefix-match-unit 16``).  A hit at position x needs the GDN + conv state after
  exactly x tokens, x % 16 == 0.
* Worker: after every step (hook on ``MambaSpecDecodeGPUContext.run_fused_postprocess_align``) a request whose
  accepted tokens cross a 16-boundary B writes the exact state after B tokens (slot-0 + replay of ``bias``
  log entries, same arithmetic as gdn_ss_align) and the matching conv window into a private *ring* of
  ``R`` vLLM blocks per mamba group (ring index (B/16) % R).  R = 2, or 3 when the EAGLE last-block drop is on.
  Ring blocks are ordinary pool blocks owned by the request (not in its block table).
* Scheduler: when a request finishes (stop / length), the state for ``Ls = floor16(num_tokens - 1)`` (minus one
  unit under the EAGLE drop) is in ring[(Ls/16) % R]; that block is registered under the prefix hash of the
  first Ls tokens (hash of the 16-unit chain, exactly what a later prompt's lookup computes) and handed back to
  the pool, the other ring blocks are released.  Attention groups (full / sliding-window / drafter) get an
  alias hash on the request's partial last block (their KV is already complete), no copy.
* Off-by-one: the last sampled token (EOS / length) is never forwarded, so computed tokens = num_tokens - 1.
  Requests that run one wasted step under async scheduling can only overwrite ring slot (B+16) % R, never the
  slots of Ls or Ls-16 (a step advances at most 16 tokens), hence the ring.
* Only 16-boundaries crossed during decode are snapshotted (device-side ``vf`` threshold = prefill end);
  boundaries inside the prompt are already served by the prompt-tail machinery.

Requires: uni executor (scheduler and worker in one process), V2 model runner, gdn_ss_align (single-state GDN,
bf16 state), align mode, ``--prefix-match-unit 16``.  All edits are monkey patches; stock behaviour with the flag off.
"""

from __future__ import annotations

import logging
import os
import threading

import torch

log = logging.getLogger("atdev_serve")
UNIT = 16                                            # prefix-hash unit
SNAP = int(os.environ.get("ATDEV_EOR_SNAP", "16"))    # snapshot spacing (multiple of UNIT)
assert SNAP % UNIT == 0
INT_MAX = 2**31 - 1
RESERVE_BLOCKS = int(os.environ.get("ATDEV_EOR_RESERVE", "32"))
DEBUG = os.environ.get("ATDEV_EOR_DEBUG", "0") == "1"
NOKERNEL = os.environ.get("ATDEV_EOR_NOKERNEL", "0") == "1"   # profiling: allocate rings, skip the snapshot kernels
TIMING = os.environ.get("ATDEV_EOR_TIME", "0") == "1"          # profiling: cuda-event time of the snapshot kernels
NOALLOC = os.environ.get("ATDEV_EOR_NOALLOC", "0") == "1"      # profiling: kernels run on a dummy request set (no rings)
HEADROOM = float(os.environ.get("ATDEV_EOR_HEADROOM", "0"))     # keep this fraction of the pool free (incl. evictable) when allocating rings
KEEP = int(os.environ.get("ATDEV_EOR_KEEP", "0"))                 # keep at most this many registered checkpoints (FIFO); older ones are uncached
PLANNED = os.environ.get("ATDEV_EOR_PLANNED", "0") == "1"       # two-stage snapshot: plan kernel + kernels over the active rows only
NACT_CTA = int(os.environ.get("ATDEV_EOR_NACT", "8"))
NOREG = os.environ.get("ATDEV_EOR_NOREG", "0") == "1"         # profiling: allocate rings, never register
NOALIAS = os.environ.get("ATDEV_EOR_NOALIAS", "0") == "1"     # profiling: register mamba blocks only
RING = int(os.environ.get("ATDEV_EOR_RING", "0"))               # force ring size (0 = auto)

import collections
REGISTERED: collections.deque = collections.deque()
LEDGER: dict = {}
CFG: dict = {}
STATS = dict(alloc=0, alloc_skip=0, reg=0, reg_short=0, reg_status=0, preempt_or_abort=0, attn_alias=0, snaps_calls=0)
_LOCK = threading.Lock()

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


def enabled() -> bool:
    return os.environ.get("ATDEV_EOR_CKPT", "0") == "1"


# ----------------------------------------------------------------------------------------------
# worker kernels
# ----------------------------------------------------------------------------------------------
if triton is not None:

    @triton.jit
    def _rank1(h, k, v, decay, beta):
        h = h * decay
        delta = (v - tl.sum(h * k[None, :], axis=1)) * beta
        return h + delta[:, None] * k[None, :]

    @triton.jit
    def _decide(row, idx_map_ptr, acc_ptr, sidx_ptr, nc_ptr, vf_ptr, ids_ptr, g,
                R: tl.constexpr, G: tl.constexpr, UNIT: tl.constexpr):
        req = tl.load(idx_map_ptr + row)
        ok = req >= 0
        req = tl.maximum(req, 0)
        a = tl.load(acc_ptr + req).to(tl.int32)
        nc = tl.load(nc_ptr + req).to(tl.int32)
        src = tl.load(sidx_ptr + req).to(tl.int32)
        vf = tl.load(vf_ptr + req)
        running = nc - a + 1
        b_pos = (nc // UNIT) * UNIT
        bias = b_pos - running
        ring = (b_pos // UNIT) % R
        dst = tl.load(ids_ptr + (req * R + ring) * G + g)
        act = ok & (bias >= 0) & (b_pos > vf) & (dst >= 0)
        return act, src, bias, dst

    @triton.jit
    def _snap_temporal_kernel(
        base_ptr, bstride_ptr, grp_ptr, bt_ptrs_ptr, bt_stride,
        idx_map_ptr, acc_ptr, sidx_ptr, nc_ptr, vf_ptr, ids_ptr, num_reqs,
        R: tl.constexpr, G: tl.constexpr, UNIT: tl.constexpr,
        H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BV: tl.constexpr, LOG_BASE: tl.constexpr, BUF: tl.constexpr,
        V_OFF: tl.constexpr, D_OFF: tl.constexpr, B_OFF: tl.constexpr,
    ):
        row = tl.program_id(0)
        layer = tl.program_id(1)
        hv = tl.program_id(2)
        if row >= num_reqs:
            return
        g = tl.load(grp_ptr + layer).to(tl.int32)
        act, src, bias, dst = _decide(row, idx_map_ptr, acc_ptr, sidx_ptr, nc_ptr, vf_ptr, ids_ptr, g, R, G, UNIT)
        if not act:
            return
        base = tl.load(base_ptr + layer)
        bstride = tl.load(bstride_ptr + layer)
        bt = tl.load(bt_ptrs_ptr + g.to(tl.int64)).to(tl.pointer_type(tl.int32)) + row * bt_stride
        s0 = tl.load(bt + src).to(tl.int64)
        s1 = tl.load(bt + src + 1).to(tl.int64)
        p0 = (base + s0 * bstride).to(tl.pointer_type(tl.bfloat16))
        a1 = base + s1 * bstride
        lg = a1.to(tl.pointer_type(tl.float32))
        par = a1.to(tl.pointer_type(tl.int32))
        pt = (base + dst.to(tl.int64) * bstride).to(tl.pointer_type(tl.bfloat16))
        NV: tl.constexpr = V // BV
        GQ: tl.constexpr = HV // H
        offs_k = tl.arange(0, K)
        hk = hv // GQ
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
    def _snap_conv_kernel(
        base_ptr, bstride_ptr, grp_ptr, cw_ptr, inner_ptr, rows_ptr, rstride_ptr, bt_ptrs_ptr, bt_stride,
        idx_map_ptr, acc_ptr, sidx_ptr, nc_ptr, vf_ptr, ids_ptr, num_reqs,
        R: tl.constexpr, G: tl.constexpr, UNIT: tl.constexpr,
        DIM_FIRST: tl.constexpr, BLK: tl.constexpr,
    ):
        row = tl.program_id(0)
        layer = tl.program_id(1)
        if row >= num_reqs:
            return
        g = tl.load(grp_ptr + layer).to(tl.int32)
        act, src, bias, dst = _decide(row, idx_map_ptr, acc_ptr, sidx_ptr, nc_ptr, vf_ptr, ids_ptr, g, R, G, UNIT)
        if not act:
            return
        base = tl.load(base_ptr + layer)
        bstride = tl.load(bstride_ptr + layer)
        cw = tl.load(cw_ptr + layer)
        bt = tl.load(bt_ptrs_ptr + g.to(tl.int64)).to(tl.pointer_type(tl.int32)) + row * bt_stride
        sb = tl.load(bt + src).to(tl.int64)
        src_addr = base + sb * bstride
        dst_addr = base + dst.to(tl.int64) * bstride
        offs = tl.arange(0, BLK)
        ntok = cw - bias
        if DIM_FIRST:
            # [dim, state_len]: row r token t at r*row_stride + t*2 bytes (bf16)
            nrows = tl.load(rows_ptr + layer)
            rstride = tl.load(rstride_ptr + layer)
            for t in range(0, ntok):
                for r0 in range(0, nrows, BLK):
                    rr = r0 + offs
                    m = rr < nrows
                    sp = (src_addr + rr * rstride + (t + bias) * 2).to(tl.pointer_type(tl.uint16))
                    dp = (dst_addr + rr * rstride + t * 2).to(tl.pointer_type(tl.uint16))
                    tl.store(dp, tl.load(sp, mask=m), mask=m)
        else:
            inner = tl.load(inner_ptr + layer)  # elements per token
            tok_bytes = inner * 2
            total = ntok.to(tl.int64) * inner
            sp0 = (src_addr + bias.to(tl.int64) * tok_bytes).to(tl.pointer_type(tl.uint16))
            dp0 = dst_addr.to(tl.pointer_type(tl.uint16))
            for i in range(0, total, BLK):
                m = (i + offs) < total
                tl.store(dp0 + i + offs, tl.load(sp0 + i + offs, mask=m), mask=m)

    @triton.jit
    def _plan_kernel(idx_map_ptr, acc_ptr, sidx_ptr, nc_ptr, vf_ptr, num_reqs, plan_ptr, cnt_ptr,
                     UNIT: tl.constexpr, BLOCK: tl.constexpr):
        """one CTA: compact list of rows that cross a 16-boundary in the decode region.
        plan[0..cnt) = row; plan[BLOCK + i] = src col, plan[2*BLOCK + i] = bias, plan[3*BLOCK + i] = ring-pos (B // UNIT), plan[4*BLOCK + i] = req"""
        offs = tl.arange(0, BLOCK)
        m = offs < num_reqs
        req = tl.load(idx_map_ptr + offs, mask=m, other=-1)
        ok = m & (req >= 0)
        r = tl.maximum(req, 0)
        a = tl.load(acc_ptr + r, mask=ok, other=1).to(tl.int32)
        nc = tl.load(nc_ptr + r, mask=ok, other=0).to(tl.int32)
        src = tl.load(sidx_ptr + r, mask=ok, other=0).to(tl.int32)
        vf = tl.load(vf_ptr + r, mask=ok, other=2147483647)
        running = nc - a + 1
        b_pos = (nc // UNIT) * UNIT
        bias = b_pos - running
        act = ok & (bias >= 0) & (b_pos > vf)
        pos = tl.cumsum(act.to(tl.int32), axis=0) - 1
        tl.store(plan_ptr + pos, offs, mask=act)
        tl.store(plan_ptr + BLOCK + pos, src, mask=act)
        tl.store(plan_ptr + 2 * BLOCK + pos, bias, mask=act)
        tl.store(plan_ptr + 3 * BLOCK + pos, b_pos // UNIT, mask=act)
        tl.store(plan_ptr + 4 * BLOCK + pos, r, mask=act)
        tl.store(cnt_ptr, tl.sum(act.to(tl.int32), axis=0))

    @triton.jit
    def _snap_temporal_planned(
        base_ptr, bstride_ptr, grp_ptr, bt_ptrs_ptr, bt_stride, plan_ptr, cnt_ptr, ids_ptr,
        R: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr, NACT: tl.constexpr,
        H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BV: tl.constexpr, LOG_BASE: tl.constexpr, BUF: tl.constexpr,
        V_OFF: tl.constexpr, D_OFF: tl.constexpr, B_OFF: tl.constexpr,
    ):
        slot = tl.program_id(0)
        layer = tl.program_id(1)
        hv = tl.program_id(2)
        cnt = tl.load(cnt_ptr)
        if slot >= cnt:
            return
        g = tl.load(grp_ptr + layer).to(tl.int32)
        base = tl.load(base_ptr + layer)
        bstride = tl.load(bstride_ptr + layer)
        btp = tl.load(bt_ptrs_ptr + g.to(tl.int64)).to(tl.pointer_type(tl.int32))
        NV: tl.constexpr = V // BV
        GQ: tl.constexpr = HV // H
        offs_k = tl.arange(0, K)
        hk = hv // GQ
        for i in range(slot, cnt, NACT):
            row = tl.load(plan_ptr + i)
            src = tl.load(plan_ptr + BLOCK + i)
            bias = tl.load(plan_ptr + 2 * BLOCK + i)
            ring = tl.load(plan_ptr + 3 * BLOCK + i) % R
            req = tl.load(plan_ptr + 4 * BLOCK + i)
            dst = tl.load(ids_ptr + (req * R + ring) * G + g)
            if dst >= 0:
                bt = btp + row * bt_stride
                s0 = tl.load(bt + src).to(tl.int64)
                s1 = tl.load(bt + src + 1).to(tl.int64)
                p0 = (base + s0 * bstride).to(tl.pointer_type(tl.bfloat16))
                a1 = base + s1 * bstride
                lg = a1.to(tl.pointer_type(tl.float32))
                par = a1.to(tl.pointer_type(tl.int32))
                pt = (base + dst.to(tl.int64) * bstride).to(tl.pointer_type(tl.bfloat16))
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
    def _snap_conv_planned(
        base_ptr, bstride_ptr, grp_ptr, cw_ptr, inner_ptr, rows_ptr, rstride_ptr, bt_ptrs_ptr, bt_stride,
        plan_ptr, cnt_ptr, ids_ptr,
        R: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr, NACT: tl.constexpr,
        DIM_FIRST: tl.constexpr, BLK: tl.constexpr,
    ):
        slot = tl.program_id(0)
        layer = tl.program_id(1)
        cnt = tl.load(cnt_ptr)
        if slot >= cnt:
            return
        g = tl.load(grp_ptr + layer).to(tl.int32)
        base = tl.load(base_ptr + layer)
        bstride = tl.load(bstride_ptr + layer)
        cw = tl.load(cw_ptr + layer)
        btp = tl.load(bt_ptrs_ptr + g.to(tl.int64)).to(tl.pointer_type(tl.int32))
        offs = tl.arange(0, BLK)
        for i in range(slot, cnt, NACT):
            row = tl.load(plan_ptr + i)
            src = tl.load(plan_ptr + BLOCK + i)
            bias = tl.load(plan_ptr + 2 * BLOCK + i)
            ring = tl.load(plan_ptr + 3 * BLOCK + i) % R
            req = tl.load(plan_ptr + 4 * BLOCK + i)
            dst = tl.load(ids_ptr + (req * R + ring) * G + g)
            if dst >= 0:
                sb = tl.load(btp + row * bt_stride + src).to(tl.int64)
                src_addr = base + sb * bstride
                dst_addr = base + dst.to(tl.int64) * bstride
                ntok = cw - bias
                if DIM_FIRST:
                    nrows = tl.load(rows_ptr + layer)
                    rstride = tl.load(rstride_ptr + layer)
                    for t in range(0, ntok):
                        for r0 in range(0, nrows, BLK):
                            rr = r0 + offs
                            m = rr < nrows
                            sp = (src_addr + rr * rstride + (t + bias) * 2).to(tl.pointer_type(tl.uint16))
                            dp = (dst_addr + rr * rstride + t * 2).to(tl.pointer_type(tl.uint16))
                            tl.store(dp, tl.load(sp, mask=m), mask=m)
                else:
                    inner = tl.load(inner_ptr + layer)
                    total = ntok.to(tl.int64) * inner
                    sp0 = (src_addr + bias.to(tl.int64) * inner * 2).to(tl.pointer_type(tl.uint16))
                    dp0 = dst_addr.to(tl.pointer_type(tl.uint16))
                    for j in range(0, total, BLK):
                        m = (j + offs) < total
                        tl.store(dp0 + j + offs, tl.load(sp0 + j + offs, mask=m), mask=m)


# ----------------------------------------------------------------------------------------------
# worker side
# ----------------------------------------------------------------------------------------------
class _Worker:
    def __init__(self) -> None:
        self.ids: torch.Tensor | None = None  # [max_reqs, R, G] int32 block ids (-1 = none)
        self.vf: torch.Tensor | None = None  # [max_reqs] int32: first snapshot-eligible position
        self.tables: dict | None = None
        self.snapshots = 0

    def ensure(self, device, max_reqs: int) -> bool:
        if not CFG:
            return False
        if self.ids is None:
            R, G = CFG["R"], CFG["G"]
            self.ids = torch.full((max_reqs, R, G), -1, dtype=torch.int32, device=device)
            self.vf = torch.full((max_reqs,), INT_MAX, dtype=torch.int32, device=device)
        return True

    def on_add(self, runner, scheduler_output) -> None:
        if not self.ensure(runner.device, runner.max_num_reqs):
            return
        R, G = CFG["R"], CFG["G"]
        for nr in scheduler_output.scheduled_new_reqs:
            idx = runner.req_states.req_id_to_index[nr.req_id]
            with _LOCK:
                e = LEDGER.get(nr.req_id)
            if e is None:
                self.ids[idx].fill_(-1)
                self.vf[idx] = INT_MAX
            else:
                t = torch.tensor(e["ids"], dtype=torch.int32)  # [R][G]
                self.ids[idx].copy_(t)
                self.vf[idx] = e["valid_from"]

    def build_tables(self, ctx) -> dict:
        from vllm.utils.gpu_sync_debug import gpu_sync_allowed
        from vllm.v1.worker.mamba_utils import is_conv_state_dim_first

        from atdev_serve import gdn_ss_align
        from atdev_serve.gdn_single_state import log_layout

        g = gdn_ss_align.GEOM
        if not g:
            raise RuntimeError("eor_ckpt: no SingleStateGDNAttention layer constructed (GEOM empty)")
        n_state = g["HV"] * g["V"] * g["K"]
        with gpu_sync_allowed():
            cw = ctx.state_conv_widths.cpu()
            esz = ctx.state_elem_sizes.cpu()
            inner = ctx.state_inner_sizes.cpu()
            tsel = torch.nonzero(cw == 0).flatten()
            csel = torch.nonzero(cw > 0).flatten()
            if not bool((inner[tsel] == n_state).all()) or not bool((esz[tsel] == 2).all()):
                raise RuntimeError("eor_ckpt: temporal states must be bf16 [HV, V, K]")
            if not bool((esz[csel] == 2).all()):
                raise RuntimeError("eor_ckpt: conv states must be 2-byte elements")
            dev = ctx.state_base_addrs.device
            tsel_d, csel_d = tsel.to(dev), csel.to(dev)
            tb = dict(
                lay=log_layout(g["H"], g["HV"], g["K"], g["V"], 32, g["max_log"]),
                nt=int(tsel.numel()), nc=int(csel.numel()),
                t_base=ctx.state_base_addrs[tsel_d].contiguous(),
                t_bstride=ctx.state_block_strides[tsel_d].contiguous(),
                t_grp=ctx.state_group_indices[tsel_d].to(torch.int32).contiguous(),
                c_base=ctx.state_base_addrs[csel_d].contiguous(),
                c_bstride=ctx.state_block_strides[csel_d].contiguous(),
                c_grp=ctx.state_group_indices[csel_d].to(torch.int32).contiguous(),
                c_cw=ctx.state_conv_widths[csel_d].to(torch.int32).contiguous(),
                c_inner=ctx.state_inner_sizes[csel_d].contiguous(),
                c_rows=ctx.state_dim_row_count[csel_d].to(torch.int32).contiguous(),
                c_rstride=ctx.state_dim_row_stride[csel_d].contiguous(),
                dim_first=bool(is_conv_state_dim_first()),
            )
        if ctx.num_groups != CFG["G"]:
            raise RuntimeError(f"eor_ckpt: ctx mamba groups {ctx.num_groups} != ledger groups {CFG['G']}")
        log.warning("atdev eor_ckpt: %d temporal + %d conv state tensors, R=%d G=%d, conv dim_first=%s",
                    tb["nt"], tb["nc"], CFG["R"], CFG["G"], tb["dim_first"])
        return tb

    def snap(self, ctx, num_reqs, acc, state_idx, nc, idx_map) -> None:
        if self.ids is None or not ctx.is_initialized or NOKERNEL:
            return
        if TIMING:
            if not hasattr(self, "_ev"):
                self._ev, self._tot, self._n = [], 0.0, 0
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
            self._snap(ctx, num_reqs, acc, state_idx, nc, idx_map)
            e1.record(); self._ev.append((e0, e1)); self._n += 1
            if len(self._ev) >= 2000:
                torch.cuda.synchronize(); ms = sum(a.elapsed_time(b) for a, b in self._ev); self._tot += ms
                log.warning("atdev eor_ckpt TIMING: %d snap calls, avg %.1f us/call (total %.1f ms)", self._n, 1000 * self._tot / self._n, self._tot)
                self._ev = []
            return
        self._snap(ctx, num_reqs, acc, state_idx, nc, idx_map)

    def _snap_planned(self, ctx, num_reqs, acc, state_idx, nc, idx_map) -> None:
        tb = self.tables
        if tb is None:
            tb = self.tables = self.build_tables(ctx)
        if not hasattr(self, "plan"):
            self.PB = 1024
            self.plan = torch.zeros(5 * self.PB, dtype=torch.int32, device=self.ids.device)
            self.cnt = torch.zeros(1, dtype=torch.int32, device=self.ids.device)
        g, lay = _geom(), tb["lay"]
        R, G, NACT = CFG["R"], CFG["G"], NACT_CTA
        _plan_kernel[(1,)](idx_map, acc, state_idx, nc, self.vf, num_reqs, self.plan, self.cnt, UNIT=SNAP, BLOCK=self.PB, num_warps=4)
        _snap_temporal_planned[(NACT, tb["nt"], g["HV"])](
            tb["t_base"], tb["t_bstride"], tb["t_grp"], ctx.block_table_ptrs, ctx.block_table_stride_req, self.plan, self.cnt, self.ids,
            R=R, G=G, BLOCK=self.PB, NACT=NACT, H=g["H"], HV=g["HV"], K=g["K"], V=g["V"], BV=32,
            LOG_BASE=lay["log_base"], BUF=lay["buf"], V_OFF=lay["v_off"], D_OFF=lay["d_off"], B_OFF=lay["b_off"], num_warps=4)
        _snap_conv_planned[(NACT, tb["nc"])](
            tb["c_base"], tb["c_bstride"], tb["c_grp"], tb["c_cw"], tb["c_inner"], tb["c_rows"], tb["c_rstride"],
            ctx.block_table_ptrs, ctx.block_table_stride_req, self.plan, self.cnt, self.ids,
            R=R, G=G, BLOCK=self.PB, NACT=NACT, DIM_FIRST=tb["dim_first"], BLK=256, num_warps=4)
        self.snapshots += 1

    def _snap(self, ctx, num_reqs, acc, state_idx, nc, idx_map) -> None:
        if PLANNED:
            return self._snap_planned(ctx, num_reqs, acc, state_idx, nc, idx_map)
        tb = self.tables
        if tb is None:
            tb = self.tables = self.build_tables(ctx)
        g, lay = _geom(), tb["lay"]
        R, G = CFG["R"], CFG["G"]
        _snap_temporal_kernel[(num_reqs, tb["nt"], g["HV"])](
            tb["t_base"], tb["t_bstride"], tb["t_grp"], ctx.block_table_ptrs, ctx.block_table_stride_req,
            idx_map, acc, state_idx, nc, self.vf, self.ids, num_reqs,
            R=R, G=G, UNIT=SNAP, H=g["H"], HV=g["HV"], K=g["K"], V=g["V"], BV=32,
            LOG_BASE=lay["log_base"], BUF=lay["buf"], V_OFF=lay["v_off"], D_OFF=lay["d_off"], B_OFF=lay["b_off"],
            num_warps=4)
        _snap_conv_kernel[(num_reqs, tb["nc"])](
            tb["c_base"], tb["c_bstride"], tb["c_grp"], tb["c_cw"], tb["c_inner"], tb["c_rows"], tb["c_rstride"],
            ctx.block_table_ptrs, ctx.block_table_stride_req,
            idx_map, acc, state_idx, nc, self.vf, self.ids, num_reqs,
            R=R, G=G, UNIT=SNAP, DIM_FIRST=tb["dim_first"], BLK=256, num_warps=4)
        self.snapshots += 1


def _geom() -> dict:
    from atdev_serve import gdn_ss_align

    return gdn_ss_align.GEOM


_W = _Worker()


def _install_worker() -> None:
    from vllm.v1.worker import mamba_utils as mu
    from vllm.v1.worker.gpu import model_runner as mr

    cls = mu.MambaSpecDecodeGPUContext
    if not getattr(cls, "_atdev_eor", False):
        orig_post = cls.run_fused_postprocess_align

        def run_fused_postprocess_align(self, num_reqs, num_accepted_tokens_gpu, state_idx_gpu,
                                        new_num_computed_tokens_gpu, idx_mapping):
            orig_post(self, num_reqs, num_accepted_tokens_gpu, state_idx_gpu, new_num_computed_tokens_gpu,
                      idx_mapping)
            if num_reqs and idx_mapping is not None:
                _W.snap(self, num_reqs, num_accepted_tokens_gpu, state_idx_gpu, new_num_computed_tokens_gpu,
                        idx_mapping)

        cls.run_fused_postprocess_align = run_fused_postprocess_align
        cls._atdev_eor = True
    rcls = mr.GPUModelRunner
    if not getattr(rcls, "_atdev_eor", False):
        orig_add = rcls.add_requests

        def add_requests(self, scheduler_output):
            orig_add(self, scheduler_output)
            if scheduler_output.scheduled_new_reqs:
                _W.on_add(self, scheduler_output)

        rcls.add_requests = add_requests
        rcls._atdev_eor = True


# ----------------------------------------------------------------------------------------------
# scheduler side
# ----------------------------------------------------------------------------------------------
def _mamba_mgrs(kcm):
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    out = []
    for gid, m in enumerate(kcm.coordinator.single_type_managers):
        if isinstance(m, MambaManager) and m.mamba_cache_mode == "align" and m.enable_caching:
            out.append((gid, m))
    return out


def _eagle_drop(kcm) -> bool:
    return any(bool(getattr(m, "use_eagle", False)) for m in kcm.coordinator.single_type_managers)


def _alloc(kcm, request) -> None:
    mm = _mamba_mgrs(kcm)
    if not mm:
        return
    R = RING or (3 if (_eagle_drop(kcm) and SNAP < 32) else 2)
    G = len(mm)
    if not CFG:
        CFG.update(R=R, G=G, gids=[g for g, _ in mm])
        log.warning("atdev eor_ckpt: scheduler side active, R=%d, mamba groups %s, eagle_drop=%s",
                    R, CFG["gids"], R == 3)
    pool = kcm.block_pool
    need = R * G
    if pool.get_num_free_blocks() < need + RESERVE_BLOCKS + HEADROOM * pool.num_gpu_blocks:
        STATS["alloc_skip"] += 1
        return
    blocks = pool.get_new_blocks(need)
    ring_blocks = [[blocks[r * G + gi] for gi in range(G)] for r in range(R)]
    prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
    with _LOCK:
        LEDGER[request.request_id] = dict(
            blocks=ring_blocks, ids=[[b.block_id for b in row] for row in ring_blocks], valid_from=prefill_end)
    STATS["alloc"] += 1


def _release(kcm, entry, keep=None) -> None:
    pool = kcm.block_pool
    blocks = [b for row in entry["blocks"] for b in row if keep is None or b.block_id not in keep]
    if blocks:
        pool.free_blocks(blocks)


def _finish(kcm, request, entry) -> None:
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
    from vllm.v1.request import RequestStatus

    st = request.status
    if st not in (RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED):
        STATS["preempt_or_abort"] += 1
        _release(kcm, entry)
        return
    pool = kcm.block_pool
    mm = _mamba_mgrs(kcm)
    R, G = CFG["R"], CFG["G"]
    drop = UNIT if _eagle_drop(kcm) else 0
    e_comp = request.num_tokens - 1  # the last sampled token is never forwarded
    l16 = e_comp // UNIT * UNIT
    ls = (l16 - drop) // SNAP * SNAP
    hashes = request.block_hashes
    if NOREG or ls <= entry["valid_from"] or ls < UNIT or len(hashes) < l16 // UNIT or len(mm) != G:
        STATS["reg_short"] += 1
        _release(kcm, entry)
        return
    ring = (ls // SNAP) % R
    keep = set()
    for gi, (gid, mgr) in enumerate(mm):
        blk = entry["blocks"][ring][gi]
        pool.cache_partial_block(request, blk, ls, gid, mgr.block_size, replace_existing_hashes=True)
        keep.add(blk.block_id)
    # attention-type groups: alias hash at l16 on the request's (partial) last block
    n_alias = 0
    hbs = pool.hash_block_size
    for gid, mgr in enumerate(kcm.coordinator.single_type_managers):
        if any(gid == g for g, _ in mm):
            continue
        if not mgr.kv_cache_spec.prefix_cacheable or not mgr.enable_caching:
            continue
        bs = mgr.block_size
        if NOALIAS or bs <= hbs or l16 % bs == 0:
            continue
        blocks = mgr.req_to_blocks.get(request.request_id)
        if not blocks:
            continue
        idx = l16 // bs
        if idx >= len(blocks) or blocks[idx].is_null:
            continue
        key = make_block_hash_with_group_id(hashes[l16 // UNIT - 1], mgr.kv_cache_group_id)
        pool._insert_block_hash(key, blocks[idx], num_tokens=l16)
        n_alias += 1
    STATS["reg"] += 1
    STATS["attn_alias"] += n_alias
    if DEBUG:
        log.warning("atdev eor_ckpt: register req=%s num_tokens=%d Ls=%d l16=%d ring=%d aliases=%d valid_from=%d",
                    request.request_id, request.num_tokens, ls, l16, ring, n_alias, entry["valid_from"])
    # release: unused ring blocks first, the registered ones last (evicted last)
    _release(kcm, entry, keep=keep)
    pool.free_blocks([entry["blocks"][ring][gi] for gi in range(G)])
    if KEEP:
        REGISTERED.append([entry["blocks"][ring][gi] for gi in range(G)])
        while len(REGISTERED) > KEEP:
            for b in REGISTERED.popleft():
                if b.ref_cnt == 0 and b.block_hash is not None:  # stale checkpoint nobody hit: uncache and make it the first reuse candidate
                    pool._maybe_evict_cached_block(b)
                    pool.free_block_queue.remove(b)
                    pool.free_block_queue.prepend_n([b])


def _install_sched() -> None:
    from vllm.v1.core import kv_cache_manager as km

    cls = km.KVCacheManager
    if getattr(cls, "_atdev_eor", False):
        return
    orig_alloc = cls.allocate_slots
    orig_free = cls.free
    orig_pop = cls.pop_blocks_for_free

    def allocate_slots(self, request, *a, **k):
        res = orig_alloc(self, request, *a, **k)
        if res is not None and request.request_id not in LEDGER and self.enable_caching:
            try:
                _alloc(self, request)
            except Exception:  # never break scheduling
                log.exception("eor_ckpt: ledger alloc failed")
        return res

    def _take(self, request):
        with _LOCK:
            return LEDGER.pop(request.request_id, None)

    def free(self, request):
        e = _take(self, request)
        if e is not None:
            try:
                _finish(self, request, e)
            except Exception:
                log.exception("eor_ckpt: finish failed")
        orig_free(self, request)

    def pop_blocks_for_free(self, request):
        e = _take(self, request)
        if e is not None:
            try:
                _finish(self, request, e)
            except Exception:
                log.exception("eor_ckpt: finish failed")
        return orig_pop(self, request)

    cls.allocate_slots = allocate_slots
    cls.free = free
    cls.pop_blocks_for_free = pop_blocks_for_free
    cls._atdev_eor = True


def stats_str() -> str:
    return " ".join(f"{k}={v}" for k, v in STATS.items())


def install() -> None:
    if not enabled():
        return
    if os.environ.get("ATDEV_GDN_SS_ALIGN", "0") != "1":
        raise RuntimeError("ATDEV_EOR_CKPT=1 needs ATDEV_GDN_SS_ALIGN=1 (single-state GDN under align mode)")
    _install_sched()
    _install_worker()
    log.warning("atdev eor_ckpt: installed (unit %d)", UNIT)
