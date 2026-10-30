"""LL5 (ATDEV_OPT=ll5_fuse): remaining bit-identical c=1 glue items (LEDGER LL5-*), each opt-in:

  ATDEV_LL5_CN=1   DFlash2 drafter: DFlashGroupedConv.finish (vLLM Triton grouped conv, 10 launches / step) fused
                   into the Inductor add + RMSNorm kernel that follows it. Replaces DFlash2 model forward with the
                   same module calls, except per layer
                     attention_conv.finish + post_attention_layernorm      -> op atdev_ll5_cna
                     mlp_conv.finish + next input_layernorm (final norm)   -> op atdev_ll5_cnb
                   The Triton kernels below are the Inductor kernels of the stock compiled drafter (pinned configs:
                   _0 XBLOCK 1 / R0_BLOCK 4096 / 16 warps, _2 and the final norm R0_BLOCK 2048 / 16 warps; sources in
                   results/LL5/ref) with the conv value computed in-register by the conv kernel's expressions and
                   rounded to bf16 where the stock chain stores it. Inductor keeps the residual of the post-attention
                   norm unrounded (fp32 conv_attn + residual) inside the next kernel, so op A returns the bf16 conv
                   output and op B re-forms the same fp32 sum.
  ATDEV_LL5_OVL=1  next-step target prep overlapped with the drafter graph: the part of GPUModelRunner.prepare_inputs
                   before combine_sampled_and_draft_tokens (idx_mapping / cu_num_logits / query_start_loc HtoD,
                   expand_idx_mapping, prepare_pos_seq_lens) runs on a side stream that waits on an event recorded
                   right before the DFlash query graph replay (i.e. after prepare_dflash_inputs, the last reader of
                   the target positions / query_start_loc of the previous step); the main stream joins it before
                   combine_sampled_and_draft_tokens (the first op that needs the drafts). Same kernels, same inputs.
                   Decode-only steps without new requests (no staged req-state writes), no LoRA / PCP / adaptive
                   verification. Side-stream tensors of the InputBatch get record_stream(main).
                   =2: also the target KV groups' gather_block_tables / compute_slot_mappings on the side stream
                   (steps without block-table updates; the drafter groups must be the last groups): the drafter
                   groups' block-table gather stays on main (after the drafter graph that reads those rows), their
                   slot-mapping rows are skipped (dead: prepare_dflash_inputs rewrites the whole row before any
                   reader), main joins the side stream right after compute_slot_mappings (before tree_pos shifts the
                   positions the slot kernel reads).
  ATDEV_LL5_TAIL=1 LL1-tail (needs ATDEV_LL1_TAIL=1) with the final torch.topk(sorted) emulation in registers
                   (csrc/ll5_tail.cu: ballot ranks for the sbtopk gather order + the same bitonic network on shuffles
                   instead of a serial lane-0 gather and shared-memory swaps).
  ATDEV_LL5_TREE=1 gdn_tree.build_tree_gpu (N = 16 nodes, K = 16, no chain / mix) as a CUDA kernel (csrc/ll5_tree.cu)
                   with the Triton kernel's arithmetic (ex2.approx, libdevice logf, fma contraction, butterfly sums,
                   tie-left argmax); each node's lattice row is normalised once instead of every iteration.
  ATDEV_LL5_DG=1   drop a dead gather in GPUModelRunner.sample_tokens: set_draft_tokens(input_batch,
                   req_states.draft_tokens[idx_mapping]) only reads the gathered tensor's shape unless the batch has
                   structured-output requests -> pass the view req_states.draft_tokens[:num_reqs] then (same shape;
                   -2 eager kernels in e2). Source-level patch of sample_tokens (one expression), checked verbatim.
  ATDEV_LL5_GAPS=n diagnostic (no profiler): CUDA events around every CUDAGraph.replay; every n steps log the median
                   T graph, D graph, e1 (T end -> D start) and e2 (D end -> next T start) GPU times.
"""
from __future__ import annotations

import functools
import logging
import os

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

OPT_NAME = "ll5_fuse"
log = logging.getLogger("atdev_serve")

CN = os.environ.get("ATDEV_LL5_CN", "0") == "1"
OVL = int(os.environ.get("ATDEV_LL5_OVL", "0"))
GAPS = int(os.environ.get("ATDEV_LL5_GAPS", "0"))
TAIL = os.environ.get("ATDEV_LL5_TAIL", "0") == "1"
TREE = os.environ.get("ATDEV_LL5_TREE", "0") == "1"
DG = os.environ.get("ATDEV_LL5_DG", "0") == "1"
_N: dict = {}


def _once(key: str, msg: str, *a) -> None:
    if not _N.get(key):
        _N[key] = 1
        log.warning("atdev %s: " + msg, OPT_NAME, *a)


# ---------------------------------------------------------------------------------------------------------------
# LL5-cn: conv.finish + add-RMSNorm
# ---------------------------------------------------------------------------------------------------------------


@triton.jit
def _conv1(x_ptr, x_stride, d_ptr, d_stride_row, d_stride_tap, b_ptr, b_stride_tap, x0, r0_offset, r0_1, m, xmask,
           BS: tl.constexpr, GS: tl.constexpr, R0_BLOCK: tl.constexpr):
    """== _dflash2_grouped_conv_kernel (TAPS = 2) for elements (row x0, channel r0_1), fp32 before the bf16 store.
    The per-group delta is loaded once per group ([1, R0_BLOCK / GS]) and broadcast (same values)."""
    position = x0 % BS
    delta_row = d_ptr + x0 * d_stride_row
    x_row = x_ptr + x0 * x_stride
    g = r0_offset // GS + tl.arange(0, R0_BLOCK // GS)[None, :]
    gm = (g < 5120 // GS) & xmask
    d0 = tl.load(delta_row + g, mask=gm, other=0.0)
    d0 = tl.reshape(tl.broadcast_to(d0[:, :, None], [1, R0_BLOCK // GS, GS]), [1, R0_BLOCK])
    tap_mask = m & (position >= 1)
    d1 = tl.load(delta_row + d_stride_tap + g, mask=gm & (position >= 1), other=0.0)
    d1 = tl.reshape(tl.broadcast_to(d1[:, :, None], [1, R0_BLOCK // GS, GS]), [1, R0_BLOCK])
    accumulator = (
        tl.load(b_ptr + r0_1, mask=m, other=0.0).to(tl.float32)
        + d0.to(tl.float32)
    ) * tl.load(x_row + r0_1, mask=m, other=0.0).to(tl.float32)
    coefficient = tl.load(b_ptr + b_stride_tap + r0_1, mask=tap_mask, other=0.0).to(tl.float32) + d1.to(tl.float32)
    x = tl.load(x_ptr + (x0 - 1) * x_stride + r0_1, mask=tap_mask, other=0.0).to(tl.float32)
    accumulator += coefficient * x
    return accumulator


@triton.jit
def _cn_chunk(x_ptr, x_stride, d_ptr, d_stride_row, d_stride_tap, b_ptr, b_stride_tap, x0, xmask, r0_offset,
              BS: tl.constexpr, GS: tl.constexpr, R0_BLOCK: tl.constexpr):
    r0_1 = r0_offset + tl.arange(0, R0_BLOCK)[None, :]
    m = (r0_1 < 5120) & xmask
    cv = _conv1(x_ptr, x_stride, d_ptr, d_stride_row, d_stride_tap, b_ptr, b_stride_tap, x0, r0_offset, r0_1, m,
                xmask, BS, GS, R0_BLOCK)
    return cv.to(tl.bfloat16), r0_1, m


@triton.jit
def _ll5_cna_kernel(x_ptr, d_ptr, b_ptr, res_ptr, w_ptr, conv_ptr, out_ptr, xnumel, x_stride, d_stride_row,
                    d_stride_tap, b_stride_tap, BS: tl.constexpr, GS: tl.constexpr, XBLOCK: tl.constexpr,
                    R0_BLOCK: tl.constexpr):
    # == conv.finish (attention side) + triton_red_fused_atdev_ll1_kpc_fused_add_rms_norm_select_0 (R0_BLOCK 4096:
    # its 2 loop iterations unrolled, the second pass reuses the first pass' fp32 values: same per-element ops,
    # same [XBLOCK, R0_BLOCK] accumulator and tl.sum)
    tl.static_assert(R0_BLOCK == 4096)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x0 = xindex
    c0, r0, m0 = _cn_chunk(x_ptr, x_stride, d_ptr, d_stride_row, d_stride_tap, b_ptr, b_stride_tap, x0, xmask, 0,
                           BS, GS, R0_BLOCK)
    c1, r1, m1 = _cn_chunk(x_ptr, x_stride, d_ptr, d_stride_row, d_stride_tap, b_ptr, b_stride_tap, x0, xmask,
                           R0_BLOCK, BS, GS, R0_BLOCK)
    s0 = tl.load(res_ptr + (r0 + 5120 * x0), m0, other=0.0).to(tl.float32)
    s1 = tl.load(res_ptr + (r1 + 5120 * x0), m1, other=0.0).to(tl.float32)
    w0 = tl.load(w_ptr + r0, r0 < 5120, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + r1, r1 < 5120, other=0.0).to(tl.float32)
    tl.store(conv_ptr + (r0 + 5120 * x0), c0, m0)
    tl.store(conv_ptr + (r1 + 5120 * x0), c1, m1)
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    t40 = c0.to(tl.float32) + s0
    _tmp7 = tl.where(m0, _tmp7 + t40 * t40, _tmp7)
    t41 = c1.to(tl.float32) + s1
    _tmp7 = tl.where(m1, _tmp7 + t41 * t41, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp14 = tl.full([1, 1], 5120.0, tl.float32)
    tmp15 = (tmp7 / tmp14)
    tmp16 = tl.full([1, 1], 1e-06, tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.store(out_ptr + (r0 + 5120 * x0), (t40 * tmp18) * w0, m0)
    tl.store(out_ptr + (r1 + 5120 * x0), (t41 * tmp18) * w1, m1)


@triton.jit
def _ll5_cnb_kernel(x_ptr, d_ptr, b_ptr, ca_ptr, res_ptr, w_ptr, resout_ptr, out_ptr, xnumel, x_stride,
                    d_stride_row, d_stride_tap, b_stride_tap, STORE_RES: tl.constexpr, BS: tl.constexpr,
                    GS: tl.constexpr, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    # == conv.finish (mlp side) + triton_red_fused_atdev_ll1_kpc_fused_add_rms_norm_select_2 (STORE_RES) /
    #    triton_red_fused_fused_add_rms_norm_2 (final norm); R0_BLOCK 2048: 3 iterations unrolled
    tl.static_assert(R0_BLOCK == 2048)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x0 = xindex
    c0, r0, m0 = _cn_chunk(x_ptr, x_stride, d_ptr, d_stride_row, d_stride_tap, b_ptr, b_stride_tap, x0, xmask, 0,
                           BS, GS, R0_BLOCK)
    c1, r1, m1 = _cn_chunk(x_ptr, x_stride, d_ptr, d_stride_row, d_stride_tap, b_ptr, b_stride_tap, x0, xmask,
                           R0_BLOCK, BS, GS, R0_BLOCK)
    c2, r2, m2 = _cn_chunk(x_ptr, x_stride, d_ptr, d_stride_row, d_stride_tap, b_ptr, b_stride_tap, x0, xmask,
                           2 * R0_BLOCK, BS, GS, R0_BLOCK)
    a0 = tl.load(ca_ptr + (r0 + 5120 * x0), m0, other=0.0).to(tl.float32)
    a1 = tl.load(ca_ptr + (r1 + 5120 * x0), m1, other=0.0).to(tl.float32)
    a2 = tl.load(ca_ptr + (r2 + 5120 * x0), m2, other=0.0).to(tl.float32)
    s0 = tl.load(res_ptr + (r0 + 5120 * x0), m0, other=0.0).to(tl.float32)
    s1 = tl.load(res_ptr + (r1 + 5120 * x0), m1, other=0.0).to(tl.float32)
    s2 = tl.load(res_ptr + (r2 + 5120 * x0), m2, other=0.0).to(tl.float32)
    w0 = tl.load(w_ptr + r0, r0 < 5120, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + r1, r1 < 5120, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + r2, r2 < 5120, other=0.0).to(tl.float32)
    t90 = c0.to(tl.float32) + (a0 + s0)
    t91 = c1.to(tl.float32) + (a1 + s1)
    t92 = c2.to(tl.float32) + (a2 + s2)
    if STORE_RES:
        tl.store(resout_ptr + (r0 + 5120 * x0), t90, m0)
        tl.store(resout_ptr + (r1 + 5120 * x0), t91, m1)
        tl.store(resout_ptr + (r2 + 5120 * x0), t92, m2)
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp12 = tl.where(m0, _tmp12 + t90 * t90, _tmp12)
    _tmp12 = tl.where(m1, _tmp12 + t91 * t91, _tmp12)
    _tmp12 = tl.where(m2, _tmp12 + t92 * t92, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp25 = tl.full([1, 1], 5120.0, tl.float32)
    tmp26 = (tmp12 / tmp25)
    tmp27 = tl.full([1, 1], 1e-06, tl.float32)
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tl.store(out_ptr + (r0 + 5120 * x0), (t90 * tmp29) * w0, m0)
    tl.store(out_ptr + (r1 + 5120 * x0), (t91 * tmp29) * w1, m1)
    tl.store(out_ptr + (r2 + 5120 * x0), (t92 * tmp29) * w2, m2)


A_CFG = dict(XBLOCK=1, R0_BLOCK=4096, num_warps=16, num_stages=1)
B_CFG = dict(XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1)


def _ok(x, delta, base, res, w) -> bool:
    return (x.dim() == 2 and x.shape[1] == 5120 and x.stride(1) == 1 and x.dtype == torch.bfloat16
            and res.is_contiguous() and res.shape == x.shape and res.dtype == torch.bfloat16
            and delta.dim() == 3 and delta.shape[1] == 2 and delta.shape[2] == 320 and delta.stride(2) == 1
            and delta.dtype == torch.bfloat16 and tuple(base.shape) == (2, 5120) and base.stride(1) == 1
            and base.dtype == torch.bfloat16 and w.is_contiguous() and w.numel() == 5120)


def cna_run(x, delta, base, res, w, block_size: int):
    """(norm_out, conv_out) == conv.finish(x) -> fused_add_rms_norm(conv, res, w) of the compiled stock drafter."""
    t = x.shape[0]
    conv = torch.empty((t, 5120), dtype=x.dtype, device=x.device)
    out = torch.empty((t, 5120), dtype=x.dtype, device=x.device)
    if t:
        _ll5_cna_kernel[(t,)](x, delta, base, res, w, conv, out, t, x.stride(0), delta.stride(0), delta.stride(1),
                              base.stride(0), BS=block_size, GS=16, **A_CFG)
    return out, conv


def cnb_run(x, delta, base, conv_a, res, w, block_size: int, store_res: bool):
    """(residual_out, norm_out): residual_out = bf16(conv_m + (conv_a + res)) (None unless store_res)."""
    t = x.shape[0]
    out = torch.empty((t, 5120), dtype=x.dtype, device=x.device)
    rout = torch.empty((t, 5120), dtype=x.dtype, device=x.device) if store_res else out
    if t:
        _ll5_cnb_kernel[(t,)](x, delta, base, conv_a, res, w, rout, out, t, x.stride(0), delta.stride(0),
                              delta.stride(1), base.stride(0), STORE_RES=store_res, BS=block_size, GS=16, **B_CFG)
    return rout, out


def _stock_finish(x, delta, base, block_size):
    from vllm.model_executor.models import qwen3_dflash2 as q2
    return q2.dflash2_grouped_conv_impl(x, delta, base, block_size, 16)


def _cna_op(x: torch.Tensor, delta: torch.Tensor, base: torch.Tensor, res: torch.Tensor, w: torch.Tensor,
            block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not _ok(x, delta, base, res, w):
        raise RuntimeError(f"atdev {OPT_NAME}: cna layout x {tuple(x.shape)} {x.stride()} delta "
                           f"{tuple(delta.shape)} {delta.stride()}")
    _once("cna", "LL5-cn conv.finish + add-RMSNorm (attention side) ENGAGED")
    return cna_run(x, delta, base, res, w, block_size)


def _cna_fake(x, delta, base, res, w, block_size):
    return x.new_empty(x.shape[0], 5120), x.new_empty(x.shape[0], 5120)


def _cnb_op(x: torch.Tensor, delta: torch.Tensor, base: torch.Tensor, conv_a: torch.Tensor, res: torch.Tensor,
            w: torch.Tensor, block_size: int, store_res: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if not (_ok(x, delta, base, res, w) and conv_a.is_contiguous() and conv_a.shape == x.shape):
        raise RuntimeError(f"atdev {OPT_NAME}: cnb layout x {tuple(x.shape)} {x.stride()}")
    _once("cnb", "LL5-cn conv.finish + add-RMSNorm (mlp side, store_res=%s) ENGAGED", store_res)
    rout, out = cnb_run(x, delta, base, conv_a, res, w, block_size, store_res)
    return (rout if store_res else x.new_empty(0)), out


def _cnb_fake(x, delta, base, conv_a, res, w, block_size, store_res):
    return (x.new_empty(x.shape[0], 5120) if store_res else x.new_empty(0)), x.new_empty(x.shape[0], 5120)


def _model_ok(model) -> bool:
    from vllm.model_executor.models import qwen3_dflash2 as q2
    if not model.layers or model.config.hidden_size != 5120:
        return False
    for layer in model.layers:
        if not isinstance(layer, q2.DFlash2Qwen3DecoderLayer):
            return False
        for cv in (layer.attention_conv, layer.mlp_conv):
            if cv.taps != 2 or cv.group_size != 16 or cv.base_kernel.dtype != torch.bfloat16:
                return False
        for n in (layer.input_layernorm, layer.post_attention_layernorm):
            if n.weight.dtype != torch.bfloat16 or abs(n.variance_epsilon - 1e-6) > 0 or getattr(n, "has_weight", True) is False:
                return False
    return model.norm.weight.dtype == torch.bfloat16 and model.norm.variance_epsilon == 1e-6


def _install_cn() -> None:
    from vllm.model_executor.models import qwen3_dflash2 as q2
    from vllm.utils.torch_utils import direct_register_custom_op

    if not hasattr(torch.ops.vllm, "atdev_ll5_cna"):
        direct_register_custom_op(op_name="atdev_ll5_cna", op_func=_cna_op, fake_impl=_cna_fake)
        direct_register_custom_op(op_name="atdev_ll5_cnb", op_func=_cnb_op, fake_impl=_cnb_fake)
    cls = q2.DFlash2Qwen3Model
    if getattr(cls.forward, "_atdev_ll5", False):
        return
    orig = cls.forward
    orig_init = cls.__init__

    @functools.wraps(orig_init)
    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        self._atdev_ll5_ok = _model_ok(self)
        log.warning("atdev %s: LL5-cn drafter model check %s", OPT_NAME, self._atdev_ll5_ok)

    cls.__init__ = __init__

    def forward(self, input_ids, positions, input_embeds=None):
        if not self._atdev_ll5_ok:
            return orig(self, input_ids, positions, input_embeds)
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        layers = self.layers
        n = len(layers)
        res = input_embeds
        hn = layers[0].input_layernorm(input_embeds)
        for i in range(n):
            layer = layers[i]
            x, coef = layer.attention_conv.prepare(hn)
            a = layer.self_attn(positions=positions, hidden_states=x)
            ac = layer.attention_conv
            norm_a, conv_a = torch.ops.vllm.atdev_ll5_cna(a, coef, ac.base_kernel[1], res,
                                                          layer.post_attention_layernorm.weight, ac.block_size)
            x, coef = layer.mlp_conv.prepare(norm_a)
            m = layer.mlp(x)
            mc = layer.mlp_conv
            last = i + 1 == n
            w_next = self.norm.weight if last else layers[i + 1].input_layernorm.weight
            r, hn = torch.ops.vllm.atdev_ll5_cnb(m, coef, mc.base_kernel[1], conv_a, res, w_next, mc.block_size,
                                                 not last)
            res = r
        return hn

    forward._atdev_ll5 = True
    cls.forward = forward


# ---------------------------------------------------------------------------------------------------------------
# LL5-ovl: next-step prep on a side stream, overlapped with the drafter graph
# ---------------------------------------------------------------------------------------------------------------


class _Ovl:
    bt = False  # this step: target-group gather / slot mappings on the side stream
    d0 = -1
    n2 = 0
    side = None
    ev = None
    pending = False
    main = None
    active = False
    n = 0
    skip = 0


O = _Ovl()


def _ovl_join() -> None:
    if O.active:
        O.main.wait_stream(O.side)
        torch.cuda.set_stream(O.main)
        O.active = False


def _draft_suffix(runner) -> int:
    """First KV-group id of the drafter if the drafter groups are exactly the last groups, else 0."""
    sp = getattr(runner, "speculator", None)
    gids = sorted(getattr(sp, "draft_kv_cache_group_ids", []) or [])
    g = runner.block_tables.num_kv_cache_groups
    if not gids or gids != list(range(gids[0], g)) or gids[0] == 0 or runner.block_tables.cp_size != 1:
        log.warning("atdev %s: LL5-ovl2 drafter groups %s of %d are not a suffix -> level 1 only", OPT_NAME, gids, g)
        return 0
    log.warning("atdev %s: LL5-ovl2 target groups 0..%d on the side stream, drafter groups %s on main", OPT_NAME,
                gids[0] - 1, gids)
    return gids[0]


def _install_bt() -> None:
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.worker.gpu import block_table as btm

    B = btm.BlockTables
    orig_g = B.gather_block_tables
    orig_s = B.compute_slot_mappings

    def gather_block_tables(self, idx_mapping, num_reqs_padded, out=None, out_ptrs=None):
        if not (O.bt and O.active is False and out is None and torch.cuda.current_stream() == O.main):
            return orig_g(self, idx_mapping, num_reqs_padded, out=out, out_ptrs=out_ptrs)
        g, d0 = self.num_kv_cache_groups, O.d0
        num_reqs = idx_mapping.shape[0]
        nb = self.num_blocks.gpu
        with torch.cuda.stream(O.side):
            btm._gather_block_tables_kernel[(d0, num_reqs_padded)](
                idx_mapping, self.block_table_ptrs, self.input_block_table_ptrs, self.block_table_strides, nb,
                nb.stride(0), num_reqs, BLOCK_SIZE=1024)
        btm._gather_block_tables_kernel[(g - d0, num_reqs_padded)](
            idx_mapping, self.block_table_ptrs[d0:], self.input_block_table_ptrs[d0:], self.block_table_strides[d0:],
            nb[d0:], nb.stride(0), num_reqs, BLOCK_SIZE=1024)
        return tuple(bt[:num_reqs_padded] for bt in self.input_block_tables)

    def compute_slot_mappings(self, idx_mapping, query_start_loc, positions, num_tokens_padded, out=None):
        if not (O.bt and O.active is False and out is None and torch.cuda.current_stream() == O.main):
            return orig_s(self, idx_mapping, query_start_loc, positions, num_tokens_padded, out=out)
        num_reqs = idx_mapping.shape[0]
        sm = self.slot_mappings
        with torch.cuda.stream(O.side):
            btm._compute_slot_mappings_kernel[(O.d0, num_reqs + 1)](
                sm.shape[1], idx_mapping, query_start_loc, positions, self.block_table_ptrs,
                self.block_table_strides, self.block_sizes_tensor, self.kernel_block_sizes_tensor,
                self.slot_mapping_enabled, self.dcp_sharded, sm, sm.stride(0), self.cp_rank, CP_SIZE=self.cp_size,
                CP_INTERLEAVE=self.cp_interleave, PAD_ID=PAD_SLOT_ID, TRITON_BLOCK_SIZE=1024)
        O.main.wait_stream(O.side)
        O.bt = False
        O.n2 += 1
        if O.n2 == 1:
            log.warning("atdev %s: LL5-ovl2 target-group gather / slot mappings on the side stream ENGAGED",
                        OPT_NAME)
        return sm[:, :num_tokens_padded]

    B.gather_block_tables = gather_block_tables
    B.compute_slot_mappings = compute_slot_mappings


def _install_ovl() -> None:
    from vllm.v1.worker.gpu import model_runner as mr
    from vllm.v1.worker.gpu.spec_decode.dflash import cudagraph as dcg

    G = dcg.DFlashCudaGraphManager
    if getattr(G.run_fullgraph, "_atdev_ll5", False):
        return
    orig_run = G.run_fullgraph

    def run_fullgraph(self, desc):
        if not torch.cuda.is_current_stream_capturing():
            if O.side is None:
                O.side = torch.cuda.Stream()
            O.ev = torch.cuda.Event()
            O.ev.record()
            O.pending = True
        return orig_run(self, desc)

    run_fullgraph._atdev_ll5 = True
    G.run_fullgraph = run_fullgraph

    orig_comb = mr.combine_sampled_and_draft_tokens

    def combine_sampled_and_draft_tokens(*a, **kw):
        _ovl_join()
        return orig_comb(*a, **kw)

    mr.combine_sampled_and_draft_tokens = combine_sampled_and_draft_tokens

    cls = mr.GPUModelRunner
    orig_pi = cls.prepare_inputs

    def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc, num_active_loras):
        pend, O.pending = O.pending, False
        O.bt = False
        ok = (pend and not batch_req_state.has_prefill and not scheduler_output.scheduled_new_reqs
              and bool(scheduler_output.scheduled_spec_decode_tokens) and not getattr(self, "lora_config", None)
              and getattr(self, "pcp_manager", None) is None and getattr(self, "adaptive_verification", None) is None
              and getattr(self, "fast_prefill", None) is None and not torch.cuda.is_current_stream_capturing())
        if not ok:
            if pend:
                O.skip += 1
            return orig_pi(self, scheduler_output, batch_req_state, batch_desc, num_active_loras)
        O.bt = False
        if OVL >= 2:
            so = scheduler_output
            upd = bool(so.block_table_updates) or any(b is not None for b in so.scheduled_cached_reqs.new_block_ids)
            if O.d0 < 0:
                O.d0 = _draft_suffix(self)
            O.bt = (not upd) and O.d0 > 0
        O.main = torch.cuda.current_stream()
        O.side.wait_event(O.ev)
        torch.cuda.set_stream(O.side)
        O.active = True
        try:
            ib = orig_pi(self, scheduler_output, batch_req_state, batch_desc, num_active_loras)
        finally:
            _ovl_join()  # no-op unless combine was not reached
        for v in vars(ib).values():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                v.record_stream(O.main)
        O.n += 1
        if O.n == 1:
            log.warning("atdev %s: LL5-ovl prep overlapped with the drafter graph ENGAGED", OPT_NAME)
        return ib

    cls.prepare_inputs = prepare_inputs
    if OVL >= 2:
        _install_bt()


# ---------------------------------------------------------------------------------------------------------------
# LL5-gaps: no-profiler e1 / e2 timing
# ---------------------------------------------------------------------------------------------------------------


def _install_gaps() -> None:
    import statistics

    Gr = torch.cuda.CUDAGraph
    if getattr(Gr.replay, "_atdev_ll5", False):
        return
    orig = Gr.replay
    pend: list = []
    dur: dict = {}
    acc = {"T": [], "D": [], "e1": [], "e2": []}
    st = {"steps": 0}

    def resolve():
        while pend and pend[0][2].query():
            gid, a, b = pend.pop(0)
            d = a.elapsed_time(b) * 1e3
            if "tg" not in st:
                dur.setdefault(gid, []).append(d)
                big = sorted(((statistics.median(v), k) for k, v in dur.items() if len(v) >= 30), reverse=True)
                if len(big) >= 2:
                    st["tg"], st["dg"] = big[0][1], big[1][1]
                continue
            if gid == st["tg"]:
                if st.get("d") is not None:
                    acc["e2"].append(st["d"].elapsed_time(a) * 1e3)
                    st["steps"] += 1
                acc["T"].append(d)
                st["t"], st["d"] = b, None
            elif gid == st["dg"]:
                if st.get("t") is not None:
                    acc["e1"].append(st["t"].elapsed_time(a) * 1e3)
                acc["D"].append(d)
                st["d"], st["t"] = b, None
        if st["steps"] >= GAPS:
            log.warning("atdev %s GAPS: %d steps medians: T %.1f D %.1f e1 %.1f e2 %.1f us (means e1 %.1f e2 %.1f)"
                        " ovl %d skip %d", OPT_NAME, st["steps"],
                        *(statistics.median(acc[k]) if acc[k] else -1 for k in ("T", "D", "e1", "e2")),
                        *(statistics.mean(acc[k]) if acc[k] else -1 for k in ("e1", "e2")), O.n, O.skip)
            for k in acc:
                acc[k] = []
            st["steps"] = 0

    def replay(self):
        if torch.cuda.is_current_stream_capturing():
            return orig(self)
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        r = orig(self)
        b.record()
        pend.append((id(self), a, b))
        resolve()
        return r

    replay._atdev_ll5 = True
    Gr.replay = replay


_TEXT = None


def text5():
    global _TEXT
    if _TEXT is None:
        from torch.utils.cpp_extension import load
        _TEXT = load(name="atdev_ll5_tail", sources=[os.path.join(os.path.dirname(__file__), "csrc", "ll5_tail.cu")],
                     extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"], verbose=False)
    return _TEXT


_TREXT = None


def trext():
    global _TREXT
    if _TREXT is None:
        from torch.utils.cpp_extension import load
        _TREXT = load(name="atdev_ll5_tree", sources=[os.path.join(os.path.dirname(__file__), "csrc", "ll5_tree.cu")],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"], verbose=False)
    return _TREXT


def tree_ok(scores, cand, num_nodes, chain, unary, mix, pw) -> bool:
    use_mix = unary is not None and (mix != 0.0 or pw != 1.0)
    return (not chain and not use_mix and num_nodes == 16 and scores.dim() == 4 and scores.shape[2] == 16
            and scores.shape[3] == 16 and scores.is_contiguous() and scores.dtype == torch.float32
            and cand.is_contiguous() and cand.dtype == torch.int64)


def _install_tree() -> None:
    from atdev_serve import gdn_tree as gt

    orig = gt.build_tree_gpu
    if getattr(orig, "_atdev_ll5", False):
        return

    def build_tree_gpu(scores, cand, rmap, rmap_stride, tok_out, meta_parent, meta_depth, meta_anc, meta_conv,
                       num_nodes, chain=False, temperature=1.0, unary=None, mix=0.0, pw=1.0):
        if not tree_ok(scores, cand, num_nodes, chain, unary, mix, pw):
            return orig(scores, cand, rmap, rmap_stride, tok_out, meta_parent, meta_depth, meta_anc, meta_conv,
                        num_nodes, chain=chain, temperature=temperature, unary=unary, mix=mix, pw=pw)
        _once("tree", "LL5-tree CUDA tree build ENGAGED")
        trext().tree_build(scores, cand, rmap, int(rmap_stride), tok_out, meta_parent, meta_depth, meta_anc,
                           meta_conv, 1.0 / float(temperature))

    build_tree_gpu._atdev_ll5 = True
    gt.build_tree_gpu = build_tree_gpu


def _install_dg() -> None:
    import inspect
    import textwrap
    from vllm.v1.worker.gpu import model_runner as mr

    cls = mr.GPUModelRunner
    if getattr(cls, "_atdev_ll5_dg", False):
        return
    src = textwrap.dedent(inspect.getsource(inspect.unwrap(cls.sample_tokens)))
    if not src.startswith("@torch.inference_mode()\n@step_eplb_after()\ndef sample_tokens("):
        raise RuntimeError(f"atdev {OPT_NAME}: LL5-dg unexpected sample_tokens source head")
    old = "self.req_states.draft_tokens[input_batch.idx_mapping],\n"
    if src.count(old) != 1 or src.count("input_batch,\n" + " " * 12 + old) != 1:
        raise RuntimeError(f"atdev {OPT_NAME}: LL5-dg sample_tokens source mismatch")
    src = src.replace(old, "(self.req_states.draft_tokens[input_batch.idx_mapping]"
                           " if input_batch.has_structured_output_reqs"
                           " else self.req_states.draft_tokens[: input_batch.num_reqs]),\n")
    g = dict(vars(mr))
    exec(compile(src, mr.__file__ + " (atdev ll5_dg)", "exec"), g)
    cls.sample_tokens = g["sample_tokens"]
    cls._atdev_ll5_dg = True
    log.warning("atdev %s: LL5-dg dead draft-token gather in sample_tokens removed ENGAGED", OPT_NAME)


def register() -> None:
    if DG:
        _install_dg()
    if TREE:
        trext()
        _install_tree()
    if TAIL:
        from atdev_serve import ll1_fuse
        ll1_fuse._TEXT = text5()  # ll1_fuse.tail_run -> text().tail2 of the LL5 build
        log.warning("atdev %s: LL5-tail register sort ENGAGED (ll1 tail ext replaced)", OPT_NAME)
    if CN:
        _install_cn()
    if OVL:
        _install_ovl()
    if GAPS:
        _install_gaps()
    log.warning("atdev %s: registered (cn=%d ovl=%d gaps=%d tail=%d tree=%d dg=%d)", OPT_NAME, CN, OVL, GAPS, TAIL,
                TREE, DG)
