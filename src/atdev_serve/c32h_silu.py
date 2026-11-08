"""C32h (ATDEV_C32H_SILU=1, with ATDEV_OPT ...,fused_quant on the c=32 W4A4 stack): the SiLU.mul -> s4
producer between the MLP gate_up and down GEMMs (act_a4 / _silu_a4_kernel, 64 launches per step) is folded
into the gate_up MarlinA4 GEMM epilogue (A4_SILU=1 build of csrc/marlin_plus_a4, mp_a4_ne.h A4Silu):

  load:     gate_up weight + scales (Marlin int8-path layout) interleaved per n8 tile, in storage (no repack):
            GEMM n8 tile 2p <- gate columns 8p..8p+7, 2p+1 <- up columns 8p..8p+7. A 64-column GEMM tile then
            holds g and u of 32 y columns (= one k32 chunk of the down input) in the same mma fragments.
  epilogue: (after the split-K gather) the g / u tile goes to smem as bf16 (= the deployed bf16 output), each
            thread computes the 8 y = bf16(silu(g) u) of one s4 word (act_a4's op sequence), keeps them in smem and
            publishes the tile's per-row max |y| to PA as a (value, call epoch) 64-bit pair (per-CTA epoch counters
            in `cnt`, no tickets). The tile's quant (A4_SILU_DEFER=1: after the next slice's pipeline prologue)
            waits for the 3 other tiles of its 128-group (all CTAs co-resident), scale = group amax / 7, and the
            correctly rounded quotient via a branch-free fma correction (bit-identical to div.rn) -> s4 words +
            scales straight into the down GEMM's A buffers (pk2 / s2, separate from the gate_up input).
  no bf16 gate_up output, no act_a4 launch.

Numerics: y / s4 / scales are bit-identical to act_a4 on the GEMM's bf16 output; the interleave moves columns
between split-K tiles, so with the bf16 gather partials (A4_C32C_PB=1) g / u differ from the
un-interleaved GEMM at the bf16-rounding level -> gated. Every other M (prefill, buckets without a4 / tn 64 /
equal tmb) runs the unfused ops on the interleaved weights and un-interleaves the gate_up output first.
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("atdev_serve")
_B: dict = {}
_STATS = {"fused": 0, "fallback": 0}
# extra defines of the A4_SILU build, e.g. "A4_SILU_DEFER=1,A4_SILU_UNR=2" (LEDGER C32h-mb: quant after the next
# slice's prologue, +16 tmb * 88 B smem (DEFER 2: * 152 B) -> not for tmb 8 at 2 CTAs/SM)
_DEFS = os.environ.get("ATDEV_C32H_DEFS", "")
_DEFER = "A4_SILU_DEFER=1" in _DEFS or "A4_SILU_DEFER=2" in _DEFS


def enabled() -> bool:
    return os.environ.get("ATDEV_C32H_SILU", "0") == "1"


def _src_blocks(n: int, device) -> torch.Tensor:
    """GEMM n8 tile T' <- logical n8 tile: 2p <- p (gate), 2p + 1 <- n/16 + p (up)."""
    p = torch.arange(n // 16, device=device)
    return torch.stack([p, p + n // 16], 1).reshape(-1)


def perm_storage(mq: torch.Tensor, ms: torch.Tensor, n: int):
    """interleave a Marlin int8-path (a8) weight / scale pair per n8 tile (checked == repack of the permuted
    logical weight: per k32 row a 64-column block is 64 int4 = [warp half 2][lane 32][word 4], n8 tile =
    half * 4 + word; scales per group row and block: 8 int4 = [half 2][lane%4 4][word 4][2] int16)."""
    src = _src_blocks(n, mq.device)
    rows = mq.numel() // (n * 4)
    v = mq.reshape(rows, n // 64, 2, 32, 4).permute(0, 1, 2, 4, 3).reshape(rows, n // 8, 32)
    mq2 = v[:, src].reshape(rows, n // 64, 2, 4, 32).permute(0, 1, 2, 4, 3).reshape(mq.shape).contiguous()
    g = ms.shape[0]
    sv = ms.reshape(g, n // 64, 2, 4, 4, 2).permute(0, 1, 2, 4, 3, 5).reshape(g, n // 8, 4, 2)
    ms2 = sv[:, src].reshape(g, n // 64, 2, 4, 4, 2).permute(0, 1, 2, 4, 3, 5).reshape(ms.shape).contiguous()
    return mq2, ms2


def unperm(gu: torch.Tensor) -> torch.Tensor:
    """interleaved gate_up GEMM output [M, n] -> logical [gate | up]"""
    m, n = gu.shape
    v = gu.view(m, n // 16, 2, 8)
    return torch.cat([v[:, :, 0].reshape(m, n // 2), v[:, :, 1].reshape(m, n // 2)], 1)


def _mod():
    from atdev_serve import marlin_plus_a4_build
    return marlin_plus_a4_build.load_silu(_DEFS)


def _bufs(dev, n_gu: int, k_dn: int):
    from atdev_serve import marlin_plus as mp
    key = (dev.index, n_gu, k_dn)
    b = _B.get(key)
    if b is None:
        tmb = mp.A4_MAX_TMB
        b = _B[key] = dict(
            pk=torch.zeros(k_dn // 64 * tmb * 32 * 4, dtype=torch.int32, device=dev),
            s=torch.zeros(k_dn // 128 * 16 * tmb, dtype=torch.float32, device=dev),
            PA=torch.zeros(n_gu // 64 * 4 * 16 * tmb, dtype=torch.float32, device=dev),  # x2: TAG pairs
            cnt=torch.zeros(max(1024, 2 * n_gu // 256), dtype=torch.int32, device=dev),  # per-CTA epochs
        )
    return b


def plan(n_gu: int, k_gu: int, n_dn: int, k_dn: int, m: int):
    """(gate_up cfg, down entry) of the fused path, or None"""
    from atdev_serve import fused_quant as fq
    eg = fq._a4_ent(n_gu, k_gu, m)
    ed = fq._a4_ent(n_dn, k_dn, m)
    if eg is None or ed is None:
        return None
    cg = list(eg["cfg"])
    if cg[0] != 64 or cg[4] != ed["cfg"][4] or m > 16 * cg[4] or n_gu != 2 * k_dn or n_gu % 256:
        return None
    if _DEFER and cg[4] > 6:  # 2 CTAs/SM: tmb 8 + the deferred y tile exceeds the smem budget
        return None
    return cg, ed


def mlp(xq: torch.Tensor, xs: torch.Tensor, gu_q: torch.Tensor, gu_s: torch.Tensor, dn_q: torch.Tensor,
        dn_s: torch.Tensor, gs_dn: torch.Tensor, n_gu: int, k_gu: int, n_dn: int, k_dn: int) -> torch.Tensor:
    """down(silu_mul(gate_up(x))) with interleaved gate_up weights; x = the a4 fragments in marlin_plus' shared
    buffers (xq / xs: placeholders) or int8 (non-a4 M)."""
    from atdev_serve import fused_quant as fq
    from atdev_serve import marlin_plus as mp
    m = xq.shape[0]
    pl = plan(n_gu, k_gu, n_dn, k_dn, m) if m > 0 else None
    if pl is None:
        _STATS["fallback"] += 1
        gu = fq.fq_linear(xq, xs, gu_q, gu_s, n_gu, k_gu)
        q, s = fq.silu_quant(unperm(gu), gs_dn, None, n_dn)
        return fq.fq_linear(q, s, dn_q, dn_s, n_dn, k_dn)
    _STATS["fused"] += 1
    cg, ed = pl
    b = mp.bufs(xq.device)
    if "A4_PAR_REDUCE" in os.environ.get("ATDEV_A4_DEFINES", ""):  # as fused_quant.fq_linear
        if "a4_locks" not in b:
            b["a4_locks"] = torch.zeros(1024, dtype=torch.int32, device=xq.device)
            b["a4_ctmp"] = torch.empty(64 << 18, dtype=torch.float32, device=xq.device)
        lk, ct = b["a4_locks"], b["a4_ctmp"]
    else:
        lk, ct = b["locks"], b["c_tmp"]
    c = _bufs(xq.device, n_gu, k_dn)
    _mod().gemm_a4_silu(b["a4_pk"], b["a4_s"], gu_q, gu_s, lk, ct, m, n_gu, k_gu, cg,
                        c["pk"], c["s"], gs_dn, c["PA"], c["cnt"])
    out = torch.empty((m, n_dn), dtype=torch.bfloat16, device=xq.device)
    return mp.mod_a4().gemm_a4(c["pk"], c["s"], dn_q, dn_s, out, lk, ct, m, n_dn, k_dn, list(ed["cfg"]))


def _mlp_fake(xq, xs, gu_q, gu_s, dn_q, dn_s, gs_dn, n_gu, k_gu, n_dn, k_dn):
    return torch.empty((xq.shape[0], n_dn), dtype=torch.bfloat16, device=xq.device)


def permute_model(model) -> int:
    """interleave the gate_up weights of every decoder layer the fused_quant forward drives (after load)."""
    from atdev_serve import fused_quant as fq
    from vllm.model_executor.models import qwen3_next as qn
    cnt = 0
    for mod in model.modules():
        if not isinstance(mod, qn.Qwen3NextDecoderLayer):
            continue
        if mod.layer_scale or getattr(mod, "use_attn_reduce_scatter_for_moe", False) or not fq._layer_eligible(mod):
            continue
        gu = mod.mlp.gate_up_proj
        if getattr(gu, "_atdev_gu_perm", False):
            continue
        k = fq._mp_kernel(gu)
        kk, n = k.config.partition_weight_shape
        if n % 256:
            continue
        w_q, w_s, _ = k._get_weight_params(gu)
        q2, s2 = perm_storage(w_q, w_s, n)
        w_q.data.copy_(q2)
        w_s.data.copy_(s2)
        del q2, s2
        gu._atdev_gu_perm = True
        cnt += 1
    torch.cuda.empty_cache()
    return cnt


_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    from vllm.model_executor.model_loader import base_loader
    from vllm.utils.torch_utils import direct_register_custom_op
    from atdev_serve import compile_cache
    direct_register_custom_op("atdev_c32h_mlp", mlp, fake_impl=_mlp_fake)
    compile_cache._SALTS["c32h_silu"] = "1"
    orig_pwal = base_loader.process_weights_after_loading

    def process_weights_after_loading(model, model_config, target_device):
        orig_pwal(model, model_config, target_device)
        c = permute_model(model)
        if c:
            _mod()  # build / load outside any capture
            log.warning("atdev c32h_silu: gate_up weights interleaved per n8 tile on %d layers "
                        "(SiLU.mul -> s4 in the gate_up MarlinA4 epilogue)", c)

    base_loader.process_weights_after_loading = process_weights_after_loading
