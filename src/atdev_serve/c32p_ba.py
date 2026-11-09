"""C32p: GDN in_proj_ba on a side CUDA stream, concurrent with the causal conv
(ATDEV_C32P_BA=1, needs fused_quant + gdn_single_state CUDA verify). Bit-identical (same cuBLAS GEMM).

In the bcpsz stack each GDN layer runs, on one stream, qkvz MarlinA4 GEMM -> in_proj_ba (cuBLAS Kernel2
5.9 us + splitKreduce 2.5 us) -> conv_update (7.0 us) -> GDN, although ba (reads the bf16 normed x only)
and the conv (reads qkvz only) are independent; only the GDN kernel reads both.

* fused_quant._gdn_forward passes the bf16 layer input xb [M, hidden] in place of ba to the core op
  (the op is opaque to Dynamo; the width tells the two apart);
* the patched _forward_core_fused_norm_packed (inside the op, so the fork AND the join are in the same
  FULL-graph capture / eager call): on the single-state CUDA spec-verify path it records a fork on the
  current stream, runs torch.mm(xb, W_ba^T, out=ba) on a side stream (= F.linear's mm, same cuBLAS
  call), and arms a join that the GDN launch (gdn_single_state_verify_cuda) performs first, so the conv
  overlaps the ba GEMM; any other path (prefill, mixed, stock) computes ba serially before the body;
* a join still armed at the end of the op is performed there (never leaves the side stream open).

ATDEV_C32P_EARLY=1: the fork is recorded before the qkvz GEMM instead (ba also overlaps the GEMM tail);
needs the GEMM issued from inside the op as well -> not implemented (the GEMM is traced by Dynamo).
"""
from __future__ import annotations

import logging
import os

import torch

from atdev_serve import c32n_ba

OPT_NAME = "c32p_ba"
log = logging.getLogger("atdev_serve")
_SIDE: dict = {}
_JOIN: list = [None]  # (current stream, event) of the armed join
_STATS = {"fork": 0, "serial": 0, "late_join": 0}
# C32n2: ATDEV_C32N_BA=side -> c32n_ba.py (cuBLAS-exact one-launch ba kernel)
_NPC = int(os.environ.get("ATDEV_C32N_BA_NPC", "4"))


def _n2_mode() -> str:
    return c32n_ba.mode()


def enabled() -> bool:
    return os.environ.get("ATDEV_C32P_BA", "0") == "1"


def _side(dev):
    s = _SIDE.get(dev.index)
    if s is None:
        s = _SIDE[dev.index] = torch.cuda.Stream(device=dev)
    return s


def _fork_ok(layer, x) -> bool:
    """True iff the op body reaches gdn_single_state_verify_cuda (the join point) before any reader of ba:
    mirrors QwenGatedDeltaNetAttention._forward_core_fused_norm + SingleStateGDNAttention's CUDA branch."""
    from vllm.forward_context import get_forward_context
    md = get_forward_context().attn_metadata
    md = md.get(layer.prefix) if isinstance(md, dict) else None
    if md is None or getattr(md, "num_prefills", 1) != 0:
        return False
    if getattr(layer, "atdev_impl", None) != "cuda":
        return False
    try:
        if not layer._can_use_fused_gdn_mtp_decode(md):
            return False
    except Exception:  # noqa: BLE001
        return False
    si = md.spec_state_indices_tensor
    return layer.kv_cache[1].dtype == torch.bfloat16 and si is not None and si.size(1) <= 8 and \
        x.dim() == 2 and x.stride(1) == 1


def join() -> None:
    j = _JOIN[0]
    if j is not None:
        _JOIN[0] = None
        j[0].wait_event(j[1])


_HOOK: list = [False]


def _hook_verify() -> None:
    """Wrap gdn_single_state.gdn_single_state_verify_cuda as it is at the first op call (after every
    registration, so gdn_a4_fold's wrapper is inside ours): the join precedes any GDN launch."""
    from atdev_serve import gdn_single_state as gss
    _HOOK[0] = True
    o_verify = gss.gdn_single_state_verify_cuda

    def gdn_single_state_verify_cuda(**kw):
        join()
        return o_verify(**kw)

    gdn_single_state_verify_cuda._atdev_c32p = True
    gss.gdn_single_state_verify_cuda = gdn_single_state_verify_cuda


def register() -> None:
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn_mod
    from atdev_serve import compile_cache
    cls = gdn_mod.QwenGatedDeltaNetAttention
    if getattr(cls, "_atdev_c32p", False):
        return
    cls._atdev_c32p = True
    compile_cache.salt_aot_cache(OPT_NAME, __file__)
    o_packed = cls._forward_core_fused_norm_packed

    def _forward_core_fused_norm_packed(self, mixed_qkvz, ba, core_attn_out):
        w = self.in_proj_ba.weight
        if ba.dim() != 2 or ba.shape[-1] != w.shape[1] or ba.shape[-1] == w.shape[0]:
            return o_packed(self, mixed_qkvz, ba, core_attn_out)  # a real ba (other callers)
        x = ba
        if not _HOOK[0]:
            _hook_verify()
        out = torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)
        nm = _n2_mode()
        ncfg = None
        if nm and c32n_ba.table(w) is None and not torch.cuda.is_current_stream_capturing():
            c32n_ba.ensure(w)  # first eager call (profile run): calibrate vs torch.mm before any capture
        if nm and _fork_ok(self, x) and c32n_ba.ok_tensors(x, w):
            ncfg = c32n_ba.cfg_for(w, x.shape[0])
        if _fork_ok(self, x):
            cur = torch.cuda.current_stream(x.device)
            side = _side(x.device)
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                if ncfg is not None:  # C32n2 side: one low-CTA launch, cuBLAS-exact
                    c32n_ba.run(x, w, out, ncfg[0], ncfg[1], 0, _NPC)
                else:
                    torch.mm(x, w.t(), out=out)
                ev = torch.cuda.Event()
                ev.record(side)
            _JOIN[0] = (cur, ev)
            _STATS["fork"] += 1
            if _STATS["fork"] == 1:
                log.warning("atdev %s: in_proj_ba on a side stream, joined before the GDN kernel (M=%d)",
                            OPT_NAME, x.shape[0])
        else:
            torch.mm(x, w.t(), out=out)
            _STATS["serial"] += 1
        try:
            return o_packed(self, mixed_qkvz, out, core_attn_out)
        finally:
            if _JOIN[0] is not None:
                _STATS["late_join"] += 1
                if _STATS["late_join"] == 1:
                    log.warning("atdev %s: join at the end of the op (GDN verify not reached)", OPT_NAME)
                join()

    cls._forward_core_fused_norm_packed = _forward_core_fused_norm_packed

    log.warning("atdev %s: registered (ATDEV_C32P_BA=1: GDN in_proj_ba on a side stream || conv_update)",
                OPT_NAME)
