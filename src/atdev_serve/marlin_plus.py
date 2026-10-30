"""marlin_plus: vLLM's own Marlin kernel, rebuilt with a tunable launch config
and a fused activation-quant prologue (ATDEV_OPT=marlin_plus).

Why: at c=32 (MTP k=1, M=64) the W4A8 Marlin GEMMs are 53% of the step at
0.56-0.95 TB/s, and every linear pays two extra launches (Triton per-token
int8 quant + Inductor `a_s * input_global_scale`, ~5 us). An earlier Triton
replacement lost 2-3x (LEDGER A7-w4, W48-mb); this keeps Marlin's inner loop
(int4->int8 lop3 dequant, mma.sync, striped split-K) and changes only:

1. launch config per (shape, M-bucket): thread tile (k, n, threads), pipeline
   stages (4/6/8 instead of 4) and CTAs per SM -- vLLM hard-codes stages=4 and a
   3-entry priority list (csrc marlin.cu determine_exec_config);
2. int8 path: the per-token quantisation runs inside the GEMM (every CTA
   quantises ~1 row, then a grid barrier) -> 2 fewer launches per linear;
   same formula as vLLM's `_per_token_quant_int8`.

Weights/scales are exactly vLLM's Marlin tensors (no extra memory). Kernel
source: src/atdev_serve/csrc/marlin_plus (vLLM v0.31.0 csrc, Apache-2.0),
built by marlin_plus_build.py. Per-shape configs: ATDEV_MP_CONFIG json
(scripts/bench_marlin_plus.py output -> scripts/mp_make_config.py); shapes/M
not in the table run stock vLLM Marlin.
"""
from __future__ import annotations

import functools
import json
import os

import torch

OPT_NAME = "marlin_plus"
SUPPORTED_VLLM_VERSION = "0.31.0"
DEFAULT_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "marlin_plus_cfg.json")
FQ_MAX_M = 256       # fused-quant scratch rows; larger M (prefill) -> stock path
FQ_MAX_K = 17408
# P3 U8 (bit-exact): W4A8 ms kernels take B as unsigned nibbles + C = -8*Sa; Sa (exact int32 per
# (row, 128-group) sums of the int8 activation codes) comes from the fq prologue or the fused_quant
# producers. Needs the ms module built with ATDEV_MS_U8=1 (same env var).
U8 = os.environ.get("ATDEV_MS_U8", "0") == "1"

_TABLE: dict | None = None
_BUF: dict = {}
_SEEN: set = set()
_MOD = None


def m_bucket(m: int, a8: bool) -> str:
    if not a8 and m <= 8:
        return "8"
    if m > 128:
        return "none"  # prefill / large batches: stock Marlin (untuned)
    return str(min((m + 15) // 16, 4) * 16) if m <= 64 else "big"


def table() -> dict:
    """{"a8"|"bf16": {"N,K": {bucket: {"cfg": [6 ints], "fq": bool}}}}"""
    global _TABLE
    if _TABLE is None:
        path = os.environ.get("ATDEV_MP_CONFIG", DEFAULT_CFG)
        _TABLE = json.load(open(path)) if os.path.exists(path) else {}
    return _TABLE


@functools.lru_cache(maxsize=4096)
def bucket(n: int, k: int, m: int, a8: bool) -> str:
    """m_bucket, refined for 64 < m <= 128: a shape entry may carry numeric buckets "80".."128"
    (P3 k=2: W4A4 tmb=6 at M=96, tmb=8 at M=128); the smallest numeric bucket >= m wins, else
    "big". Tables without numeric buckets above 64 route exactly as before."""
    b = m_bucket(m, a8)
    if b == "big":
        t = table().get("a8" if a8 else "bf16", {}).get(f"{n},{k}", {})
        wide = sorted(int(x) for x in t if x.isdigit() and 64 < int(x) <= 128 and int(x) >= m)
        if wide:
            return str(wide[0])
    return b


def lookup(n: int, k: int, m: int, a8: bool):
    t = table().get("a8" if a8 else "bf16", {}).get(f"{n},{k}", {})
    return t.get(bucket(n, k, m, a8))


_MOD_MS = None


def mod_ms():
    """REFRAME A2 M-split W4A8 kernels (separate extension, marlin_plus_ms_build)."""
    global _MOD_MS
    if _MOD_MS is None:
        from atdev_serve import marlin_plus_ms_build
        _MOD_MS = marlin_plus_ms_build.load()
    return _MOD_MS


_MOD_A4 = None
A4_MAX_K = 17408
A4_MAX_TMB = 8  # tmb 6 / 8 = the M=96 / 128 buckets (MTP k=2 / k=3 at c=32)


def mod_a4():
    """P3 W4A4 kernels (s4 activations, separate extension, marlin_plus_a4_build)."""
    global _MOD_A4
    if _MOD_A4 is None:
        from atdev_serve import marlin_plus_a4_build
        _MOD_A4 = marlin_plus_a4_build.load()
    return _MOD_A4


def mod():
    global _MOD
    if _MOD is None:
        from atdev_serve import marlin_plus_build
        _MOD = marlin_plus_build.load()
    return _MOD


def bufs(dev: torch.device) -> dict:
    b = _BUF.get(dev)
    if b is None:
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        b = _BUF[dev] = dict(
            locks=torch.zeros(sms * 4, dtype=torch.int32, device=dev),
            c_tmp=torch.empty(sms * 4 * 64 * 256, dtype=torch.float32, device=dev),
            fq_aq=torch.empty(FQ_MAX_M * FQ_MAX_K, dtype=torch.int8, device=dev),
            fq_as=torch.empty(FQ_MAX_M, dtype=torch.float32, device=dev),
            fq_bar=torch.zeros(2, dtype=torch.int32, device=dev),
            fq_sa=torch.zeros(FQ_MAX_M * (FQ_MAX_K // 128), dtype=torch.int32, device=dev),
            a_sa=torch.zeros(FQ_MAX_M * (FQ_MAX_K // 128), dtype=torch.int32, device=dev),
            sa_for=0,
            # W4A4: s4 A fragments [K/64][tmb][32][4] words + per-(group,row) scales
            a4_pk=torch.zeros(A4_MAX_K // 64 * A4_MAX_TMB * 32 * 4, dtype=torch.int32, device=dev),
            a4_s=torch.zeros(A4_MAX_K // 128 * 16 * A4_MAX_TMB, dtype=torch.float32, device=dev),
        )
    return b


def _stock(x, mq, ms, gs, n, k, a8):
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear)
    from vllm.scalar_type import scalar_types
    zp = torch.empty(0, dtype=torch.int, device=x.device)
    return apply_gptq_marlin_linear(
        input=x, weight=mq, weight_scale=ms, weight_zp=zp, workspace=None,
        wtype=scalar_types.uint4b8, output_size_per_partition=n,
        input_size_per_partition=k, input_global_scale=gs if a8 else None, bias=None,
        input_dtype=torch.int8 if a8 else None)


def mp_linear(x: torch.Tensor, mq: torch.Tensor, ms: torch.Tensor, gs: torch.Tensor,
              n: int, k: int, a8: bool) -> torch.Tensor:
    m = x.shape[0]
    ent = lookup(n, k, m, a8) if m > 0 else None
    if ent is None or (a8 and ent.get("fq") and m > FQ_MAX_M):
        return _stock(x, mq, ms, gs, n, k, a8)
    key = (n, k, bucket(n, k, m, a8), a8)
    if key not in _SEEN:
        _SEEN.add(key)
        import logging
        logging.getLogger("atdev_serve").warning(
            "atdev marlin_plus: N=%d K=%d M=%d (bucket %s, a8=%s) -> cfg %s fq=%s ms=%s a4=%s",
            n, k, m, key[2], a8, ent["cfg"], ent.get("fq", False), ent.get("ms", False),
            ent.get("a4", False))
    b = bufs(x.device)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if x.stride(-1) != 1 or x.stride(0) % 8:
        x = x.contiguous()
    cfg = list(ent["cfg"])
    if not a8:
        return mod().gemm(x, mq, ms, None, out, b["locks"], b["c_tmp"], n, k, cfg, True, False)
    gsv = gs.reshape(1).float()
    if ent.get("a4"):  # W4A4 (P3): cfg [tn, threads, stages, bps, tmb, MW, DEFER], bf16 x in
        tmb = cfg[4]
        if m > 16 * tmb or k > A4_MAX_K or tmb > A4_MAX_TMB:
            return _stock(x, mq, ms, gs, n, k, a8)
        a4 = mod_a4()
        a4.quant_a4(x, gsv, b["a4_pk"], b["a4_s"], tmb)
        if "A4_PAR_REDUCE" in os.environ.get("ATDEV_A4_DEFINES", ""):
            # parallel split-K reduction: per-column tickets + one fp32 partial slot per slice
            if "a4_locks" not in b:
                b["a4_locks"] = torch.zeros(1024, dtype=torch.int32, device=x.device)
                b["a4_ctmp"] = torch.empty(64 << 18, dtype=torch.float32, device=x.device)
            return a4.gemm_a4(b["a4_pk"], b["a4_s"], mq, ms, out, b["a4_locks"], b["a4_ctmp"], m, n, k,
                              cfg)
        return a4.gemm_a4(b["a4_pk"], b["a4_s"], mq, ms, out, b["locks"], b["c_tmp"], m, n, k, cfg)
    if ent.get("ms"):  # cfg: [tk, tn, threads, stages, bps, tmb, MW, DEFER]
        if ent.get("fq"):
            return mod_ms().gemm_ms(x, mq, ms, None, out, b["locks"], b["c_tmp"], n, k, cfg,
                                    gsv, b["fq_aq"], b["fq_as"], b["fq_bar"], b["fq_sa"] if U8 else None)
        from vllm.model_executor.layers.quantization.utils.int8_utils import per_token_quant_int8
        a_q, a_s = per_token_quant_int8(x)
        return mod_ms().gemm_ms(a_q, mq, ms, (a_s.view(-1) * gsv).contiguous(), out, b["locks"],
                                b["c_tmp"], n, k, cfg)
    if ent.get("fq"):
        return mod().gemm(x, mq, ms, None, out, b["locks"], b["c_tmp"], n, k, cfg, True, False,
                          gsv, b["fq_aq"], b["fq_as"], b["fq_bar"])
    from vllm.model_executor.layers.quantization.utils.int8_utils import per_token_quant_int8
    a_q, a_s = per_token_quant_int8(x)
    return mod().gemm(a_q, mq, ms, (a_s.view(-1) * gsv).contiguous(), out, b["locks"],
                      b["c_tmp"], n, k, cfg, True, False)


_WS = {}


def mp_linear_q(a_q: torch.Tensor, a_s: torch.Tensor, mq: torch.Tensor, ms: torch.Tensor,
                n: int, k: int) -> torch.Tensor:
    """W4A8 GEMM on activations already quantised outside the op (ATDEV_MP_QUANT=torch:
    the per-token quant is plain torch code, so Inductor can fuse it into the
    producer -- RMSNorm / SiLU-mul -- instead of a separate launch or an in-GEMM
    prologue + grid barrier). a_s already includes input_global_scale."""
    m = a_q.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a_q.device)
    if m == 0:
        return out
    ent = lookup(n, k, m, True)
    b = bufs(a_q.device)
    if ent is None:
        from vllm.scalar_type import scalar_types
        ws = _WS.get(a_q.device)
        if ws is None:
            from vllm.model_executor.layers.quantization.utils.marlin_utils import (
                marlin_make_workspace_new)
            ws = _WS[a_q.device] = marlin_make_workspace_new(a_q.device)
        return torch.ops._C.marlin_gemm(a_q, None, mq, None, ms, a_s, None, None, ws,
                                        scalar_types.uint4b8.id, m, n, k, False, True, False)
    key = (n, k, bucket(n, k, m, True), "q")
    if key not in _SEEN:
        _SEEN.add(key)
        import logging
        logging.getLogger("atdev_serve").warning(
            "atdev marlin_plus(q): N=%d K=%d M=%d (bucket %s) -> cfg %s ms=%s", n, k, m, key[2],
            ent["cfg"], ent.get("ms", False))
    cfg = list(ent["cfg"])
    if ent.get("ms"):
        sa = b["a_sa"] if (U8 and b.get("sa_for") == a_q.data_ptr()) else None
        return mod_ms().gemm_ms(a_q, mq, ms, a_s, out, b["locks"], b["c_tmp"], n, k, cfg,
                                None, None, None, None, sa)
    return mod().gemm(a_q, mq, ms, a_s, out, b["locks"], b["c_tmp"], n, k, cfg, True, False)


def _fake_q(a_q, a_s, mq, ms, n, k):
    return torch.empty((a_q.shape[0], n), dtype=torch.bfloat16, device=a_q.device)


def quant_torch(x: torch.Tensor, gs: torch.Tensor):
    """Same formula as vllm int8_utils._per_token_quant_int8 (round half away from 0),
    written as torch ops so Inductor can fuse it with the producer."""
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    v = xf * (127.0 / amax)
    a_q = (torch.sign(v) * torch.floor(v.abs() + 0.5)).to(torch.int8)
    a_s = (amax / 127.0).view(-1) * gs.reshape(1).float()
    return a_q, a_s


def _fake(x, mq, ms, gs, n, k, a8):
    return torch.empty((x.shape[0], n), dtype=x.dtype, device=x.device)


_REGISTERED = False
_CLS = None
_DRAFTER_LOADING = False  # set while the DFlash drafter loads: its linears stay stock bf16


def kernel_class():
    global _CLS
    if _CLS is not None:
        return _CLS
    from vllm.model_executor.kernels.linear.mixed_precision.marlin import MarlinLinearKernel
    from vllm.scalar_type import scalar_types

    class MarlinPlusLinearKernel(MarlinLinearKernel):
        """Stock Marlin weights; apply via marlin_plus (tuned launch + fused quant)."""

        @classmethod
        def can_implement(cls, c):
            ok, why = super().can_implement(c)
            if not ok:
                return ok, why
            k, n = c.partition_weight_shape
            if _DRAFTER_LOADING or c.weight_type != scalar_types.uint4b8:
                # drafter linears and 8-bit (fc8) weights: claim them so vLLM does not
                # switch them to int8 activations (W8A8 unsupported); run stock bf16 Marlin
                return True, None
            if c.zero_points or c.group_size != 128:
                return False, "marlin_plus: uint4b8 g128 symmetric only"
            # at selection time act_type is params_dtype (bf16); vLLM switches it
            # to int8 (VLLM_MARLIN_INPUT_DTYPE) only for `kernel_type is
            # MarlinLinearKernel`, so __init__ below re-applies that for us.
            if c.act_type not in (torch.bfloat16, torch.int8):
                return False, "marlin_plus: bf16 (W4A16) or int8 (W4A8) activations only"
            if n % 64 or k % 128:
                return False, f"marlin_plus: unpadded shapes only (K={k}, N={n})"
            return True, None

        def __init__(self, c, *args, **kwargs):
            super().__init__(c, *args, **kwargs)
            from vllm.model_executor.layers.quantization.utils.marlin_utils import (
                get_marlin_input_dtype)
            self._atdev_stock = _DRAFTER_LOADING or c.weight_type != scalar_types.uint4b8
            if self._atdev_stock:
                return  # bf16 activations, stock Marlin (W4A16 / W8A16)
            dt = get_marlin_input_dtype()
            if dt is not None and c.act_type == torch.bfloat16:
                if dt != torch.int8:
                    raise RuntimeError(f"marlin_plus: VLLM_MARLIN_INPUT_DTYPE {dt} unsupported")
                c.act_type = dt  # same as compressed_tensors_wNa16 does for Marlin

        def apply_weights(self, layer, x, bias=None):
            if self._atdev_stock:
                return super().apply_weights(layer, x, bias)
            c = self.config
            a8 = c.act_type == torch.int8
            gs = getattr(layer, "input_global_scale", None)
            if bias is not None or (a8 and gs is None) or x.dtype != torch.bfloat16:
                return super().apply_weights(layer, x, bias)
            k, n = c.partition_weight_shape
            w_q, w_s, _ = self._get_weight_params(layer)
            x2 = x.reshape(-1, x.shape[-1])
            if gs is None:
                gs = w_s  # unused placeholder (bf16 path)
            if a8 and os.environ.get("ATDEV_MP_QUANT") == "torch":
                a_q, a_s = quant_torch(x2, gs)
                out = torch.ops.vllm.atdev_mp_linear_q(a_q, a_s, w_q, w_s, n, k)
            else:
                out = torch.ops.vllm.atdev_mp_linear(x2, w_q, w_s, gs, n, k, a8)
            return out.reshape(x.shape[:-1] + (n,))

    _CLS = MarlinPlusLinearKernel
    return _CLS


def register() -> None:
    global _REGISTERED
    import vllm
    from vllm.logger import init_logger
    from vllm.model_executor.kernels import linear as kl
    from vllm.model_executor.kernels.linear.mixed_precision.marlin import MarlinLinearKernel
    from vllm.platforms import PlatformEnum
    from vllm.utils.torch_utils import direct_register_custom_op

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written for vLLM {SUPPORTED_VLLM_VERSION}, "
                           f"found {vllm.__version__}")
    from atdev_serve.compile_cache import salt_aot_cache
    salt_aot_cache(OPT_NAME, __file__)
    if os.environ.get("ATDEV_MP_QUANT"):  # traced graph differs -> separate AOT artifact
        from atdev_serve import compile_cache
        compile_cache._SALTS[OPT_NAME] += "+q=" + os.environ["ATDEV_MP_QUANT"]
    if not _REGISTERED:
        direct_register_custom_op("atdev_mp_linear", mp_linear, fake_impl=_fake)
        direct_register_custom_op("atdev_mp_linear_q", mp_linear_q, fake_impl=_fake_q)
        _REGISTERED = True
    # drafter (DFlash) linears keep bf16 activations even under VLLM_MARLIN_INPUT_DTYPE=int8
    try:
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
        orig_load = DFlashSpeculator.load_draft_model
        if not getattr(orig_load, "_atdev_mp", False):
            def load_draft_model(self, *args, **kwargs):
                global _DRAFTER_LOADING
                _DRAFTER_LOADING = True
                try:
                    return orig_load(self, *args, **kwargs)
                finally:
                    _DRAFTER_LOADING = False
            load_draft_model._atdev_mp = True
            DFlashSpeculator.load_draft_model = load_draft_model
    except ImportError:
        pass
    cls = kernel_class()
    lst = kl._POSSIBLE_KERNELS[PlatformEnum.CUDA]
    if cls not in lst:
        lst.insert(lst.index(MarlinLinearKernel), cls)
    kl._LINEAR_BACKEND_KERNEL_MAP["marlin"].add(cls)
    n_a8 = len(table().get("a8", {}))
    n_bf = len(table().get("bf16", {}))
    init_logger(__name__).warning(
        "atdev %s: registered (config %s: %d int8 shapes, %d bf16 shapes)", OPT_NAME,
        os.environ.get("ATDEV_MP_CONFIG", DEFAULT_CFG), n_a8, n_bf)
