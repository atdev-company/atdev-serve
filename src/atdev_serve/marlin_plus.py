"""marlin_plus: vLLM's own Marlin kernel, rebuilt with a per-shape launch config
(ATDEV_OPT=marlin_plus), plus the W4A4 (s4 activation) GEMM used at c=32.

Marlin's inner loop (int4 dequant, mma.sync, striped split-K) is unchanged; what changes is the
launch config per (shape, M-bucket): thread tile (k, n, threads), pipeline stages (4/6/8 instead
of vLLM's fixed 4) and CTAs per SM. vLLM hard-codes stages=4 and a 3-entry priority list
(csrc marlin.cu determine_exec_config).

With int8 activations (VLLM_MARLIN_INPUT_DTYPE=int8) a table entry may instead select the W4A4
kernel (marlin_plus_a4_build): the bf16 activation is quantised to s4 per (row, 128-group) and
multiplied with the same int4 weights.

Weights/scales are exactly vLLM's Marlin tensors (no extra memory). Kernel source:
csrc/marlin_plus (vLLM v0.31.0 csrc, Apache-2.0), built by marlin_plus_build.py. Per-shape
configs: the ATDEV_MP_CONFIG json; shapes / M not in the table run stock vLLM Marlin.
"""
from __future__ import annotations

import functools
import json
import os

import torch

OPT_NAME = "marlin_plus"
SUPPORTED_VLLM_VERSION = "0.31.0"
DEFAULT_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "marlin_plus_cfg.json")

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
    """{"a8"|"bf16": {"N,K": {bucket: {"cfg": [ints], "a4": bool}}}}"""
    global _TABLE
    if _TABLE is None:
        path = os.environ.get("ATDEV_MP_CONFIG", DEFAULT_CFG)
        _TABLE = json.load(open(path)) if os.path.exists(path) else {}
    return _TABLE


@functools.lru_cache(maxsize=4096)
def bucket(n: int, k: int, m: int, a8: bool) -> str:
    """m_bucket, refined for 64 < m <= 128: a shape entry may carry numeric buckets "80".."128"
    (W4A4 tmb=6 at M=96, tmb=8 at M=128); the smallest numeric bucket >= m wins, else "big"."""
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


_MOD_A4 = None
A4_MAX_K = 17408
A4_MAX_TMB = 8  # tmb 6 / 8 = the M=96 / 128 buckets


def mod_a4():
    """W4A4 kernels (s4 activations, separate extension, marlin_plus_a4_build)."""
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
            # W4A4: s4 A fragments [K/64][tmb][32][4] words + per-(group,row) scales
            a4_pk=torch.zeros(A4_MAX_K // 64 * A4_MAX_TMB * 32 * 4, dtype=torch.int32, device=dev),
            a4_s=torch.zeros(A4_MAX_K // 128 * 16 * A4_MAX_TMB, dtype=torch.float32, device=dev),
        )
    return b


def a4_scratch(b: dict, dev: torch.device):
    """Split-K scratch of the W4A4 GEMM (parallel reduction build: per-column tickets + one fp32
    partial slot per slice), allocated on first use."""
    if "a4_locks" not in b:
        b["a4_locks"] = torch.zeros(1024, dtype=torch.int32, device=dev)
        b["a4_ctmp"] = torch.empty(64 << 18, dtype=torch.float32, device=dev)
    return b["a4_locks"], b["a4_ctmp"]


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
    if ent is None or (a8 and not ent.get("a4")):
        return _stock(x, mq, ms, gs, n, k, a8)
    key = (n, k, bucket(n, k, m, a8), a8)
    if key not in _SEEN:
        _SEEN.add(key)
        import logging
        logging.getLogger("atdev_serve").warning(
            "atdev marlin_plus: N=%d K=%d M=%d (bucket %s, a8=%s) -> cfg %s a4=%s",
            n, k, m, key[2], a8, ent["cfg"], ent.get("a4", False))
    b = bufs(x.device)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if x.stride(-1) != 1 or x.stride(0) % 8:
        x = x.contiguous()
    cfg = list(ent["cfg"])
    if not a8:
        return mod().gemm(x, mq, ms, None, out, b["locks"], b["c_tmp"], n, k, cfg, True, False)
    # W4A4: cfg [tn, threads, stages, bps, tmb, MW, DEFER], bf16 x in
    tmb = cfg[4]
    if m > 16 * tmb or k > A4_MAX_K or tmb > A4_MAX_TMB:
        return _stock(x, mq, ms, gs, n, k, a8)
    a4 = mod_a4()
    a4.quant_a4(x, gs.reshape(1).float(), b["a4_pk"], b["a4_s"], tmb)
    locks, ctmp = a4_scratch(b, x.device)
    return a4.gemm_a4(b["a4_pk"], b["a4_s"], mq, ms, out, locks, ctmp, m, n, k, cfg)


_WS = {}


def mp_linear_q(a_q: torch.Tensor, a_s: torch.Tensor, mq: torch.Tensor, ms: torch.Tensor,
                n: int, k: int) -> torch.Tensor:
    """Stock vLLM W4A8 Marlin GEMM on activations already quantised outside the op (a_s already
    includes input_global_scale). Used by fused_quant for the M buckets without a W4A4 entry."""
    m = a_q.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a_q.device)
    if m == 0:
        return out
    from vllm.scalar_type import scalar_types
    ws = _WS.get(a_q.device)
    if ws is None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new)
        ws = _WS[a_q.device] = marlin_make_workspace_new(a_q.device)
    return torch.ops._C.marlin_gemm(a_q, None, mq, None, ms, a_s, None, None, ws,
                                    scalar_types.uint4b8.id, m, n, k, False, True, False)


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
    if not _REGISTERED:
        direct_register_custom_op("atdev_mp_linear", mp_linear, fake_impl=_fake)
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
