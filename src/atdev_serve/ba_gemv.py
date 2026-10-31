"""ba_gemv: the GDN layer's `in_proj_ba` 5120->96 bf16 GEMV as one
deterministic split-K Triton launch (ATDEV_OPT=ba_gemv).

Why: cuBLAS runs this 1 MB GEMV as `ampere_s16816gemm` + `splitKreduce`
(5.0 + 3.3 us at M=16, 5.2 + 3.3 us at M=64), i.e. two latency-bound
launches per GDN layer, 48 per step. Here
grid (N/16, SPLIT_K) programs each compute a [M, 16] fp32 partial over K/SPLIT_K
on tensor cores, write it to a workspace and take a ticket; the last arriver of
a column block sums the SPLIT_K partials in fixed order (k-slice 0..S-1), writes
bf16 and resets the ticket (CUDA-graph safe, no memset). Output differs from
cuBLAS only by fp32 summation order.

Active for decode-sized M (<= 64); larger M (prefill) stays on F.linear.
"""
from __future__ import annotations

import logging

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

OPT_NAME = "ba_gemv"
SUPPORTED_VLLM_VERSION = "0.31.0"
MAX_M = 64
BN = 16
_CFG = (8, 128, 4)  # (split_k, block_k, num_warps)
SPLIT_MAX = 32
log = logging.getLogger("atdev_serve")

if triton is not None:

    @triton.jit
    def _ba_gemv_kernel(x_ptr, stride_x, w_ptr, stride_w, o_ptr, stride_o, ws_ptr, tick_ptr,
                        M, N, K,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        SPLIT: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = tl.arange(0, BM)
        cols = pid_n * BN + tl.arange(0, BN)
        rmask = rows < M
        kk = tl.arange(0, BK)
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        k0 = pid_k * (K // SPLIT)
        for ks in range(0, K // SPLIT, BK):
            kidx = k0 + ks + kk
            xk = tl.load(x_ptr + rows[:, None] * stride_x + kidx[None, :],
                         mask=rmask[:, None], other=0.0)                       # [BM, BK]
            wk = tl.load(w_ptr + cols[None, :] * stride_w + kidx[:, None])     # [BK, BN]
            acc = tl.dot(xk, wk, acc)
        ws = ws_ptr + (pid_k * BM + rows)[:, None] * N + cols[None, :]
        tl.store(ws, acc)
        tl.debug_barrier()
        t = tl.atomic_add(tick_ptr + pid_n, 1, sem="acq_rel")
        if t == SPLIT - 1:
            tot = tl.zeros([BM, BN], dtype=tl.float32)
            for s in tl.static_range(SPLIT):
                tot += tl.load(ws_ptr + (s * BM + rows)[:, None] * N + cols[None, :])
            tl.store(o_ptr + rows[:, None] * stride_o + cols[None, :],
                     tot.to(o_ptr.dtype.element_ty), mask=rmask[:, None])
            tl.atomic_xchg(tick_ptr + pid_n, 0)


_BUF: dict = {}


def _bufs(dev: torch.device, n: int):
    b = _BUF.get(dev.index)
    if b is None:
        b = _BUF[dev.index] = dict(
            ws=torch.empty(SPLIT_MAX * MAX_M * 256, dtype=torch.float32, device=dev),
            tick=torch.zeros(64, dtype=torch.int32, device=dev),
        )
    assert n <= 256 and n // BN <= 64
    return b


def ba_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [M, K] bf16, w [N, K] bf16 -> [M, N] bf16 (= F.linear(x, w))."""
    split, bk, warps = _CFG
    m, k = x.shape
    n = w.shape[0]
    if (m == 0 or m > MAX_M or n % BN or k % (split * bk) or split > SPLIT_MAX
            or w.stride(1) != 1 or x.stride(1) != 1):
        return torch.nn.functional.linear(x, w)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    b = _bufs(x.device, n)
    bm = max(16, triton.next_power_of_2(m))
    _ba_gemv_kernel[(n // BN, split)](
        x, x.stride(0), w, w.stride(0), out, out.stride(0), b["ws"], b["tick"], m, n, k,
        BM=bm, BN=BN, BK=bk, SPLIT=split, num_warps=warps)
    return out


def _op(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return ba_gemv(x, w)


def _fake(x, w):
    return torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)


def _method_class():
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    class _BaMethod(UnquantizedLinearMethod):
        """The GDN layer's in_proj_ba UnquantizedLinearMethod with apply() routed
        to the Triton GEMV (class swapped in place, so isinstance checks and
        process_weights_after_loading behave as stock)."""

        def apply(self, layer, x, bias=None):
            if (bias is not None or x.dtype != torch.bfloat16
                    or layer.weight.dtype != torch.bfloat16 or x.shape[-1] != layer.weight.shape[1]):
                return super().apply(layer, x, bias)
            x2 = x.reshape(-1, x.shape[-1])
            out = torch.ops.vllm.atdev_ba_gemv(x2, layer.weight)
            return out.reshape(x.shape[:-1] + (layer.weight.shape[0],))

    return _BaMethod, UnquantizedLinearMethod


_REGISTERED = False


def register() -> None:
    global _REGISTERED
    import vllm
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn_mod
    from vllm.utils.torch_utils import direct_register_custom_op

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written for vLLM {SUPPORTED_VLLM_VERSION}, "
                           f"found {vllm.__version__}")
    if triton is None:
        raise RuntimeError(f"atdev {OPT_NAME}: Triton required")
    if _REGISTERED:
        return
    _REGISTERED = True
    direct_register_custom_op("atdev_ba_gemv", _op, fake_impl=_fake)
    ba_cls, unq_cls = _method_class()
    cls = gdn_mod.QwenGatedDeltaNetAttention
    orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        lin = getattr(self, "in_proj_ba", None)
        if lin is not None and type(lin.quant_method) is unq_cls:
            lin.quant_method.__class__ = ba_cls
            if not getattr(cls, "_atdev_ba_logged", False):
                cls._atdev_ba_logged = True
                log.warning("atdev %s: in_proj_ba -> deterministic split-K Triton GEMV (M<=%d)",
                            OPT_NAME, MAX_M)

    cls.__init__ = __init__
    log.info("atdev %s: registered", OPT_NAME)
