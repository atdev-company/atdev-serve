"""W4A16 MTP drafter layer (ATDEV_OPT=mtp_w4).

The RedHatAI Qwen3.8-27B-INT4 checkpoint keeps its MTP module in bf16
(``quantization_config.ignore = ["re:^mtp.*"]``): fc 10240->5120 plus one
full decoder layer, ~0.85 GB of bf16 linear weights read on every draft
pass (c32-analysis.md: part of the 4.1 ms/step cuBLAS bucket). The MTP only
*proposes* tokens; the target verifies them, so a lower-precision MTP cannot
change generated text, only the acceptance rate.

``register()`` wraps ``UnquantizedLinearMethod.process_weights_after_loading``
/ ``apply``: every unquantized linear whose prefix is under ``mtp.`` gets an
int4 g128 symmetric (RTN) Marlin copy (``shadow_head.ShadowHead``, the same
GPTQ-Marlin W4A16 kernel the target uses) and its forward runs on it; the
bf16 weight is kept (ATDEV_MTP_W4_FREE=1 drops it).

Env knobs:
  ATDEV_MTP_W4_BITS   4 (default) or 8
  ATDEV_MTP_W4_MATCH  regex on the layer prefix (default ``(^|\\.)mtp\\.``)
  ATDEV_MTP_W4_MS     "tk,tn,threads,stages,bps,tmb,MW,DEFER": eager calls with 64 < M <= 16*tmb
                      (mtp_prune's first pass at c=32 MTP k=2: M=96 fc / qkv) run one weight pass on
                      the marlin_plus M-split W4A8 kernel (int8 per-token activations, an extra
                      int8-path int4 copy of the layers matching ATDEV_MTP_W4_MS_MATCH, default fc /
                      qkv_proj) instead of stock Marlin's 64 + 32 row W4A16 launches. Drafter-only
                      numerics; every other call keeps the W4A16 copy.
"""

from __future__ import annotations

import os
import re

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

OPT_NAME = "mtp_w4"
_ATTR = "_atdev_w4"
_STATS = {"layers": 0, "bf16_bytes": 0, "q_bytes": 0}


def _eligible(layer: torch.nn.Module, pattern: re.Pattern) -> bool:
    w = getattr(layer, "weight", None)
    prefix = getattr(layer, "prefix", "") or ""
    return (
        isinstance(w, torch.Tensor)
        and w.is_cuda
        and w.dim() == 2
        and w.dtype in (torch.bfloat16, torch.float16)
        and bool(pattern.search(prefix))
        and w.shape[0] % 64 == 0
        and w.shape[1] % 128 == 0
        and getattr(layer, "tp_size", 1) == 1
    )


def register() -> None:
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from atdev_serve.shadow_head import ShadowHead, ShadowHeadConfig

    if getattr(UnquantizedLinearMethod, "_atdev_mtp_w4", False):
        return
    bits = int(os.environ.get("ATDEV_MTP_W4_BITS", "4"))
    pattern = re.compile(os.environ.get("ATDEV_MTP_W4_MATCH", r"(^|\.)mtp\."))
    a8 = os.environ.get("ATDEV_MTP_W4_A8", "0") == "1"  # W4A8 (int8 per-token act quant) at M<=64
    cfg = ShadowHeadConfig(bits=bits, group_size=128, screen=1, a8=a8)
    ms_cfg = [int(v) for v in os.environ.get("ATDEV_MTP_W4_MS", "").split(",") if v]
    ms_pat = re.compile(os.environ.get("ATDEV_MTP_W4_MS_MATCH", r"\.(fc|qkv_proj)$"))
    orig_pwal = UnquantizedLinearMethod.process_weights_after_loading
    orig_apply = UnquantizedLinearMethod.apply

    def process_weights_after_loading(self, layer):
        orig_pwal(self, layer)
        if not _eligible(layer, pattern):
            return
        w = layer.weight.data
        shadow = ShadowHead(w, cfg)
        # plain buffers on the layer (traced by torch.compile like any weight)
        layer.register_buffer("atdev_w4_qweight", shadow.qweight, persistent=False)
        layer.register_buffer("atdev_w4_scales", shadow.scales, persistent=False)
        layer.register_buffer("atdev_w4_zp", shadow.zp, persistent=False)
        if shadow.gs is not None:
            layer.register_buffer("atdev_w4_gs", shadow.gs, persistent=False)
        setattr(layer, _ATTR, (shadow.size_n, shadow.size_k))
        _STATS["layers"] += 1
        _STATS["bf16_bytes"] += w.numel() * w.element_size()
        _STATS["q_bytes"] += shadow.nbytes
        if ms_cfg and ms_pat.search(layer.prefix or ""):
            sh8 = shadow if a8 else ShadowHead(w, ShadowHeadConfig(bits=bits, group_size=128,
                                                                   screen=1, a8=True))
            layer.register_buffer("atdev_ms_qweight", sh8.qweight, persistent=False)
            layer.register_buffer("atdev_ms_scales", sh8.scales, persistent=False)
            layer.register_buffer("atdev_ms_gs", sh8.gs, persistent=False)
        if os.environ.get("ATDEV_MTP_W4_FREE", "0") == "1":
            layer.weight.data = torch.empty(0, w.shape[1], dtype=w.dtype, device=w.device)
        logger.warning(
            "atdev mtp_w4: %s -> int%d g128 Marlin (%d layers, %.3f GB bf16 -> %.3f GB)",
            layer.prefix, bits, _STATS["layers"], _STATS["bf16_bytes"] / 1e9,
            _STATS["q_bytes"] / 1e9)


    def _ms(layer, x, nk):
        from vllm.model_executor.layers.quantization.utils.int8_utils import per_token_quant_int8

        from atdev_serve import marlin_plus as mpl
        b = mpl.bufs(x.device)
        if x.stride(-1) != 1:
            x = x.contiguous()
        a_q, a_s = per_token_quant_int8(x)
        out = torch.empty((x.shape[0], nk[0]), dtype=x.dtype, device=x.device)
        return mpl.mod_ms().gemm_ms(a_q, layer.atdev_ms_qweight, layer.atdev_ms_scales,
                                    (a_s.view(-1) * layer.atdev_ms_gs).contiguous(), out,
                                    b["locks"], b["c_tmp"], nk[0], nk[1], ms_cfg)

    def apply(self, layer, x, bias=None):
        nk = getattr(layer, _ATTR, None)
        if nk is None:
            return orig_apply(self, layer, x, bias)
        # eager calls only (mtp_prune's first pass); traced graphs keep the stock op
        if (ms_cfg and not torch.compiler.is_compiling() and x.dim() == 2
                and 64 < x.shape[0] <= 16 * ms_cfg[5] and hasattr(layer, "atdev_ms_qweight")):
            y = _ms(layer, x, nk)
            return y + bias if bias is not None else y
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            apply_gptq_marlin_linear,
        )

        gs = getattr(layer, "atdev_w4_gs", None)
        y = apply_gptq_marlin_linear(
            input=x, weight=layer.atdev_w4_qweight, weight_scale=layer.atdev_w4_scales,
            weight_zp=layer.atdev_w4_zp, workspace=None, wtype=cfg.quant_type,
            output_size_per_partition=nk[0], input_size_per_partition=nk[1],
            use_fp32_reduce=True, input_global_scale=gs,
            input_dtype=torch.int8 if gs is not None else None)
        return y + bias if bias is not None else y

    UnquantizedLinearMethod.process_weights_after_loading = process_weights_after_loading
    UnquantizedLinearMethod.apply = apply
    UnquantizedLinearMethod._atdev_mtp_w4 = True
    logger.info("atdev mtp_w4: registered (bits=%d, a8=%s, match=%s)", bits, a8, pattern.pattern)
