"""W4A16 MTP drafter layer (ATDEV_OPT=mtp_w4).

The RedHatAI Qwen3.8-27B-INT4 checkpoint keeps its MTP module in bf16
(``quantization_config.ignore = ["re:^mtp.*"]``): fc 10240->5120 plus one
full decoder layer, ~0.85 GB of bf16 linear weights read on every draft
pass. The MTP only *proposes* tokens; the target verifies them, so a
lower-precision MTP cannot change generated text, only the acceptance rate.

``register()`` wraps ``UnquantizedLinearMethod.process_weights_after_loading``
/ ``apply``: every unquantized linear whose prefix is under ``mtp.`` gets an
int4 g128 symmetric (RTN) Marlin copy (``shadow_head.ShadowHead``, the same
GPTQ-Marlin W4A16 kernel the target uses) and its forward runs on it; the
bf16 weight is kept.
"""

from __future__ import annotations

import re

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

OPT_NAME = "mtp_w4"
_ATTR = "_atdev_w4"
_STATS = {"layers": 0, "bf16_bytes": 0, "q_bytes": 0}
_MATCH = re.compile(r"(^|\.)mtp\.")


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
    bits = 4
    pattern = _MATCH
    cfg = ShadowHeadConfig(bits=bits, group_size=128, screen=1)
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
        setattr(layer, _ATTR, (shadow.size_n, shadow.size_k))
        _STATS["layers"] += 1
        _STATS["bf16_bytes"] += w.numel() * w.element_size()
        _STATS["q_bytes"] += shadow.nbytes
        logger.warning(
            "atdev mtp_w4: %s -> int%d g128 Marlin (%d layers, %.3f GB bf16 -> %.3f GB)",
            layer.prefix, bits, _STATS["layers"], _STATS["bf16_bytes"] / 1e9,
            _STATS["q_bytes"] / 1e9)

    def apply(self, layer, x, bias=None):
        nk = getattr(layer, _ATTR, None)
        if nk is None:
            return orig_apply(self, layer, x, bias)
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            apply_gptq_marlin_linear,
        )

        y = apply_gptq_marlin_linear(
            input=x, weight=layer.atdev_w4_qweight, weight_scale=layer.atdev_w4_scales,
            weight_zp=layer.atdev_w4_zp, workspace=None, wtype=cfg.quant_type,
            output_size_per_partition=nk[0], input_size_per_partition=nk[1],
            use_fp32_reduce=True, input_global_scale=None, input_dtype=None)
        return y + bias if bias is not None else y

    UnquantizedLinearMethod.process_weights_after_loading = process_weights_after_loading
    UnquantizedLinearMethod.apply = apply
    UnquantizedLinearMethod._atdev_mtp_w4 = True
    logger.info("atdev mtp_w4: registered (bits=%d, match=%s)", bits, pattern.pattern)
