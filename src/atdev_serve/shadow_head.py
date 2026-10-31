"""Screened draft head (ATDEV_OPT=shadow_head).

Drafters only need a short candidate list from the lm_head, not full logits:

* DFlash2 calls ``LogitsProcessor.get_top_k_tokens`` (top-16 per mask
  position, then the candidate selector walks them).
* MTP heads call ``LogitsProcessor.get_top_tokens`` (argmax) when
  ``use_local_argmax_reduction`` is set in the speculative config.

Stock vLLM computes these with a full bf16 GEMM over the shared target
lm_head (248320 x 5120 bf16 = 2.54 GB for Qwen3.8), i.e. ~1.6 ms per draft
pass on an A100. This module keeps an int4 g128 *shadow* of the lm_head
(0.66 GB) and computes:

1. approximate logits with the shadow (Marlin W4A16 GEMM),
2. the top-``screen`` (128) approximate candidates per row,
3. exact bf16 logits for only those candidates (row gather + bmm), and
4. the final top-k / argmax among them.

The result equals the stock result whenever the true top-k is inside the
approximate top-``screen``. Only draft tokens are affected; the target's
verify logits are computed separately, so only the acceptance rate could move.

Delivery: ``LogitsProcessor`` is a ``PluggableLayer``; ``register()`` swaps in
``ScreenedLogitsProcessor`` via ``PluggableLayer.register_oot``. The subclass
only overrides the two top-token methods, so ``forward`` (target logits) is
untouched. ``ShadowHead`` is also the low-bit Marlin copy used by draft_vocab,
verify_split_head and mtp_w4.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.logits_processor import LogitsProcessor, _topk
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.scalar_type import ScalarType, scalar_types

logger = init_logger(__name__)

OPT_NAME = "shadow_head"

_QUANT_TYPES: dict[int, ScalarType] = {
    4: scalar_types.uint4b8,
    8: scalar_types.uint8b128,
}
_SUPPORTED_GROUP_SIZES = (-1, 32, 64, 128)
# Attribute on the lm_head module that caches its shadow. A plain attribute,
# not a registered buffer, so state_dict / weight loading never see it.
_SHADOW_ATTR = "_atdev_shadow_head"


@dataclass(frozen=True)
class ShadowHeadConfig:
    bits: int = 4
    group_size: int = 128
    screen: int = 128
    a8: bool = False  # int8 activations (Marlin W4A8 path: per-token quant + int16 scales)

    def __post_init__(self) -> None:
        if self.bits not in _QUANT_TYPES:
            raise ValueError(f"shadow head bits must be 4 or 8, got {self.bits}")
        if self.group_size not in _SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"shadow head group size must be one of {_SUPPORTED_GROUP_SIZES}, "
                f"got {self.group_size}"
            )
        if self.screen < 1:
            raise ValueError(f"shadow head screen must be >= 1, got {self.screen}")

    @property
    def quant_type(self) -> ScalarType:
        return _QUANT_TYPES[self.bits]


# ---------------------------------------------------------------------------
# Quantization (pure torch; matches vLLM's quant_utils.quantize_weights +
# gptq_pack for symmetric GPTQ types, but chunked and device-agnostic so the
# full 248k-row lm_head can be quantized on the GPU without a 5 GB temporary).
# ---------------------------------------------------------------------------


def _pack_rows(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack [K, N] unsigned codes along K into [K * bits / 32, N] int32."""
    pack_factor = 32 // bits
    k, n = q.shape
    q = q.to(torch.int64).view(k // pack_factor, pack_factor, n)
    shifts = torch.arange(0, 32, bits, device=q.device, dtype=torch.int64)
    packed = (q << shifts.view(1, -1, 1)).sum(dim=1)
    # Reinterpret as two's-complement int32 (values are < 2**32).
    packed = torch.where(packed >= 2**31, packed - 2**32, packed)
    return packed.to(torch.int32)


@torch.no_grad()
def quantize_and_pack(
    weight: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    chunk_rows: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric GPTQ quantization of an [N, K] (out x in) weight.

    Returns ``(packed, scales)``: ``packed`` is the GPTQ row-packed
    [K * bits / 32, N] int32 tensor (codes biased by ``quant_type.bias``) and
    ``scales`` is [K / group, N] in ``weight.dtype``. Same math as
    ``quant_utils.quantize_weights(w.T, quant_type, group_size)``.
    """
    assert weight.dim() == 2 and weight.is_floating_point()
    n, k = weight.shape
    bits = quant_type.size_bits
    group = k if group_size == -1 else group_size
    if k % group or k % (32 // bits):
        raise ValueError(f"K={k} must be divisible by group={group} and pack factor")
    num_groups = k // group
    max_q, min_q, bias = quant_type.max(), quant_type.min(), quant_type.bias

    packed = torch.empty((k * bits // 32, n), dtype=torch.int32, device=weight.device)
    scales = torch.empty((num_groups, n), dtype=weight.dtype, device=weight.device)
    for start in range(0, n, chunk_rows):
        end = min(n, start + chunk_rows)
        # [Nc, K] -> [Nc, G, group] in fp32 (quantize_weights works in the
        # weight dtype; fp32 here only removes double rounding of the scale).
        w = weight[start:end].float().view(end - start, num_groups, group)
        s = torch.maximum(
            (w.amax(dim=-1) / max_q).abs(), (w.amin(dim=-1) / min_q).abs()
        ).to(weight.dtype)  # [Nc, G]
        s_f = s.float().clamp(min=torch.finfo(torch.float32).tiny)
        q = torch.round(w / s_f.unsqueeze(-1)).clamp(min_q, max_q).to(torch.int32)
        q = (q + bias).view(end - start, k).t().contiguous()  # [K, Nc]
        packed[:, start:end] = _pack_rows(q, bits)
        scales[:, start:end] = s.t()
    return packed, scales


# ---------------------------------------------------------------------------
# Shadow head
# ---------------------------------------------------------------------------


class ShadowHead:
    """Low-bit copy of an lm_head weight with an approximate-logits Marlin GEMM
    (W4A16 / W8A16, or W4A8 with ``config.a8``)."""

    backend = "marlin"

    def __init__(self, weight: torch.Tensor, config: ShadowHeadConfig) -> None:
        self.config = config
        self.size_n, self.size_k = weight.shape
        self.dtype = weight.dtype

        packed, scales = quantize_and_pack(
            weight, config.quant_type, config.group_size
        )
        self._init_marlin(packed, scales)

    def _init_marlin(self, packed: torch.Tensor, scales: torch.Tensor) -> None:
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_empty,
            marlin_pad_qweight,
            marlin_pad_scales,
            marlin_padded_nk,
            marlin_permute_scales,
        )

        c = self.config
        size_n, size_k = self.size_n, self.size_k
        padded_n, padded_k = marlin_padded_nk(size_n, size_k, c.group_size)
        self.padded_n = padded_n
        a8 = bool(getattr(c, "a8", False)) and c.bits == 4
        self.a8 = a8
        self.qweight = ops.gptq_marlin_repack(
            marlin_pad_qweight(packed, size_n, size_k, padded_n, padded_k),
            size_k=padded_k,
            size_n=padded_n,
            num_bits=c.bits,
            is_a_8bit=a8,
        )
        del packed
        self.scales = marlin_permute_scales(
            marlin_pad_scales(
                scales, size_n, size_k, padded_n, padded_k, c.group_size
            ),
            size_k=padded_k,
            size_n=padded_n,
            group_size=c.group_size,
            is_a_8bit=a8,
        )
        self.gs = None
        if a8:
            from vllm.model_executor.layers.quantization.utils.marlin_utils import (
                marlin_act_int8_process_scales)
            self.scales, gs = marlin_act_int8_process_scales(self.scales)
            self.gs = gs.reshape(1).float().contiguous()
        self.zp = marlin_make_empty(self.qweight.device)
        self.nbytes = (
            self.qweight.numel() * 4 + self.scales.numel() * self.scales.element_size()
        )

    def approx_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[M, K] -> approximate [M, N] logits in the activation dtype."""
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            apply_gptq_marlin_linear,
        )

        a8 = self.a8
        return apply_gptq_marlin_linear(
            input=hidden_states,
            weight=self.qweight,
            weight_scale=self.scales,
            weight_zp=self.zp,
            workspace=None,
            wtype=self.config.quant_type,
            output_size_per_partition=self.size_n,
            input_size_per_partition=self.size_k,
            use_fp32_reduce=True,
            input_global_scale=self.gs if a8 else None,
            input_dtype=torch.int8 if a8 else None,
        )


def screened_top_k(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    shadow: ShadowHead,
    k: int,
    num_pad: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k of ``hidden_states @ weight.T`` via shadow screening.

    Returns ``(ids int64 [M, k], values [M, k])``. Values are the exact logits
    in ``weight.dtype`` (what the stock bf16 GEMM produces), sorted descending.
    The last ``num_pad`` vocabulary columns are excluded, as in
    ``LogitsProcessor``.
    """
    num_rows = hidden_states.shape[0]
    vocab = weight.shape[0]
    screen = min(max(shadow.config.screen, k), vocab - num_pad)

    approx = shadow.approx_logits(hidden_states)
    if num_pad > 0:
        approx[..., -num_pad:] = -float("inf")
    _, cand = _topk(approx, screen)  # [M, C]
    cand = cand.to(torch.int64)

    # Exact rescoring of the candidates: gather their bf16 rows and do a
    # batched dot product (M * C * K * 2 bytes, ~9 MB at M=7, C=128).
    rows = weight.index_select(0, cand.reshape(-1)).view(num_rows, screen, -1)
    exact = torch.bmm(
        rows, hidden_states.to(weight.dtype).unsqueeze(-1)
    ).squeeze(-1)  # [M, C], weight.dtype (fp32 accumulate, rounded like F.linear)

    values, idx = torch.topk(exact, k, dim=-1, sorted=True)
    return cand.gather(-1, idx), values


# ---------------------------------------------------------------------------
# PluggableLayer override
# ---------------------------------------------------------------------------


class ScreenedLogitsProcessor(LogitsProcessor):
    """LogitsProcessor whose draft-only top-token paths use a shadow head.

    ``forward`` (full logits: target verify, probabilistic drafting) is the
    stock implementation. Anything the screened path does not cover (TP > 1,
    embedding bias, log-probs, non-default head dtype, quantized lm_head,
    CPU) falls back to the stock method.
    """

    _shadow_config: ShadowHeadConfig | None = None
    # Whether the draft-only top-token paths use the shadow (draft_vocab turns
    # it off when it installs this class without ATDEV_OPT shadow_head).
    _draft_screen_enabled: bool = True
    # ATDEV_OPT=draft_vocab: draft top-k over a frequent-token subset
    # (atdev_serve.draft_vocab) takes precedence over the shadow screening.
    _draft_vocab_enabled: bool = False

    def _draft_vocab_top_k(self, lm_head, hidden_states, k, embedding_bias):
        if not (self._draft_vocab_enabled and hidden_states.shape[0] > 0
                and self._can_screen(lm_head, hidden_states, embedding_bias)):
            return None
        if self.scale <= 0.0 and self.scale != 1.0:
            return None
        from atdev_serve import draft_vocab

        dv = draft_vocab.head_for(lm_head, hidden_states)
        if dv is None:
            return None
        ids, values = dv.top_k(hidden_states, k)
        return ids + lm_head.shard_indices.org_vocab_start_index, values

    @classmethod
    def shadow_config(cls) -> ShadowHeadConfig:
        if cls._shadow_config is None:
            cls._shadow_config = ShadowHeadConfig()
        return cls._shadow_config

    def _shadow_for(
        self,
        lm_head: torch.nn.Module,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
    ) -> ShadowHead | None:
        shadow = getattr(lm_head, _SHADOW_ATTR, None)
        if shadow is not None:
            return shadow
        if not self._can_screen(lm_head, hidden_states, embedding_bias):
            return None
        if hidden_states.is_cuda and torch.cuda.is_current_stream_capturing():
            # Never quantize inside a graph capture. The profile/dummy run
            # builds the shadow before capture; reaching here means it did not.
            logger.warning_once(
                "atdev shadow_head: lm_head shadow missing during CUDA graph "
                "capture; using the stock draft head."
            )
            return None
        weight = lm_head.weight
        shadow = self._build_shadow(weight.data)
        setattr(lm_head, _SHADOW_ATTR, shadow)
        logger.info(
            "atdev shadow_head: built int%d g%d shadow of lm_head %s "
            "(%.2f GB vs %.2f GB bf16), draft screen=%d",
            shadow.config.bits,
            shadow.config.group_size,
            tuple(weight.shape),
            shadow.nbytes / 1e9,
            weight.numel() * weight.element_size() / 1e9,
            shadow.config.screen,
        )
        return shadow

    def _build_shadow(self, weight: torch.Tensor):
        """Factory for the cached shadow."""
        return ShadowHead(weight, self.shadow_config())

    def _can_screen(
        self,
        lm_head: torch.nn.Module,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
    ) -> bool:
        weight = getattr(lm_head, "weight", None)
        return (
            embedding_bias is None
            and getattr(lm_head, "tp_size", 1) == 1
            and isinstance(
                getattr(lm_head, "quant_method", None),
                (UnquantizedEmbeddingMethod, UnquantizedLinearMethod),
            )
            and isinstance(weight, torch.Tensor)
            and weight.dtype in (torch.bfloat16, torch.float16)
            and (self.head_dtype is None or self.head_dtype == hidden_states.dtype)
            and hidden_states.dim() == 2
            and hidden_states.dtype == weight.dtype
        )

    def get_top_k_tokens(
        self,
        lm_head,
        hidden_states: torch.Tensor,
        k: int,
        embedding_bias: torch.Tensor | None = None,
        *,
        return_log_probs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not return_log_probs:
            r = self._draft_vocab_top_k(lm_head, hidden_states, k, embedding_bias)
            if r is not None:
                ids, values = r
                values = values.float()
                if self.scale != 1.0:
                    values = values * self.scale
                if self.soft_cap is not None:
                    values = torch.tanh(values / self.soft_cap) * self.soft_cap
                return ids, values
        shadow = (
            None
            if return_log_probs
            or hidden_states.shape[0] == 0
            or not self._draft_screen_enabled
            else self._shadow_for(lm_head, hidden_states, embedding_bias)
        )
        if shadow is None:
            return super().get_top_k_tokens(
                lm_head,
                hidden_states,
                k,
                embedding_bias,
                return_log_probs=return_log_probs,
            )
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local top-k reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        ids, values = screened_top_k(
            hidden_states,
            lm_head.weight,
            shadow,
            k,
            num_pad=lm_head.shard_indices.num_org_vocab_padding,
        )
        ids = ids + lm_head.shard_indices.org_vocab_start_index
        # Same post-processing as the stock non-log-prob path.
        values = values.float()
        if self.scale != 1.0:
            values = values * self.scale
        if self.soft_cap is not None:
            values = torch.tanh(values / self.soft_cap) * self.soft_cap
        return ids, values

    def get_top_tokens(
        self,
        lm_head,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r = self._draft_vocab_top_k(lm_head, hidden_states, 1, embedding_bias)
        if r is not None:
            return r[0].squeeze(-1)
        shadow = (
            None
            if hidden_states.shape[0] == 0 or not self._draft_screen_enabled
            else self._shadow_for(lm_head, hidden_states, embedding_bias)
        )
        if shadow is None:
            return super().get_top_tokens(lm_head, hidden_states, embedding_bias)
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local argmax reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        ids, _ = screened_top_k(
            hidden_states,
            lm_head.weight,
            shadow,
            1,
            num_pad=lm_head.shard_indices.num_org_vocab_padding,
        )
        # Scale and soft cap are monotonic, so the argmax is unchanged.
        return ids.squeeze(-1) + lm_head.shard_indices.org_vocab_start_index


def register() -> None:
    """Route every LogitsProcessor construction to ScreenedLogitsProcessor."""
    LogitsProcessor.register_oot(ScreenedLogitsProcessor)
    cfg = ScreenedLogitsProcessor.shadow_config()
    logger.info(
        "atdev shadow_head: registered (bits=%d group=%d screen=%d)",
        cfg.bits,
        cfg.group_size,
        cfg.screen,
    )
