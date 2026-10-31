"""Standalone CUDA GDN causal_conv1d_update for the c=32 MTP verify path (ATDEV_OPT=c32_conv).

csrc/c32_conv.cu (2 channels / thread, grid dim/256 x requests) reproduces vLLM's Triton
``_causal_conv1d_update_kernel`` op sequence for width 4 + SiLU + no bias + spec decoding (varlen
query_start_loc, num_accepted_tokens, null block): bf16 products via fma.rn.bf16(x, w, -0), fp32 adds in
tap order from 0, SiLU = acc / (1 + ex2.approx(-acc * log2e)) with div.full, the same state-window roll.
Bit-identical (out + conv state vs the Triton kernel).

The hook replaces the module-level ``causal_conv1d_update`` name in qwen_gdn_linear_attn; every call
that is not exactly the supported form (bias, other activation / width / dtype, no query_start_loc,
state_len != width - 1 + max_query_len - 1, unaligned views, out=...) goes to the stock function.

It also installs the ll1_qkg kernel (vLLM's fused q/k RMSNorm + RoPE + gate kernel with the
norm-independent loads hoisted; arithmetic unchanged) for the target / MTP attention layers.
"""
from __future__ import annotations

import logging
import os

import torch

OPT_NAME = "c32_conv"
SUPPORTED_VLLM_VERSION = "0.31.0"
log = logging.getLogger("atdev_serve")
_EXT: list = [None]
_STATS = {"run": 0, "stock": 0}


def ext():
    if _EXT[0] is None:
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(__file__), "csrc", "c32_conv.cu")
        _EXT[0] = load(name="atdev_c32_conv", sources=[src], verbose=False,
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80", "-std=c++17"])
    return _EXT[0]


def _ok(x, conv_state, weight, bias, activation, csi, nacc, qsl, max_q, kw) -> bool:
    if bias is not None or activation not in ("silu", "swish", True) or csi is None or nacc is None or qsl is None:
        return False
    if any(kw.get(k) is not None for k in ("out", "block_idx_last_scheduled_token", "initial_state_idx")):
        return False
    if x.dtype != torch.bfloat16 or conv_state.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False
    if x.dim() != 2 or conv_state.dim() != 3 or weight.dim() != 2 or weight.size(1) != 4:
        return False
    dim = conv_state.size(1)
    if not (1 <= max_q <= 4) or conv_state.size(2) != 3 + max_q - 1 or dim % 256 or x.size(1) != dim:
        return False
    if weight.size(0) != dim or not weight.is_contiguous():
        return False
    if x.stride(1) != 1 or x.stride(0) % 2 or x.data_ptr() % 4:
        return False
    if conv_state.stride(1) != 1 or conv_state.stride(0) % 2 or conv_state.stride(2) % 2 or conv_state.data_ptr() % 4:
        return False
    for t in (csi, nacc, qsl):
        if t.dtype != torch.int32 or t.dim() != 1:
            return False
    return qsl.stride(0) == 1 and nacc.stride(0) == 1 and qsl.numel() == csi.numel() + 1 and \
        nacc.numel() >= csi.numel()


def install() -> None:
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gm
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import NULL_BLOCK_ID

    stock = gm.causal_conv1d_update
    if getattr(stock, "_atdev_c32_conv", False):
        return
    ext()

    def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None, conv_state_indices=None,
                             num_accepted_tokens=None, query_start_loc=None, max_query_len=-1,
                             null_block_id=NULL_BLOCK_ID, **kw):
        if _ok(x, conv_state, weight, bias, activation, conv_state_indices, num_accepted_tokens,
               query_start_loc, max_query_len, kw):
            ext().conv_update(x, conv_state, weight, conv_state_indices, query_start_loc, num_accepted_tokens,
                              conv_state_indices.numel(), int(null_block_id), int(max_query_len), False)
            if _STATS["run"] == 0:
                log.warning("atdev %s: CUDA conv_update engaged (M=%d, reqs=%d, T<=%d)", OPT_NAME,
                            x.size(0), conv_state_indices.numel(), max_query_len)
            _STATS["run"] += 1
            return x
        _STATS["stock"] += 1
        return stock(x, conv_state, weight, bias, activation, conv_state_indices, num_accepted_tokens,
                     query_start_loc, max_query_len, null_block_id, **kw)

    causal_conv1d_update._atdev_c32_conv = True
    gm.causal_conv1d_update = causal_conv1d_update


def register() -> None:
    import vllm
    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written for vLLM {SUPPORTED_VLLM_VERSION}")
    install()
    # bit-identical load hoisting in the fused q/k norm + rope + gate kernel
    from vllm.model_executor.models import qwen3_next as q3n
    from atdev_serve import compile_cache, ll1_qkg
    q3n.fused_qk_rmsnorm_rope_gate = ll1_qkg.fused_qk_rmsnorm_rope_gate_ll1
    compile_cache._SALTS["c32_qkg"] = "1"  # traced attention forward launches a different Triton kernel
    log.warning("atdev %s: qwen3_next fused q/k norm + rope + gate -> ll1_qkg", OPT_NAME)
    log.warning("atdev %s: qwen_gdn_linear_attn.causal_conv1d_update -> csrc/c32_conv.cu for the spec-decode "
                "form (bit-identical), stock otherwise", OPT_NAME)
