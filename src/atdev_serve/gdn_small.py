"""ATDEV_GDN_SMALL=<T>: single-sequence GDN prefill chunks of <= T tokens use the recurrent kernel instead of the chunked FLA path.

The chunked path launches ~11 Triton kernels per GDN layer (556 per engine step with 48 layers: ~55 ms of CPU launch time that
dominates the step for the 2-64 token tails of a prefix-cache hit), the recurrent kernel one.  Same recurrence (decode uses it),
fp32 state in registers, bf16 final state like the chunk path; only rounding order differs.  Applies to forward_native
(Triton/FLA backend) when cu_seqlens describes one sequence; everything else falls through to the stock path.
"""
import logging
import os

import torch

log = logging.getLogger("atdev_serve")


def install():
    limit = int(os.environ.get("ATDEV_GDN_SMALL", "0"))
    if limit <= 0:
        return
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as m
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule
    cls = m.ChunkGatedDeltaRule
    if getattr(cls, "_atdev_gdn_small", False):
        return
    cls._atdev_gdn_small = True
    orig = cls.forward_native

    def forward_native(self, q, k, v, g, beta, initial_state, output_final_state, cu_seqlens=None, chunk_indices=None,
                       chunk_offsets=None, use_qk_l2norm_in_kernel=True, core_attn_out=None):
        if (cu_seqlens is not None and cu_seqlens.shape[0] == 2 and q.shape[0] == 1 and q.shape[1] <= limit
                and output_final_state and initial_state is not None and initial_state.shape[0] == 1
                and not use_qk_l2norm_in_kernel and core_attn_out is None):
            o, ht = fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta, scale=None, initial_state=initial_state, inplace_final_state=False,
                cu_seqlens=cu_seqlens, ssm_state_indices=None, num_accepted_tokens=None, use_qk_l2norm_in_kernel=False)
            return o, ht[-1:]  # state after the last token: [1, HV, V, K]
        return orig(self, q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state, output_final_state=output_final_state,
                    cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, core_attn_out=core_attn_out)

    cls.forward_native = forward_native
    log.warning("atdev-serve: ATDEV_GDN_SMALL=%d (recurrent GDN kernel for single-sequence prefill chunks <= %d tokens)", limit, limit)
