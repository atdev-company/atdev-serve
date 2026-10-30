#!/usr/bin/env bash
# Best stock vLLM 0.31.0 configuration at c=32 (the denominator of the c=32 ratio; 1757.1 agg tok/s).
# Same RedHatAI INT4 target, built-in MTP head k=1, W4A8 Marlin (VLLM_MARLIN_INPUT_DTYPE=int8),
# TRITON_ATTN, bf16 mamba SSM cache, no prefix caching. Do NOT enable ATDEV_OPT here.
set -euo pipefail
unset ATDEV_OPT
export VLLM_MARLIN_INPUT_DTYPE=int8
exec "${VLLM:-vllm}" serve "${MODEL:-RedHatAI/Qwen3.8-27B-INT4}" \
  --port "${PORT:-8100}" --served-model-name "${SERVED_NAME:-m}" \
  --max-model-len 8192 --kv-cache-dtype bfloat16 --language-model-only --gpu-memory-utilization 0.90 \
  --mamba-ssm-cache-dtype bfloat16 --attention-backend TRITON_ATTN --no-enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' "$@"
