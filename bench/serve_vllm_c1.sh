#!/usr/bin/env bash
# Best stock vLLM 0.31.0 configuration at c=1 (the denominator of the c=1 ratio; 196.5 mean tok/s).
# Same RedHatAI INT4 target, stock z-lab DFlash2 drafter (k=7, vLLM fp8 drafter quantization),
# bf16 mamba SSM cache. Do NOT enable ATDEV_OPT here.
set -euo pipefail
unset ATDEV_OPT
exec "${VLLM:-vllm}" serve "${MODEL:-RedHatAI/Qwen3.8-27B-INT4}" \
  --port "${PORT:-8100}" --served-model-name "${SERVED_NAME:-m}" \
  --max-model-len 8192 --kv-cache-dtype bfloat16 --language-model-only --gpu-memory-utilization 0.90 \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":7,"quantization":"fp8"}' \
  --mamba-ssm-cache-dtype bfloat16 "$@"
