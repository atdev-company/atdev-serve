#!/usr/bin/env bash
# Stock vLLM 0.31.0 with default settings (no speculative decoding, no tuning flags): the first two rungs
# of the README "baseline -> best" ladder.
#   MODEL=Qwen/Qwen3.8-27B            original bf16 checkpoint  (c=1 28.9 tok/s, c=32 671.8 tok/s)
#   MODEL=RedHatAI/Qwen3.8-27B-INT4   INT4 checkpoint (default) (c=1 74.9 tok/s, c=32 1264.7 tok/s)
# Only the port and the served model name are set. Do NOT enable ATDEV_OPT here.
set -euo pipefail
unset ATDEV_OPT
exec "${VLLM:-vllm}" serve "${MODEL:-RedHatAI/Qwen3.8-27B-INT4}" \
  --port "${PORT:-8100}" --served-model-name "${SERVED_NAME:-m}" "$@"
