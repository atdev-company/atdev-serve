#!/usr/bin/env bash
# atdev-serve, c=32 (32 concurrent streams) full-stack configuration for Qwen3.8-27B on one A100 80GB.
#
# Reproduces the c=32 headline (96 prompts at concurrency 32, aggregate output 2,730.9 tok/s vs
# 1757.1 for the best vLLM 0.31.0 configuration): RedHatAI INT4 target with W4A4 activation
# quantization on the large verify GEMMs, the model's built-in MTP head (k=3 drafts with
# confidence-scheduled variable-length verification), plus the atdev engine kernels.
# No extra model files are needed for this stack.
#
# usage:  ./serve_c32.sh [extra vllm serve args...]
# env:    MODEL (RedHatAI/Qwen3.8-27B-INT4)  PORT (8100)  SERVED_NAME (m)  PYTHON (python)  VLLM (vllm)
#         CUDA_VISIBLE_DEVICES (0)
# The first start JIT-compiles the CUDA kernels (needs nvcc, see README).
set -euo pipefail

MODEL=${MODEL:-RedHatAI/Qwen3.8-27B-INT4}
PORT=${PORT:-8100}
SERVED_NAME=${SERVED_NAME:-m}
PYTHON=${PYTHON:-python}
PKG=$("$PYTHON" -c 'import atdev_serve, os; print(os.path.dirname(atdev_serve.__file__))')

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ---- which engine features are on (vLLM general plugin atdev_serve.register:register_all) ----
export ATDEV_OPT=draft_vocab,gdn_single_state,verify_split_head,mtp_w4,marlin_plus,fused_quant,vsplit_mtp,gdn_a4_fold,mtp_prune,c32_attn,di1_varlen,c32_conv,c32k_fuse

# ---- vLLM built-in: int8 activations into Marlin (W4A8) ----
export VLLM_MARLIN_INPUT_DTYPE=int8

# ---- settings that differ from serve_c1.sh ----
export ATDEV_MP_CONFIG=$PKG/marlin_plus_a4k2q_cfg.json # Marlin W4A8 / W4A4 launch configs
export ATDEV_GDN_IMPL=cuda                             # CUDA GDN single-state verify kernel
export ATDEV_DRAFT_VOCAB_FUSED=1                       # MTP draft vocabulary (k=1 argmax): fused screen + exact rescoring kernels
export ATDEV_VSPLIT_A8=1                               # verify head tail: int8 activations

exec "${VLLM:-vllm}" serve "$MODEL" \
  --port "$PORT" --served-model-name "$SERVED_NAME" \
  --max-model-len 8192 --kv-cache-dtype bfloat16 --language-model-only \
  --gpu-memory-utilization 0.90 --mamba-ssm-cache-dtype bfloat16 \
  --attention-backend TRITON_ATTN --no-enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' \
  "$@"
