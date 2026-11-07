#!/usr/bin/env bash
# atdev-serve, c=32 (32 concurrent streams) full-stack configuration for Qwen3.8-27B on one A100 80GB.
#
# Selected block-Hadamard W4A4 stack with the model's built-in MTP head (k=3).
# Requires one rotated INT4 sidecar; no external drafter or unrotated preset.
# C32_HADAMARD (alias HADW): local file or Hub repo:file; defaults to the released sidecar.
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
export ATDEV_OPT=a4_sim,draft_vocab,gdn_single_state,verify_split_head,mtp_w4,marlin_plus,fused_quant,vsplit_mtp,gdn_a4_fold,mtp_prune,c32_attn,di1_varlen,c32_conv,c32k_fuse

# Exactly the selected had3 rotation, including all fused activation producers.
export ATDEV_A4_REAL=had3
export ATDEV_A4_HADW=${C32_HADAMARD:-${HADW:-atdev-company/Qwen3.8-27B-atdev-c32:hadamard/qwen38_27b_hadamard_gptq_d10.safetensors}}
export ATDEV_GDN_SRC=$PKG/csrc/gdn_var/gdn_ss12h.cu
export ATDEV_FQ_A4_NORM=16,8 ATDEV_FQ_A4_SILU=16,4
export ATDEV_C32_MK=1 ATDEV_C32K_SCR=1 ATDEV_C32C_CSUM=1
export ATDEV_GDN_CORE_EMPTY=1 ATDEV_C32_QKG=1 ATDEV_FQ_A4_CUDA=1
export ATDEV_A4_SET=cfg
export ATDEV_A4_DEFINES=A4_FOLD2=1,A4_MAGICQ=1,A4_SCALAR_CTMP=1,A4_PAR_REDUCE=6,A4_C32C_PB=1
export ATDEV_DI1_CG_MAXREQS=32 ATDEV_DI1_BUDGET=rows:96 ATDEV_DI1_STATS=0
export ATDEV_VSPLIT_FAST=1 ATDEV_VSPLIT_TAIL_MP=64,256,256,6,1,4
export ATDEV_VSPLIT_A4SCR=1 ATDEV_VSPLIT_A4SCR_CFG=16,4
unset ATDEV_DRAFTER_QUANT_PATH
echo 'atdev profile: c32-hadamard-w4a4 (built-in MTP)' >&2

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
  --override-generation-config '{"temperature":0}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' \
  "$@"
