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

# ---- GEMM kernels (marlin_plus W4A8 / W4A4) ----
export ATDEV_MP_CONFIG=$PKG/marlin_plus_a4k2q_cfg.json
export ATDEV_MP_DEFINES=MP_FOLD=1,MP_PURE_ASM=0
export ATDEV_A4_SET=cfg
export ATDEV_A4_DEFINES=A4_FOLD2=1,A4_MAGICQ=1,A4_SCALAR_CTMP=1,A4_PAR_REDUCE=6,A4_C32C_PB=1
export ATDEV_C32C_CSUM=1
export ATDEV_C32_MK=1
export ATDEV_FQ_A4_CUDA=1

# ---- GDN (linear attention) single-state verify kernel ----
export ATDEV_GDN_IMPL=cuda
export ATDEV_GDN_SRC=csrc/gdn_var/gdn_ss12.cu          # relative to the atdev_serve package dir
export ATDEV_GDN_CORE_EMPTY=1
export ATDEV_C32_QKG=1

# ---- draft vocabulary / verify head ----
export ATDEV_DRAFT_VOCAB_SIZE=65536
export ATDEV_DRAFT_VOCAB_BITS=4
export ATDEV_DRAFT_VOCAB_FUSED=1
export ATDEV_VSPLIT_SIZE=65536
export ATDEV_VSPLIT_BITS=4
export ATDEV_VSPLIT_A8=1
export ATDEV_VSPLIT_FAST=1
export ATDEV_VSPLIT_TAIL_MP=64,256,256,6,1,4
export ATDEV_VSPLIT_A4SCR=1
export ATDEV_VSPLIT_A4SCR_CFG=16,4
export ATDEV_C32K_SCR=1

# ---- DI1: confidence-scheduled variable-length MTP verification ----
export ATDEV_DI1_CG_MAXREQS=32
export ATDEV_DI1_BUDGET=rows:96
export ATDEV_DI1_STATS=0

exec "${VLLM:-vllm}" serve "$MODEL" \
  --port "$PORT" --served-model-name "$SERVED_NAME" \
  --max-model-len 8192 --kv-cache-dtype bfloat16 --language-model-only \
  --gpu-memory-utilization 0.90 --mamba-ssm-cache-dtype bfloat16 \
  --attention-backend TRITON_ATTN --no-enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' \
  "$@"
