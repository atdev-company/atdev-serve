#!/usr/bin/env bash
# atdev-serve, c=1 (single-stream) full-stack configuration for Qwen3.8-27B on one A100 80GB.
#
# Reproduces the c=1 headline (96 prompts, mean decode 408.9 tok/s vs 196.5 for the best vLLM 0.31.0
# configuration): RedHatAI INT4 target + 3-bit overlay (W3) + re-trained, W4A16-quantized DFlash2
# drafter (d16) with tree verification, plus the atdev engine kernels.
#
# usage:  ./serve_c1.sh [extra vllm serve args...]
# env:    MODEL        target checkpoint           (default RedHatAI/Qwen3.8-27B-INT4)
#         HF_REPO      HF repo with the model-side files (default atdev-company/Qwen3.8-27B-atdev-c1)
#         W3_OVERLAY   W3 sidecar (default $HF_REPO:w3/): repo:folder/, repo:file, or comma-separated
#                      local .safetensors files
#         DRAFTER      quantized drafter (default $HF_REPO:drafter/): repo:folder/ or local dir
#         PORT (8100)  SERVED_NAME (m)  PYTHON (python)  VLLM (vllm)  CUDA_VISIBLE_DEVICES (0)
# The first start JIT-compiles the CUDA kernels (needs nvcc, see README); later starts reuse
# TORCH_EXTENSIONS_DIR.
set -euo pipefail

MODEL=${MODEL:-RedHatAI/Qwen3.8-27B-INT4}
HF_REPO=${HF_REPO:-atdev-company/Qwen3.8-27B-atdev-c1}  # one repo: w3/ (overlay) + drafter/ (drafter)
W3_OVERLAY=${W3_OVERLAY:-$HF_REPO:w3/}
DRAFTER=${DRAFTER:-$HF_REPO:drafter/}
PORT=${PORT:-8100}
SERVED_NAME=${SERVED_NAME:-m}
PYTHON=${PYTHON:-python}
PKG=$("$PYTHON" -c 'import atdev_serve, os; print(os.path.dirname(atdev_serve.__file__))')

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ---- which engine features are on (vLLM general plugin atdev_serve.register:register_all) ----
export ATDEV_OPT=gdn_single_state,dflash_tree,attn_split_verify,gdn_meta_fast,shadow_head,draft_vocab,marlin_plus,ba_gemv,c1_fuse,drafter_ctxfold,w3_mixed,verify_split_head,drafter_w4,tree_attn,ll1_fuse,ll5_fuse

# ---- model-side files (full stack) ----
export ATDEV_W3_SIDECAR=$W3_OVERLAY                    # 3-bit g128 GPTQ overlay (MLP, GDN in/out, attention q/k/v/o)
export ATDEV_DRAFTER_QUANT_PATH=$DRAFTER               # replaces z-lab/Qwen3.8-27B-DFlash2 at load

# ---- settings that differ from serve_c32.sh ----
export ATDEV_MP_CONFIG=$PKG/marlin_plus_cfg.json       # Marlin launch configs (W4A16)
export ATDEV_DRAFT_VOCAB_TOPKX=1                       # drafter top-k over the draft vocabulary as two Triton kernels (= torch.topk)
export ATDEV_VSPLIT_FBITS=8                            # verify head: int8 rows for the 64k frequent tokens
export ATDEV_VSPLIT_STREAMS=1                          # verify head: tail GEMM on a side stream, overlapped with the frequent-row GEMM

exec "${VLLM:-vllm}" serve "$MODEL" \
  --port "$PORT" --served-model-name "$SERVED_NAME" \
  --max-model-len 8192 --kv-cache-dtype bfloat16 --language-model-only \
  --gpu-memory-utilization 0.90 --mamba-ssm-cache-dtype bfloat16 \
  --attention-backend TRITON_ATTN --no-enable-prefix-caching \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":15}' \
  "$@"
