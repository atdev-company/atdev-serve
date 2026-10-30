#!/usr/bin/env bash
# atdev-serve, c=1 (single-stream) full-stack configuration for Qwen3.8-27B on one A100 80GB.
#
# Reproduces the c=1 headline (96 prompts, mean decode 408.9 tok/s vs 196.5 for the best vLLM 0.31.0
# configuration): RedHatAI INT4 target + 3-bit overlay (W3) + re-trained, W4A16-quantized DFlash2
# drafter (d16) with tree verification, plus the atdev engine kernels.
#
# usage:  ./serve_c1.sh [extra vllm serve args...]
# env:    MODEL        target checkpoint           (default RedHatAI/Qwen3.8-27B-INT4)
#         W3_OVERLAY   W3 sidecar: HF repo id, repo:file, or comma-separated local .safetensors files
#         DRAFTER      quantized drafter: HF repo id or local dir
#         PORT (8100)  SERVED_NAME (m)  PYTHON (python)  VLLM (vllm)  CUDA_VISIBLE_DEVICES (0)
# The first start JIT-compiles the CUDA kernels (needs nvcc, see README); later starts reuse
# TORCH_EXTENSIONS_DIR.
set -euo pipefail

MODEL=${MODEL:-RedHatAI/Qwen3.8-27B-INT4}
W3_OVERLAY=${W3_OVERLAY:-atdev-company/Qwen3.8-27B-atdev-W3-overlay}
DRAFTER=${DRAFTER:-atdev-company/Qwen3.8-27B-atdev-DFlash2-d16}
PORT=${PORT:-8100}
SERVED_NAME=${SERVED_NAME:-m}
PYTHON=${PYTHON:-python}
PKG=$("$PYTHON" -c 'import atdev_serve, os; print(os.path.dirname(atdev_serve.__file__))')

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ---- which engine features are on (vLLM general plugin atdev_serve.register:register_all) ----
export ATDEV_OPT=gdn_single_state,dflash_tree,attn_split_verify,gdn_meta_fast,shadow_head,draft_vocab,marlin_plus,ba_gemv,c1_fuse,drafter_ctxfold,w3_mixed,verify_split_head,drafter_w4,tree_attn,ll1_fuse,ll5_fuse

# ---- model-side files (full stack) ----
export ATDEV_W3_SIDECAR=$W3_OVERLAY                    # 3-bit g128 GPTQ overlay for the layers below
export ATDEV_W3_LAYERS='mlp\.|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$|self_attn\.(q|k|v|o)_proj$'
export ATDEV_DRAFTER_QUANT_PATH=$DRAFTER               # replaces z-lab/Qwen3.8-27B-DFlash2 at load

# ---- kernels / launch configs ----
export ATDEV_MP_CONFIG=$PKG/marlin_plus_cfg.json
export ATDEV_MP_DEFINES=MP_FOLD=1,MP_PURE_ASM=0
export ATDEV_W3_DEFINES=W3_FASTDQ=1
export ATDEV_W3_CONFIG=$PKG/w3_fastdq_cfg.json
export ATDEV_E6W3_CONFIG=$PKG/w3_c1_cfg_v2tune.json
export ATDEV_C1W3_X=ffm,pc,s0
export ATDEV_C1_CORE_EMPTY=1
export ATDEV_C1_BA_STREAM=2

# ---- draft vocabulary / verify head (candidate selection on a 64k frequent-token subset) ----
export ATDEV_DRAFT_VOCAB_SIZE=65536
export ATDEV_DRAFT_VOCAB_BITS=4
export ATDEV_DRAFT_VOCAB_TOPKX=1
export ATDEV_VSPLIT_SIZE=65536
export ATDEV_VSPLIT_BITS=4
export ATDEV_VSPLIT_FBITS=8
export ATDEV_VSPLIT_FAST=1
export ATDEV_VSPLIT_STREAMS=1
export ATDEV_VSPLIT_TAIL_CFG=128,128,256,6,1,4

# ---- tree verification (GDN tree kernel, tree attention, post-processing) ----
export ATDEV_TREE_CUDA=1
export ATDEV_TREE_CUDA_SRC=$PKG/csrc/gdn_var/gdn_tree_c1g_f4.cu
export ATDEV_TREE_CUDA_DEFS=GQ_PAD=1,GQ_LOGLATE=1,GQ_EARLY=1,GQ_META=1,GQ_OSPLIT=1,GQ_NRS=1
export ATDEV_GDN_META_LEVEL=2
export ATDEV_TREE_POST_FUSED=1
export ATDEV_TREE_ACCEPT_GRAPH=1
export ATDEV_TREE_ATTN_GATE=1
export ATDEV_TREE_ATTN_KV=1
export ATDEV_TREE_ATTN_EXACT=2
export ATDEV_TREE_AG_FORK=1
export ATDEV_CTXFOLD_FORK=1

# ---- drafter-graph glue fusions (bit-identical) ----
export ATDEV_LL1_CTX=1 ATDEV_LL1_CBUF=1 ATDEV_LL1_KPC=1 ATDEV_LL1_SEL=1 ATDEV_LL1_TAIL=1 ATDEV_LL1_QKG=1
export ATDEV_LL5_CN=1 ATDEV_LL5_DG=1 ATDEV_LL5_TAIL=1 ATDEV_LL5_OVL=1

exec "${VLLM:-vllm}" serve "$MODEL" \
  --port "$PORT" --served-model-name "$SERVED_NAME" \
  --max-model-len 8192 --kv-cache-dtype bfloat16 --language-model-only \
  --gpu-memory-utilization 0.90 --mamba-ssm-cache-dtype bfloat16 \
  --attention-backend TRITON_ATTN --no-enable-prefix-caching \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":15}' \
  "$@"
