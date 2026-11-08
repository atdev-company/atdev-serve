#!/usr/bin/env bash
# atdev-serve, c=32 (32 concurrent streams) full-stack configuration for Qwen3.8-27B on one A100 80GB.
#
# RedHatAI INT4 target with W4A4 activation quantization on the large verify GEMMs and the model's
# built-in MTP head (k=3 drafts, confidence-scheduled variable-length verification). No external drafter.
#
# usage:  ./serve_c32.sh [--profile single|multiturn] [extra vllm serve args...]
# profiles:
#   multiturn   (default) the Hadamard W4A4 stack with prefix caching for 32 concurrent multi-turn sessions (agent loops,
#               chat histories), chunked-prefill kernels (FlashAttention-2 / FlashInfer, rotation-aware W4A4 prefill GEMM),
#               16384-token context, max-num-batched-tokens 4096, start-up warmup. Requires the rotated INT4 sidecar.
#   single      the previous launch: the same Hadamard W4A4 stack without prefix caching, 8192-token context.
#   Both: C32_HADAMARD (alias HADW) = local sidecar file or Hub repo:file; defaults to the released sidecar.
# env:    MODEL (RedHatAI/Qwen3.8-27B-INT4)  PORT (8100)  SERVED_NAME (m)  PYTHON (python)  VLLM (vllm)
#         CUDA_VISIBLE_DEVICES (0)
#         MAX_MODEL_LEN  context length (multiturn: 16384, single: 8192)
#         NO_WARMUP=1  skip the start-up JIT warmup (multiturn only)
#         ATDEV_EOR_CKPT=1  optional end-of-reply state checkpoint (multiturn only, OFF by default). Lets the next
#                      turn reuse the previous reply as well (live-reply traces: 5-12% faster wall time at 32
#                      sessions). Costs 1.3-2.7% on traces where replies cannot be reused and needs 3% more GPU
#                      memory (the script then uses --gpu-memory-utilization 0.93, snapshot spacing 32).
#         ATDEV_PF_A4=0     multiturn: disable the W4A4 prefill GEMM (more accurate, about 20% slower).
# The first start JIT-compiles the CUDA kernels (needs nvcc, see README).
set -euo pipefail

C32_DEFAULT_PROFILE=multiturn
PROFILE=${PROFILE:-$C32_DEFAULT_PROFILE}
if [[ ${1:-} == --profile ]]; then PROFILE=${2:-}; shift 2 || true
elif [[ ${1:-} == --profile=* ]]; then PROFILE=${1#--profile=}; shift; fi
case "$PROFILE" in multiturn|single) ;; *) echo "invalid profile '$PROFILE': use multiturn or single" >&2; exit 2 ;; esac

MODEL=${MODEL:-RedHatAI/Qwen3.8-27B-INT4}
PORT=${PORT:-8100}
SERVED_NAME=${SERVED_NAME:-m}
PYTHON=${PYTHON:-python}
PKG=$("$PYTHON" -c 'import atdev_serve, os; print(os.path.dirname(atdev_serve.__file__))')

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ---- which engine features are on (vLLM general plugin atdev_serve.register:register_all) ----
export ATDEV_OPT=a4_sim,draft_vocab,gdn_single_state,verify_split_head,mtp_w4,marlin_plus,fused_quant,vsplit_mtp,gdn_a4_fold,mtp_prune,c32_attn,di1_varlen,c32_conv,c32k_fuse

if [[ "$PROFILE" == single ]]; then
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
  echo 'atdev profile: c32-hadamard-w4a4 (built-in MTP), single' >&2
else
  # ---- multi-turn bundle: the Hadamard W4A4 stack above + prefix caching + chunked-prefill kernels + rotation-aware 4-bit prefill ----
  # (frozen bundle: results/C32MT/best_env.txt + results/A4ACC arm A)
  export ATDEV_A4_REAL=had3
  export ATDEV_A4_HADW=${C32_HADAMARD:-${HADW:-atdev-company/Qwen3.8-27B-atdev-c32:hadamard/qwen38_27b_hadamard_gptq_d10.safetensors}}
  export ATDEV_GDN_SRC=$PKG/csrc/gdn_var/gdn_ss15h.cu  # CUDA GDN verify kernel with the folded W4A4 out_proj quantization (+ meta fast path)
  export ATDEV_GDN_META=1
  export ATDEV_GDN_SS_ALIGN=1                          # single-state GDN verify under prefix caching (--mamba-cache-mode align)
  export ATDEV_C32_MK=1 ATDEV_C32K_SCR=1 ATDEV_C32C_CSUM=1 ATDEV_C32P_BA=1
  export ATDEV_GDN_CORE_EMPTY=1 ATDEV_C32_QKG=1 ATDEV_FQ_A4_CUDA=1
  export ATDEV_A4_SET=cfg
  export ATDEV_A4_DEFINES=A4_FOLD2=1,A4_MAGICQ=1,A4_SCALAR_CTMP=1,A4_PAR_REDUCE=6,A4_C32C_PB=1,A4_S0=1,A4_PTRINC=1,A4_PTRINC_ST=6
  export ATDEV_DI1_CG_MAXREQS=32 ATDEV_DI1_BUDGET=rows:96 ATDEV_DI1_STATS=0
  export ATDEV_VSPLIT_FAST=1 ATDEV_VSPLIT_TAIL_MP=64,256,256,6,1,4
  export ATDEV_VSPLIT_A4SCR=1 ATDEV_VSPLIT_A4SCR_CFG=16,4
  unset ATDEV_DRAFTER_QUANT_PATH
  # chunked-prefill side (what the multi-turn gain comes from)
  export ATDEV_PF_FA2=1                                # prefill / mixed attention: FlashAttention-2 for q >= 32
  export ATDEV_PF_FI=1                                 # FlashInfer fa2 paged attention for the prefill rows of mixed steps
  export ATDEV_PF_MP=1                                 # tuned Marlin launch configs for prefill M > 128
  export ATDEV_TOK_CACHE=1                             # frontend: per-segment tokenization cache for growing chat histories
  # 4-bit prefill GEMM on the Hadamard-rotated weights (A4ACC arm A): rotation fused with the activation quantization,
  # s4 fragments emitted directly, activation clip 0.9, layer 0 kept on W4A8. ATDEV_PF_A4=0: prefill through the int8 path (slower).
  export ATDEV_PF_A4=${ATDEV_PF_A4:-1} ATDEV_PF_A4_RQV=${ATDEV_PF_A4_RQV:-1}
  export ATDEV_A4ACC_FUSE=1 ATDEV_A4ACC_S4=1 ATDEV_A4ACC_CLIP=0.9
  export ATDEV_PF_A4_SKIP='*:0-0'
  echo "atdev profile: c32-multiturn (Hadamard W4A4, built-in MTP, PF_A4=$ATDEV_PF_A4)" >&2
fi

# ---- vLLM built-in: int8 activations into Marlin (W4A8) ----
export VLLM_MARLIN_INPUT_DTYPE=int8

# ---- settings that differ from serve_c1.sh ----
export ATDEV_MP_CONFIG=$PKG/marlin_plus_a4k2q_cfg.json # Marlin W4A8 / W4A4 launch configs
export ATDEV_GDN_IMPL=cuda                             # CUDA GDN single-state verify kernel
export ATDEV_DRAFT_VOCAB_SIZE=65536 ATDEV_DRAFT_VOCAB_BITS=4 ATDEV_VSPLIT_SIZE=65536 ATDEV_VSPLIT_BITS=4  # 64k draft vocabulary / exact verify rows, int4 screen and tail
export ATDEV_DRAFT_VOCAB_FUSED=1                       # MTP draft vocabulary (k=1 argmax): fused screen + exact rescoring kernels
export ATDEV_VSPLIT_A8=1                               # verify head tail: int8 activations

if [[ "$PROFILE" == multiturn ]]; then
  export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1}
  MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
  GMU=0.90
  if [[ "${ATDEV_EOR_CKPT:-0}" == 1 ]]; then
    export ATDEV_EOR_CKPT=1 ATDEV_EOR_SNAP=${ATDEV_EOR_SNAP:-32} ATDEV_EOR_PLANNED=${ATDEV_EOR_PLANNED:-1}
    GMU=0.93
    echo "atdev: end-of-reply checkpoint ON (optional): snapshot spacing $ATDEV_EOR_SNAP, --gpu-memory-utilization $GMU" >&2
  fi
  MODE_ARGS=(--max-num-seqs 32 --max-num-batched-tokens 4096
             --enable-prefix-caching --mamba-cache-mode align --enable-mamba-shared-prefix-checkpoint --prefix-match-unit 16)
  if [[ "${NO_WARMUP:-0}" != 1 ]]; then
    # compiles the Triton kernels of the first requests (single and 32-way batch shapes) while the server is idle
    "$PYTHON" -m atdev_serve.warmup --port "$PORT" --model "$SERVED_NAME" --concurrency 32 --max-tokens 32 --watch-pid $$ &
  fi
else
  MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
  GMU=0.90
  MODE_ARGS=(--no-enable-prefix-caching)
fi

exec "${VLLM:-vllm}" serve "$MODEL" \
  --port "$PORT" --served-model-name "$SERVED_NAME" \
  --max-model-len "$MAX_MODEL_LEN" --kv-cache-dtype bfloat16 --language-model-only \
  --gpu-memory-utilization "$GMU" --mamba-ssm-cache-dtype bfloat16 \
  --attention-backend TRITON_ATTN "${MODE_ARGS[@]}" \
  --override-generation-config '{"temperature":0}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' \
  "$@"
