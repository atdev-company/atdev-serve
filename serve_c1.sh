#!/usr/bin/env bash
# atdev-serve, c=1 (single-stream) full-stack configuration for Qwen3.8-27B on one A100 80GB.
#
# RedHatAI INT4 target + 3-bit overlay (W3) + re-trained, W4A16-quantized DFlash2 drafter (d16) with
# tree verification, plus the atdev engine kernels.
#
# usage:  ./serve_c1.sh [--profile multiturn|single] [extra vllm serve args...]
# profiles:
#   multiturn (default)  prefix caching for repeated conversations and agent loops, 131072-token context,
#                        chunked-prefill kernels (FlashAttention-2 / FlashInfer, W3 int8 GEMM), JIT warmup at
#                        start. Long multi-turn traces run 2.3-3.8x faster than the best stock vLLM; single-request
#                        decode is about 3.6% slower than `single` (394.4 vs 409.5 tok/s mean, 96 prompts).
#   single               the original c=1 launch (408.9 tok/s mean decode vs 196.5 for the best vLLM 0.31.0
#                        configuration): no prefix caching, 8192-token context.
# env:    MODEL        target checkpoint           (default RedHatAI/Qwen3.8-27B-INT4)
#         HF_REPO      HF repo with the model-side files (default atdev-company/Qwen3.8-27B-atdev-c1)
#         W3_OVERLAY   W3 sidecar (default $HF_REPO:w3/): repo:folder/, repo:file, or comma-separated
#                      local .safetensors files
#         DRAFTER      quantized drafter (default $HF_REPO:drafter/): repo:folder/ or local dir
#         C1_MODE      fast (default, W3) | accuracy (no W3 overlay; implies --profile single, see serve_c1_accuracy.sh)
#         MAX_MODEL_LEN  context length (multiturn: 131072, single: 8192)
#         NO_WARMUP=1  skip the start-up JIT warmup (multiturn only)
#         ATDEV_EOR_CKPT=1  optional end-of-reply state checkpoint (multiturn only, OFF by default): the next
#                      turn also reuses the reply of the previous turn. Neutral on the measured c=1 traces
#                      (the reply is short next to the new content of a turn); see README, "Optional".
#         PORT (8100)  SERVED_NAME (m)  PYTHON (python)  VLLM (vllm)  CUDA_VISIBLE_DEVICES (0)
# The first start JIT-compiles the CUDA kernels (needs nvcc, see README); later starts reuse
# TORCH_EXTENSIONS_DIR.
set -euo pipefail

C1_MODE=${C1_MODE:-fast}  # fast (original W3) is the default; accuracy/noW3 keeps the DFlash stack.
case "$C1_MODE" in fast|accuracy|noW3) ;; *) echo "invalid C1_MODE: use fast or accuracy" >&2; exit 2 ;; esac

PROFILE=${PROFILE:-}
if [[ ${1:-} == --profile ]]; then PROFILE=${2:-}; shift 2 || true
elif [[ ${1:-} == --profile=* ]]; then PROFILE=${1#--profile=}; shift; fi
if [[ -z "$PROFILE" ]]; then  # default: multiturn, except the explicit accuracy preset (single, as before)
  if [[ "$C1_MODE" == fast ]]; then PROFILE=multiturn; else PROFILE=single; fi
fi
case "$PROFILE" in multiturn|single) ;; *) echo "invalid profile '$PROFILE': use multiturn or single" >&2; exit 2 ;; esac
if [[ "$PROFILE" == multiturn && "$C1_MODE" != fast ]]; then
  echo "C1_MODE=$C1_MODE is only available with --profile single (prefix caching was validated with the W3 stack only)" >&2; exit 2
fi

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

if [[ "$C1_MODE" == fast ]]; then
  echo "atdev profile: c1-fast (original W3), $PROFILE" >&2
else
  # Match the accuracy variant by changing only the W3 feature and its overlay pointer.
  export ATDEV_OPT=${ATDEV_OPT/,w3_mixed/}
  unset ATDEV_W3_SIDECAR
  echo "atdev profile: c1-accuracy (no W3 overlay, same DFlash stack), $PROFILE" >&2
fi

# ---- settings that differ from serve_c32.sh ----
export ATDEV_MP_CONFIG=$PKG/marlin_plus_cfg.json       # Marlin launch configs (W4A16)
export ATDEV_DRAFT_VOCAB_SIZE=65536 ATDEV_DRAFT_VOCAB_BITS=4   # draft vocabulary: 64k most frequent tokens, int4 screen + exact rescoring
export ATDEV_VSPLIT_SIZE=65536 ATDEV_VSPLIT_BITS=4  # verify head: 64k exact rows + int4 tail
export ATDEV_DRAFT_VOCAB_TOPKX=1                       # drafter top-k over the draft vocabulary as two Triton kernels (= torch.topk)
export ATDEV_VSPLIT_FBITS=8                            # verify head: int8 rows for the 64k frequent tokens
export ATDEV_VSPLIT_STREAMS=1                          # verify head: tail GEMM on a side stream, overlapped with the frequent-row GEMM

if [[ "$PROFILE" == multiturn ]]; then
  # ---- multi-turn: prefix caching for the tree stack + chunked-prefill kernels (frozen bundle, results/C1FINAL) ----
  export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1}
  export ATDEV_DFT_APC=1                               # tree verification under prefix caching (needs --mamba-cache-mode align)
  export ATDEV_PF_FUSE=1                               # Triton kernels for the prefill-side epilogues of the W3 GEMMs
  export ATDEV_PF_GLUE=1                               # prefill SiLU*mul / residual-RMSNorm fused with the int8 activation quantization
  export ATDEV_PF_FA2=1                                # prefill attention: FlashAttention-2 for q >= 32 (rounding-level difference to TRITON_ATTN)
  export ATDEV_PF_FA2_GATHER=1                         #   ... with gathered KV for long single-request chunks
  export ATDEV_PF_FA2_FI=1                             #   ... FlashInfer split-KV for short tails (q 2..299) at long context
  export ATDEV_PF_FA2_FI_MINQ=2
  export ATDEV_PF_FA2_FI_PINRING=1                     #   ... pinned plan buffers in a ring (fixes a CUDA illegal-memory-access race)
  export ATDEV_SWA_FINE=1                              # fine-grained prefix hits for the drafter's sliding-window layers
  export ATDEV_W3_DQ=1 ATDEV_W3_DQ_MIN_M=1536          # prefill M >= 1536: W3 dequantized to int8 once per layer, int8 GEMM
  export ATDEV_W3_I8=1 ATDEV_W3_I8_MIN_M=64            # 64 <= M < 1536: int8 activations into the W3 GEMM
  export ATDEV_TOK_CACHE=1                             # frontend: per-segment tokenization cache for growing chat histories
  export ATDEV_GDN_SMALL=32                            # GDN prefill chunks of <= 32 tokens: recurrent kernel instead of chunked FLA
  export ATDEV_TREE_HSPL=auto                          # tree-attention head split picked per context length
  MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
  SPEC='{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":15,"disable_eagle_block_drop":true}'
  MODE_ARGS=(--enable-prefix-caching --mamba-cache-mode align --enable-mamba-shared-prefix-checkpoint --prefix-match-unit 16
             --max-num-batched-tokens 2048)
  if [[ "${ATDEV_EOR_CKPT:-0}" == 1 ]]; then
    export ATDEV_EOR_CKPT=1 ATDEV_EOR_SNAP=${ATDEV_EOR_SNAP:-16}
    echo "atdev: end-of-reply checkpoint ON (optional; neutral at c=1 on the measured traces)" >&2
  fi
  if [[ "${NO_WARMUP:-0}" != 1 ]]; then
    # compiles the Triton kernels of the first request while the server is idle (logs "atdev warmup: ...")
    "$PYTHON" -m atdev_serve.warmup --port "$PORT" --model "$SERVED_NAME" --watch-pid $$ &
  fi
else
  MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
  SPEC='{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":15}'
  MODE_ARGS=(--no-enable-prefix-caching)
fi

exec "${VLLM:-vllm}" serve "$MODEL" \
  --port "$PORT" --served-model-name "$SERVED_NAME" \
  --max-model-len "$MAX_MODEL_LEN" --kv-cache-dtype bfloat16 --language-model-only \
  --gpu-memory-utilization 0.90 --mamba-ssm-cache-dtype bfloat16 \
  --attention-backend TRITON_ATTN "${MODE_ARGS[@]}" \
  --override-generation-config '{"temperature":0}' \
  --speculative-config "$SPEC" \
  "$@"
