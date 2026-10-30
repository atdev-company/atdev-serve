#!/usr/bin/env bash
# Start a server, run the fixed workload, stop the server.
#
# usage:  bench/run_bench.sh <stack> <out-dir>
#   stack: c1 | c32            (atdev full stack, ../serve_c1.sh / ../serve_c32.sh)
#          vllm_c1 | vllm_c32  (best stock vLLM baselines, bench/serve_vllm_c*.sh)
#          default_c1 | default_c32  (stock vLLM defaults, bench/serve_vllm_default.sh; set
#                       MODEL=Qwen/Qwen3.8-27B for the bf16 rung, default MODEL is the INT4 checkpoint)
# env:    REPS   c=1 repetitions over the 96 prompts (default 2; the headline is the mean over reps)
#         ACC=1  also run GSM8K (ACC_N problems, default 300; ACC_CONC parallel requests, default 1 for
#                c1 stacks, 32 for c32 stacks)
#         LONGCTX=1  also run the 36-item long-context retrieval gate
#         PORT (8100)  PYTHON (python)  CLIENT_CPUS (optional taskset CPU list for the client)
# Metric: c=1 -> mean (and median) per-request decode tok/s = (completion_tokens - 1) / (t_last - t_first)
#         c=32 -> aggregate output tok/s = total completion tokens / wall time of the 96-request run
set -uo pipefail
STACK=${1:?stack}; OUT=${2:?out-dir}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd)
PORT=${PORT:-8100}; PY=${PYTHON:-python}
case $STACK in
  c1) SERVE=$ROOT/serve_c1.sh; CONC=1 ;;
  c32) SERVE=$ROOT/serve_c32.sh; CONC=32 ;;
  vllm_c1) SERVE=$HERE/serve_vllm_c1.sh; CONC=1 ;;
  vllm_c32) SERVE=$HERE/serve_vllm_c32.sh; CONC=32 ;;
  default_c1) SERVE=$HERE/serve_vllm_default.sh; CONC=1 ;;
  default_c32) SERVE=$HERE/serve_vllm_default.sh; CONC=32 ;;
  *) echo "unknown stack $STACK"; exit 2 ;;
esac
mkdir -p "$OUT"
TS=(); [ -n "${CLIENT_CPUS:-}" ] && TS=(taskset -c "$CLIENT_CPUS")
PORT=$PORT setsid bash "$SERVE" > "$OUT/server.log" 2>&1 &
SP=$!
cleanup() { kill -TERM -- -$SP 2>/dev/null; sleep 5; kill -KILL -- -$SP 2>/dev/null; }
trap cleanup EXIT
# first start compiles the CUDA kernels: allow up to 60 min
for _ in $(seq 1 720); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  kill -0 $SP 2>/dev/null || { echo "server died"; tail -30 "$OUT/server.log"; exit 4; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "server not healthy"; exit 5; }
reps=${REPS:-2}; [ "$CONC" -gt 1 ] && reps=1
for r in $(seq 1 "$reps"); do
  "${TS[@]}" "$PY" "$HERE/client.py" --base-url "http://127.0.0.1:$PORT" --model "${SERVED_NAME:-m}" \
    --prompts "$HERE/prompts96.jsonl" --concurrency "$CONC" --max-tokens "${MAXTOK:-1024}" \
    --out "$OUT/c${CONC}_r${r}.json" | tee -a "$OUT/summary.jsonl"
done
if [ -n "${LONGCTX:-}" ]; then
  "${TS[@]}" "$PY" "$HERE/acc_longctx.py" --base-url "http://127.0.0.1:$PORT" --out "$OUT/longctx.json" | tee -a "$OUT/summary.jsonl"
fi
if [ -n "${ACC:-}" ]; then
  "${TS[@]}" "$PY" "$HERE/acc_gsm8k.py" --base-url "http://127.0.0.1:$PORT" --n "${ACC_N:-300}" \
    --concurrency "${ACC_CONC:-$CONC}" --out "$OUT/acc.json" | tee -a "$OUT/summary.jsonl"
fi
