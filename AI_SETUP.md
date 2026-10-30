# AI_SETUP.md: end-to-end setup instructions for AI coding agents

You are an AI agent asked to set up **atdev-serve** (a vLLM 0.31.0 plugin that serves Qwen3.8-27B
fast on one NVIDIA A100 80GB). Execute the phases below in order. Each step lists the command, the
expected result, and what to do on failure. Do not skip the checks. Report the final values listed in
Phase 7 to the user.

Repository: `https://github.com/atdev-company/atdev-serve`. Human guide: `docs/USER_GUIDE.md`.

## Rules

* **R1. Ask before acting** when a step needs `sudo`, installs system packages, stops another process,
  or uses a GPU that is already busy. Never kill processes you did not start.
* **R2. One GPU.** Use exactly one A100 80GB. Default GPU index 0, or the one the user names. Export
  `CUDA_VISIBLE_DEVICES=<index>` for every command and do not change it afterwards. A busy GPU means
  more than 1 GB already used or any compute process listed by `nvidia-smi`.
* **R3. Long first start.** The first server start compiles CUDA kernels and downloads model files
  (about 32 GiB for c1, about 18 GiB for c32; the stacks are defined in Phase 1). Allow up to 2 hours. Do not interrupt it: interrupting can leave stale build locks (see F6).
* **R4. Paths.** Work inside the repository root (`$REPO`). Use absolute paths in your own scripts.
* **R5. Secrets.** If a Hugging Face token is needed, ask the user to provide it or run
  `huggingface-cli login`. Never print it or write it to files in the repository.

## Phase 0: get the repository

```bash
export REPO="${REPO:-$HOME/atdev-serve}"
git clone https://github.com/atdev-company/atdev-serve "$REPO"   # skip if already present
cd "$REPO"
test -f pyproject.toml && test -f serve_c1.sh && echo REPO_OK
```

Expected: `REPO_OK`.

## Phase 1: preflight (read-only)

Run each check and record the result. `nvidia-smi` ignores `CUDA_VISIBLE_DEVICES` and lists all
GPUs: judge P2 / P4 only by the row for the chosen index.

| id | command | pass condition | on failure |
|---|---|---|---|
| P1 | `uname -sm` | `Linux x86_64` | stop: unsupported platform |
| P2 | `nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader` | at least one row with `A100` and `81920 MiB` (or `80` GB) | no A100 80GB: stop and tell the user this release supports only the A100 80GB (sm_80) |
| P3 | same output, driver column | driver major version >= 580 | ask the user to upgrade the NVIDIA driver (R1) |
| P4 | `nvidia-smi -i <idx> --query-compute-apps=pid,used_memory --format=csv,noheader` (also print `nvidia-smi --query-gpu=index,uuid,memory.used --format=csv` for the index/UUID mapping) | empty output (no compute process) and < 1 GB used on that GPU | pick another free A100 or ask the user (R1, R2) |
| P5 | `df -Pk "$REPO" "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null` | >= 80 GB free where the venv and the HF cache live | ask the user for a disk with space and relocate the caches there (optional, set before install/start; atdev-env.sh has commented lines): `HF_HOME`, `TORCH_EXTENSIONS_DIR`, `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, `XDG_CACHE_HOME` (FlashInfer uses it), `PIP_CACHE_DIR` |
| P6 | `curl -s -o /dev/null -w '%{http_code}' https://huggingface.co/api/models/RedHatAI/Qwen3.8-27B-INT4` | `200` | offline: see F9 |
| P7 | `docker info >/dev/null 2>&1 && docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi -L` | prints the GPUs | Docker with GPU is not usable: take the venv path |
| P8 | `command -v python3.12 \|\| command -v uv` | a path is printed | venv path needs Python 3.12 or uv: ask to install (R1) or use Docker |
| P9 | `for d in "$CUDA_HOME" /usr/local/cuda-13.0 /usr/local/cuda; do [ -x "$d/bin/nvcc" ] && "$d/bin/nvcc" --version \| grep -q 'release 13.0' && echo "CUDA_DIR=$d" && break; done` | a `CUDA_DIR=` line (that directory has nvcc `release 13.0`) | venv path needs the CUDA 13.0 toolkit: ask to install it (R1), or use Docker, or try F2 |
| P10 | `g++ --version \| head -1` | prints a version | `sudo apt install build-essential` (R1) |

**Choose the install path.** If P7 passes, use **Phase 2A (Docker)**. If not, use **Phase 2B
(venv)**, which requires P8, P9 and P10 to pass.

**Choose the stack.** `c1` is tuned for c=1, i.e. 1 client (a single user, one request at a time);
`c32` is tuned for c=32, i.e. 32 concurrent clients (32 requests in flight). Use `c1` (lowest latency
per request) unless the user asked for many concurrent users or for throughput, in which case use `c32`. Set `STACK=c1` or `STACK=c32`.

## Phase 2A: install with Docker

```bash
cd "$REPO"
export ATDEV_GPU=<gpu index from P4> ATDEV_PORT=8100
docker compose --profile "$STACK" build
docker compose --profile "$STACK" run --rm "atdev-$STACK" check
```

Expected: the `check` output includes `vllm 0.31.0`, a `cuda available True` line and an nvcc line
with `release 13.0`. On a build failure, record the last 50 lines and go to F10.

Then start it, detached:

```bash
docker compose --profile "$STACK" up -d
docker compose --profile "$STACK" logs -f --tail 20     # watch for the lines in Phase 4
```

Continue with Phase 4. Use `docker compose --profile "$STACK" logs` wherever Phase 4 reads
`$LOG`.

## Phase 2B: install with a virtualenv

```bash
cd "$REPO"
./install.sh --dry-run                   # read-only; review the [FAIL] / [warn] lines
./install.sh --cuda-home "$CUDA_DIR"     # CUDA_DIR from P9
```

Expected: exit code 0, a line starting with `vllm 0.31.0 | torch 2.13.0+cu130 | cuda 13.0 | plugin ok`,
and a new file `$REPO/atdev-env.sh`.

| exit code | meaning | action |
|---|---|---|
| 3 | preflight failed | read the `[FAIL]` lines, fix them (R1), re-run |
| 4 | pip install failed | check network / disk. Re-run (it reuses `.venv`). If vllm fails to resolve, confirm Python is 3.12 |
| 5 | verification failed | see F3 / F4 |

Verify independently:

```bash
source "$REPO/atdev-env.sh"
python -c "import vllm, atdev_serve; print(vllm.__version__)"          # -> 0.31.0
python -c "import importlib.metadata as m; print([e.value for e in m.entry_points(group='vllm.general_plugins')])"
# -> list contains 'atdev_serve.register:register_all'
which vllm                                                              # -> $REPO/.venv/bin/vllm
nvcc --version | grep -q 'release 13.0' && echo NVCC_OK                 # -> NVCC_OK
```

## Phase 3: start the server (venv path)

```bash
source "$REPO/atdev-env.sh"
export CUDA_VISIBLE_DEVICES=<gpu index from P4>
LOG="$REPO/server_$STACK.log"
nohup setsid bash -c 'echo $$ > "$0"; exec ./serve_'"$STACK"'.sh' "$REPO/server_$STACK.pid" > "$LOG" 2>&1 &
# the PID file holds the process-group leader (used by Phase 8)
```

## Phase 4: wait for readiness

Poll every 30 s for up to 2 hours:

```bash
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:8100/health >/dev/null && { echo READY; break; }
  grep -qE 'Traceback|Error:|server exited' "$LOG" && { echo FAILED; break; }
  kill -0 "$(cat "$REPO/server_$STACK.pid")" 2>/dev/null || { echo FAILED; break; }
  sleep 30
done
```

Docker path: replace the `grep` line with
`docker compose --profile "$STACK" logs --no-color 2>&1 | grep -qE 'Traceback|Error:' && { echo FAILED; break; }`
and the `kill -0` line with
`[ -n "$(docker compose --profile "$STACK" ps -q --status running)" ] || { echo FAILED; break; }`.

While waiting, these log lines are expected, in roughly this order:

1. `atdev-serve: enabled optimizations: [...]` (the plugin is active). If it is missing after the
   model starts loading, go to F4.
2. About 5 minutes with a quiet log while the kernels compile (first start only): `nvcc` / `cicc`
   processes are using CPU. This is normal (R3).
3. Model download (first start only), then weight loading
4. `torch.compile` and FlashInfer `Building JIT module sampling`
5. CUDA graph capture, then `Application startup complete`

Triton JIT warnings on the first request are normal.

On `FAILED`, the grep stops at the first match, which is often not the cause: read the last error
lines (`grep -E 'Error|Exception' "$LOG" | tail -5`) and match them against Phase 6.

Harmless warnings: unauthenticated Hugging Face Hub requests, and vLLM `Can't pickle ...` for the AOT
cache.

## Phase 5: verify the server

```bash
curl -s http://127.0.0.1:8100/v1/models
# expected: JSON with "id": "m"

curl -s http://127.0.0.1:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"m","messages":[{"role":"user","content":"What is 17*23? Answer with the number only."}],
  "temperature":0,"max_tokens":512,"chat_template_kwargs":{"enable_thinking":false}}'
# expected: a completion whose content contains 391
```

The model's `generation_config.json` defaults to temperature 1.0; the checks here pass
`"temperature":0` explicitly. Pass the temperature explicitly for reproducible results.

Optional performance check. Run Phase 3 (first start) before the benchmark, so the kernels and
models are cached. It starts its own server, so stop the one from Phase 3 / 2A first (Phase 8). It
needs the `bench` extra, which install.sh installs:

```bash
source "$REPO/atdev-env.sh"
export CUDA_VISIBLE_DEVICES=<gpu index from P4>    # in this shell too; the serve scripts default to 0
bench/run_bench.sh "$STACK" "$REPO/results/agent_$STACK"
# c1  -> compare mean_decode_tps (not median, ~5% lower by design) with about 408.9 on an A100 80GB SXM (within ~5%)
# c32 -> prints agg_output_tps;  expected about 2,730.9
```

Docker path (same expectations; the results directory stays inside the removed container, the
numbers are printed):

```bash
docker compose --profile "$STACK" run --rm "atdev-$STACK" bench/run_bench.sh "$STACK" /tmp/agent_$STACK
```

## Phase 6: failure branches

| id | symptom in log / output | action |
|---|---|---|
| F1 | `CUDA_HOME environment variable is not set`, `nvcc: not found`, `No such file ... nvcc` | `source atdev-env.sh`. If it is still missing, re-run `./install.sh --cuda-home <dir with bin/nvcc>` |
| F2 | no system CUDA toolkit and the user cannot install one | `./install.sh --pip-nvcc` (experimental: pins NVIDIA's nvcc / crt / nvvm 13.0 wheels in the venv and sets `CUDA_HOME` to `site-packages/nvidia/cu13` in `atdev-env.sh`). If the JIT build fails with it, use Docker |
| F3 | `ModuleNotFoundError: atdev_serve` or the entry point is missing | the wrong Python is in use: `source atdev-env.sh`; `which python vllm` must point into `.venv` |
| F4 | no `enabled optimizations` line | same as F3; also check that `ATDEV_OPT` is set (the serve scripts export it) |
| F5 | `PermissionError` under `site-packages/atdev_serve/csrc` | the package dir is not writable: reinstall into a user-owned venv (install.sh does this) |
| F6 | log stalls at `Waiting for lock` / `waiting for ... lock` | a stale build lock from an interrupted start. Stop the server, run `find "${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}" -name lock -delete`, and restart |
| F7 | `CUDA out of memory` while loading | the GPU is not free (P4). Pick another GPU or ask the user (R1) |
| F8 | `GatedRepoError` / `401` / `RepositoryNotFound` | ask the user for HF access / token (R5), then retry |
| F9 | no internet | the models must already be local: set `MODEL=<dir>` (and for c1 `W3_OVERLAY=<dir>/w3/w3_gptqact_d30_mlpo.safetensors,<dir>/w3/w3_gptqact_d30_qkv.safetensors` `DRAFTER=<dir>/drafter`, where `<dir>` is a local copy of `atdev-company/Qwen3.8-27B-atdev-c1`) and `HF_HUB_OFFLINE=1` |
| F10 | Docker build fails on `apt-get` or `pip` | retry once (network). If the base image tag is not found, use `--build-arg CUDA_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04` |
| F11 | `unsupported GNU version` from nvcc | install a g++ version supported by nvcc 13.0 and export `CC` / `CXX` (R1) |
| F12 | `unknown ATDEV_OPT entries` | `ATDEV_OPT` was edited by hand. Restore it from the serve script |
| F13 | speed far below the expectation | check that the GPU is an idle A100 80GB (P2 / P4), the request uses `temperature: 0`, and the measurement is not of the first requests after start |
| F14 | `ipc path ... is longer than 107 characters` (ZMQError) | vLLM's IPC socket lives under `TMPDIR`; export a short `TMPDIR` (e.g. `export TMPDIR=/tmp/atdev`) and restart |

If a failure is not in this table, stop and report the last error block and the outputs of P2 and P9
to the user.

## Phase 7: report to the user

Report: the install path (Docker or venv), the stack (`c1` / `c32`), the GPU index, the endpoint URL
(`http://<host>:8100/v1`, model name `m`), the time the first start took, the result of the Phase 5
math check, and the benchmark numbers if they were run. Give the user the commands to stop and
restart the server (Phase 8).

## Phase 8: stop and restart

```bash
# venv
kill -TERM -- -"$(cat "$REPO/server_$STACK.pid")"      # stops the process group started in Phase 3
# Docker
docker compose --profile "$STACK" down
```

Restarting uses the same commands as Phase 3 / Phase 2A. It is fast after the first start, because
the kernels and models are cached.
