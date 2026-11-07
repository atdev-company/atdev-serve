# AI_SETUP.md: end-to-end setup instructions for AI coding agents

You are an AI agent asked to set up **atdev-serve** (a vLLM 0.31.0 plugin that serves Qwen3.8-27B
fast on one NVIDIA A100 80GB). Execute the phases below in order. Each step lists the command, the
expected result, and what to do on failure. Do not skip the checks. Report the final values listed in
Phase 7 to the user.

Repository: `https://github.com/atdev-company/atdev-serve`. Human guide: `docs/USER_GUIDE.md`.

## Tested environment reference

The project's inspected stack is **Python 3.12.13 / PyTorch 2.13.0+cu130 /
vLLM 0.31.0 / CUDA toolkit 13.0 (nvcc V13.0.88)** on an **A100-SXM4-80GB** with
NVIDIA driver **580.126.20**. See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for the
complete version snapshot and reporting commands. Record the actual installed
versions during setup; the observed patch versions are not additional installer
pins. Do not infer the installed CUDA toolkit from the CUDA label in `nvidia-smi`;
check both `torch.version.cuda` and `nvcc --version`.

## Rules

* **R1. Ask before acting** when a step needs `sudo`, installs system packages, stops another process,
  or uses a GPU that is already busy. Never kill processes you did not start.
* **R2. One GPU.** Use exactly one A100 80GB. Default GPU index 0, or the one the user names. Export
  `CUDA_VISIBLE_DEVICES=<index>` for every command and do not change it afterwards. A busy GPU means
  more than 1 GB already used or any compute process listed by `nvidia-smi`.
* **R3. Long first start.** The first server start compiles CUDA kernels and downloads model files
  (about 32 GiB for C1 fast; C1 accuracy skips the 12.5 GB W3 overlay; C32 needs the INT4 base plus a roughly 12.5 GB Hadamard sidecar; profiles are defined in Phase 1). Allow up to 2 hours. Do not interrupt it: interrupting can leave stale build locks (see F6).
* **R4. Paths.** Work inside the repository root (`$REPO`). Use absolute paths in your own scripts.
* **R5. Secrets.** Reuse an existing authorized Hub login or user-provided `HF_TOKEN` first. If access is still needed, ask the user to provide it or run
  `huggingface-cli login`. Never print it or write it to files in the repository.

## Phase 0: get the repository

```bash
export REPO="${REPO:-$HOME/atdev-serve}"
git clone https://github.com/atdev-company/atdev-serve "$REPO"   # skip if already present
cd "$REPO"
test -f pyproject.toml && test -f serve_c1.sh && echo REPO_OK
```

Expected: `REPO_OK`.

For an existing installation, preserve local changes, update the checkout with `git pull --ff-only`,
then refresh the installed plugin explicitly (the package version may still be `0.1.0`):

```bash
source "$REPO/atdev-env.sh"
python -m pip install --no-deps --force-reinstall "$REPO"
```

Restart only a server you own. The current C32 profile requires its additional Hadamard sidecar;
C1 accuracy needs the existing drafter but no W3 overlay. Docker users rebuild with `--build`.

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
per request, original W3 maximum-speed profile) unless the user asks for improved measured broad accuracy: then use `c1-accuracy`, which disables W3 but retains the same drafter and engine. For many concurrent users or throughput, use `c32`, which ships only Hadamard W4A4 with built-in MTP. Set `STACK=c1`, `STACK=c1-accuracy`, or `STACK=c32`. Do not silently change a user-selected profile. The up-to-14.2x headline belongs only to C1 fast; inspect [docs/ACCURACY.md](docs/ACCURACY.md) for all score and speed differences.

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
case "$STACK" in
  c1) START_SCRIPT=serve_c1.sh ;;
  c1-accuracy) START_SCRIPT=serve_c1_accuracy.sh ;;
  c32) START_SCRIPT=serve_c32.sh ;;
  *) echo "Choose c1, c1-accuracy or c32"; exit 2 ;;
esac
nohup setsid bash -c 'echo $$ > "$1"; exec "./$2"' _ "$REPO/server_$STACK.pid" "$START_SCRIPT" > "$LOG" 2>&1 &
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

Verify the selected profile before accepting readiness:

- `c1`: W3 mixed loading must be enabled; both original W3 sidecars and the quantized d16 drafter are used.
- `c1-accuracy`: W3 mixed loading must be absent; the same d16 drafter remains enabled. No W3 overlay should be requested or downloaded.
- `c32`: Hadamard W4A4 (`ATDEV_A4_REAL=had3`) and the selected sidecar must load. Built-in MTP remains enabled; no DFlash/external drafter is added. Do not fall back to unrotated W4A4 or W4A8 after an error.

Triton JIT warnings on the first request are normal.

On `FAILED`, the grep stops at the first match, which is often not the cause: read the last error
lines (`grep -E 'Error|Exception' "$LOG" | tail -5`) and match them against Phase 6.

Harmless warnings: unauthenticated Hugging Face Hub requests, and vLLM `Can't pickle ...` for the AOT
cache.

### Optional 262,144-token context

The default launch limit is 8,192 total tokens (input + output). If the user requests the full native context, start **C1 Fast W3 or C32 Hadamard W4A4** with `--max-model-len 262144` appended to its launch command. If a server is already running, stop only the owned server before restarting it with the new limit.

Both profiles passed a single-A100-80GB test with **261,583 input + 21 output = 261,604 total tokens**, correctly retrieving three codes from the beginning, middle and end of a synthetic long input. Keep request `max_tokens` within the remaining context. Published speed benchmarks still use the 8,192-context preset and an output cap of 1,024 tokens.

## Phase 5: verify the server

```bash
curl -s http://127.0.0.1:8100/v1/models
# expected: JSON with "id": "m"

curl -s http://127.0.0.1:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"m","messages":[{"role":"user","content":"What is 17*23? Answer with the number only."}],
  "temperature":0,"max_tokens":512,"chat_template_kwargs":{"enable_thinking":false}}'
# expected: a completion whose content contains 391
```

The model's original `generation_config.json` defaults to temperature 1.0, but the released
`serve_c1.sh` / `serve_c1_accuracy.sh` / `serve_c32.sh` launch presets override the server default
to **temperature 0** with `--override-generation-config '{"temperature":0}'`. Requests that omit
`temperature` therefore use greedy decoding. An explicitly supplied request temperature takes
precedence. The examples still pass `"temperature":0` explicitly for reproducible comparisons;
`temperature > 0` has not been validated for the published speed or quality.

Optional speed check (send it twice; the first requests after a start are slower):

```bash
python3 - <<'PY'
import json, time, urllib.request
body = {"model": "m", "temperature": 0, "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Write a Python function that merges two sorted lists, with tests."}]}
req = urllib.request.Request("http://127.0.0.1:8100/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
t = time.time(); r = json.load(urllib.request.urlopen(req)); dt = time.time() - t
n = r["usage"]["completion_tokens"]; print(f"{n} tokens in {dt:.1f} s = {n / dt:.0f} tok/s")
PY
```

Expected on an idle A100 80GB with the `c1` stack: a few hundred tok/s for one request (the
published 408.9 is the original C1-fast mean over 96 prompts; single prompts vary). C1 accuracy measured 357.17 tok/s over two repetitions. C32 Hadamard measured 2,688.13 aggregate tok/s at c=32 in one run; do not compare that with a single-request speed. The `c32` stack is tuned for many
concurrent requests, so one request alone is slower there. The full measurement method is in
`docs/BENCHMARKS.md`.

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
| F9 | no internet | the models must already be local; for C1 accuracy omit W3 and use the same local drafter; for C32 set `HADW=/models/atdev-c32/hadamard/qwen38_27b_hadamard_gptq_d10.safetensors` and pre-cache the INT4 base. C1 fast: set `MODEL=<dir>` (and for c1 `W3_OVERLAY=<dir>/w3/w3_gptqact_d30_mlpo.safetensors,<dir>/w3/w3_gptqact_d30_qkv.safetensors` `DRAFTER=<dir>/drafter`, where `<dir>` is a local copy of `atdev-company/Qwen3.8-27B-atdev-c1`) and `HF_HUB_OFFLINE=1` |
| F10 | Docker build fails on `apt-get` or `pip` | retry once (network). If the base image tag is not found, use `--build-arg CUDA_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04` |
| F11 | `unsupported GNU version` from nvcc | install a g++ version supported by nvcc 13.0 and export `CC` / `CXX` (R1) |
| F12 | `unknown ATDEV_OPT entries` | `ATDEV_OPT` was edited by hand. Restore it from the serve script |
| F13 | speed far below the expectation | check that the GPU is an idle A100 80GB (P2 / P4), the request uses `temperature: 0`, and the measurement is not of the first requests after start |
| F14 | `ipc path ... is longer than 107 characters` (ZMQError) | vLLM's IPC socket lives under `TMPDIR`; export a short `TMPDIR` (e.g. `export TMPDIR=/tmp/atdev`) and restart |

If a failure is not in this table, stop and report the last error block and the outputs of P2 and P9
to the user.

## Phase 7: report to the user

Report: the install path (Docker or venv), the exact profile (`c1` fast / `c1-accuracy` / `c32` Hadamard W4A4), the GPU index, the endpoint URL
(`http://<host>:8100/v1`, model name `m`), the time the first start took, the result of the Phase 5
math check, and the speed check result if it was run. Give the user the commands to stop and
restart the server (Phase 8). Include the active target precision, sidecar/drafter sources, and whether the observed speed uses single-request decode or aggregate throughput. Report any errors or accuracy differences; do not label a short math answer as full accuracy validation. Link the measured profile table in `docs/ACCURACY.md`.

## Phase 8: stop and restart

```bash
# venv
kill -TERM -- -"$(cat "$REPO/server_$STACK.pid")"      # stops the process group started in Phase 3
# Docker
docker compose --profile "$STACK" down
```

Restarting uses the same commands as Phase 3 / Phase 2A. It is fast after the first start, because
the kernels and models are cached.
