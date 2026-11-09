# AI_SETUP.md: end-to-end setup instructions for AI coding agents

You are an AI agent asked to set up **atdev-serve** (a vLLM 0.31.0 plugin that serves Qwen3.8-27B
fast on one NVIDIA A100 80GB). Execute the phases below in order. Each step lists the command, the
expected result, and what to do on failure. Do not skip the checks. Report the final values listed in
Phase 7 to the user.

Repository: `https://github.com/atdev-company/atdev-serve`. Human guide: `docs/USER_GUIDE.md`.

## Pick one profile first (decision guide)

atdev-serve ships **four profiles**. The name says what it is for: **c** = concurrent clients and the number is how many it is tuned for; **m** = multi-turn (the server keeps the state of a growing
conversation and prefills only what is new), **s** = single-turn (independent requests, no cache). Ask the user the two questions below, then use the table. Do not silently change a profile the user already chose.

1. **How many people or programs will talk to the server at the same time?** One (or occasionally two) means `c1*`; many at once (a team, a fleet of agents, an API) means `c32*`.
2. **Do the requests repeat a long prefix?** A coding agent, a long chat, or a document Q&A session sends the same history again with something added each turn: choose `*m`. One-shot calls with short, unrelated prompts: choose `*s`.

| Setup | STACK value | Real use case | Command |
|---|---|---|---|
| **c1m** | `c1m` | one person with a coding agent, a long chat, or document Q&A over a long file; context grows to 100k+ tokens | `./serve.sh c1m` |
| **c1s** | `c1s` | one-shot calls with short, unrelated prompts (classification, extraction, quick questions, scripts that call the model one prompt at a time) | `./serve.sh c1s` |
| **c32m** | `c32m` | a shared team assistant or a fleet of agents, many conversations that each grow turn by turn | `./serve.sh c32m` |
| **c32s** | `c32s` | batch jobs and API traffic made of many independent requests (summarize 1,000 documents, evaluation runs, a public endpoint) | `./serve.sh c32s` |

If the user wants the best measured broad accuracy for single-user work and accepts a lower speed, offer **c1s accuracy** (`STACK=c1s-accuracy`, `./serve_c1_accuracy.sh`: no 3-bit overlay, 357 tok/s, benchmark scores within about 0.3-0.6 points of stock on MMLU-Pro and Korean knowledge).
If the user is unsure and has one user: `c1m`. It costs about 3.6% on short unrelated prompts (394.4 vs 409.5 mean decode tok/s over 96 prompts) and is much faster whenever a prefix repeats.

Measured facts to base the choice on (all on one A100 80GB, vLLM 0.31.0, greedy, full stack; details in [README.md](README.md) and [docs/MULTITURN.md](docs/MULTITURN.md)):

* c1m: a 2 x 38-turn agent trace up to 111k tokens runs at 82.2 / 159.4 / 198.6 output tok/s at `max_tokens` 128 / 512 / 2048 against 34.7 / 52.0 / 58.4 for the fastest stock vLLM setting (2.37x / 3.07x / 3.40x; whole-trace wall time 116.0 / 205.2 / 301.2 s against 268.6 / 623.1 / 1,136.3 s).
* c1s: 408.9 tok/s mean decode per request, 2.08x the best stock vLLM (196.5).
* c32m: 32 concurrent 7-turn sessions reach 386.3 / 857.3 / 1,096.1 aggregate output tok/s against 245.5 / 543.5 / 719.2 for the fastest stock setting (1.57x / 1.58x / 1.52x; wall time 73.4 / 117.6 / 199.1 s against 115.1 / 184.3 / 314.0 s).
* c32s: 2,688.1 aggregate output tok/s, 1.53x the best stock vLLM (single measured run).
* KV cache: measured in the server log at start-up: c1m 284,160 tokens at its 131,072-token window; c1s 172,398 at 8,192 (c1s accuracy 166,803); c32m 514,755 at 16,384; c32s 423,424 at 8,192. c1m has the largest cache of the two single-user setups.

## The four setups

Each setup is a separate, self-contained recipe: pick **one**, set `STACK` to its value, then run Phases 1-8 once.

### Setup c1m: one user, multi-turn (`STACK=c1m`)

* **Fits:** an agent loop or chat where each request resends the conversation so far. Long context is the point: 131,072-token window (the measured trace reaches 111k tokens).
* **What you get:** prefix caching (cache hit rate 0.95 on the trace), faster prefill kernels, a start-up warmup that logs `atdev warmup: done`. Single requests decode at 394.4 tok/s.
* **Downloads:** INT4 target (19 GB) + W3 overlay (12.5 GB) + drafter (1.35 GB) from `atdev-company/Qwen3.8-27B-atdev-c1`.
* **Caveats to tell the user:** 3.6-3.7% slower than c1s on independent short prompts; at c=1 its context window is 131,072 against 8,192 for c1s; the prefix cache only helps when the conversation prefix is unchanged; greedy decoding only; the c1 W3 overlay carries a measured 2.5-3.3 point gap on MMLU-Pro and Korean knowledge against stock INT4.
* **Optional:** `ATDEV_EOR_CKPT=1` also reuses the server's own previous reply (neutral at c=1 on the measured traces).

### Setup c1s: one user, single-turn (`STACK=c1s`)

* **Fits:** independent requests with short prompts, where nothing repeats; or when you need the longest window: `--max-model-len 262144` was tested (261,583 input tokens).
* **What you get:** 408.9 tok/s decode per request (2.08x best stock vLLM), no prefix caching, 8,192-token default window.
* **Downloads:** same as c1m. `c1s accuracy` (`STACK=c1s-accuracy`) skips the W3 overlay.

### Setup c32m: many users, multi-turn (`STACK=c32m`)

* **Fits:** up to 32 conversations in flight that each grow (shared assistant, agent fleet). 16,384-token window, prefix caching for MTP with the shared-prefix checkpoint.
* **What you get:** 1.52x-1.58x the output tok/s of the best stock vLLM on 32 concurrent 7-turn sessions, a start-up warmup with 32-way waves.
* **Downloads:** INT4 target (19 GB) + the Hadamard sidecar (about 12.5 GB) from `atdev-company/Qwen3.8-27B-atdev-c32`. No separate drafter.
* **Caveats:** IFEval prompt strict is 2.0 points below stock INT4 (run-to-run spread up to about 4 points); outputs are not byte-identical to stock; greedy decoding only. `ATDEV_EOR_CKPT=1` (off by default) helps live conversations at the cost of 3% more GPU memory.

### Setup c32s: many users, single-turn (`STACK=c32s`)

* **Fits:** batch jobs and API traffic of independent requests. 8,192-token default window (262,144 tested).
* **What you get:** 2,688.1 aggregate output tok/s at 32 in flight (1.53x best stock vLLM, single measured run), Hadamard W4A4 accuracy in the class of stock INT4.
* **Downloads:** same as c32m.

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
  (about 32 GiB for c1m / c1s; c1s accuracy skips the 12.5 GB W3 overlay; c32m / c32s need the INT4 base plus a roughly 12.5 GB Hadamard sidecar; profiles are defined in the decision guide). The install tests measured 17-19 minutes for a first start on a clean machine (a 34-35 GB download included; install itself about 8 minutes); allow up to 2 hours on a slow network. Do not interrupt it: interrupting can leave stale build locks (see F6).
* **R4. Paths.** Work inside the repository root (`$REPO`). Use absolute paths in your own scripts.
* **R5. Secrets.** Reuse an existing authorized Hub login or user-provided `HF_TOKEN` first. If access is still needed, ask the user to provide it or run
  `huggingface-cli login`. Never print it or write it to files in the repository.

## Phase 0: get the repository

```bash
export REPO="${REPO:-$HOME/atdev-serve}"
git clone https://github.com/atdev-company/atdev-serve "$REPO"   # skip if already present
cd "$REPO"
test -f pyproject.toml && test -f serve.sh && echo REPO_OK
```

Expected: `REPO_OK`. If the checkout is not in `$HOME/atdev-serve`, set `REPO` to its real location before every command in this guide (every phase uses `$REPO`).

For an existing installation, preserve local changes, update the checkout with `git pull --ff-only`,
then refresh the installed plugin explicitly (the package version may still be `0.1.0`):

```bash
source "$REPO/atdev-env.sh"
python -m pip install --no-deps --force-reinstall "$REPO"
```

Restart only a server you own. c32m / c32s require the additional Hadamard sidecar;
c1s accuracy needs the existing drafter but no W3 overlay. Docker users rebuild with `--build`.

## Phase 1: preflight (read-only)

Run each check and record the result. `nvidia-smi` ignores `CUDA_VISIBLE_DEVICES` and lists all
GPUs: judge P2 / P4 only by the row for the chosen index.

| id | command | pass condition | on failure |
|---|---|---|---|
| P1 | `uname -sm` | `Linux x86_64` | stop: unsupported platform |
| P2 | `nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader` | at least one row with `A100` and `81920 MiB` (or `80` GB) | no A100 80GB: stop and tell the user this release supports only the A100 80GB (sm_80) |
| P3 | same output, driver column | driver major version >= 580 | ask the user to upgrade the NVIDIA driver (R1) |
| P4 | `nvidia-smi -i <idx> --query-compute-apps=pid,used_memory --format=csv,noheader` (also print `nvidia-smi --query-gpu=index,uuid,memory.used --format=csv` for the index/UUID mapping) | empty output (no compute process) and < 1 GB used on that GPU. Treat memory used >= 1 GB as "in use" even when the compute-apps list is empty (another container or user can hold memory without being listed) | pick another free A100 or ask the user (R1, R2). If the GPU is only partly in use and the user agrees to share it, lower `--gpu-memory-utilization` (append e.g. `--gpu-memory-utilization 0.60` to the launch command; the default is 0.90 / 0.93) so that the value times 80 GB fits in the free memory with about 8 GiB margin; very small values leave too little KV cache and the server will refuse to start |
| P5 | `df -Pk "$REPO" "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null` | >= 80 GB free where the venv and the HF cache live | ask the user for a disk with space and relocate the caches there (optional, set before install/start; atdev-env.sh has commented lines): `HF_HOME`, `TORCH_EXTENSIONS_DIR`, `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, `XDG_CACHE_HOME` (FlashInfer uses it), `PIP_CACHE_DIR` |
| P6 | `curl -s -o /dev/null -w '%{http_code}' https://huggingface.co/api/models/RedHatAI/Qwen3.8-27B-INT4` | `200` | `429` can be transient (rate limit): wait a minute and retry. Otherwise offline: see F9 |
| P7 | `docker info >/dev/null 2>&1 && docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi -L` | prints the GPUs | Docker with GPU is not usable: take the venv path |
| P8 | `command -v python3.12 \|\| command -v uv` | a path is printed | venv path needs Python 3.12 or uv: ask to install (R1) or use Docker |
| P9 | `for d in "$CUDA_HOME" /usr/local/cuda-13.0 /usr/local/cuda; do [ -x "$d/bin/nvcc" ] && "$d/bin/nvcc" --version \| grep -q 'release 13.0' && echo "CUDA_DIR=$d" && break; done` | a `CUDA_DIR=` line (that directory has nvcc `release 13.0`) | venv path needs the CUDA 13.0 toolkit: ask to install it (R1), or use Docker, or try F2 |
| P10 | `g++ --version \| head -1` | prints a version | `sudo apt install build-essential` (R1) |
| P11 | `PORT="${PORT:-8100}"; (ss -ltn "sport = :$PORT" 2>/dev/null \| grep -q LISTEN) && echo PORT_BUSY \|\| echo PORT_FREE` | `PORT_FREE` | choose another port: `export PORT=<free port>` for the Python route (`ATDEV_PORT` for Docker) and use it in every URL below |

**Choose the install path.** If P7 passes, use **Phase 2A (Docker)**. If not, use **Phase 2B
(venv)**, which requires P8, P9 and P10 to pass.

**Choose the stack.** Use the decision guide at the top. Set exactly one of `STACK=c1m`, `STACK=c1s`, `STACK=c1s-accuracy`, `STACK=c32m`, `STACK=c32s`. Docker uses the compose profile with the matching name:
Docker uses the compose profile of the same name (the older names `c1`, `c1-single`, `c1-accuracy`, `c32`, `c32-single` still work as aliases); set `COMPOSE_PROFILE`:

```bash
case "$STACK" in c1m|c1s|c1s-accuracy|c32m|c32s) COMPOSE_PROFILE=$STACK;; *) echo "bad STACK"; exit 2;; esac
```

The up-to-14.2x headline belongs only to c1s; the multi-turn speedups (up to 3.40x for c1m and 1.58x for c32m over the fastest stock setting, in output tok/s) are full-stack results measured on scripted conversations; inspect [docs/ACCURACY.md](docs/ACCURACY.md) and [docs/MULTITURN.md](docs/MULTITURN.md) for all score and speed differences.

## Phase 2A: install with Docker

```bash
cd "$REPO"
export ATDEV_GPU=<gpu index from P4> ATDEV_PORT=8100   # ATDEV_PORT is the host port (Docker's name for PORT)
export PORT=$ATDEV_PORT                                 # the check URLs below use PORT
docker compose --profile "$COMPOSE_PROFILE" build
docker compose --profile "$COMPOSE_PROFILE" run --rm "atdev-$COMPOSE_PROFILE" check
```

Expected: the `check` output includes `vllm 0.31.0`, a `cuda available True` line and an nvcc line
with `release 13.0`. On a build failure, record the last 50 lines and go to F10.

Then start it, detached:

```bash
docker compose --profile "$COMPOSE_PROFILE" up -d
docker compose --profile "$COMPOSE_PROFILE" logs -f --tail 20     # watch for the lines in Phase 4
```

Continue with Phase 4. Use `docker compose --profile "$COMPOSE_PROFILE" logs` wherever Phase 4 reads
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

`serve.sh` is the entry point: `./serve.sh c1m | c1s | c32m | c32s` (c1s accuracy has its own script). Check the port first (P11), and run the commands below from `$REPO`.

```bash
source "$REPO/atdev-env.sh"
export CUDA_VISIBLE_DEVICES=<gpu index from P4>
export PORT="${PORT:-8100}"        # the launch scripts read PORT; the check URLs below use it too
LOG="$REPO/server_$STACK.log"
case "$STACK" in
  c1m|c1s|c32m|c32s) START_SCRIPT=serve.sh; START_ARG=$STACK ;;     # ./serve.sh <name>
  c1s-accuracy) START_SCRIPT=serve_c1_accuracy.sh; START_ARG= ;;
  *) echo "Choose c1m, c1s, c1s-accuracy, c32m or c32s"; exit 2 ;;
esac
nohup setsid bash -c 'echo $$ > "$1"; exec "./$2" $3' _ "$REPO/server_$STACK.pid" "$START_SCRIPT" "$START_ARG" > "$LOG" 2>&1 &
# the PID file holds the process-group leader (used by Phase 8)
```

## Phase 4: wait for readiness

Poll every 30 s (the install tests took 17-19 minutes from a clean machine, including a 34-35 GB model download and about 8 minutes of install; allow longer on a slow link):

```bash
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:${PORT:-8100}/health >/dev/null && { echo READY; break; }
  grep -qE 'Traceback|Error:|server exited' "$LOG" && { echo FAILED; break; }
  kill -0 "$(cat "$REPO/server_$STACK.pid")" 2>/dev/null || { echo FAILED; break; }
  sleep 30
done
```

Docker path: replace the `grep` line with
`docker compose --profile "$COMPOSE_PROFILE" logs --no-color 2>&1 | grep -qE 'Traceback|Error:' && { echo FAILED; break; }`
and the `kill -0` line with
`[ -n "$(docker compose --profile "$COMPOSE_PROFILE" ps -q --status running)" ] || { echo FAILED; break; }`.

While waiting, these log lines are expected, in roughly this order:

1. `atdev-serve: enabled optimizations: [...]` (the plugin is active). If it is missing after the
   model starts loading, go to F4.
2. A few minutes with a quiet log while the kernels compile (first start only): `nvcc` / `cicc`
   processes are using CPU. This is normal (R3).
3. Model download (first start only), then weight loading
4. `torch.compile` and FlashInfer `Building JIT module sampling`
5. CUDA graph capture, then `Application startup complete`

Verify the selected profile before accepting readiness:

- `c1m` and `c1s`: W3 mixed loading must be enabled; both original W3 sidecars and the quantized d16 drafter are used. `c1m` additionally logs `atdev warmup: done` a short while after readiness (wait for it before measuring).
- `c1s-accuracy`: W3 mixed loading must be absent; the same d16 drafter remains enabled. No W3 overlay should be requested or downloaded.
- `c32m` and `c32s`: Hadamard W4A4 (`ATDEV_A4_REAL=had3`) and the selected sidecar must load (`c32m` also logs the warmup line). Built-in MTP remains enabled; no DFlash/external drafter is added. Do not fall back to unrotated W4A4 or W4A8 after an error.

Triton JIT warnings on the first request are normal.

On `FAILED`, the grep stops at the first match, which is often not the cause: read the last error
lines (`grep -E 'Error|Exception' "$LOG" | tail -5`) and match them against Phase 6.

Harmless warnings: unauthenticated Hugging Face Hub requests, and vLLM `Can't pickle ...` for the AOT
cache.

### Optional 262,144-token context

The default launch limit of `c1s` and `c32s` is 8,192 total tokens (input + output); `c1m` starts with 131,072 and `c32m` with 16,384, and neither was tested with 262,144. If the user requests the full native context, start **`c1s` or `c32s`** with `--max-model-len 262144` appended to its launch command. If a server is already running, stop only the owned server before restarting it with the new limit.

Both profiles passed a single-A100-80GB test with **261,583 input + 21 output = 261,604 total tokens**, correctly retrieving three codes from the beginning, middle and end of a synthetic long input. Keep request `max_tokens` within the remaining context. Published speed benchmarks still use the 8,192-context preset and an output cap of 1,024 tokens.

## Phase 5: verify the server

`serve.sh` (Phase 3) is the entry point that started this server; every URL below uses `${PORT:-8100}`.

```bash
curl -s http://127.0.0.1:${PORT:-8100}/v1/models
# expected: JSON with "id": "m" and this profile's context window in "max_model_len"
# c1m 131072, c1s 8192, c1s accuracy 8192, c32m 16384, c32s 8192 (confirms the server you reached is your own launch, not another one on the port)

curl -s http://127.0.0.1:${PORT:-8100}/v1/chat/completions -H 'Content-Type: application/json' -d '{
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

### Checking prefix-cache hits (m profiles)

Wait for `atdev warmup: done` in the log before measuring anything. `usage.prompt_tokens_details` is `null` by default: the launchers do not pass `--enable-prompt-tokens-details`. Appending that flag to the launch command (for example `./serve.sh c1m --enable-prompt-tokens-details`) fills `usage.prompt_tokens_details.cached_tokens`; we confirmed this on c1m, where a repeated document reported 8,160 of 8,268 prompt tokens cached, and on c1s, which reports 0. Without the flag, read the server counter instead:

```bash
m() { curl -s "http://127.0.0.1:${PORT:-8100}/metrics" | awk '/^vllm:prefix_cache_hits_total/ {print $2}'; }
before=$(m)
# ... send the second request of a conversation (the same system prompt and history, a new last user message) ...
after=$(m); echo "cache hit tokens: $((${after%.*} - ${before%.*}))"
```

A positive delta means the prefix was reused. The s profiles have no prefix caching, so the counter stays at 0.

Optional speed check (send it twice; the first requests after a start are slower):

```bash
python3 - <<'PY'
import json, os, time, urllib.request
body = {"model": "m", "temperature": 0, "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Write a Python function that merges two sorted lists, with tests."}]}
req = urllib.request.Request(f"http://127.0.0.1:{os.environ.get('PORT', '8100')}/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
t = time.time(); r = json.load(urllib.request.urlopen(req)); dt = time.time() - t
n = r["usage"]["completion_tokens"]; print(f"{n} tokens in {dt:.1f} s = {n / dt:.0f} tok/s")
PY
```

Expected on an idle A100 80GB with `c1s` (about 394 tok/s with `c1m`): a few hundred tok/s for one request (the
published 408.9 is the c1s mean over 96 prompts; single prompts vary). c1s accuracy measured 357.17 tok/s over two repetitions. c32s measured 2,688.13 aggregate tok/s at c=32 in one run (c32m is measured on conversation traces, not on this workload); do not compare that with a single-request speed. The c32 profiles are tuned for many
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
| F9 | no internet | the models must already be local; for c1s accuracy omit W3 and use the same local drafter; for C32 set `HADW=/models/atdev-c32/hadamard/qwen38_27b_hadamard_gptq_d10.safetensors` and pre-cache the INT4 base. c1m / c1s: set `MODEL=<dir>` (and for c1 `W3_OVERLAY=<dir>/w3/w3_gptqact_d30_mlpo.safetensors,<dir>/w3/w3_gptqact_d30_qkv.safetensors` `DRAFTER=<dir>/drafter`, where `<dir>` is a local copy of `atdev-company/Qwen3.8-27B-atdev-c1`) and `HF_HUB_OFFLINE=1` |
| F10 | Docker build fails on `apt-get` or `pip` | retry once (network). If the base image tag is not found, use `--build-arg CUDA_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04` |
| F11 | `unsupported GNU version` from nvcc | install a g++ version supported by nvcc 13.0 and export `CC` / `CXX` (R1) |
| F12 | `unknown ATDEV_OPT entries` | `ATDEV_OPT` was edited by hand. Restore it from the serve script |
| F13 | speed far below the expectation | check that the GPU is an idle A100 80GB (P2 / P4), the request uses `temperature: 0`, and the measurement is not of the first requests after start |
| F14 | `ipc path ... is longer than 107 characters` (ZMQError) | vLLM's IPC socket lives under `TMPDIR`; export a short `TMPDIR` (e.g. `export TMPDIR=/tmp/atdev`) and restart |

If a failure is not in this table, stop and report the last error block and the outputs of P2 and P9
to the user.

## Phase 7: report to the user

Report: the install path (Docker or venv), the exact profile (`c1m` / `c1s` / `c1s-accuracy` / `c32m` / `c32s`), the GPU index, the endpoint URL
(`http://<host>:${PORT:-8100}/v1`, model name `m`), the time the first start took, the result of the Phase 5
math check, and the speed check result if it was run. Give the user the commands to stop and
restart the server (Phase 8). Include the active target precision, sidecar/drafter sources, and whether the observed speed uses single-request decode or aggregate throughput. Report any errors or accuracy differences; do not label a short math answer as full accuracy validation. Link the measured profile table in `docs/ACCURACY.md`.

## Phase 8: stop and restart

```bash
# venv
kill -TERM -- -"$(cat "$REPO/server_$STACK.pid")"      # stops the process group started in Phase 3
sleep 30; kill -0 "$(cat "$REPO/server_$STACK.pid")" 2>/dev/null && kill -TERM -- -"$(cat "$REPO/server_$STACK.pid")"   # still up after 30 s: send TERM again
# Docker
docker compose --profile "$COMPOSE_PROFILE" down
```

Restarting uses the same commands as Phase 3 / Phase 2A. It is fast after the first start, because
the kernels and models are cached.
