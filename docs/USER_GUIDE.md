# atdev-serve user guide

This guide takes you from an empty machine to a running, OpenAI-compatible Qwen3.8-27B server on one
A100 80GB. Follow the steps in order.

* [1. What you need](#1-what-you-need)
* [2. Choose a stack](#2-choose-a-stack)
* [3a. Install and run with Docker](#3a-install-and-run-with-docker)
* [3b. Install and run with a Python virtualenv](#3b-install-and-run-with-a-python-virtualenv)
* [4. Send requests](#4-send-requests)
* [5. Check that it is working well](#5-check-that-it-is-working-well)
* [6. Troubleshooting](#6-troubleshooting)
* [7. Settings you can change](#7-settings-you-can-change)

## 1. What you need

| | requirement | how to check |
|---|---|---|
| GPU | NVIDIA **A100 80GB** (other GPUs are not supported) | `nvidia-smi` lists `A100-SXM4-80GB` or `A100 80GB PCIe` |
| Driver | NVIDIA driver **>= 580** (CUDA 13.0) | `nvidia-smi` top-right shows `CUDA Version: 13.0` or higher |
| OS | Linux x86_64 | `uname -sm` |
| Disk | **80 GB free** (models ~37 GB, Python env ~9 GB, kernel cache ~2 GB, headroom) | `df -h ~` |
| Network | access to `huggingface.co` (models download on the first start) | `curl -I https://huggingface.co` |
| Docker path | Docker + NVIDIA Container Toolkit | `docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi` |
| venv path | Python 3.12, CUDA 13.0 toolkit with `nvcc`, `g++` | `python3.12 --version`, `nvcc --version`, `g++ --version` |

The GPU must be free: the server reserves 90% of its memory.

## 2. Choose a stack

| if you... | run | what you get |
|---|---|---|
| serve one user at a time, or a few, and care about how fast each answer streams | **c1** (`serve_c1.sh`) | about 408.9 tokens/s per request (vs 196.5 with the best stock vLLM settings) |
| serve many users at once and care about total throughput | **c32** (`serve_c32.sh`) | about 2,730.9 tokens/s in total at 32 parallel requests (vs 1,757 with the best stock vLLM settings) |

Both stacks accept any number of requests, but each is tuned for its load. Run one stack per GPU.

## 3a. Install and run with Docker

You do not need CUDA or Python on the host, only Docker with GPU support.

```bash
git clone https://github.com/atdev-company/atdev-serve
cd atdev-serve
docker compose --profile c1 up --build          # or: --profile c32
```

* The **first** run builds the image (it downloads about 10 GB of Python packages), downloads the
  models into a Docker volume, and compiles the CUDA kernels. Expect it to take
  10 minutes or more. Later runs start in a few minutes.
* The server is ready when the log shows `Application startup complete`.
* To use a GPU other than 0: `ATDEV_GPU=1 docker compose --profile c1 up`. To use another port:
  `ATDEV_PORT=9000 ...`.
* Stop with `Ctrl+C` or `docker compose --profile c1 down`. The volumes (`hf-cache`, `torch-ext`)
  keep the models and compiled kernels for next time.

Without compose:

```bash
docker build -f docker/Dockerfile -t atdev-serve:0.1.0 .
docker run --gpus '"device=0"' --ipc=host -p 8100:8100 \
  -v atdev-hf:/root/.cache/huggingface -v atdev-ext:/root/.cache/torch_extensions \
  atdev-serve:0.1.0 c1
docker run --rm --gpus '"device=0"' atdev-serve:0.1.0 check    # versions, GPU and nvcc inside the image
```

## 3b. Install and run with a Python virtualenv

### Step 1: get the code

```bash
git clone https://github.com/atdev-company/atdev-serve
cd atdev-serve
```

### Step 2: run the installer

```bash
./install.sh --dry-run     # optional: shows the checks and the commands, changes nothing
./install.sh               # checks the machine, then creates .venv and installs everything
```

The installer:

1. Checks Linux, Python 3.12 (or `uv`), the GPU and driver, `nvcc` 13.0, `g++`, free disk and
   Hugging Face access. If something is missing it stops and tells you how to fix it, before
   installing anything.
2. Creates `.venv` and installs `vllm==0.31.0`, then this package.
3. Checks that vLLM can find the plugin.
4. Writes `atdev-env.sh`, which activates the venv and sets `CUDA_HOME`.

Useful options: `--cuda-home /usr/local/cuda-13.0` if `nvcc` is not found automatically,
`--venv DIR` to install elsewhere, `--editable` for development, and `--warmup c1` to compile the
kernels and download the models right away. Run `./install.sh --help` for the full list.

Manual install, if you prefer:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install vllm==0.31.0
pip install ".[bench]"
export CUDA_HOME=/usr/local/cuda          # the CUDA 13.0 toolkit with bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
```

### Step 3: start the server

```bash
source atdev-env.sh
./serve_c1.sh            # or ./serve_c32.sh
```

* The first start compiles the CUDA kernels (this needs `nvcc`) and downloads the models. Expect it to
  take 10 minutes or more. Wait for `Application startup complete`.
* The log should contain `atdev-serve: enabled optimizations: [...]`. If it does not, the plugin is
  not active (see [Troubleshooting](#6-troubleshooting)).
* Change the port or the model name the API reports with `PORT=9000 SERVED_NAME=qwen ./serve_c1.sh`.
  Extra arguments are passed to `vllm serve`, for example `./serve_c1.sh --api-key mysecret`.

## 4. Send requests

The server speaks the OpenAI API at `http://127.0.0.1:8100/v1`. The model name is `m` unless you set
`SERVED_NAME`.

**curl**

```bash
curl http://127.0.0.1:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "m",
  "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
  "temperature": 0,
  "max_tokens": 256
}'
```

**Python (OpenAI client)**

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8100/v1", api_key="not-needed")

# streaming, so you see tokens as they are generated
stream = client.chat.completions.create(
    model="m",
    messages=[{"role": "user", "content": "Explain speculative decoding in three sentences."}],
    temperature=0,          # greedy: the configuration the speedups were measured with
    max_tokens=512,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
```

**Thinking mode.** Qwen3.8 can produce a reasoning section before the answer. To turn it off, pass
`extra_body={"chat_template_kwargs": {"enable_thinking": False}}` in the Python client, or add
`"chat_template_kwargs": {"enable_thinking": false}` to the JSON body.

**Sampling.** The speedups and accuracy checks were measured with `temperature: 0` (greedy). Requests
with `temperature > 0` still run, but this release has not validated them for speed or quality.

## 5. Check that it is working well

```bash
curl -s http://127.0.0.1:8100/health && echo healthy
curl -s http://127.0.0.1:8100/v1/models
```

Measure speed on your machine with the included benchmark (it starts and stops its own server, so stop
yours first):

```bash
source atdev-env.sh
bench/run_bench.sh c1 results/my_c1       # prints mean / median decode tok/s per request
bench/run_bench.sh c32 results/my_c32     # prints aggregate tok/s
```

You should get numbers close to the README (within a few percent on an A100 80GB SXM).

## 6. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| The first start sits for a long time after loading weights | CUDA kernels are compiling (normal on the first start) | Wait. Watch the CPU with `top` (`nvcc` / `cicc` processes). Later starts reuse `~/.cache/torch_extensions` |
| `nvcc: not found`, `CUDA_HOME environment variable is not set`, or `No CUDA runtime is found` during the build | no CUDA toolkit on `PATH` | Install the CUDA 13.0 toolkit (toolkit only, no driver), then `export CUDA_HOME=/usr/local/cuda-13.0; export PATH=$CUDA_HOME/bin:$PATH`, or re-run `./install.sh --cuda-home ...`. No toolkit and no root access: `./install.sh --pip-nvcc` (experimental: NVIDIA's CUDA 13.0 nvcc wheels inside the venv). Or use Docker |
| Build error mentioning `g++` / `cc1plus` | no host compiler | `sudo apt install build-essential` |
| `unsupported GNU version` from nvcc | host g++ newer than nvcc 13.0 supports | install a supported g++ (for example g++-13) and `export CC=gcc-13 CXX=g++-13` |
| Build hangs at `Waiting for lock` | an earlier start was killed during compilation and left a lock file | stop all servers, then delete the `lock` files: `find ~/.cache/torch_extensions -name lock -delete` |
| `PermissionError` writing `.../atdev_serve/csrc/.../gen...` | the package is installed in a read-only location | install into a virtualenv you own (the installer does this) |
| `No space left on device` | models or kernel cache filled the disk | free space, or point the caches elsewhere: `export HF_HOME=/big/disk/hf TORCH_EXTENSIONS_DIR=/big/disk/torch_ext` |
| `CUDA out of memory` at start | another process is using the GPU | `nvidia-smi` and stop it. The server needs about 90% of the 80 GB |
| `unknown ATDEV_OPT entries` | typo in `ATDEV_OPT` | use the names in the launch scripts |
| No `enabled optimizations` line in the log | the plugin is not installed in the Python that runs `vllm` | `which vllm` should point into your venv. Run `source atdev-env.sh` |
| `401` / `403` / `RepositoryNotFound` from Hugging Face | no access to a model repo | `huggingface-cli login` (or set `HF_TOKEN`) and check the repo names in `serve_c1.sh` |
| Offline machine | models cannot download | download them elsewhere and pass local paths: `MODEL=/models/Qwen3.8-27B-INT4 W3_OVERLAY=/models/w3 DRAFTER=/models/d16 ./serve_c1.sh` |
| Docker: `could not select device driver "nvidia"` | NVIDIA Container Toolkit missing | install it and restart Docker |
| Slower than expected | different GPU, another job on the GPU, or measurement with `temperature > 0` | run `bench/run_bench.sh` on an idle A100 80GB |

## 7. Settings you can change

The launch scripts read these environment variables:

| variable | default | meaning |
|---|---|---|
| `PORT` | 8100 | HTTP port |
| `SERVED_NAME` | `m` | the model name the API reports |
| `CUDA_VISIBLE_DEVICES` | 0 | which GPU to use |
| `MODEL` | `RedHatAI/Qwen3.8-27B-INT4` | target checkpoint (HF id or local path) |
| `W3_OVERLAY` (c1) | `atdev-company/Qwen3.8-27B-atdev-W3-overlay` | 3-bit overlay (HF id or local `.safetensors` paths) |
| `DRAFTER` (c1) | `atdev-company/Qwen3.8-27B-atdev-DFlash2-d16` | drafter (HF id or local dir) |
| `HF_HOME` | `~/.cache/huggingface` | model download cache |
| `TORCH_EXTENSIONS_DIR` | `~/.cache/torch_extensions` | compiled kernel cache |

The context length is 8192 tokens (`--max-model-len 8192`), which is the setting the stacks were tuned
and measured with. You can pass a different value as an extra argument, but it has not been
benchmarked.
