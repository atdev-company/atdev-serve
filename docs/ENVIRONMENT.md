# Tested environment

This is the project's observed development/test environment, inspected on **2026-11-07**.
The released C1 Fast W3, C1 accuracy and C32 Hadamard W4A4 profiles target **one NVIDIA
A100 80GB (SXM, sm_80)** per server. This table records the environment used in this
project; it is not a compatibility matrix for every GPU or package combination.

| Component | Observed version / configuration |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB, 81,920 MiB, compute capability 8.0 (`sm_80`) |
| OS userland | Ubuntu 24.04.3 LTS, x86_64 |
| Linux kernel | 6.8.0-138-generic |
| NVIDIA driver | 580.126.20 |
| Python | 3.12.13 |
| PyTorch | `2.13.0+cu130` (`torch.__version__`) |
| PyTorch CUDA build target | `13.0` (`torch.version.cuda`) |
| CUDA toolkit / compiler | CUDA 13.0, `nvcc V13.0.88` |
| Host C++ compiler | `g++ 13.3.0` |
| vLLM | `0.31.0`, unmodified pip wheel; atdev-serve loads as a plugin |
| Triton | `3.7.1` |
| FlashInfer Python package | `flashinfer-python 0.7.0.post1` |
| Transformers | `5.17.0` |
| NumPy | `2.3.5` |

The installer already selects vLLM 0.31.0 and the matching PyTorch/CUDA stack. The
additional package patch versions above are an observed snapshot, not new installer
pins or minimum requirements. Historical benchmark records do not establish that
every ancillary package patch version stayed identical across all runs.

## CUDA versions: what each number means

- **`torch.version.cuda`** identifies the CUDA version used to build PyTorch. In
  this environment it is 13.0, corresponding to the `+cu130` wheel.
- **`nvcc --version`** identifies the local CUDA toolkit compiler used to JIT-build
  the custom CUDA extensions. The tested compiler is CUDA 13.0 / V13.0.88.
- **The CUDA version shown by `nvidia-smi`** describes driver-supported CUDA
  compatibility; it does not identify the installed toolkit or the PyTorch build.
  Record the driver version separately.

A PyTorch CUDA wheel alone does not provide the complete supported extension-build
setup: the native virtualenv installation also needs the CUDA toolkit, `nvcc`,
headers and a host C++ compiler. The supplied Docker image provides the toolkit and
build dependencies inside the container; the host still needs a compatible NVIDIA
driver. See the [installation guide](../AI_SETUP.md).

## Inspect your installed environment

Run these from the same activated environment that starts the server. They report
versions without launching an inference server:

```bash
python - <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version
import torch

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
for package in ("vllm", "triton", "flashinfer-python", "transformers", "numpy"):
    try:
        print(f"{package}: {version(package)}")
    except PackageNotFoundError:
        print(f"{package}: not installed")
PY
nvcc --version
g++ --version
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
```

Use the same environment for installation, CUDA extension compilation and server
launch. Preserve these outputs and the startup log when reporting a setup or
performance issue.

## Reproducing performance

Matching software and GPU versions is only part of reproducing a measurement.
Also match the selected profile, input/output lengths, concurrency, cache policy,
warm-up and timing definition. The 408.9 tok/s C1 headline measures decode speed,
whereas the 2,688.13 tok/s C32 result is aggregate output across 32 concurrent
clients. Supporting a 262,144-token context does not establish those same speeds
at that context length. See [benchmark methodology](BENCHMARKS.md) and
[profile accuracy results](ACCURACY.md).
