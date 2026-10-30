#!/usr/bin/env bash
# atdev-serve installer: preflight checks -> Python 3.12 venv -> vllm==0.31.0 -> atdev-serve -> verification.
#
# usage:  ./install.sh [options]
#   --venv DIR        virtualenv location (default: ./.venv)
#   --python BIN      Python 3.12 interpreter (default: python3.12; uv-managed Python if missing and uv exists)
#   --editable        pip install -e . (default: regular install of this checkout)
#   --no-bench        skip the benchmark extra (aiohttp)
#   --cuda-home DIR   CUDA 13.0 toolkit with bin/nvcc (default: $CUDA_HOME, else /usr/local/cuda-13.0, /usr/local/cuda)
#   --pip-nvcc        EXPERIMENTAL: no system toolkit -> use NVIDIA's CUDA 13.0 nvcc wheels inside the venv
#                     (pinned nvidia-cuda-nvcc / -crt / nvvm 13.0.*; a JIT build + link was tested, a full
#                     server start with it was not)
#   --skip-gpu-check  do not require nvidia-smi / an A100 (build-only machines, CI)
#   --warmup c1|c32   after installing, start that stack once (JIT-compiles the kernels and downloads the
#                     models), wait until /health answers, then stop it. Needs the GPU. Can take a long time.
#   --dry-run         run the read-only checks and print every install command without executing it
#   -h, --help
#
# Writes ./atdev-env.sh (source it to get the venv + CUDA_HOME + PATH) and prints the next steps.
# Exit codes: 0 ok, 2 bad usage, 3 preflight failed, 4 install failed, 5 verification failed, 6 warmup failed.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV=$HERE/.venv
PY_BIN=${PYTHON:-python3.12}
EDITABLE=0; BENCH=1; DRY=0; CUDA_EXPLICIT=0; SKIP_GPU=0; PIP_NVCC=0; WARMUP=""
CUDA_DIR=${CUDA_HOME:-}
VLLM_VERSION=0.31.0
MIN_DISK_GB=80

while [ $# -gt 0 ]; do
  case $1 in
    --venv) VENV=$(realpath -m "$2"); shift 2 ;;
    --python) PY_BIN=$2; shift 2 ;;
    --editable) EDITABLE=1; shift ;;
    --no-bench) BENCH=0; shift ;;
    --cuda-home) CUDA_DIR=$2; CUDA_EXPLICIT=1; shift 2 ;;
    --pip-nvcc) PIP_NVCC=1; shift ;;
    --skip-gpu-check) SKIP_GPU=1; shift ;;
    --warmup) WARMUP=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
case $WARMUP in ""|c1|c32) ;; *) echo "--warmup must be c1 or c32" >&2; exit 2 ;; esac

ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; FAILS=$((FAILS + 1)); }
run()  { if [ $DRY = 1 ]; then printf '  + %s\n' "$*"; else "$@"; fi; }
FAILS=0

echo "== atdev-serve installer ($( [ $DRY = 1 ] && echo dry run || echo live ))"
echo "   repo: $HERE"
echo "   venv: $VENV"

# ---------------------------------------------------------------- preflight (read-only)
echo "== preflight"
[ "$(uname -s)" = Linux ] && ok "Linux $(uname -r)" || bad "Linux required (found $(uname -s))"
[ -f "$HERE/pyproject.toml" ] && [ -d "$HERE/src/atdev_serve" ] && ok "repo checkout found" \
  || bad "run install.sh from the atdev-serve checkout (pyproject.toml / src/atdev_serve missing)"

# Python 3.12
USE_UV=0
if command -v "$PY_BIN" >/dev/null 2>&1 && "$PY_BIN" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' 2>/dev/null; then
  ok "Python $("$PY_BIN" -c 'import platform; print(platform.python_version())') ($(command -v "$PY_BIN"))"
elif command -v uv >/dev/null 2>&1; then
  USE_UV=1; ok "no $PY_BIN on PATH; uv found -> uv will provide Python 3.12"
else
  bad "Python 3.12 not found. Install python3.12 + python3.12-venv (apt: sudo apt install python3.12 python3.12-venv python3.12-dev) or uv (https://docs.astral.sh/uv/)"
fi
if [ $USE_UV = 0 ] && command -v "$PY_BIN" >/dev/null 2>&1 && ! "$PY_BIN" -c 'import venv, ensurepip' 2>/dev/null; then
  if command -v uv >/dev/null 2>&1; then USE_UV=1; warn "python venv/ensurepip missing -> using uv"
  else bad "python3.12 venv module missing (apt: sudo apt install python3.12-venv)"; fi
fi

# GPU
if [ $SKIP_GPU = 1 ]; then
  warn "GPU check skipped (--skip-gpu-check)"
elif command -v nvidia-smi >/dev/null 2>&1; then
  GPUS=$(nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true)
  if [ -z "$GPUS" ]; then bad "nvidia-smi found but no GPU visible"
  else
    while IFS= read -r g; do echo "         GPU $g"; done <<< "$GPUS"
    if grep -q 'A100.*80' <<< "$GPUS"; then ok "A100 80GB present"
    else warn "no A100 80GB found: the kernels are built and tuned for sm_80 / A100 80GB only"; fi
    DRV=$(head -1 <<< "$GPUS" | awk -F', ' '{print $4}')
    DRV_MAJOR=${DRV%%.*}
    if [ -n "$DRV_MAJOR" ] && [ "$DRV_MAJOR" -ge 580 ] 2>/dev/null; then ok "driver $DRV (CUDA 13.0 needs >= 580)"
    else warn "driver $DRV: CUDA 13.0 (torch cu130) needs driver >= 580"; fi
  fi
else
  bad "nvidia-smi not found (NVIDIA driver missing). Use --skip-gpu-check on a build-only machine"
fi

# CUDA toolkit / nvcc
find_cuda() {
  if [ $CUDA_EXPLICIT = 1 ]; then [ -x "$CUDA_DIR/bin/nvcc" ] && echo "$CUDA_DIR"; return; fi
  for d in "$CUDA_DIR" /usr/local/cuda-13.0 /usr/local/cuda "$(dirname "$(dirname "$(command -v nvcc 2>/dev/null || echo /x/x/x)")")"; do
    [ -n "$d" ] && [ -x "$d/bin/nvcc" ] && { echo "$d"; return; }
  done
}
CUDA_FOUND=$(find_cuda || true)
if [ -n "$CUDA_FOUND" ]; then
  NVV=$("$CUDA_FOUND/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
  if [ "$NVV" = 13.0 ]; then ok "nvcc $NVV at $CUDA_FOUND"
  else warn "nvcc $NVV at $CUDA_FOUND: CUDA 13.0 is the tested toolkit (torch 2.13.0+cu130)"; fi
  CUDA_DIR=$CUDA_FOUND
elif [ $PIP_NVCC = 1 ]; then
  warn "no system nvcc; --pip-nvcc: will use NVIDIA's CUDA 13.0 nvcc wheels inside the venv (EXPERIMENTAL)"
else
  bad "CUDA 13.0 toolkit with nvcc not found (set --cuda-home or CUDA_HOME). Install it from
         https://developer.nvidia.com/cuda-13-0-0-download-archive (toolkit only, no driver), use the
         Docker image instead, or try --pip-nvcc (experimental)"
fi
command -v g++ >/dev/null 2>&1 && ok "g++ $(g++ -dumpversion)" \
  || bad "g++ not found (apt: sudo apt install build-essential)"
command -v git >/dev/null 2>&1 || warn "git not found (only needed to clone/update the repo)"

# disk
for d in "$(dirname "$VENV")" "${HF_HOME:-$HOME/.cache/huggingface}"; do
  p=$d; while [ ! -d "$p" ]; do p=$(dirname "$p"); done
  free=$(df -Pk "$p" | awk 'NR==2 {print int($4 / 1048576)}')
  if [ "$free" -ge $MIN_DISK_GB ]; then ok "disk: ${free} GB free at $p"
  else warn "disk: only ${free} GB free at $p (venv ~9 GB, models ~37 GB, kernel cache ~2 GB)"; fi
done

# HF Hub reachability (models download on first start)
if command -v curl >/dev/null 2>&1; then
  code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' https://huggingface.co/api/models/RedHatAI/Qwen3.8-27B-INT4 || true)
  [ "$code" = 200 ] && ok "Hugging Face Hub reachable" \
    || warn "Hugging Face Hub not reachable (HTTP $code): models must then be present in HF_HOME / given as local paths"
fi

if [ $FAILS -gt 0 ]; then
  echo "== preflight: $FAILS problem(s). Fix them and re-run (nothing was installed)."
  [ $DRY = 1 ] || exit 3
  echo "   (dry run: continuing to show the install commands)"
fi

# ---------------------------------------------------------------- install
echo "== install"
if [ -x "$VENV/bin/python" ]; then
  ok "reusing existing venv $VENV"
elif [ $USE_UV = 1 ]; then
  run uv venv --python 3.12 --seed "$VENV"
else
  run "$PY_BIN" -m venv "$VENV"
fi
VPY=$VENV/bin/python
if [ $USE_UV = 1 ]; then PIP=(uv pip install --python "$VPY"); else PIP=("$VPY" -m pip install); fi
[ $USE_UV = 1 ] || run "$VPY" -m pip install --upgrade pip
run "${PIP[@]}" "vllm==$VLLM_VERSION" || { echo "vllm install failed"; exit 4; }
TARGET="$HERE"; [ $BENCH = 1 ] && TARGET="${HERE}[bench]"
if [ $EDITABLE = 1 ]; then run "${PIP[@]}" -e "$TARGET" || exit 4
else run "${PIP[@]}" "$TARGET" || exit 4; fi
if [ -z "$CUDA_FOUND" ] && [ $PIP_NVCC = 1 ]; then
  # torch 2.13.0+cu130 pulls NVIDIA's cuda-toolkit wheels into site-packages/nvidia/cu13 (headers, cudart),
  # but its nvcc / crt / nvvm wheels can resolve to a newer 13.x whose headers do not match cudart 13.0
  # ("CUDA compiler and CUDA toolkit headers are incompatible"). Pin them to 13.0 (= nvcc V13.0.88, the
  # toolkit the published numbers were built with) and add the unversioned libcudart.so the linker needs.
  run "${PIP[@]}" "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*" || exit 4
  if [ $DRY = 0 ]; then
    NVCC=$(find "$VENV" -path '*/nvidia/cu13/bin/nvcc' -type f | head -1)
    [ -n "$NVCC" ] || { echo "no nvidia/cu13/bin/nvcc in the venv after installing the nvcc wheels"; exit 4; }
    CUDA_DIR=$(dirname "$(dirname "$NVCC")")
    if [ ! -e "$CUDA_DIR/lib/libcudart.so" ] && [ -e "$CUDA_DIR/lib/libcudart.so.13" ]; then
      ln -s libcudart.so.13 "$CUDA_DIR/lib/libcudart.so"
    fi
    warn "using pip nvcc $("$NVCC" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p') at $CUDA_DIR (EXPERIMENTAL)"
  else
    echo "  + ln -s libcudart.so.13 <venv>/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so"
    CUDA_DIR='<venv>/lib/python3.12/site-packages/nvidia/cu13'
  fi
fi

# ---------------------------------------------------------------- env file
ENVF=$HERE/atdev-env.sh
echo "== writing $ENVF"
if [ $DRY = 1 ]; then
  echo "  + (would write) source $VENV/bin/activate; export CUDA_HOME=$CUDA_DIR; PATH=\$CUDA_HOME/bin:\$PATH"
else
  cat > "$ENVF" <<EOF
# generated by install.sh -- usage: source atdev-env.sh
source "$VENV/bin/activate"
export CUDA_HOME="$CUDA_DIR"
export PATH="\$CUDA_HOME/bin:\$PATH"
export TORCH_EXTENSIONS_DIR="\${TORCH_EXTENSIONS_DIR:-\$HOME/.cache/torch_extensions}"
# optional: relocate caches to a disk with space (uncomment and edit)
# export HF_HOME=/data/cache/huggingface             # model downloads
# export TORCH_EXTENSIONS_DIR=/data/cache/torch_extensions
# export VLLM_CACHE_ROOT=/data/cache/vllm            # torch.compile cache
# export TRITON_CACHE_DIR=/data/cache/triton
# export XDG_CACHE_HOME=/data/cache                  # FlashInfer JIT cache uses it
# export PIP_CACHE_DIR=/data/cache/pip
EOF
fi

# ---------------------------------------------------------------- verify
echo "== verify"
VERIFY='
import importlib.metadata as md, os, sys
import vllm, atdev_serve
eps = [e.value for e in md.entry_points(group="vllm.general_plugins") if e.name == "atdev"]
assert vllm.__version__ == "'$VLLM_VERSION'", "vllm " + vllm.__version__
assert eps == ["atdev_serve.register:register_all"], eps
pkg = os.path.dirname(atdev_serve.__file__)
assert os.access(pkg, os.W_OK), "package dir not writable: " + pkg
import torch
print("vllm", vllm.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda, "| plugin ok |", pkg)
'
if [ $DRY = 1 ]; then
  echo "  + $VPY -c '<import vllm + atdev_serve, check entry point and writable package dir>'"
else
  "$VPY" -c "$VERIFY" || { echo "verification failed"; exit 5; }
fi

# ---------------------------------------------------------------- optional warmup
if [ -n "$WARMUP" ]; then
  echo "== warmup: starting serve_$WARMUP.sh once (kernel JIT build + model download)"
  if [ $DRY = 1 ]; then
    echo "  + source $ENVF && PORT=8199 ./serve_$WARMUP.sh  (until /health answers, then stop)"
  else
    LOG=$HERE/warmup_$WARMUP.log
    # shellcheck disable=SC1090
    ( source "$ENVF" && PORT=8199 setsid bash "$HERE/serve_$WARMUP.sh" ) > "$LOG" 2>&1 &
    SP=$!; t0=$(date +%s)
    trap 'kill -TERM -- -$SP 2>/dev/null || true' EXIT
    until curl -sf http://127.0.0.1:8199/health >/dev/null 2>&1; do
      kill -0 $SP 2>/dev/null || { echo "server exited during warmup, see $LOG"; tail -30 "$LOG"; exit 6; }
      [ $(( $(date +%s) - t0 )) -gt 7200 ] && { echo "warmup timed out after 2 h, see $LOG"; exit 6; }
      sleep 10
    done
    echo "  warmup done in $(( ($(date +%s) - t0) / 60 )) min (log: $LOG)"
    kill -TERM -- -$SP 2>/dev/null || true; trap - EXIT
  fi
fi

echo "== done"
cat <<EOF
Next:
  source atdev-env.sh
  ./serve_c1.sh      # single-user stack  (or ./serve_c32.sh for 32 concurrent requests)
  # the first start compiles the CUDA kernels and downloads the models; wait for "Application startup complete"
  curl http://127.0.0.1:8100/v1/models
EOF
