#!/usr/bin/env bash
# Container entrypoint.  c1m | c1s | c1s-accuracy | c32m | c32s [extra vllm serve args]  -> launch that profile (./serve.sh <name>)
#                        c1 | c1-single | c1-accuracy | c32 | c32-single                   -> old names, same launch as c1m | c1s | c1s-accuracy | c32m | c32s
#                        check                                                            -> print versions / GPU / nvcc and exit
#                        anything else                                                    -> run it as a command (e.g. bash)
set -euo pipefail
cd "${ATDEV_HOME:-/opt/atdev-serve}"
stack=${1:-c1m}
[ $# -gt 0 ] && shift
case $stack in
  c1m|c1)                exec ./serve.sh c1m "$@" ;;
  c1s|c1-single)         exec ./serve.sh c1s "$@" ;;
  c1s-accuracy|c1-accuracy) exec ./serve_c1_accuracy.sh "$@" ;;
  c32m|c32)              exec ./serve.sh c32m "$@" ;;
  c32s|c32-single)       exec ./serve.sh c32s "$@" ;;
  check)
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || echo "no GPU visible"
    nvcc --version | grep -m1 'release' || echo "nvcc not found"
    exec python -c "import torch, vllm, atdev_serve; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda available', torch.cuda.is_available())" ;;
  *) exec "$stack" "$@" ;;
esac
