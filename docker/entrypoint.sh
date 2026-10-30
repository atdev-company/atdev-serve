#!/usr/bin/env bash
# Container entrypoint.  c1 | c32 [extra vllm serve args]  -> launch that stack
#                        check                              -> print versions / GPU / nvcc and exit
#                        anything else                      -> run it as a command (e.g. bash)
set -euo pipefail
cd /opt/atdev-serve
stack=${1:-c1}
[ $# -gt 0 ] && shift
case $stack in
  c1)  exec ./serve_c1.sh "$@" ;;
  c32) exec ./serve_c32.sh "$@" ;;
  check)
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || echo "no GPU visible"
    nvcc --version | tail -1
    exec python -c "import torch, vllm, atdev_serve; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda available', torch.cuda.is_available())" ;;
  *) exec "$stack" "$@" ;;
esac
