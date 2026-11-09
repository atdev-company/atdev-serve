#!/usr/bin/env bash
# atdev-serve launcher: ./serve.sh <c1m|c1s|c32m|c32s> [extra vllm serve args...]
#
#   c1m   1 client, multi-turn   -> ./serve_c1.sh  --profile multiturn
#   c1s   1 client, single-turn  -> ./serve_c1.sh  --profile single
#   c32m  32 clients, multi-turn -> ./serve_c32.sh --profile multiturn
#   c32s  32 clients, single-turn-> ./serve_c32.sh --profile single
#
# c = concurrent clients, the number is what the profile is tuned for; m = multi-turn (prefix caching), s = single-turn.
# This script only picks the existing launcher and its --profile; every environment variable (PORT, MODEL, C1_MODE, HADW, NO_WARMUP,
# ATDEV_EOR_CKPT, ATDEV_PF_A4, ...) and every extra argument is passed through unchanged.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
name=${1:-}
[[ $# -gt 0 ]] && shift
case "$name" in
  c1m)  exec "$here/serve_c1.sh"  --profile multiturn "$@" ;;
  c1s)  exec "$here/serve_c1.sh"  --profile single "$@" ;;
  c32m) exec "$here/serve_c32.sh" --profile multiturn "$@" ;;
  c32s) exec "$here/serve_c32.sh" --profile single "$@" ;;
  *) echo "usage: ./serve.sh <c1m|c1s|c32m|c32s> [extra vllm serve args...]" >&2
     echo "  c1m  1 client, multi-turn    c1s  1 client, single-turn" >&2
     echo "  c32m 32 clients, multi-turn  c32s 32 clients, single-turn" >&2
     exit 2 ;;
esac
