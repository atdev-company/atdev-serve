#!/usr/bin/env bash
# Explicit no-W3 accuracy preset; serve_c1.sh remains fast by default.
set -euo pipefail
export C1_MODE=accuracy
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/serve_c1.sh" "$@"
