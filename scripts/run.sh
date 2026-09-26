#!/usr/bin/env bash
set -euo pipefail
lightdelta_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${lightdelta_root}/scripts/activate.sh"
cd "${lightdelta_root}"
exec "$@"
