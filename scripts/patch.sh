#!/usr/bin/env bash
# Non-interactive patch shortcut. Delegates to the Python CLI so the logic
# stays in a single place.

set -euo pipefail

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -P "$SCRIPT_DIR/.." && pwd)"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "✖ No active virtualenv — run 'source venv/bin/activate' first." >&2
  exit 1
fi

cd "$ROOT"
exec "${VIRTUAL_ENV}/bin/python" -m ultralytics_xpu patch
