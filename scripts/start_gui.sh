#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
exec megax-gui --host "${MEGAX_GUI_HOST:-0.0.0.0}" --port "${MEGAX_GUI_PORT:-18555}" "$@"
