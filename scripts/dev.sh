#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# A complete model environment takes precedence over the preview-only environment.
if [[ -x .venv-model/bin/python ]]; then
  python_bin=.venv-model/bin/python
elif [[ -x .venv/bin/python ]]; then
  python_bin=.venv/bin/python
else
  echo 'Install the Python environment first; see README.md.' >&2
  exit 1
fi
"$python_bin" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 &
api_pid=$!
cleanup() { kill "$api_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
npm run dev -- --host 127.0.0.1
