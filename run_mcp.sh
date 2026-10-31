#!/usr/bin/env bash
# Launch the sys12 MCP server (stdio). Cursor starts MCP with a stripped
# environment, so recover HOME before `set -u`.
set -euo pipefail
if [ -z "${HOME:-}" ]; then
  HOME="$(eval echo ~"$(id -un)")"
  export HOME
fi
ROOT="$(cd "$(dirname "$0")" && pwd)"
export FASTMCP_SHOW_SERVER_BANNER=false
export PYTHONUNBUFFERED=1

if [ ! -x "$ROOT/venv/bin/python" ]; then
  echo "sys12: missing $ROOT/venv. Run: python3 -m venv venv && ./venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

exec "$ROOT/venv/bin/python" "$ROOT/sys12_mcp.py"
