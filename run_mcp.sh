#!/bin/bash
# sys12 MCP server (stdio) for OpenCode.
set -euo pipefail
cd "$(dirname "$0")"

# API key: environment first, then the oMLX App Support env file.
if [[ -z "${OMLX_API_KEY:-}" ]]; then
  ENV_FILE="${HOME}/Library/Application Support/omlx/.env"
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
  fi
fi

exec ./.venv/bin/python sys12_mcp.py
