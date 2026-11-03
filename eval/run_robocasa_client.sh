#!/usr/bin/env bash
# Copyright 2026 VinRobotics - Apache-2.0
# Forward the documented Python options; this client never needs GGUF weights.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROBOCASA_PYTHON:-${SCRIPT_DIR}/sim/robocasa/robocasa_uv/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: RoboCasa Python not found: $PYTHON" >&2
    echo "Run: bash eval/sim/robocasa/setup_robocasa.sh" >&2
    exit 1
fi
exec "$PYTHON" "$SCRIPT_DIR/client/run_robocasa_client_direct.py" "$@"
