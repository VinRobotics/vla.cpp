#!/usr/bin/env bash
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# run_libero_wsl.sh - one-command LIBERO eval for the full serving stack on WSL2.
#
# Starts vla-server (CUDA) and drives run_sim_client_direct.py for a chosen
# task-id / episode count, both co-resident in one WSL distro, then stops the
# server. See docs/backend/wsl.md for the WSL2 + CUDA setup this assumes
# (built vla-server, installed LIBERO venv, user in the `render` group).
#
# Usage:
#   bash eval/run_libero_wsl.sh              # prompts for task-id + episodes
#   bash eval/run_libero_wsl.sh 3 5          # task-id 3, 5 episodes (no prompt)
#   TASK_SUITE=libero_spatial bash eval/run_libero_wsl.sh 0 1
#
# Overridable via environment:
#   VLA_GGUF     path to the served GGUF
#                (default: $HOME/data/vrfai/smolvla-libero-gguf/smolvla-libero.gguf)
#   VLA_ARCH     arch preset for the client (default: smolvla)
#   TASK_SUITE   LIBERO suite (default: libero_object)
#   N_ACTION_STEPS  actions replayed per predicted chunk (default: 1)
#   CUDA_HOME    CUDA toolkit dir (default: /usr/local/cuda-12.6)
#   OUTPUT_DIR   where videos/summaries land (default: $HOME/outputs/wsl_smoke)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

VLA_ARCH="${VLA_ARCH:-smolvla}"
TASK_SUITE="${TASK_SUITE:-libero_object}"
N_ACTION_STEPS="${N_ACTION_STEPS:-1}"
VLA_GGUF="${VLA_GGUF:-$HOME/data/vrfai/smolvla-libero-gguf/smolvla-libero.gguf}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/outputs/wsl_smoke}"

SERVER_BIN="$REPO_ROOT/build/vla-server"
VENV_PY="$REPO_ROOT/eval/sim/libero/libero_uv/.venv/bin/python"
CLIENT="$REPO_ROOT/eval/client/run_sim_client_direct.py"
BIND_ADDR="${BIND_ADDR:-tcp://*:5555}"
CLIENT_ADDR="${CLIENT_ADDR:-tcp://localhost:5555}"

# --- pick task-id + episode count: CLI args first, else prompt, else default ---
TASK_ID="${1:-}"
N_EPISODES="${2:-}"

ask() {  # ask <var> <prompt> <default>; only prompts on an interactive terminal
    local __var="$1" __prompt="$2" __def="$3" __ans=""
    if [ -z "${!__var}" ]; then
        if [ -t 0 ]; then
            read -r -p "$__prompt [$__def]: " __ans
        fi
        printf -v "$__var" '%s' "${__ans:-$__def}"
    fi
}

ask TASK_ID   "Task id (0-9) in suite '$TASK_SUITE'" 0
ask N_EPISODES "Number of episodes to run"           1

# --- validate numeric input ---
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: task-id must be a non-negative integer (got '$TASK_ID')" >&2; exit 1
fi
if ! [[ "$N_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: episodes must be a positive integer (got '$N_EPISODES')" >&2; exit 1
fi

# --- preflight ---
[ -x "$SERVER_BIN" ] || { echo "ERROR: vla-server not built at $SERVER_BIN (see docs/backend/wsl.md)" >&2; exit 1; }
[ -x "$VENV_PY" ]    || { echo "ERROR: LIBERO venv missing; run: bash eval/sim/libero/setup_libero.sh" >&2; exit 1; }
[ -f "$VLA_GGUF" ]   || { echo "ERROR: GGUF not found at $VLA_GGUF (set VLA_GGUF=...)" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
SERVER_LOG="$OUTPUT_DIR/vla-server.log"

echo "[config] arch=$VLA_ARCH suite=$TASK_SUITE task-id=$TASK_ID episodes=$N_EPISODES"
echo "[config] gguf=$VLA_GGUF"
echo "[config] output=$OUTPUT_DIR"

# --- start server, stop it on any exit ---
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[cleanup] stopping vla-server (pid=$SERVER_PID)"
        kill -INT "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 10); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.5; done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[server] starting (CUDA)..."
"$SERVER_BIN" --bind "$BIND_ADDR" "$VLA_GGUF" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 180); do
    if grep -q 'ready' "$SERVER_LOG" 2>/dev/null; then echo "[server] ready"; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: vla-server exited before ready; see $SERVER_LOG" >&2; tail -n 30 "$SERVER_LOG" >&2; exit 1
    fi
    sleep 1
done
grep -iE 'backend =|CUDA \(device' "$SERVER_LOG" | head -1 || true

# --- run the sim client (EGL/GPU render path) ---
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 VLA_ARCH="$VLA_ARCH"
echo "[client] running $N_EPISODES episode(s) on $TASK_SUITE/task_$TASK_ID..."
"$VENV_PY" "$CLIENT" \
    --arch "$VLA_ARCH" \
    --vla-addr "$CLIENT_ADDR" \
    --task "$TASK_SUITE" --task-id "$TASK_ID" \
    --n-episodes "$N_EPISODES" --n-action-steps "$N_ACTION_STEPS" \
    --output-dir "$OUTPUT_DIR"

echo "[done] outputs under $OUTPUT_DIR/$VLA_ARCH/$TASK_SUITE/task_$TASK_ID/"
