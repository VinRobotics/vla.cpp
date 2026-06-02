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

# Run the ALOHA ROS2 inference node against a running vla-server.
#
# Run setup_aloha_env.sh once first, then:
#   bash eval/run_aloha_client.sh
#   bash eval/run_aloha_client.sh --task "pick up the cup and put it on the plate"
#   bash eval/run_aloha_client.sh --dual-arm --executed-length 16
#   bash eval/run_aloha_client.sh --async-infer                             # fully hidden prefetch
#   bash eval/run_aloha_client.sh --async-infer --async-trigger-delay-ms 300  # fresher obs snapshot

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# VENV="$REPO/eval/aloha_uv/.venv"

# if [[ ! -f "$VENV/bin/activate" ]]; then
#     echo "ERROR: venv not found at $VENV" >&2
#     echo "       Run:  bash eval/setup_aloha_env.sh" >&2
#     exit 1
# fi
# source "$VENV/bin/activate"

STATS="$REPO/weights/vrfai/gr00t-n1d6-aloha-leftarm-gguf/statistics.json"
VLA_ADDR="${VLA_ADDR:-tcp://localhost:5555}"

if [[ ! -f "$STATS" ]]; then
    echo "ERROR: statistics.json not found: $STATS" >&2
    exit 1
fi

echo "Starting ALOHA inference node"
echo "  arch:         gr00t_n1_6"
echo "  embodiment:   new_embodiment"
echo "  stats-json:   $STATS"
echo "  vla-addr:     $VLA_ADDR"
echo ""

python "$REPO/eval/client/run_ALOHA_client_direct.py" \
    --arch          gr00t_n1_6 \
    --embodiment    new_embodiment \
    --stats-json    "$STATS" \
    --vla-addr      "$VLA_ADDR" \
    --image-size    252 \
    --n-action-steps  16 \
    --executed-length  8 \
    --smooth-step     20 \
    "$@"
