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

# Outer driver for the vla.cpp vs torch.compile latency matrix: deals the six
# models across the available GPUs and runs all four variants for each.
#
# One server at a time per GPU, each on its own port (5700 + gpu), so the two
# lanes never contend for VRAM or for the ZMQ port.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

N_STEPS="${N_STEPS:-200}"
VARIANTS="${VARIANTS:-vla.cpp,eager,compile-default,compile-reduce-overhead}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/latency_compare}"

# Dealt so each lane carries a comparable total model size (GGUF bytes):
#   lane 0: smolvla 1.1G + evo1 1.6G + gr00t_n1_6 9.2G  = 11.9G
#   lane 1: pi0 6.5G + gr00t_n1_5 6.9G + gr00t_n1_7 6.3G = 19.7G
# Lane 0 also carries the two cheapest models, so it finishes its first two
# entries quickly and starts the 9.2G one while lane 1 is still on pi0.
LANE0="${LANE0:-smolvla evo1 gr00t_n1_6}"
LANE1="${LANE1:-pi0 gr00t_n1_5 gr00t_n1_7}"

mkdir -p "${OUTPUT_ROOT}"
DRIVER_LOG="${OUTPUT_ROOT}/driver.log"
: > "${DRIVER_LOG}"

run_lane() {
    local gpu="$1"; shift
    local models="$1"
    for m in ${models}; do
        echo "[lane${gpu}] === ${m} start $(date +%H:%M:%S) ===" | tee -a "${DRIVER_LOG}"
        bash "${SCRIPT_DIR}/run_latency_compare.sh" \
            -m "${m}" -g "${gpu}" -n "${N_STEPS}" \
            -v "${VARIANTS}" -o "${OUTPUT_ROOT}" \
            >"${OUTPUT_ROOT}/_${m}.driver.log" 2>&1
        echo "[lane${gpu}] === ${m} done rc=$? $(date +%H:%M:%S) ===" | tee -a "${DRIVER_LOG}"
    done
}

run_lane 0 "${LANE0}" &
P0=$!
run_lane 1 "${LANE1}" &
P1=$!

wait "${P0}"; wait "${P1}"
echo "ALL DONE $(date +%H:%M:%S)" | tee -a "${DRIVER_LOG}"
