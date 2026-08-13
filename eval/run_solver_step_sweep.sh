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

# Solver-step (T) sweep for GR00T-N1.7 on libero_object, both stacks.
#
# The flow-matching head integrates over T solver steps (checkpoint default
# T=4). This sweep varies ONLY T and reports vla.cpp against the PyTorch
# reference at each value, so the comparison is a parity measurement rather
# than a capability one.
#
# One environment variable sets T on both sides:
#   - vla.cpp : VLA_NUM_STEPS, read in src/models/gr00tn1d7.cpp at load
#   - PyTorch : VLA_NUM_STEPS, read in eval/pytorch_ref/policies/gr00t_n17/
#               __init__.py, which assigns action_head.num_inference_timesteps
#               (assigning to config there is inert -- the denoise loop reads
#               the instance attribute)
#
# Everything else is held fixed: same LIBERO env, seed 42, 500-step cap, and
# n_action_steps=16 on BOTH stacks, so chunk-replay cadence cannot leak into
# the SR delta.
#
# Phases run in ORDER (PyTorch fully completes before vla.cpp), matching
# run_libero_compare.sh. Running them concurrently on separate GPUs looks
# tempting -- the servers do not share a device -- but the LIBERO simulator is
# CPU-bound and two clients contend, which inflates the client-side ms/step
# both stacks report. Measured: GPUs sat at 42%/0% while wall-clock per task
# roughly doubled. Serial is both cleaner and, here, faster.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-n <N_EPISODES>] [-T <STEPS>] [-o <OUTPUT_ROOT>]
                        [-i <GGUF_ROOT>] [-c <CKPT_ROOT>] [-P] [-C]

  -n N_EPISODES   episodes per task-id (default: 10 => 100 per cell)
  -T STEPS        space-separated solver-step values (default: "1 2 4 8 16")
  -o OUTPUT_ROOT  results root (default: ${REPO_ROOT}/outputs/solver_sweep)
  -i GGUF_ROOT    root of the vrfai GGUF dirs (default: /mnt/data/hf_data/vrfai)
  -c CKPT_ROOT    root of the PyTorch checkpoints (default: /mnt/data/hf_data)
  -g GPU_CPP      CUDA device for the vla.cpp phase (default: 0)
  -G GPU_PT       CUDA device for the PyTorch phase (default: 1)
  -P              PyTorch phase only
  -C              vla.cpp phase only
  -h              show this help
EOF
}

N_EPISODES="10"
STEPS="1 2 4 8 16"
OUTPUT_ROOT=""
GGUF_ROOT="/mnt/data/hf_data/vrfai"
CKPT_ROOT="/mnt/data/hf_data"
GPU_CPP="0"
GPU_PT="1"
RUN_PT=1
RUN_CPP=1

while getopts ":n:T:o:i:c:g:G:PCh" opt; do
    case "${opt}" in
        n) N_EPISODES="${OPTARG}" ;;
        T) STEPS="${OPTARG}" ;;
        o) OUTPUT_ROOT="${OPTARG}" ;;
        i) GGUF_ROOT="${OPTARG}" ;;
        c) CKPT_ROOT="${OPTARG}" ;;
        g) GPU_CPP="${OPTARG}" ;;
        G) GPU_PT="${OPTARG}" ;;
        P) RUN_CPP=0 ;;
        C) RUN_PT=0 ;;
        h) usage; exit 0 ;;
        \?) echo "ERROR: unknown option -${OPTARG}" >&2; usage >&2; exit 1 ;;
        :)  echo "ERROR: option -${OPTARG} requires an argument" >&2; usage >&2; exit 1 ;;
    esac
done

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/solver_sweep}"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"
LOG_DIR="${OUTPUT_ROOT}/_driver_logs"
mkdir -p "${LOG_DIR}"

echo "[config] STEPS=${STEPS}  N_EPISODES=${N_EPISODES}"
echo "[config] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[config] GPU_CPP=${GPU_CPP}  GPU_PT=${GPU_PT}"

# The vla.cpp phase needs the server built once, before any concurrent run.
if (( RUN_CPP )); then
    echo "[build] cmake --build build"
    cmake --build "${REPO_ROOT}/build" -j"$(nproc)" || {
        echo "ERROR: vla-server build failed" >&2; exit 1; }
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# ---------------------------------------------------------------------------
# vla.cpp phase: one T at a time on GPU_CPP.
# ---------------------------------------------------------------------------
sweep_cpp() {
    for t in ${STEPS}; do
        local out="${OUTPUT_ROOT}/T${t}/vla_cpp"
        local log="${LOG_DIR}/vla_cpp.T${t}.log"
        mkdir -p "${out}"
        echo "[vla_cpp] T=${t} -> ${log}"
        SKIP_BUILD=1 \
        VLA_NUM_STEPS="${t}" \
        CUDA_VISIBLE_DEVICES="${GPU_CPP}" \
        BIND_ADDR="tcp://*:5620" \
        CLIENT_ADDR="tcp://localhost:5620" \
        bash "${SCRIPT_DIR}/run_libero.sh" \
            -m gr00t_n1_7 -i "${GGUF_ROOT}" -o "${out}" -n "${N_EPISODES}" \
            >"${log}" 2>&1
        if [[ $? -eq 0 ]]; then echo "[vla_cpp] T=${t}: OK"; else
            echo "[vla_cpp] T=${t}: FAILED (see ${log})" >&2; fi
    done
}

# ---------------------------------------------------------------------------
# PyTorch phase: one T at a time on GPU_PT.
# ---------------------------------------------------------------------------
sweep_pt() {
    for t in ${STEPS}; do
        local out="${OUTPUT_ROOT}/T${t}/pytorch"
        local log="${LOG_DIR}/pytorch.T${t}.log"
        mkdir -p "${out}"
        echo "[pytorch] T=${t} -> ${log}"
        VLA_NUM_STEPS="${t}" \
        bash "${SCRIPT_DIR}/run_libero_pytorch.sh" \
            -m gr00t_n1_7 -g "${GPU_PT}" -p 5621 \
            -i "${CKPT_ROOT}" -o "${out}" -n "${N_EPISODES}" \
            >"${log}" 2>&1
        if [[ $? -eq 0 ]]; then echo "[pytorch] T=${t}: OK"; else
            echo "[pytorch] T=${t}: FAILED (see ${log})" >&2; fi
    done
}

if (( RUN_PT ));  then echo "=== PHASE 1/2 - PyTorch reference ==="; sweep_pt;  fi
if (( RUN_CPP )); then echo "=== PHASE 2/2 - vla.cpp GGUF ===";      sweep_cpp; fi

echo "===================================================="
echo "Done. Results under ${OUTPUT_ROOT}"
echo "Aggregate with: python eval/collect_solver_sweep.py -o ${OUTPUT_ROOT}"
