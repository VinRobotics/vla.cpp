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

# Measure inference latency and GPU/host memory of the UPSTREAM PyTorch BitVLA
# policy on LIBERO-Object, in two passes:
#
#   1. bench  — the policy alone, no simulator. Fixed observation replayed N
#               times. MuJoCo's offscreen renderer is not on the device, so the
#               memory figures are the model's own.
#   2. libero — BitVLA's own run_libero_eval_bitnet.py driven for a short
#               rollout. Confirms the checkpoint actually solves the suite and
#               gives in-the-loop latency; its GPU memory includes the renderer.
#
# Both passes report two timers: the whole get_action() call (which on this
# stack includes a TensorFlow JPEG round-trip and a lanczos3 resize per image)
# and model.predict_action() alone, which is what lines up with vla.cpp's
# server-reported latency.
#
# Prerequisite: eval/setup_bitvla_ref_env.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CKPT="${CKPT:-/mnt/data/hf_data/hongyuw/ft-bitvla-bitsiglipL-224px-libero_object-bf16}"
VENV="${VENV:-/mnt/data/vla_sr_compare/venvs/bitvla_ref}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-c CKPT] [-g GPU] [-n N_STEPS] [-e EPISODES] [-s SUITE]
                        [-o OUTPUT_ROOT] [-p PHASES]

  -c CKPT          BitVLA finetuned checkpoint dir (default: ${CKPT})
  -g GPU           CUDA device index (default: 0)
  -n N_STEPS       timed policy calls in the sim-free bench (default: 200)
  -e EPISODES      episodes per task in the LIBERO pass (default: 1 -> 10 total)
  -s SUITE         libero suite (default: libero_object)
  -o OUTPUT_ROOT   default: \${REPO_ROOT}/outputs/bitvla_pytorch
  -p PHASES        comma-separated subset of: bench, libero (default: both)
  -h               show this help

Env overrides: CKPT, VENV
EOF
}

GPU="0"
N_STEPS="200"
EPISODES="1"
SUITE="libero_object"
OUTPUT_ROOT=""
PHASES="bench,libero"

while getopts ":c:g:n:e:s:o:p:h" opt; do
    case "${opt}" in
        c) CKPT="${OPTARG}" ;;
        g) GPU="${OPTARG}" ;;
        n) N_STEPS="${OPTARG}" ;;
        e) EPISODES="${OPTARG}" ;;
        s) SUITE="${OPTARG}" ;;
        o) OUTPUT_ROOT="${OPTARG}" ;;
        p) PHASES="${OPTARG}" ;;
        h) usage; exit 0 ;;
        \?) echo "ERROR: unknown option -${OPTARG}" >&2; usage >&2; exit 1 ;;
        :)  echo "ERROR: option -${OPTARG} requires an argument" >&2; usage >&2; exit 1 ;;
    esac
done

PY="${VENV}/bin/python"
if [[ ! -x "${PY}" ]]; then
    echo "ERROR: no interpreter at ${PY}. Run eval/setup_bitvla_ref_env.sh first." >&2
    exit 1
fi
if [[ ! -d "${CKPT}" ]]; then
    echo "ERROR: checkpoint dir not found: ${CKPT}" >&2
    exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/bitvla_pytorch}"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"

# MUJOCO_GL=egl for headless rendering; HF_HUB_OFFLINE so a local checkpoint is
# never re-resolved against the Hub mid-run.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2

has_phase() { [[ ",${PHASES}," == *",$1,"* ]]; }

rc=0

if has_phase bench; then
    echo "=== [1/2] sim-free latency + memory (${N_STEPS} timed calls) ==="
    "${PY}" "${SCRIPT_DIR}/bitvla_ref/bench_bitvla_pytorch.py" \
        --pretrained-checkpoint "${CKPT}" \
        --task-suite-name "${SUITE}" \
        --n-steps "${N_STEPS}" \
        --output "${OUTPUT_ROOT}/bench_${SUITE}.json" \
        2>&1 | tee "${OUTPUT_ROOT}/bench_${SUITE}.log"
    rc=$(( rc + ${PIPESTATUS[0]} ))
fi

if has_phase libero; then
    echo "=== [2/2] LIBERO rollout (${EPISODES} episode(s) x 10 tasks) ==="
    "${PY}" "${SCRIPT_DIR}/bitvla_ref/run_libero_bitvla_pytorch.py" \
        --pretrained-checkpoint "${CKPT}" \
        --task-suite-name "${SUITE}" \
        --num-trials-per-task "${EPISODES}" \
        --log-dir "${OUTPUT_ROOT}/logs" \
        --output "${OUTPUT_ROOT}/libero_${SUITE}.json" \
        2>&1 | tee "${OUTPUT_ROOT}/libero_${SUITE}.log"
    rc=$(( rc + ${PIPESTATUS[0]} ))
fi

echo
echo "Reports under ${OUTPUT_ROOT}"
exit "${rc}"
