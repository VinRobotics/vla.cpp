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

# How far the solver-step count T moves the action chunk, at fixed noise.
#
# Companion to run_solver_step_sweep.sh: that one measures task success, this
# one measures the numerical displacement behind it. vla_predict_check feeds
# fixed images / language / state / noise (see tests/predict_check.cpp), so
# every run at a given T is bit-reproducible and the only thing that changes
# between rows is T. Same instrument as the E4 quantisation table.
#
# This is a WITHIN-stack measurement: chunk(T) against chunk(T=4), the
# checkpoint default. It is deliberately not a cross-stack max|delta|, which
# would require feeding predict_check's synthetic tokens through the PyTorch
# processor and is a different (and much more error-prone) harness.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GGUF="${GGUF:-/mnt/data/hf_data/vrfai/gr00tn1d7-libero-gguf/libero_object/gr00tn1d7-libero-object.gguf}"
BIN="${BIN:-${REPO_ROOT}/build/tests/vla_predict_check}"
OUT="${OUT:-${REPO_ROOT}/outputs/solver_sweep/displacement}"
STEPS="${STEPS:-1 2 4 8 16}"
ITERS="${ITERS:-20}"
# predict_check defaults to 224; gr00t_n1_7's tower wants 256 and silently
# returns an EMPTY chunk on a mismatch ("image view is 224x224, expected
# 256x256" on stderr, action_len=0 on stdout), so this must match the arch.
IMG_SIZE="${IMG_SIZE:-256}"
N_IMAGES="${N_IMAGES:-2}"

for f in "${BIN}" "${GGUF}"; do
    [[ -e "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done

mkdir -p "${OUT}"
echo "[config] GGUF=${GGUF}"
echo "[config] STEPS=${STEPS}  ITERS=${ITERS}  OUT=${OUT}"

for t in ${STEPS}; do
    echo "[T=${t}] running vla_predict_check"
    VLA_NUM_STEPS="${t}" \
    VLA_BENCH_ITERS="${ITERS}" \
    VLA_IMG_SIZE="${IMG_SIZE}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "${BIN}" "${GGUF}" "" "${N_IMAGES}" \
        > "${OUT}/T${t}.actions.txt" 2> "${OUT}/T${t}.timing.txt"
    # An empty chunk means the run is void, not just quiet -- fail loudly.
    if grep -q "^action_len=0$" "${OUT}/T${t}.actions.txt"; then
        echo "ERROR: T=${t} produced an empty action chunk; see ${OUT}/T${t}.timing.txt" >&2
        exit 1
    fi
    grep -E "predict\(\) over|last split" "${OUT}/T${t}.timing.txt" || true
done

echo "Done. Analyse with: python eval/collect_solver_displacement.py -o ${OUT}"
