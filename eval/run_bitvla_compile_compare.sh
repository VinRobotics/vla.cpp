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

# BitVLA's row for docs/corl-paper/experiments/compile_compare.md: the upstream
# PyTorch policy in eager, torch.compile default and torch.compile
# reduce-overhead, all in one session so the variants are comparable to each
# other without a between-session offset.
#
# BitVLA cannot go through eval/run_latency_compare.sh: that harness serves each
# policy from eval/pytorch_ref, and BitVLA needs OpenVLA-OFT's prismatic package
# plus BitVLA's own transformers fork. It gets the sim-free bench instead, which
# times the same two things (whole policy query, and model forward alone) with
# the same CUDA-synchronised timer.
#
# A variant that fails is expected and is recorded rather than hidden -- see the
# report. Failures do not abort the sweep.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CKPT="${CKPT:-/mnt/data/hf_data/hongyuw/ft-bitvla-bitsiglipL-224px-libero_object-bf16}"
VENV="${VENV:-/mnt/data/vla_sr_compare/venvs/bitvla_ref}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-c CKPT] [-g GPU] [-n N_STEPS] [-o OUTPUT_ROOT]
                        [-v VARIANTS] [-t COMPILE_TARGET]

  -c CKPT            BitVLA checkpoint dir (default: ${CKPT})
  -g GPU             CUDA device index (default: 0)
  -n N_STEPS         timed calls per variant (default: 200)
  -o OUTPUT_ROOT     default: \${REPO_ROOT}/outputs/bitvla_pytorch/compile
  -v VARIANTS        comma-separated subset of:
                       eager, compile-default, compile-reduce-overhead,
                       compile-max-autotune
                     (default: the first three)
  -t COMPILE_TARGET  modules | predict_action  (default: modules)
  -h                 show this help
EOF
}

GPU="0"
N_STEPS="200"
OUTPUT_ROOT=""
VARIANTS="eager,compile-default,compile-reduce-overhead"
COMPILE_TARGET="modules"

while getopts ":c:g:n:o:v:t:h" opt; do
    case "${opt}" in
        c) CKPT="${OPTARG}" ;;
        g) GPU="${OPTARG}" ;;
        n) N_STEPS="${OPTARG}" ;;
        o) OUTPUT_ROOT="${OPTARG}" ;;
        v) VARIANTS="${OPTARG}" ;;
        t) COMPILE_TARGET="${OPTARG}" ;;
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

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/bitvla_pytorch/compile}"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2

declare -a FAILED=()

IFS=',' read -ra WANTED <<< "${VARIANTS}"
for variant in "${WANTED[@]}"; do
    case "${variant}" in
        eager)                    mode="off" ;;
        compile-default)          mode="default" ;;
        compile-reduce-overhead)  mode="reduce-overhead" ;;
        compile-max-autotune)     mode="max-autotune" ;;
        *) echo "ERROR: unknown variant '${variant}'" >&2; exit 1 ;;
    esac

    echo
    echo "==================== ${variant} (mode=${mode}, target=${COMPILE_TARGET})"
    "${PY}" "${SCRIPT_DIR}/bitvla_ref/bench_bitvla_pytorch.py" \
        --pretrained-checkpoint "${CKPT}" \
        --n-steps "${N_STEPS}" \
        --compile "${mode}" \
        --compile-target "${COMPILE_TARGET}" \
        --output "${OUTPUT_ROOT}/${variant}.json" \
        > "${OUTPUT_ROOT}/${variant}.log" 2>&1
    if [[ $? -ne 0 ]]; then
        echo "FAILED: ${variant} -- see ${OUTPUT_ROOT}/${variant}.log"
        tail -n 25 "${OUTPUT_ROOT}/${variant}.log" | sed 's/^/    | /'
        FAILED+=("${variant}")
        continue
    fi
    grep -E "^(get_action|predict_action|weights|peak|parameters) " \
        "${OUTPUT_ROOT}/${variant}.log" | sed 's/^/    /'
done

echo
echo "Reports under ${OUTPUT_ROOT}"
if (( ${#FAILED[@]} )); then
    echo "Variants that did not run: ${FAILED[*]}"
fi
