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

# Side-by-side LIBERO-object success-rate sweep: upstream PyTorch policies
# first, then the vla.cpp GGUF ports, over the same 10 tasks x N episodes.
#
# Both phases use the same LIBERO env (eval/sim/libero/libero_env.py), the same
# seed (42), the same 500-step cap, and the same per-arch action-chunk replay,
# so the SR delta isolates the port.
#
# Phases run in order (PyTorch fully completes before vla.cpp). Within a phase,
# models are dealt to the available GPUs and run concurrently — one policy
# server per GPU, each on its own ZMQ port.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-n <N_EPISODES>] [-g <GPUS>] [-o <OUTPUT_ROOT>]
                        [-i <GGUF_ROOT>] [-c <CKPT_ROOT>] [-P] [-C]

  -n N_EPISODES   episodes per task-id (default: 10)
  -g GPUS         comma-separated CUDA device indices (default: 0,1)
  -o OUTPUT_ROOT  results root (default: ${REPO_ROOT}/outputs/sr_compare)
  -i GGUF_ROOT    root of the vrfai GGUF dirs (default: /mnt/data/hf_data/vrfai)
  -c CKPT_ROOT    root of the PyTorch checkpoints (default: /mnt/data/hf_data)
  -P              run the PyTorch phase only
  -C              run the vla.cpp phase only
  -h              show this help
EOF
}

N_EPISODES="10"
GPUS="0,1"
OUTPUT_ROOT=""
GGUF_ROOT="/mnt/data/hf_data/vrfai"
CKPT_ROOT="/mnt/data/hf_data"
RUN_PT=1
RUN_CPP=1

while getopts ":n:g:o:i:c:PCh" opt; do
    case "${opt}" in
        n) N_EPISODES="${OPTARG}" ;;
        g) GPUS="${OPTARG}" ;;
        o) OUTPUT_ROOT="${OPTARG}" ;;
        i) GGUF_ROOT="${OPTARG}" ;;
        c) CKPT_ROOT="${OPTARG}" ;;
        P) RUN_CPP=0 ;;
        C) RUN_PT=0 ;;
        h) usage; exit 0 ;;
        \?) echo "ERROR: unknown option -${OPTARG}" >&2; usage >&2; exit 1 ;;
        :)  echo "ERROR: option -${OPTARG} requires an argument" >&2; usage >&2; exit 1 ;;
    esac
done

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/sr_compare}"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"
PT_OUT="${OUTPUT_ROOT}/pytorch"
CPP_OUT="${OUTPUT_ROOT}/vla_cpp"
DRIVER_LOG_DIR="${OUTPUT_ROOT}/_driver_logs"
mkdir -p "${PT_OUT}" "${CPP_OUT}" "${DRIVER_LOG_DIR}"

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
N_GPU="${#GPU_LIST[@]}"
if (( N_GPU < 1 )); then
    echo "ERROR: -g needs at least one GPU index" >&2
    exit 1
fi

# BitVLA has no PyTorch policy wrapper in eval/pytorch_ref, so it appears in the
# vla.cpp phase only; its PyTorch reference is the paper's reported number.
PT_MODELS=(smolvla evo1 pi0 gr00t_n1_5 gr00t_n1_6 gr00t_n1_7)
# run_libero.sh's own -m keys (smol/bit, not smolvla/bitvla).
CPP_MODELS=(smol evo1 pi0 bit gr00t_n1_5 gr00t_n1_6 gr00t_n1_7)

echo "[config] N_EPISODES=${N_EPISODES}  GPUS=${GPUS}"
echo "[config] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[config] GGUF_ROOT=${GGUF_ROOT}  CKPT_ROOT=${CKPT_ROOT}"

# ---------------------------------------------------------------------------
# Deal `models` across GPU_LIST and wait for the whole phase.
#   $1 = phase label, $2... = model keys
# `run_one <model> <gpu> <port>` must be defined by the caller.
# ---------------------------------------------------------------------------
run_phase() {
    local phase="$1"; shift
    local models=("$@")
    local pids=() names=()
    local i=0

    while (( i < ${#models[@]} )); do
        # Fill every GPU slot, then wait for the whole wave before the next.
        local wave_pids=() wave_names=()
        local slot=0
        while (( slot < N_GPU && i < ${#models[@]} )); do
            local m="${models[$i]}"
            local gpu="${GPU_LIST[$slot]}"
            local port=$((5570 + slot))
            local log="${DRIVER_LOG_DIR}/${phase}.${m}.log"
            echo "[${phase}] launch ${m} on GPU ${gpu} (port ${port}) -> ${log}"
            run_one "${m}" "${gpu}" "${port}" >"${log}" 2>&1 &
            wave_pids+=($!)
            wave_names+=("${m}")
            slot=$((slot + 1))
            i=$((i + 1))
        done
        local k=0
        for pid in "${wave_pids[@]}"; do
            if wait "${pid}"; then
                echo "[${phase}] ${wave_names[$k]}: OK"
            else
                echo "[${phase}] ${wave_names[$k]}: FAILED (see ${DRIVER_LOG_DIR}/${phase}.${wave_names[$k]}.log)" >&2
            fi
            k=$((k + 1))
        done
    done
}

# ---------------------------------------------------------------------------
# Phase 1 — PyTorch reference
# ---------------------------------------------------------------------------
if (( RUN_PT )); then
    echo "===================================================="
    echo "PHASE 1/2 — PyTorch reference (${#PT_MODELS[@]} models)"
    echo "===================================================="
    run_one() {
        bash "${SCRIPT_DIR}/run_libero_pytorch.sh" \
            -m "$1" -g "$2" -p "$3" \
            -i "${CKPT_ROOT}" -o "${PT_OUT}" -n "${N_EPISODES}"
    }
    run_phase pytorch "${PT_MODELS[@]}"
fi

# ---------------------------------------------------------------------------
# Phase 2 — vla.cpp GGUF port
# ---------------------------------------------------------------------------
if (( RUN_CPP )); then
    echo "===================================================="
    echo "PHASE 2/2 — vla.cpp GGUF (${#CPP_MODELS[@]} models)"
    echo "===================================================="
    echo "[build] cmake --build build (once, before the concurrent runs)"
    cmake --build "${REPO_ROOT}/build" -j"$(nproc)" || {
        echo "ERROR: vla-server build failed" >&2; exit 1; }

    # Every tokenizer/processor the client needs is already cached (verified
    # offline before the sweep). Staying offline avoids the Hub metadata call
    # that can wedge a load on a half-closed CDN socket — see the same note in
    # run_libero_pytorch.sh. BitVLA's tokenizer AND its dataset_statistics.json
    # both come from the local GGUF dir, so it needs no Hub access either.
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export BITVLA_TOKENIZER="${BITVLA_TOKENIZER:-${GGUF_ROOT}/bitvla-libero-gguf/libero_object}"

    run_one() {
        local m="$1" gpu="$2" port="$3"
        SKIP_BUILD=1 \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        BIND_ADDR="tcp://*:${port}" \
        CLIENT_ADDR="tcp://localhost:${port}" \
        bash "${SCRIPT_DIR}/run_libero.sh" \
            -m "${m}" -i "${GGUF_ROOT}" -o "${CPP_OUT}" -n "${N_EPISODES}"
    }
    run_phase vla_cpp "${CPP_MODELS[@]}"
fi

echo "===================================================="
echo "Done. Results under ${OUTPUT_ROOT}"
echo "  PyTorch : ${PT_OUT}"
echo "  vla.cpp : ${CPP_OUT}"
echo "Aggregate with: python eval/collect_sr_compare.py -o ${OUTPUT_ROOT}"
