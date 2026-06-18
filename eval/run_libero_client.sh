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

# Client-only LIBERO runner: drives run_sim_client_direct.py against a REMOTE
# vla-server (default tcp://192.168.1.6:5555) for pi0 + the GR00T family.
#
# Runs task-id 0..9, N_EPISODES each (default 20). vla-server holds ONE checkpoint
# at a time, so in "all" mode this pauses before each arch, prints the exact server
# command to launch on the host, and probes reachability before starting.
#
# This script touches NO model weights locally - those live on the server. The only
# files it needs locally are the per-arch dataset_statistics.json (GR00T un-norm),
# which it reads from MODELS_ROOT (-i, default $HOME/data/vrfai).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-i MODELS_ROOT] [-o OUTPUT_ROOT] [-n N_EPISODES] [-m MODEL] [-a ADDR] [-y]

  -i MODELS_ROOT   dir holding the per-model GGUF folders (for the GR00T
                   dataset_statistics.json) (default: $HOME/data/vrfai)
  -o OUTPUT_ROOT   destination for client outputs
                   (default: ${REPO_ROOT}/outputs/libero_object_sweep)
  -n N_EPISODES    episodes per task-id (default: 20)
  -m MODEL         pi0 | gr00t_n1_5 | gr00t_n1_6 | gr00t_n1_7 | all (default: all)
  -a ADDR          vla-server ZMQ address (default: tcp://192.168.1.6:5555)
                   accepts host, host:port, or a full tcp:// URL
  -y               non-interactive: skip the "is the server ready?" prompt in all-mode
  -h               show this help

Env overrides: CLIENT_ADDR, N_ACTION_STEPS_PI0, N_ACTION_STEPS_GR00T_N1_5,
               N_ACTION_STEPS_GR00T_N1_6, N_ACTION_STEPS_GR00T_N1_7
EOF
}

MODELS_ROOT="$HOME/data/vrfai"
OUTPUT_ROOT=""
N_EPISODES="20"
MODEL="all"
CLIENT_ADDR="${CLIENT_ADDR:-tcp://192.168.1.6:5555}"
AUTO_CONFIRM=0

while getopts ":i:o:n:m:a:yh" opt; do
    case "${opt}" in
        i) MODELS_ROOT="${OPTARG}" ;;
        o) OUTPUT_ROOT="${OPTARG}" ;;
        n) N_EPISODES="${OPTARG}" ;;
        m) MODEL="${OPTARG}" ;;
        a) CLIENT_ADDR="${OPTARG}" ;;
        y) AUTO_CONFIRM=1 ;;
        h) usage; exit 0 ;;
        \?) echo "ERROR: unknown option -${OPTARG}" >&2; usage >&2; exit 1 ;;
        :)  echo "ERROR: option -${OPTARG} requires an argument" >&2; usage >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

case "${MODEL}" in
    pi0|gr00t_n1_5|gr00t_n1_6|gr00t_n1_7|all) ;;
    *)
        echo "ERROR: -m must be one of: pi0 | gr00t_n1_5 | gr00t_n1_6 | gr00t_n1_7 | all (got '${MODEL}')" >&2
        exit 1
        ;;
esac

if ! [[ "${N_EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: N_EPISODES must be a positive integer (got '${N_EPISODES}')" >&2
    exit 1
fi

# Normalise the address: accept "host", "host:port", or "tcp://host:port".
case "${CLIENT_ADDR}" in
    tcp://*) ;;
    *:*)     CLIENT_ADDR="tcp://${CLIENT_ADDR}" ;;
    *)       CLIENT_ADDR="tcp://${CLIENT_ADDR}:5555" ;;
esac
_addr="${CLIENT_ADDR#tcp://}"
HOST="${_addr%%:*}"
PORT="${_addr##*:}"

VENV_PY="${REPO_ROOT}/eval/sim/libero/libero_uv/.venv/bin/python"
CLIENT="${REPO_ROOT}/eval/client/run_sim_client_direct.py"
TASK_SUITE="libero_object"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/libero_object_sweep}"

# Per-arch chunk replay (match the run_libero.sh closeout configs).
N_ACTION_STEPS_PI0="${N_ACTION_STEPS_PI0:-50}"
N_ACTION_STEPS_GR00T_N1_5="${N_ACTION_STEPS_GR00T_N1_5:-16}"
N_ACTION_STEPS_GR00T_N1_6="${N_ACTION_STEPS_GR00T_N1_6:-16}"
N_ACTION_STEPS_GR00T_N1_7="${N_ACTION_STEPS_GR00T_N1_7:-16}"

if [[ ! -x "${VENV_PY}" ]]; then
    echo "ERROR: LIBERO venv not found at ${VENV_PY}." >&2
    echo "       Run: bash eval/sim/libero/setup_libero.sh" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"

echo "[config] REPO_ROOT=${REPO_ROOT}"
echo "[config] MODELS_ROOT=${MODELS_ROOT}  (GR00T stats only; weights live on the server)"
echo "[config] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[config] N_EPISODES=${N_EPISODES}"
echo "[config] MODEL=${MODEL}"
echo "[config] SERVER=${CLIENT_ADDR}  (host=${HOST} port=${PORT})"

# TCP reachability probe via bash /dev/tcp (no nc dependency). Returns 0 if the
# server port accepts a connection.
probe_server() {
    (exec 3<>"/dev/tcp/${HOST}/${PORT}") 2>/dev/null && { exec 3>&- 3<&-; return 0; }
    return 1
}

# Print the exact server-side command for an arch (paths are on the SERVER host).
print_server_cmd() {
    local arch="$1"
    echo "  On the server (${HOST}), launch ONE of:"
    case "${arch}" in
        pi0)
            cat <<EOF
    ./build/vla-server --bind tcp://*:${PORT} \\
        \${MODELS_ROOT}/pi0-libero-finetuned-v044-gguf/mmproj-pi0-libero-finetuned-v044.gguf \\
        \${MODELS_ROOT}/pi0-libero-finetuned-v044-gguf/pi0-libero-finetuned-v044.gguf
EOF
            ;;
        gr00t_n1_5)
            cat <<EOF
    VLA_GR00T_BF16_WEIGHTS=1 VLA_GR00T_EMBODIMENT=new_embodiment \\
    ./build/vla-server --bind tcp://*:${PORT} \\
        \${MODELS_ROOT}/gr00tn1d5-libero-object-gguf/gr00tn1d5-libero-object.gguf
EOF
            ;;
        gr00t_n1_6)
            cat <<EOF
    VLA_GR00T_BF16_WEIGHTS=1 VLA_GR00T_EMBODIMENT=libero_panda \\
    ./build/vla-server --bind tcp://*:${PORT} \\
        \${MODELS_ROOT}/gr00t-n1d6-libero-gguf/gr00t-n1d6-libero.gguf
EOF
            ;;
        gr00t_n1_7)
            cat <<EOF
    VLA_GR00T_BF16_WEIGHTS=1 \\
    ./build/vla-server --bind tcp://*:${PORT} \\
        \${MODELS_ROOT}/gr00t-n1d7-libero-object-gguf/gr00t-n1d7-libero-object.gguf
EOF
            ;;
    esac
}

# Wait until the operator confirms the matching model is loaded AND the port is
# reachable. In -y mode we skip the prompt but still block until reachable.
await_server() {
    local arch="$1"
    echo "===================="
    echo "[${arch}] target server: ${CLIENT_ADDR}"
    print_server_cmd "${arch}"
    if [[ "${AUTO_CONFIRM}" -eq 0 && -t 0 ]]; then
        read -r -p "[${arch}] Press Enter once vla-server is serving '${arch}' (Ctrl-C to abort)... " _
    fi
    local tries=0
    until probe_server; do
        tries=$((tries + 1))
        if [[ $((tries % 5)) -eq 1 ]]; then
            echo "[${arch}] waiting for ${HOST}:${PORT} to accept connections ..." >&2
        fi
        sleep 2
    done
    echo "[${arch}] server reachable."
}

run_arch() {
    local arch="$1"
    local n_action_steps="$2"
    local stats_json="$3"   # "" to skip (pi0); path for gr00t

    if [[ -n "${stats_json}" && ! -f "${stats_json}" ]]; then
        echo "[skip] ${arch}: dataset_statistics.json not found at ${stats_json}" >&2
        echo "       (need it locally for un-normalization; copy it from the server or set -i)" >&2
        return 0
    fi

    await_server "${arch}"

    local client_extra=()
    [[ -n "${stats_json}" ]] && client_extra+=(--stats-json "${stats_json}")

    local out_dir="${OUTPUT_ROOT}/${arch}"
    mkdir -p "${out_dir}"

    for task_id in $(seq 0 9); do
        echo "[${arch}] task_id=${task_id}  episodes=${N_EPISODES}"
        "${VENV_PY}" "${CLIENT}" \
            --arch "${arch}" \
            --vla-addr "${CLIENT_ADDR}" \
            --task "${TASK_SUITE}" \
            --task-id "${task_id}" \
            --n-episodes "${N_EPISODES}" \
            --n-action-steps "${n_action_steps}" \
            --output-dir "${out_dir}" \
            "${client_extra[@]}"
    done
    echo "[${arch}] done"
}

should_run() { [[ "${MODEL}" == "all" || "${MODEL}" == "$1" ]]; }

if should_run pi0; then
    run_arch pi0 "${N_ACTION_STEPS_PI0}" ""
fi
if should_run gr00t_n1_5; then
    run_arch gr00t_n1_5 "${N_ACTION_STEPS_GR00T_N1_5}" \
        "${MODELS_ROOT}/gr00tn1d5-libero-object-gguf/dataset_statistics.json"
fi
if should_run gr00t_n1_6; then
    run_arch gr00t_n1_6 "${N_ACTION_STEPS_GR00T_N1_6}" \
        "${MODELS_ROOT}/gr00t-n1d6-libero-gguf/dataset_statistics.json"
fi
if should_run gr00t_n1_7; then
    run_arch gr00t_n1_7 "${N_ACTION_STEPS_GR00T_N1_7}" \
        "${MODELS_ROOT}/gr00t-n1d7-libero-object-gguf/dataset_statistics.json"
fi

echo "===================="
echo "Done. Results under ${OUTPUT_ROOT}"
echo "Collect with: ${VENV_PY} ${REPO_ROOT}/eval/collect_libero_results.py --sweep ${OUTPUT_ROOT}"
