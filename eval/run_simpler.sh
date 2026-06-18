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

# Run SimplerEnv WidowX/Bridge eval (one or more tasks, N_EPISODES each) for the
# GR00T-N1.6-bridge model. Build vla-server once, launch it (252-px GGUF +
# oxe_widowx embodiment), wait for ready, drive run_simpler_client_direct.py from
# the SimplerEnv venv, then stop the server.
#
# Counterpart to run_libero.sh. SIMPLER differs from LIBERO in three ways:
#   • CUDA is required (GR00T-N1.6 on CPU is ~2.3 s/predict - infeasible with
#     SIMPLER's re-predict-every-step loop), so SERVER_BIN defaults to build-cuda.
#   • Client venv is the SimplerEnv uv venv, not the LIBERO one.
#   • We loop over named WidowX/Bridge tasks (oxe_widowx/<task>) instead of
#     libero task-ids.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Canonical SimplerEnv WidowX/Bridge benchmark tasks (registered in
# eval/sim/simpler/simpler_env.py). "-t all" adds the drawer + sink extras.
DEFAULT_TASKS="widowx_spoon_on_towel,widowx_carrot_on_plate,widowx_stack_cube,widowx_put_eggplant_in_basket"
ALL_TASKS="${DEFAULT_TASKS},widowx_put_eggplant_in_sink,widowx_open_drawer,widowx_close_drawer"

usage() {
    cat <<EOF
Usage: $(basename "$0") -i <MODELS_ROOT> [-o <OUTPUT_ROOT>] [-n <N_EPISODES>] [-t <TASKS>] [-m <MODEL>]

  -i MODELS_ROOT   directory holding the per-model GGUF folders
                   (e.g. $HOMEnd61/data/vrfai, which holds
                   gr00t-n1d6-bridge-gguf/) [required]
  -o OUTPUT_ROOT   destination for client outputs + server logs
                   (default: ${REPO_ROOT}/outputs/simpler_widowx)
  -n N_EPISODES    episodes per task (default: 5)
  -t TASKS         comma-separated widowx task names, or "all" (7 tasks), or
                   "default" (4 canonical bridge tasks)
                   (default: ${DEFAULT_TASKS})
  -m MODEL         which model to run: gr00t_n1_6 (default; only one wired for SIMPLER)
  -h               show this help

Env overrides: BIND_ADDR, CLIENT_ADDR, SERVER_BIN, IMAGE_SIZE, FPS,
               VLA_GR00T_EMBODIMENT, N_ACTION_STEPS_GR00T_N1_6,
               GR00T_N1_6_BRIDGE_GGUF, GR00T_N1_6_BRIDGE_STATS
EOF
}

MODELS_ROOT=""
OUTPUT_ROOT=""
N_EPISODES="5"
TASKS="${DEFAULT_TASKS}"
MODEL="gr00t_n1_6"

while getopts ":i:o:n:t:m:h" opt; do
    case "${opt}" in
        i) MODELS_ROOT="${OPTARG}" ;;
        o) OUTPUT_ROOT="${OPTARG}" ;;
        n) N_EPISODES="${OPTARG}" ;;
        t) TASKS="${OPTARG}" ;;
        m) MODEL="${OPTARG}" ;;
        h) usage; exit 0 ;;
        \?) echo "ERROR: unknown option -${OPTARG}" >&2; usage >&2; exit 1 ;;
        :)  echo "ERROR: option -${OPTARG} requires an argument" >&2; usage >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

case "${MODEL}" in
    gr00t_n1_6) ;;
    *)
        echo "ERROR: -m must be gr00t_n1_6 (the only model wired for SIMPLER; got '${MODEL}')" >&2
        exit 1
        ;;
esac

case "${TASKS}" in
    all)     TASKS="${ALL_TASKS}" ;;
    default) TASKS="${DEFAULT_TASKS}" ;;
esac

if [[ -z "${MODELS_ROOT}" ]]; then
    echo "ERROR: -i <MODELS_ROOT> is required." >&2
    usage >&2
    exit 1
fi
if [[ ! -d "${MODELS_ROOT}" ]]; then
    echo "ERROR: MODELS_ROOT does not exist: ${MODELS_ROOT}" >&2
    exit 1
fi
MODELS_ROOT="$(cd "${MODELS_ROOT}" && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/simpler_widowx}"

if ! [[ "${N_EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: N_EPISODES must be a positive integer (got '${N_EPISODES}')" >&2
    exit 1
fi

# CUDA is required for GR00T-N1.6 - default to build-cuda (override with SERVER_BIN).
SERVER_BIN="${SERVER_BIN:-${REPO_ROOT}/build-cuda/vla-server}"
VENV_PY="${REPO_ROOT}/eval/sim/simpler/simpler_uv/.venv/bin/python"
CLIENT="${REPO_ROOT}/eval/client/run_simpler_client_direct.py"
BIND_ADDR="${BIND_ADDR:-tcp://*:5566}"
CLIENT_ADDR="${CLIENT_ADDR:-tcp://localhost:5566}"
IMAGE_SIZE="${IMAGE_SIZE:-252}"
FPS="${FPS:-20}"
EMBODIMENT="oxe_widowx"

# 252-px GGUF (K=81) + statistics.json live in the bridge GGUF dir, following the
# same ${MODELS_ROOT}/<model>-gguf/<model>.gguf layout as run_libero.sh. Override
# the resolved paths directly with GR00T_N1_6_BRIDGE_{GGUF,STATS}.
# NB: the GGUF must be the --vision-size 252 build (K=81); the default 224/K=64
# build gets only ~20% on widowx (the vision resolution is load-bearing here).
BRIDGE_DIR="${MODELS_ROOT}/gr00t-n1d6-bridge-gguf"
GR00T_N1_6_BRIDGE_GGUF="${GR00T_N1_6_BRIDGE_GGUF:-${BRIDGE_DIR}/gr00t-n1d6-bridge.gguf}"
GR00T_N1_6_BRIDGE_STATS="${GR00T_N1_6_BRIDGE_STATS:-${BRIDGE_DIR}/statistics.json}"

# Re-predict every env step (n_action_steps=1) matches the PyTorch reference
# (GR00TN16SimplerParser.parse_action uses chunk step 0). Raise for open-loop
# replay (faster, diverges from the reference).
N_ACTION_STEPS_GR00T_N1_6="${N_ACTION_STEPS_GR00T_N1_6:-1}"

mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"
LOG_DIR="${OUTPUT_ROOT}/_server_logs"
mkdir -p "${LOG_DIR}"

echo "[config] REPO_ROOT=${REPO_ROOT}"
echo "[config] MODELS_ROOT=${MODELS_ROOT}"
echo "[config] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[config] N_EPISODES=${N_EPISODES}"
echo "[config] TASKS=${TASKS}"
echo "[config] MODEL=${MODEL}  IMAGE_SIZE=${IMAGE_SIZE}  EMBODIMENT=${EMBODIMENT}"
echo "[config] SERVER_BIN=${SERVER_BIN}"

cd "${REPO_ROOT}"

# Build the CUDA server if its build tree is configured; otherwise rely on an
# existing binary (don't silently fall back to the CPU build/).
if [[ -f "${REPO_ROOT}/build-cuda/CMakeCache.txt" ]]; then
    echo "[build] cmake --build build-cuda"
    cmake --build build-cuda -j"$(nproc)"
else
    echo "[build] build-cuda not configured; skipping build (using existing ${SERVER_BIN})."
    echo "        To configure: cmake -B build-cuda -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON \\"
    echo "                        -DCMAKE_CUDA_ARCHITECTURES=<arch> -DCMAKE_BUILD_TYPE=Release"
fi

if [[ ! -x "${SERVER_BIN}" ]]; then
    echo "ERROR: ${SERVER_BIN} not found/executable." >&2
    echo "       Build it (see above) or set SERVER_BIN=<path to a CUDA vla-server>." >&2
    exit 1
fi
if [[ ! -x "${VENV_PY}" ]]; then
    echo "ERROR: SimplerEnv venv not found at ${VENV_PY}." >&2
    echo "       Run: bash eval/sim/simpler/setup_SimplerEnv.sh" >&2
    exit 1
fi

SERVER_PID=""
SAMPLER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[cleanup] stopping vla-server (pid=${SERVER_PID})"
        kill -INT "${SERVER_PID}" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "${SERVER_PID}" 2>/dev/null || break
            sleep 0.5
        done
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
    if [[ -n "${SAMPLER_PID}" ]] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then
        kill -TERM "${SAMPLER_PID}" 2>/dev/null || true
        wait "${SAMPLER_PID}" 2>/dev/null || true
    fi
    SAMPLER_PID=""
}
trap cleanup EXIT INT TERM

# System-wide used RAM in KiB = MemTotal - MemAvailable. On Tegra (Jetson/Orin)
# this is the ONLY way to see GPU memory: the iGPU shares system RAM and its
# cudaMalloc'd weights are NOT counted in the process's VmHWM/VmRSS.
sys_used_kib() {
    awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{ if(t&&a) print t-a; else print 0 }' \
        /proc/meminfo 2>/dev/null || echo 0
}

# Background memory sampler (see run_libero.sh for the full field reference).
# NB: on SIMPLER the server is co-resident with SAPIEN rendering, so the
# sys_used delta is an upper bound on the server's own footprint.
mem_sampler() {
    local pid="$1" out="$2" poll="${3:-1}"
    local peak_vram=0 peak_rss_kib=0 samples=0 vram_seen=0
    local peak_sys_kib=0 baseline_sys_kib=0
    local is_tegra=0
    [[ -f /etc/nv_tegra_release ]] && is_tegra=1
    export LC_ALL=C
    trap 'stop=1' TERM INT
    local stop=0
    baseline_sys_kib=$(sys_used_kib)
    peak_sys_kib=$baseline_sys_kib
    while [[ $stop -eq 0 ]] && kill -0 "$pid" 2>/dev/null; do
        local rss
        rss=$(awk '/^VmHWM:/ {print $2; exit}' "/proc/$pid/status" 2>/dev/null || true)
        if [[ -n "$rss" ]] && (( rss > peak_rss_kib )); then
            peak_rss_kib=$rss
        fi
        local vram
        vram=$(nvidia-smi --query-compute-apps=pid,used_memory \
                          --format=csv,noheader,nounits 2>/dev/null \
               | awk -F, -v p="$pid" '$1==p {gsub(/ /,"",$2); print $2; exit}' || true)
        if [[ -n "$vram" ]]; then
            vram_seen=1
            if (( vram > peak_vram )); then peak_vram=$vram; fi
        fi
        local sys
        sys=$(sys_used_kib)
        if [[ -n "$sys" ]] && (( sys > peak_sys_kib )); then peak_sys_kib=$sys; fi
        samples=$((samples + 1))
        sleep "$poll"
    done
    local vram_json="null"
    if (( vram_seen )); then vram_json="$peak_vram"; fi
    local rss_mib peak_sys_mib base_sys_mib delta_sys_mib
    rss_mib=$(awk      -v k="$peak_rss_kib"     'BEGIN { printf "%.1f", k/1024.0 }')
    peak_sys_mib=$(awk -v k="$peak_sys_kib"     'BEGIN { printf "%.1f", k/1024.0 }')
    base_sys_mib=$(awk -v k="$baseline_sys_kib" 'BEGIN { printf "%.1f", k/1024.0 }')
    delta_sys_mib=$(awk -v p="$peak_sys_kib" -v b="$baseline_sys_kib" \
                        'BEGIN { d=p-b; if (d<0) d=0; printf "%.1f", d/1024.0 }')
    printf '{"pid": %d, "peak_vram_mib": %s, "peak_rss_mib": %s, "peak_sys_used_mib": %s, "baseline_sys_used_mib": %s, "sys_used_delta_mib": %s, "is_tegra": %d, "samples": %d}\n' \
        "$pid" "$vram_json" "$rss_mib" "$peak_sys_mib" "$base_sys_mib" "$delta_sys_mib" "$is_tegra" "$samples" > "$out"
    echo "[mem-sampler] wrote $out  (vram=${vram_json} MiB  rss=${rss_mib} MiB  sys_peak=${peak_sys_mib} MiB  sys_delta=${delta_sys_mib} MiB  tegra=${is_tegra}  samples=${samples})"
}

start_server() {
    local log="$1"; shift
    : > "${log}"
    "${SERVER_BIN}" --bind "${BIND_ADDR}" "$@" >"${log}" 2>&1 &
    SERVER_PID=$!
    echo "[server] pid=${SERVER_PID} log=${log}"

    local mem_out="${log%.log}.mem.json"
    mem_sampler "${SERVER_PID}" "${mem_out}" 1 &
    SAMPLER_PID=$!
    echo "[mem-sampler] pid=${SAMPLER_PID} out=${mem_out}"
    for _ in $(seq 1 600); do
        if grep -q "bound to .* ready" "${log}" 2>/dev/null; then
            echo "[server] ready"
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "ERROR: vla-server exited before becoming ready; see ${log}" >&2
            tail -n 40 "${log}" >&2 || true
            return 1
        fi
        sleep 1
    done
    echo "ERROR: vla-server did not become ready within 600s; see ${log}" >&2
    return 1
}

run_model() {
    local arch="$1"
    local gguf="$2"
    local stats_json="$3"
    local n_action_steps="$4"

    if [[ ! -f "${gguf}" ]]; then
        echo "[skip] ${arch}: GGUF not found at ${gguf}" >&2
        echo "       Build the 252-px bridge GGUF from the HF ckpt snapshot with:" >&2
        echo "       PYTHONPATH=third_party/llama.cpp/gguf-py eval/.venv/bin/python \\" >&2
        echo "         scripts/convert_gr00t_n1_6_to_gguf.py --ckpt <GR00T-N1.6-bridge ckpt dir> \\" >&2
        echo "         --vision-size 252 --out ${gguf}" >&2
        echo "       then copy that ckpt's statistics.json next to it. (Or set" >&2
        echo "       GR00T_N1_6_BRIDGE_GGUF / GR00T_N1_6_BRIDGE_STATS to existing paths.)" >&2
        return 1
    fi
    if [[ ! -f "${stats_json}" ]]; then
        echo "[skip] ${arch}: statistics.json not found at ${stats_json} (set GR00T_N1_6_BRIDGE_STATS)" >&2
        return 1
    fi

    export VLA_GR00T_BF16_WEIGHTS="${VLA_GR00T_BF16_WEIGHTS:-1}"
    export VLA_GR00T_EMBODIMENT="${VLA_GR00T_EMBODIMENT:-${EMBODIMENT}}"
    echo "[${arch}] VLA_GR00T_BF16_WEIGHTS=${VLA_GR00T_BF16_WEIGHTS}  VLA_GR00T_EMBODIMENT=${VLA_GR00T_EMBODIMENT}"

    local log="${LOG_DIR}/${arch}.log"
    echo "===================="
    echo "[${arch}] gguf=${gguf}"
    echo "[${arch}] stats=${stats_json}"
    start_server "${log}" "${gguf}"

    local IFS=','
    for task in ${TASKS}; do
        task="${task#oxe_widowx/}"   # tolerate either bare name or full id
        echo "[${arch}] task=${task}  episodes=${N_EPISODES}"
        "${VENV_PY}" "${CLIENT}" \
            --arch "${arch}" \
            --vla-addr "${CLIENT_ADDR}" \
            --task-id "oxe_widowx/${task}" \
            --n-episodes "${N_EPISODES}" \
            --stats-json "${stats_json}" \
            --embodiment "${EMBODIMENT}" \
            --image-size "${IMAGE_SIZE}" \
            --n-action-steps "${n_action_steps}" \
            --fps "${FPS}" \
            --output-dir "${OUTPUT_ROOT}"
    done

    cleanup
    echo "[${arch}] done"
}

run_model gr00t_n1_6 \
    "${GR00T_N1_6_BRIDGE_GGUF}" \
    "${GR00T_N1_6_BRIDGE_STATS}" \
    "${N_ACTION_STEPS_GR00T_N1_6}"

echo "===================="
echo "Done. Results under ${OUTPUT_ROOT}"
echo "Per-task summaries: ${OUTPUT_ROOT}/gr00t_n1_6/oxe_widowx/<task>/summary.txt"
