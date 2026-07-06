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

# Run libero_object eval (task-id 0..9, N_EPISODES per task) against every VLA
# model under MODELS_ROOT. For each model: build vla-server once, launch it,
# wait for ready, drive run_sim_client_direct.py from the LIBERO venv, then
# stop the server before moving on.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") -i <MODELS_ROOT> [-o <OUTPUT_ROOT>] [-n <N_EPISODES>] [-m <MODEL>]

  -i MODELS_ROOT   directory holding the per-model GGUF folders
                   (e.g. $HOME/data/vrfai) [required]
  -o OUTPUT_ROOT   destination for client outputs + server logs
                   (default: ${REPO_ROOT}/outputs/libero_object_sweep)
  -n N_EPISODES    episodes per task-id (default: 1)
  -m MODEL         which model to run: smol | pi0 | pi05 | bit | evo1 |
                                       vla_adapter | openvla_oft |
                                       gr00t_n1_5 | gr00t_n1_6 | gr00t_n1_7 | all
                   (default: all)
  -h               show this help

Env overrides: BIND_ADDR, CLIENT_ADDR, BITVLA_TOKENIZER, GR00T_N1_6_TOKENIZER,
               GR00T_N1_5_STATS, GR00T_N1_6_STATS, GR00T_N1_7_STATS
EOF
}

MODELS_ROOT=""
OUTPUT_ROOT=""
N_EPISODES="1"
MODEL="all"

while getopts ":i:o:n:m:h" opt; do
    case "${opt}" in
        i) MODELS_ROOT="${OPTARG}" ;;
        o) OUTPUT_ROOT="${OPTARG}" ;;
        n) N_EPISODES="${OPTARG}" ;;
        m) MODEL="${OPTARG}" ;;
        h) usage; exit 0 ;;
        \?) echo "ERROR: unknown option -${OPTARG}" >&2; usage >&2; exit 1 ;;
        :)  echo "ERROR: option -${OPTARG} requires an argument" >&2; usage >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

case "${MODEL}" in
    smol|pi0|pi05|bit|evo1|vla_adapter|openvla_oft|gr00t_n1_5|gr00t_n1_6|gr00t_n1_7|all) ;;
    *)
        echo "ERROR: -m must be one of: smol | pi0 | pi05 | bit | evo1 | vla_adapter | openvla_oft | gr00t_n1_5 | gr00t_n1_6 | gr00t_n1_7 | all (got '${MODEL}')" >&2
        exit 1
        ;;
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

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/libero_object_sweep}"

if ! [[ "${N_EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: N_EPISODES must be a positive integer (got '${N_EPISODES}')" >&2
    exit 1
fi

SERVER_BIN="${REPO_ROOT}/build/vla-server"
VENV_PY="${REPO_ROOT}/eval/sim/libero/libero_uv/.venv/bin/python"
CLIENT="${REPO_ROOT}/eval/client/run_sim_client_direct.py"
BIND_ADDR="${BIND_ADDR:-tcp://*:5555}"
CLIENT_ADDR="${CLIENT_ADDR:-tcp://localhost:5555}"
TASK_SUITE="libero_object"

# bitvla auto-loads its tokenizer + dataset_statistics.json from the GGUF repo on
# the Hub. Optional override (offline): BITVLA_TOKENIZER=/path/to/bitvla-ckpt-dir
BITVLA_TOKENIZER="${BITVLA_TOKENIZER:-}"

# gr00t_n1_6 has no HF-default tokenizer; its Eagle tokenizer is vendored in the
# model dir. Defaults to the model dir; override with GR00T_N1_6_TOKENIZER.
GR00T_N1_6_TOKENIZER="${GR00T_N1_6_TOKENIZER:-}"

# GR00T-N1.5 / N1.6 / N1.7 need a dataset_statistics.json
GR00T_N1_5_STATS="${GR00T_N1_5_STATS:-}"
GR00T_N1_6_STATS="${GR00T_N1_6_STATS:-}"
GR00T_N1_7_STATS="${GR00T_N1_7_STATS:-}"
_USER_VLA_GR00T_EMBODIMENT="${VLA_GR00T_EMBODIMENT-}"

# Per-arch chunk replay (how many actions from each predicted chunk to replay
# before re-querying vla-server). Defaults below match the configurations that
# closed each arch's LIBERO sweep at the highest reported success rate.
N_ACTION_STEPS_PI0="${N_ACTION_STEPS_PI0:-50}"               # pi0_libero_finetuned_v044 cfg
N_ACTION_STEPS_PI05="${N_ACTION_STEPS_PI05:-10}"             # pi0.5 n_action_steps (chunk 50, 10 denoise steps)
N_ACTION_STEPS_VLA_ADAPTER="${N_ACTION_STEPS_VLA_ADAPTER:-8}" # VLA-Adapter action chunk
N_ACTION_STEPS_OPENVLA_OFT="${N_ACTION_STEPS_OPENVLA_OFT:-8}" # OpenVLA-OFT parallel 8-step chunk
N_ACTION_STEPS_SMOL="${N_ACTION_STEPS_SMOL:-1}"              # SmolVLA golden path (re-predict each step)
N_ACTION_STEPS_EVO1="${N_ACTION_STEPS_EVO1:-8}"              # Evo-1 (chunk replay)
N_ACTION_STEPS_BIT="${N_ACTION_STEPS_BIT:-8}"                # BitVLA NUM_ACTIONS_CHUNK
N_ACTION_STEPS_GR00T_N1_5="${N_ACTION_STEPS_GR00T_N1_5:-16}" # N1.5 lerobot closeout (10/10 on libero_object/task_0)
N_ACTION_STEPS_GR00T_N1_6="${N_ACTION_STEPS_GR00T_N1_6:-16}" # N1.6 H4 closeout (10/10 on libero_object/task_0)
N_ACTION_STEPS_GR00T_N1_7="${N_ACTION_STEPS_GR00T_N1_7:-16}" # N1.7 H4 closeout (10/10 on libero_object/task_0)

mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"
LOG_DIR="${OUTPUT_ROOT}/_server_logs"
mkdir -p "${LOG_DIR}"

echo "[config] REPO_ROOT=${REPO_ROOT}"
echo "[config] MODELS_ROOT=${MODELS_ROOT}"
echo "[config] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[config] N_EPISODES=${N_EPISODES}"
echo "[config] MODEL=${MODEL}"

cd "${REPO_ROOT}"

echo "[build] cmake --build build"
cmake --build build -j"$(nproc)"

if [[ ! -x "${SERVER_BIN}" ]]; then
    echo "ERROR: ${SERVER_BIN} not found after build." >&2
    exit 1
fi
if [[ ! -x "${VENV_PY}" ]]; then
    echo "ERROR: LIBERO venv not found at ${VENV_PY}." >&2
    echo "       Run: bash eval/sim/libero/setup_libero.sh" >&2
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
    # Mem sampler exits on its own once the server PID is gone, but signal it
    # too and wait so its final JSON write completes before we move on.
    if [[ -n "${SAMPLER_PID}" ]] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then
        kill -TERM "${SAMPLER_PID}" 2>/dev/null || true
        wait "${SAMPLER_PID}" 2>/dev/null || true
    fi
    SAMPLER_PID=""
}
trap cleanup EXIT INT TERM

# System-wide used RAM in KiB = MemTotal - MemAvailable. On Tegra (Jetson/Orin)
# this is the ONLY way to see GPU memory: the iGPU shares system RAM and its
# cudaMalloc'd weights are NOT counted in the process's VmHWM/VmRSS, and
# `nvidia-smi --query-compute-apps` is unsupported. (Cross-checked on the AGX
# Orin sweep: GR00T showed ~1.3 GiB VmHWM but really holds ~6.3 GiB of device
# weights - only the system-used delta reveals that.)
sys_used_kib() {
    awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{ if(t&&a) print t-a; else print 0 }' \
        /proc/meminfo 2>/dev/null || echo 0
}

# Background memory sampler. Polls the target PID every <poll> seconds and keeps:
#   peak_rss_mib            VmHWM (kernel peak RSS, /proc/<pid>/status) - HOST only.
#   peak_vram_mib           max per-PID VRAM (nvidia-smi --query-compute-apps) -
#                           discrete GPUs only; null on Tegra (unsupported).
#   peak_sys_used_mib       max system-wide used RAM (MemTotal-MemAvailable) -
#                           captures the unified-memory iGPU carveout VmHWM misses.
#   baseline_sys_used_mib   system-wide used RAM at sampler start (server just
#                           spawned, weights not yet loaded) - subtract for the
#                           marginal footprint. NB: on a co-resident run the peak
#                           also includes the client/sim, so the delta is an upper
#                           bound on the server's own footprint.
# Writes a JSON to <out> when the target dies or this function gets SIGTERM/SIGINT.
mem_sampler() {
    local pid="$1" out="$2" poll="${3:-1}"
    local peak_vram=0 peak_rss_kib=0 samples=0 vram_seen=0
    local peak_sys_kib=0 baseline_sys_kib=0
    local is_tegra=0
    [[ -f /etc/nv_tegra_release ]] && is_tegra=1
    # Force C locale so awk's %f prints "2.3" rather than "2,3" under
    # locales like vi_VN.UTF-8 - otherwise the JSON below is invalid.
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

    # Sidecar memory sampler (see mem_sampler() above). Output JSON lives next
    # to the server log: <log without .log>.mem.json.
    local mem_out="${log%.log}.mem.json"
    mem_sampler "${SERVER_PID}" "${mem_out}" 1 &
    SAMPLER_PID=$!
    echo "[mem-sampler] pid=${SAMPLER_PID} out=${mem_out}"
    # Wait for "ready" line (or the process dying).
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
    local model_dir="$2"
    local n_action_steps="$3"
    local stats_json="$4"     # optional dataset_statistics.json (gr00t_n1_{6,7}); "" to skip
    shift 4
    local server_args=("$@")
    local client_extra=()

    # bitvla auto-loads tokenizer + dataset_statistics.json from the GGUF repo on
    # the Hub; only pass --tokenizer when BITVLA_TOKENIZER overrides with a local dir.
    if [[ "${arch}" == "bitvla" && -n "${BITVLA_TOKENIZER}" ]]; then
        client_extra+=(--tokenizer "${BITVLA_TOKENIZER}")
    fi
    # gr00t_n1_6 has no HF-default tokenizer; point the client at the vendored
    # Eagle tokenizer in the model dir (override via GR00T_N1_6_TOKENIZER).
    if [[ "${arch}" == "gr00t_n1_6" ]]; then
        client_extra+=(--tokenizer "${GR00T_N1_6_TOKENIZER:-${model_dir}}")
    fi
    if [[ -n "${stats_json}" ]]; then
        client_extra+=(--stats-json "${stats_json}")
    fi

    if [[ "${arch}" == gr00t_n1_5 || "${arch}" == gr00t_n1_6 || "${arch}" == gr00t_n1_7 ]]; then
        export VLA_GR00T_BF16_WEIGHTS="${VLA_GR00T_BF16_WEIGHTS:-1}"
        echo "[${arch}] VLA_GR00T_BF16_WEIGHTS=${VLA_GR00T_BF16_WEIGHTS}"
    fi
    if [[ -n "${_USER_VLA_GR00T_EMBODIMENT}" ]]; then
        export VLA_GR00T_EMBODIMENT="${_USER_VLA_GR00T_EMBODIMENT}"
    else
        unset VLA_GR00T_EMBODIMENT
    fi
    if [[ "${arch}" == gr00t_n1_5 ]]; then
        export VLA_GR00T_EMBODIMENT="${VLA_GR00T_EMBODIMENT:-new_embodiment}"
        echo "[${arch}] VLA_GR00T_EMBODIMENT=${VLA_GR00T_EMBODIMENT}"
    fi
    if [[ "${arch}" == gr00t_n1_6 ]]; then
        export VLA_GR00T_EMBODIMENT="${VLA_GR00T_EMBODIMENT:-libero_panda}"
        echo "[${arch}] VLA_GR00T_EMBODIMENT=${VLA_GR00T_EMBODIMENT}"
    fi

    # openvla_oft picks its action-unnorm (server) and proprio-norm (client) stats
    # by suite key; it must be identical on both and match the suite under test.
    # The export below is inherited by the server (start_server) and the client.
    if [[ "${arch}" == openvla_oft ]]; then
        export VLA_OPENVLA_OFT_UNNORM_KEY="${VLA_OPENVLA_OFT_UNNORM_KEY:-${TASK_SUITE}_no_noops}"
        echo "[${arch}] VLA_OPENVLA_OFT_UNNORM_KEY=${VLA_OPENVLA_OFT_UNNORM_KEY}"
    else
        unset VLA_OPENVLA_OFT_UNNORM_KEY
    fi

    local log="${LOG_DIR}/${arch}.log"
    echo "===================="
    echo "[${arch}] model_dir=${model_dir}"
    echo "[${arch}] server args: ${server_args[*]}"
    start_server "${log}" "${server_args[@]}"

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

    cleanup
    echo "[${arch}] done"
}

should_run() {
    [[ "${MODEL}" == "all" || "${MODEL}" == "$1" ]]
}

if should_run pi0; then
    run_model pi0 \
        "${MODELS_ROOT}/pi0-libero-finetuned-v044-gguf" \
        "${N_ACTION_STEPS_PI0}" \
        "" \
        "${MODELS_ROOT}/pi0-libero-finetuned-v044-gguf/pi0-libero-finetuned-v044.gguf"
fi

if should_run pi05; then
    run_model pi05 \
        "${MODELS_ROOT}/pi05-libero-gguf" \
        "${N_ACTION_STEPS_PI05}" \
        "${PI05_STATS:-}" \
        "${MODELS_ROOT}/pi05-libero-gguf/pi05-libero.gguf"
fi

# smolvla: vision baked into the ckpt (no mmproj).
if should_run smol; then
    run_model smolvla \
        "${MODELS_ROOT}/smolvla-libero-gguf" \
        "${N_ACTION_STEPS_SMOL}" \
        "" \
        "${MODELS_ROOT}/smolvla-libero-gguf/smolvla-libero.gguf"
fi

# evo1: vision baked into ckpt (no mmproj)
if should_run evo1; then
    run_model evo1 \
        "${MODELS_ROOT}/evo1-libero-gguf" \
        "${N_ACTION_STEPS_EVO1}" \
        "" \
        "${MODELS_ROOT}/evo1-libero-gguf/evo1-libero.gguf"
fi

# bitvla: vision baked in; tokenizer + dataset_statistics.json auto-load from the
# GGUF repo on the Hub. Set BITVLA_TOKENIZER=<local ckpt dir> to override (offline).
if should_run bit; then
    run_model bitvla \
        "${MODELS_ROOT}/bitvla-libero-gguf/libero_object" \
        "${N_ACTION_STEPS_BIT}" \
        "" \
        "${MODELS_ROOT}/bitvla-libero-gguf/libero_object/bitvla-libero-object.gguf"
fi

# vla_adapter: Qwen2.5-0.5B + Bridge-Attention; vision baked in (no mmproj),
# tokenizer auto-loads from the base ckpt on the Hub, stats baked into the GGUF.
if should_run vla_adapter; then
    run_model vla_adapter \
        "${MODELS_ROOT}/vla-adapter-libero-object-gguf" \
        "${N_ACTION_STEPS_VLA_ADAPTER}" \
        "" \
        "${MODELS_ROOT}/vla-adapter-libero-object-gguf/libero_object/vla-adapter-libero-object.gguf"
fi

# openvla_oft: Llama-2-7B + MLPResNet head; vision baked in (no mmproj). Needs the
# client-side dataset_statistics.json (proprio norm) and VLA_OPENVLA_OFT_UNNORM_KEY
# (set per-suite in run_model) on both server and client.
if should_run openvla_oft; then
    oft_stats="${MODELS_ROOT}/openvla-oft-libero-gguf/dataset_statistics.json"
    if [[ -f "${oft_stats}" ]]; then
        run_model openvla_oft \
            "${MODELS_ROOT}/openvla-oft-libero-gguf" \
            "${N_ACTION_STEPS_OPENVLA_OFT}" \
            "${oft_stats}" \
            "${MODELS_ROOT}/openvla-oft-libero-gguf/openvla-oft-libero.gguf"
    else
        echo "[skip] openvla_oft: dataset_statistics.json not found at ${oft_stats}"
    fi
fi

# gr00t_n1_5: lerobot finetune; vision baked in; needs dataset_statistics.json
# (flat min/max, emitted next to the GGUF by scripts/convert_gr00t_n1_5_to_gguf.py).
# embodiment new_embodiment=31 is set in run_model().
if should_run gr00t_n1_5; then
    g5_stats_default="${MODELS_ROOT}/gr00tn1d5-libero-object-gguf/dataset_statistics.json"
    g5_stats="${GR00T_N1_5_STATS:-${g5_stats_default}}"
    if [[ -f "${g5_stats}" ]]; then
        run_model gr00t_n1_5 \
            "${MODELS_ROOT}/gr00tn1d5-libero-object-gguf" \
            "${N_ACTION_STEPS_GR00T_N1_5}" \
            "${g5_stats}" \
            "${MODELS_ROOT}/gr00tn1d5-libero-object-gguf/gr00tn1d5-libero-object.gguf"
    else
        echo "[skip] gr00t_n1_5: dataset_statistics.json not found at ${g5_stats}; set GR00T_N1_5_STATS to override"
    fi
fi

# gr00t_n1_6: needs GR00T_N1_6_STATS dataset_statistics.json
if should_run gr00t_n1_6; then
    g6_stats_default="${MODELS_ROOT}/gr00tn1d6-libero-gguf/dataset_statistics.json"
    g6_stats="${GR00T_N1_6_STATS:-}"
    if [[ -z "${g6_stats}" && -f "${g6_stats_default}" ]]; then
        g6_stats="${g6_stats_default}"
    fi
    if [[ -n "${g6_stats}" && -f "${g6_stats}" ]]; then
        run_model gr00t_n1_6 \
            "${MODELS_ROOT}/gr00tn1d6-libero-gguf" \
            "${N_ACTION_STEPS_GR00T_N1_6}" \
            "${g6_stats}" \
            "${MODELS_ROOT}/gr00tn1d6-libero-gguf/gr00tn1d6-libero.gguf"
    else
        echo "[skip] gr00t_n1_6: set GR00T_N1_6_STATS=<path to experiment_cfg/dataset_statistics.json> to enable"
    fi
fi

# gr00t_n1_7: needs dataset_statistics.json
if should_run gr00t_n1_7; then
    g7_stats_default="${MODELS_ROOT}/gr00tn1d7-libero-gguf/libero_object/dataset_statistics.json"
    g7_stats="${GR00T_N1_7_STATS:-${g7_stats_default}}"
    if [[ -f "${g7_stats}" ]]; then
        run_model gr00t_n1_7 \
            "${MODELS_ROOT}/gr00tn1d7-libero-gguf/libero_object" \
            "${N_ACTION_STEPS_GR00T_N1_7}" \
            "${g7_stats}" \
            "${MODELS_ROOT}/gr00tn1d7-libero-gguf/libero_object/gr00tn1d7-libero-object.gguf"
    else
        echo "[skip] gr00t_n1_7: dataset_statistics.json not found at ${g7_stats}; set GR00T_N1_7_STATS to override"
    fi
fi

echo "===================="
echo "Done. Results under ${OUTPUT_ROOT}"
