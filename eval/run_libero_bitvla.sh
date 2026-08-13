#!/usr/bin/env bash
# Run the BitVLA LIBERO sweep across THREE suites — libero_spatial, libero_goal,
# and libero_10 (LIBERO-Long) — in that order. Each suite has its OWN finetuned
# checkpoint (so its own GGUF + dataset_statistics.json), so unlike run_libero.sh
# (which sweeps many models over one suite) this script restarts vla-server once
# per suite. For each suite: launch the vla-server, wait for ready, drive
# run_sim_client_direct.py over task-id 0..9 from the LIBERO venv, then stop the
# server before moving on.
#
# Expected layout under MODELS_ROOT (one dir per suite, each holding the
# converted GGUF + dataset_statistics.json + tokenizer files):
#   <MODELS_ROOT>/libero_spatial/{bitvla-libero-spatial.gguf,dataset_statistics.json}
#   <MODELS_ROOT>/libero_goal/{bitvla-libero-goal.gguf,...}
#   <MODELS_ROOT>/libero_long/{bitvla-libero-long.gguf,...}
#   <MODELS_ROOT>/libero_object/{bitvla-libero-object.gguf,...}
# Convert each with, e.g.:
#   python scripts/convert_bitvla_to_gguf.py --ckpt <dir> --pack-int2   # -> <dir>/<gguf>

# set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-i <MODELS_ROOT>] [-o <OUTPUT_ROOT>] [-n <N_EPISODES>] [-s <SUITE>]

  -i MODELS_ROOT   directory holding the per-suite BitVLA finetune dirs
                   (default: /home/khanh/data/vrfai/bitvla-libero-gguf)
  -o OUTPUT_ROOT   destination for client outputs + server logs
                   (default: ${REPO_ROOT}/outputs/libero_bitvla_sweep)
  -n N_EPISODES    episodes per task-id (default: 20)
  -s SUITE         which suite to run: spatial | goal | long | object | all
                   (default: all — runs spatial, then goal, then long, then object)
  -h               show this help

Env overrides:
  BIND_ADDR          server bind addr    (default tcp://*:5555)
  CLIENT_ADDR        client connect addr (default tcp://localhost:5555)
  N_ACTION_STEPS     chunk replay length (default 8 = BitVLA NUM_ACTIONS_CHUNK)
  MAX_LENGTH         token budget        (default 600 = BitVLA prompt budget)
  BITVLA_GGUF_NAME   override the GGUF basename for ALL suites (default: the
                     per-suite name bitvla-libero-<suite>.gguf)
  BITVLA_TOKENIZER   local ckpt dir to override the Hub-auto-loaded tokenizer
                     (offline). Stats ALWAYS come from each suitbuide's own
                     dataset_statistics.json, never from this.
EOF
}

MODELS_ROOT="/home/khanh/data/vrfai/bitvla-libero-gguf"
OUTPUT_ROOT=""
N_EPISODES="20"
SUITE="all"

while getopts ":i:o:n:s:h" opt; do
    case "${opt}" in
        i) MODELS_ROOT="${OPTARG}" ;;
        o) OUTPUT_ROOT="${OPTARG}" ;;
        n) N_EPISODES="${OPTARG}" ;;
        s) SUITE="${OPTARG}" ;;
        h) usage; exit 0 ;;
        \?) echo "ERROR: unknown option -${OPTARG}" >&2; usage >&2; exit 1 ;;
        :)  echo "ERROR: option -${OPTARG} requires an argument" >&2; usage >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

case "${SUITE}" in
    spatial|goal|long|object|all) ;;
    *)
        echo "ERROR: -s must be one of: spatial | goal | long | object | all (got '${SUITE}')" >&2
        exit 1
        ;;
esac

if [[ ! -d "${MODELS_ROOT}" ]]; then
    echo "ERROR: MODELS_ROOT does not exist: ${MODELS_ROOT}" >&2
    exit 1
fi
MODELS_ROOT="$(cd "${MODELS_ROOT}" && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/libero_bitvla_sweep}"

if ! [[ "${N_EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: N_EPISODES must be a positive integer (got '${N_EPISODES}')" >&2
    exit 1
fi

SERVER_BIN="${REPO_ROOT}/build/vla-server"
VENV_PY="${REPO_ROOT}/eval/sim/libero/libero_uv/.venv/bin/python"
CLIENT="${REPO_ROOT}/eval/client/run_sim_client_direct.py"
BIND_ADDR="${BIND_ADDR:-tcp://*:5555}"
CLIENT_ADDR="${CLIENT_ADDR:-tcp://localhost:5555}"

N_ACTION_STEPS="${N_ACTION_STEPS:-8}"     # BitVLA NUM_ACTIONS_CHUNK
MAX_LENGTH="${MAX_LENGTH:-600}"           # BitVLA prompt token budget
BITVLA_GGUF_NAME="${BITVLA_GGUF_NAME:-}"   # empty = per-suite bitvla-libero-<suite>.gguf
BITVLA_TOKENIZER="${BITVLA_TOKENIZER:-}"

mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"
LOG_DIR="${OUTPUT_ROOT}/_server_logs"
mkdir -p "${LOG_DIR}"

echo "[config] REPO_ROOT=${REPO_ROOT}"
echo "[config] MODELS_ROOT=${MODELS_ROOT}"
echo "[config] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[config] N_EPISODES=${N_EPISODES}"
echo "[config] SUITE=${SUITE}"
echo "[config] N_ACTION_STEPS=${N_ACTION_STEPS}  MAX_LENGTH=${MAX_LENGTH}  GGUF=${BITVLA_GGUF_NAME:-<per-suite>}"

cd "${REPO_ROOT}"

echo "[build] cmake --build build"
if [[ ! -d "${REPO_ROOT}/build" ]]; then
    echo "ERROR: build/ not configured. Configure it first, e.g.:" >&2
    echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release" >&2
    exit 1
fi
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

# Background memory sampler (mirrors run_libero.sh). Polls the target PID every
# <poll> seconds and writes a JSON footprint summary when the target dies or it
# gets SIGTERM/SIGINT.
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

# run_suite <task_suite> <ckpt_dir> <label> <gguf_name>
#   task_suite  the LIBERO suite name passed to the client (libero_spatial / ...)
#   ckpt_dir    absolute dir holding the GGUF + dataset_statistics.json
#   label       short tag for log/output dir names (spatial / goal / long)
#   gguf_name   per-suite GGUF basename (overridden by BITVLA_GGUF_NAME if set)
run_suite() {
    local task_suite="$1"
    local ckpt_dir="$2"
    local label="$3"
    local gguf_name="${BITVLA_GGUF_NAME:-$4}"
    local gguf="${ckpt_dir}/${gguf_name}"
    local stats="${ckpt_dir}/dataset_statistics.json"

    echo "===================="
    echo "[${label}] task_suite=${task_suite}"
    echo "[${label}] ckpt_dir=${ckpt_dir}"

    if [[ ! -f "${gguf}" ]]; then
        echo "ERROR: GGUF not found: ${gguf}" >&2
        echo "       (set BITVLA_GGUF_NAME if your converted file has another name)" >&2
        exit 1
    fi
    if [[ ! -f "${stats}" ]]; then
        echo "ERROR: dataset_statistics.json not found: ${stats}" >&2
        exit 1
    fi

    local client_extra=()
    # Stats ALWAYS come from THIS suite's file (correct *_no_noops key + q01/q99).
    client_extra+=(--stats-json "${stats}")
    # Tokenizer is task-independent (Llama-3 BPE + specials); let it auto-load from
    # the Hub unless an offline override dir is given.
    if [[ -n "${BITVLA_TOKENIZER}" ]]; then
        client_extra+=(--tokenizer "${BITVLA_TOKENIZER}")
    fi

    start_server "${LOG_DIR}/bitvla_${label}.log" "${gguf}"

    local out_dir="${OUTPUT_ROOT}/bitvla_${label}"
    mkdir -p "${out_dir}"

    for task_id in $(seq 0 9); do
        echo "[${label}] task_id=${task_id}  episodes=${N_EPISODES}"
        "${VENV_PY}" "${CLIENT}" \
            --arch bitvla \
            --vla-addr "${CLIENT_ADDR}" \
            --task "${task_suite}" \
            --task-id "${task_id}" \
            --n-episodes "${N_EPISODES}" \
            --n-action-steps "${N_ACTION_STEPS}" \
            --max-length "${MAX_LENGTH}" \
            --output-dir "${out_dir}" \
            "${client_extra[@]}"
    done

    cleanup
    echo "[${label}] done"
}

should_run() {
    [[ "${SUITE}" == "all" || "${SUITE}" == "$1" ]]
}

# Order: spatial, then goal, then long (libero_10).
if should_run spatial; then
    run_suite libero_spatial \
        "${MODELS_ROOT}/libero_spatial" \
        spatial \
        bitvla-libero-spatial.gguf
fi

if should_run goal; then
    run_suite libero_goal \
        "${MODELS_ROOT}/libero_goal" \
        goal \
        bitvla-libero-goal.gguf
fi

# LIBERO-Long: checkpoint dir is named "libero_long", but the client task suite
# is "libero_10".
if should_run long; then
    run_suite libero_10 \
        "${MODELS_ROOT}/libero_long" \
        long \
        bitvla-libero-long.gguf
fi

if should_run object; then
    run_suite libero_object \
        "${MODELS_ROOT}/libero_object" \
        object \
        bitvla-libero-object.gguf
fi

echo "===================="
echo "Done. Results under ${OUTPUT_ROOT}"
