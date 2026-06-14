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

# Server-only LIBERO runner: builds (optional) and launches vla-server for ONE
# arch, binds it on all interfaces so a REMOTE client can connect, then blocks
# until Ctrl-C. Pair with eval/run_libero_client.sh on the other machine.
#
# vla-server holds a single checkpoint at a time, so -m takes exactly one arch
# (no "all"): pi0 | gr00t_n1_5 | gr00t_n1_6 | gr00t_n1_7. Restart with a different
# -m to serve a different model.
#
# A sidecar memory sampler writes peak RSS / system-used RAM next to the server
# log - on Tegra (Jetson/Orin) the system-used delta is the ONLY way to see the
# unified-memory iGPU carveout (VmHWM and nvidia-smi miss it).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") -m MODEL [-i MODELS_ROOT] [-a BIND_ADDR] [-o LOG_DIR] [-B]

  -m MODEL         pi0 | gr00t_n1_5 | gr00t_n1_6 | gr00t_n1_7  [required]
  -i MODELS_ROOT   dir holding the per-model GGUF folders
                   (default: $HOME/data/vrfai)
  -a BIND_ADDR     ZMQ bind address (default: tcp://*:5555)
                   accepts a port, *:port, host:port, or a full tcp:// URL.
                   Use tcp://*:PORT (not localhost) so the remote client can reach it.
  -o LOG_DIR       server log + mem JSON destination
                   (default: ${REPO_ROOT}/outputs/libero_object_sweep/_server_logs)
  -B               skip the cmake build (use the existing vla-server binary)
  -h               show this help

Env overrides: BIND_ADDR, SERVER_BIN, VLA_GR00T_BF16_WEIGHTS, VLA_GR00T_EMBODIMENT
EOF
}

MODELS_ROOT="$HOME/data/vrfai"
MODEL=""
BIND_ADDR="${BIND_ADDR:-tcp://*:5555}"
LOG_DIR=""
DO_BUILD=1

while getopts ":i:m:a:o:Bh" opt; do
    case "${opt}" in
        i) MODELS_ROOT="${OPTARG}" ;;
        m) MODEL="${OPTARG}" ;;
        a) BIND_ADDR="${OPTARG}" ;;
        o) LOG_DIR="${OPTARG}" ;;
        B) DO_BUILD=0 ;;
        h) usage; exit 0 ;;
        \?) echo "ERROR: unknown option -${OPTARG}" >&2; usage >&2; exit 1 ;;
        :)  echo "ERROR: option -${OPTARG} requires an argument" >&2; usage >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

if [[ -z "${MODEL}" ]]; then
    echo "ERROR: -m <MODEL> is required (pi0 | gr00t_n1_5 | gr00t_n1_6 | gr00t_n1_7)." >&2
    usage >&2
    exit 1
fi
case "${MODEL}" in
    pi0|gr00t_n1_5|gr00t_n1_6|gr00t_n1_7) ;;
    *)
        echo "ERROR: -m must be one of: pi0 | gr00t_n1_5 | gr00t_n1_6 | gr00t_n1_7 (got '${MODEL}')" >&2
        exit 1
        ;;
esac

if [[ ! -d "${MODELS_ROOT}" ]]; then
    echo "ERROR: MODELS_ROOT does not exist: ${MODELS_ROOT}" >&2
    exit 1
fi
MODELS_ROOT="$(cd "${MODELS_ROOT}" && pwd)"

# Normalise the bind address: accept "5555", "*:5555", "host:5555", or "tcp://...".
case "${BIND_ADDR}" in
    tcp://*)     ;;                                          # already a full URL
    *:*)         BIND_ADDR="tcp://${BIND_ADDR}" ;;           # host:port or *:port
    *[!0-9]*)    echo "ERROR: cannot parse BIND_ADDR='${BIND_ADDR}'" >&2; exit 1 ;;
    *)           BIND_ADDR="tcp://*:${BIND_ADDR}" ;;         # bare numeric port
esac

SERVER_BIN="${SERVER_BIN:-${REPO_ROOT}/build/vla-server}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/libero_object_sweep/_server_logs}"
mkdir -p "${LOG_DIR}"
LOG_DIR="$(cd "${LOG_DIR}" && pwd)"

echo "[config] REPO_ROOT=${REPO_ROOT}"
echo "[config] MODELS_ROOT=${MODELS_ROOT}"
echo "[config] MODEL=${MODEL}"
echo "[config] BIND_ADDR=${BIND_ADDR}"
echo "[config] LOG_DIR=${LOG_DIR}"
echo "[config] SERVER_BIN=${SERVER_BIN}"

cd "${REPO_ROOT}"

if [[ "${DO_BUILD}" -eq 1 ]]; then
    echo "[build] cmake --build $(dirname "${SERVER_BIN}")"
    cmake --build "$(dirname "${SERVER_BIN}")" -j"$(nproc)"
fi
if [[ ! -x "${SERVER_BIN}" ]]; then
    echo "ERROR: ${SERVER_BIN} not found/executable. Build it, point SERVER_BIN at it, or drop -B." >&2
    exit 1
fi

# ---- per-arch GGUF positional args + GR00T env -----------------------------
# pi0: mmproj + ckpt. gr00t_*: vision baked in -> just ckpt. Weights live here on
# the server, NOT on the client. (dataset_statistics.json is a CLIENT-side arg.)
SERVER_ARGS=()
case "${MODEL}" in
    pi0)
        SERVER_ARGS=(
            "${MODELS_ROOT}/pi0-libero-finetuned-v044-gguf/mmproj-pi0-libero-finetuned-v044.gguf"
            "${MODELS_ROOT}/pi0-libero-finetuned-v044-gguf/pi0-libero-finetuned-v044.gguf"
        )
        ;;
    gr00t_n1_5)
        SERVER_ARGS=("${MODELS_ROOT}/gr00t-n1d5-libero-object-gguf/gr00t-n1d5-libero-object.gguf")
        ;;
    gr00t_n1_6)
        SERVER_ARGS=("${MODELS_ROOT}/gr00t-n1d6-libero-gguf/gr00t-n1d6-libero.gguf")
        ;;
    gr00t_n1_7)
        SERVER_ARGS=("${MODELS_ROOT}/gr00t-n1d7-libero-object-gguf/gr00t-n1d7-libero-object.gguf")
        ;;
esac

# GR00T env. Honour a user-supplied VLA_GR00T_EMBODIMENT; otherwise default per
# arch (matches eval/run_libero.sh). VLA_GR00T_BF16_WEIGHTS defaults to 1 (BF16 is
# the smaller weight path - important on the Nano's 8 GB unified RAM).
_USER_VLA_GR00T_EMBODIMENT="${VLA_GR00T_EMBODIMENT-}"
if [[ "${MODEL}" == gr00t_n1_5 || "${MODEL}" == gr00t_n1_6 || "${MODEL}" == gr00t_n1_7 ]]; then
    export VLA_GR00T_BF16_WEIGHTS="${VLA_GR00T_BF16_WEIGHTS:-1}"
    echo "[${MODEL}] VLA_GR00T_BF16_WEIGHTS=${VLA_GR00T_BF16_WEIGHTS}"
fi
if [[ -n "${_USER_VLA_GR00T_EMBODIMENT}" ]]; then
    export VLA_GR00T_EMBODIMENT="${_USER_VLA_GR00T_EMBODIMENT}"
else
    unset VLA_GR00T_EMBODIMENT
fi
if [[ "${MODEL}" == gr00t_n1_5 ]]; then
    export VLA_GR00T_EMBODIMENT="${VLA_GR00T_EMBODIMENT:-new_embodiment}"
    echo "[${MODEL}] VLA_GR00T_EMBODIMENT=${VLA_GR00T_EMBODIMENT}"
fi
if [[ "${MODEL}" == gr00t_n1_6 ]]; then
    export VLA_GR00T_EMBODIMENT="${VLA_GR00T_EMBODIMENT:-libero_panda}"
    echo "[${MODEL}] VLA_GR00T_EMBODIMENT=${VLA_GR00T_EMBODIMENT}"
fi

for f in "${SERVER_ARGS[@]}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: model file not found: ${f}" >&2
        echo "       Check -i MODELS_ROOT (currently ${MODELS_ROOT})." >&2
        exit 1
    fi
done

# ---- memory sampler (Tegra-aware; copied from eval/run_libero.sh) -----------
sys_used_kib() {
    awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{ if(t&&a) print t-a; else print 0 }' \
        /proc/meminfo 2>/dev/null || echo 0
}

mem_sampler() {
    local pid="$1" out="$2" poll="${3:-1}"
    local peak_vram=0 peak_rss_kib=0 samples=0 vram_seen=0
    local peak_sys_kib=0 baseline_sys_kib=0
    local is_tegra=0
    [[ -f /etc/nv_tegra_release ]] && is_tegra=1
    export LC_ALL=C
    # Catch HUP too, so closing the SSH session / terminal still flushes a final
    # write rather than killing the sampler silently.
    trap 'stop=1' TERM INT HUP
    local stop=0
    baseline_sys_kib=$(sys_used_kib)
    peak_sys_kib=$baseline_sys_kib

    # Flush the running peak to disk atomically (temp + mv, so collect never reads
    # a half-written file). Called every poll AND once at exit, so the JSON exists
    # within ~1s of startup and survives an unclean stop (SIGKILL/SIGHUP/power loss).
    _emit() {
        local vram_json="null"
        if (( vram_seen )); then vram_json="$peak_vram"; fi
        local rss_mib peak_sys_mib base_sys_mib delta_sys_mib
        rss_mib=$(awk      -v k="$peak_rss_kib"     'BEGIN { printf "%.1f", k/1024.0 }')
        peak_sys_mib=$(awk -v k="$peak_sys_kib"     'BEGIN { printf "%.1f", k/1024.0 }')
        base_sys_mib=$(awk -v k="$baseline_sys_kib" 'BEGIN { printf "%.1f", k/1024.0 }')
        delta_sys_mib=$(awk -v p="$peak_sys_kib" -v b="$baseline_sys_kib" \
                            'BEGIN { d=p-b; if (d<0) d=0; printf "%.1f", d/1024.0 }')
        printf '{"pid": %d, "peak_vram_mib": %s, "peak_rss_mib": %s, "peak_sys_used_mib": %s, "baseline_sys_used_mib": %s, "sys_used_delta_mib": %s, "is_tegra": %d, "samples": %d}\n' \
            "$pid" "$vram_json" "$rss_mib" "$peak_sys_mib" "$base_sys_mib" "$delta_sys_mib" "$is_tegra" "$samples" > "${out}.tmp" \
            && mv -f "${out}.tmp" "$out"
    }

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
        _emit          # flush running peak every poll
        sleep "$poll"
    done
    _emit              # final flush
    echo "[mem-sampler] wrote $out  (rss_peak=$(awk -v k="$peak_rss_kib" 'BEGIN{printf "%.1f",k/1024}') MiB  sys_peak=$(awk -v k="$peak_sys_kib" 'BEGIN{printf "%.1f",k/1024}') MiB  tegra=${is_tegra}  samples=${samples})"
}

# ---- launch + supervise ----------------------------------------------------
SERVER_PID=""
SAMPLER_PID=""
TAIL_PID=""
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
    if [[ -n "${TAIL_PID}" ]] && kill -0 "${TAIL_PID}" 2>/dev/null; then
        kill -TERM "${TAIL_PID}" 2>/dev/null || true
        wait "${TAIL_PID}" 2>/dev/null || true
    fi
    TAIL_PID=""
}
trap cleanup EXIT INT TERM HUP

LOG="${LOG_DIR}/${MODEL}.log"
: > "${LOG}"
echo "===================="
echo "[${MODEL}] server args: ${SERVER_ARGS[*]}"
echo "[${MODEL}] log=${LOG}"

# Redirect to the log (clean $! = server PID); stream it to the terminal below.
"${SERVER_BIN}" --bind "${BIND_ADDR}" "${SERVER_ARGS[@]}" >"${LOG}" 2>&1 &
SERVER_PID=$!
echo "[server] pid=${SERVER_PID}"

MEM_OUT="${LOG%.log}.mem.json"
mem_sampler "${SERVER_PID}" "${MEM_OUT}" 1 &
SAMPLER_PID=$!
echo "[mem-sampler] pid=${SAMPLER_PID} out=${MEM_OUT}"

# Wait for the "ready" line (or the process dying).
ready=0
for _ in $(seq 1 600); do
    if grep -q "bound to .* ready" "${LOG}" 2>/dev/null; then
        echo "[server] ready - serving '${MODEL}' on ${BIND_ADDR}"
        echo "[server] point the client at it, e.g.:"
        echo "         bash eval/run_libero_client.sh -m ${MODEL} -a tcp://<this-host-ip>:${BIND_ADDR##*:}"
        ready=1
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: vla-server exited before becoming ready; see ${LOG}" >&2
        tail -n 40 "${LOG}" >&2 || true
        exit 1
    fi
    sleep 1
done
if [[ "${ready}" -eq 0 ]]; then
    echo "ERROR: vla-server did not become ready within 600s; see ${LOG}" >&2
    exit 1
fi

echo "[server] running. Press Ctrl-C to stop. Live log below:"
echo "===================="
# Stream the server log to the terminal until the server exits.
tail -n +1 -f "${LOG}" &
TAIL_PID=$!
wait "${SERVER_PID}"
