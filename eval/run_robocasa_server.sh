#!/usr/bin/env bash
# Copyright 2026 VinRobotics - Apache-2.0
# Serve one RoboCasa GGUF locally or to a remote simulation client.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL=smolvla
CHECKPOINT=""
BIND_ADDR="${BIND_ADDR:-tcp://*:5555}"
LOG_DIR="${REPO_ROOT}/outputs/robocasa/_server_logs"
SERVER_BIN="${SERVER_BIN:-${REPO_ROOT}/build/vla-server}"
DO_BUILD=1

usage() {
    cat <<EOF
Usage: bash eval/run_robocasa_server.sh [-m smolvla|gr00t_n1_6] [-c GGUF] [-a ADDR] [-o LOG_DIR] [-B] [-- SERVER_OPTIONS...]

  -m MODEL  Model architecture (default: smolvla)
  -c GGUF   Checkpoint; defaults to weights/smolvla_robocasa.gguf for SmolVLA.
            Required for GR00T; must contain the robocasa_panda_omron embodiment.
  -a ADDR   Bind URL, host:port, or port (default: tcp://*:5555)
  -o DIR    Server logs and memory JSON (default: outputs/robocasa/_server_logs)
  -B        Skip build; use SERVER_BIN (default: build/vla-server)

Env: SERVER_BIN, BIND_ADDR, SERVER_READY_TIMEOUT_S (default: 600).
GR00T always selects VLA_GR00T_EMBODIMENT=robocasa_panda_omron.
EOF
}
while getopts ":m:c:a:o:Bh" opt; do
    case "$opt" in
        m) MODEL="$OPTARG" ;;
        c) CHECKPOINT="$OPTARG" ;;
        a) BIND_ADDR="$OPTARG" ;;
        o) LOG_DIR="$OPTARG" ;;
        B) DO_BUILD=0 ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))
case "$MODEL" in
    smolvla) CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/weights/smolvla_robocasa.gguf}" ;;
    gr00t_n1_6)
        [[ -n "$CHECKPOINT" ]] || { echo "ERROR: GR00T requires -c GGUF" >&2; exit 1; }
        export VLA_GR00T_EMBODIMENT=robocasa_panda_omron ;;
    *) echo "ERROR: unsupported model: $MODEL" >&2; exit 1 ;;
esac
[[ -f "$CHECKPOINT" ]] || { echo "ERROR: checkpoint not found: $CHECKPOINT" >&2; exit 1; }
case "$BIND_ADDR" in
    tcp://*) ;;
    *:*) BIND_ADDR="tcp://$BIND_ADDR" ;;
    *[!0-9]*|"") echo "ERROR: invalid bind address: $BIND_ADDR" >&2; exit 1 ;;
    *) BIND_ADDR="tcp://*:$BIND_ADDR" ;;
esac
READY_TIMEOUT="${SERVER_READY_TIMEOUT_S:-600}"
[[ "$READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: invalid SERVER_READY_TIMEOUT_S" >&2; exit 1; }
if [[ "$DO_BUILD" == 1 ]]; then
    cmake --build "$(dirname "$SERVER_BIN")" --target vla-server -j"$(nproc)"
fi
[[ -x "$SERVER_BIN" ]] || { echo "ERROR: server not executable: $SERVER_BIN" >&2; exit 1; }
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${MODEL}.log"
MEM="${LOG_DIR}/${MODEL}.mem.json"

sys_used_kib() {
    awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{print t-a}' /proc/meminfo
}

sample_memory() {
    local pid="$1" out="$2" baseline peak_sys rss=0 vram=0 samples=0 tegra=0 seen=0 stop=0
    baseline=$(sys_used_kib)
    peak_sys=$baseline
    [[ ! -f /etc/nv_tegra_release ]] || tegra=1
    trap 'stop=1' TERM INT HUP
    while kill -0 "$pid" 2>/dev/null; do
        local current current_vram current_sys
        current=$(awk '/^VmHWM:/{print $2}' "/proc/$pid/status" 2>/dev/null || true)
        if [[ "$current" =~ ^[0-9]+$ ]] && ((current > rss)); then rss=$current; fi
        current_vram=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
            | awk -F, -v p="$pid" '$1==p {gsub(/ /,"",$2); print $2; exit}' || true)
        if [[ "$current_vram" =~ ^[0-9]+$ ]]; then
            seen=1
            if ((current_vram > vram)); then vram=$current_vram; fi
        fi
        current_sys=$(sys_used_kib)
        if ((current_sys > peak_sys)); then peak_sys=$current_sys; fi
        samples=$((samples + 1))
        awk -v pid="$pid" -v rss="$rss" -v vram="$vram" -v seen="$seen" \
            -v base="$baseline" -v peak="$peak_sys" -v tegra="$tegra" -v samples="$samples" \
            'BEGIN {delta=peak-base; if(delta<0)delta=0;
            printf "{\"pid\":%d,\"peak_rss_mib\":%.1f,\"peak_vram_mib\":%s,\"peak_sys_used_mib\":%.1f,\"baseline_sys_used_mib\":%.1f,\"sys_used_delta_mib\":%.1f,\"is_tegra\":%d,\"samples\":%d}\n", pid,rss/1024,seen?vram:"null",peak/1024,base/1024,delta/1024,tegra,samples}' \
            > "$out.tmp"
        mv -f "$out.tmp" "$out"
        ((stop == 0)) || break
        sleep 1
    done
}

SERVER_PID=""
SAMPLER_PID=""
TAIL_PID=""
cleanup() {
    trap - EXIT INT TERM HUP
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -INT "$SERVER_PID" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 0.25
        done
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        sleep 0.25
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    for pid in "$SAMPLER_PID" "$TAIL_PID"; do
        if [[ -n "$pid" ]]; then kill -TERM "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

echo "[server] model=$MODEL checkpoint=$CHECKPOINT bind=$BIND_ADDR log=$LOG"
"$SERVER_BIN" --bind "$BIND_ADDR" "$CHECKPOINT" "$@" > "$LOG" 2>&1 &
SERVER_PID=$!
sample_memory "$SERVER_PID" "$MEM" &
SAMPLER_PID=$!
ready=0
for ((i=0; i<READY_TIMEOUT; i++)); do
    if grep -q 'bound to .* ready' "$LOG"; then ready=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: vla-server exited before ready" >&2
        tail -n 40 "$LOG" >&2
        exit 1
    fi
    sleep 1
done
[[ "$ready" == 1 ]] || { echo "ERROR: server readiness timed out; see $LOG" >&2; exit 1; }
echo "[server] ready; run bash eval/run_robocasa_client.sh --arch $MODEL --vla-addr tcp://<server-host>:${BIND_ADDR##*:}"
tail -n +1 -f "$LOG" &
TAIL_PID=$!
wait "$SERVER_PID"
