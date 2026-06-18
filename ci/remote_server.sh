#!/usr/bin/env bash
# Copyright 2026 VinRobotics - Apache-2.0
#
# Runs ON the target server (the rtx3090 / orin / m4 host). Launches vla-server
# for ONE arch, bound on all interfaces so the orchestrator's client can reach
# it, with the sidecar mem sampler (Tegra-aware on orin; macOS variant on m4;
# discrete-GPU on rtx3090). Blocks on the server until killed (SIGINT/TERM) -
# run_remote.sh on the orchestrator starts it (nohup), waits for the port, drives
# the client, then kills it and scp's the logs back.
#
# This script lives in the repo, so the PR's code is what serves - provided
# run_remote.sh rsync'd the PR checkout here and (re)built first.
set -euo pipefail

CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CI_DIR}/.." && pwd)"
source "${CI_DIR}/lib/common.sh"
source "${CI_DIR}/config/matrix.env"

ARCH=""; MODELS_ROOT=""; SUITE="${DEFAULT_SUITE:-libero_object}"; BIND="5555"; LOG_DIR=""; PIDFILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)        ARCH="$2"; shift 2 ;;
        --models-root) MODELS_ROOT="$2"; shift 2 ;;
        --suite)       SUITE="$2"; shift 2 ;;
        --bind)        BIND="$2"; shift 2 ;;
        --logdir)      LOG_DIR="$2"; shift 2 ;;
        --pidfile)     PIDFILE="$2"; shift 2 ;;
        *) echo "ERROR: unknown arg $1" >&2; exit 1 ;;
    esac
done
[[ -n "$ARCH" && -n "$MODELS_ROOT" ]] || { echo "ERROR: --arch and --models-root required" >&2; exit 1; }

SERVER_BIN="${SERVER_BIN:-${REPO_ROOT}/build/vla-server}"
[[ -x "$SERVER_BIN" ]] || { echo "ERROR: ${SERVER_BIN} not built on this host" >&2; exit 1; }
BIND_ADDR="$(normalize_bind "$BIND")"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/ci/_server_logs}"; mkdir -p "$LOG_DIR"
# One server run per (arch, suite): tag the log so the orchestrator fetches and
# the gate aggregates each suite's run independently (per-suite weights mean a
# fresh process per suite for bitvla / gr00t_n1_7; single-suite archs => one).
LOG="${LOG_DIR}/${ARCH}.${SUITE}.log"; : > "$LOG"

# Portable array read (no `mapfile`: macOS ships bash 3.2, which lacks it).
SARGS=()
while IFS= read -r _line; do SARGS+=("$_line"); done < <(server_args_for "$ARCH" "$MODELS_ROOT" "$SUITE")
for f in "${SARGS[@]}"; do [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }; done
apply_gr00t_env "$ARCH"
apply_openvla_env "$ARCH" "$SUITE"

SERVER_PID=""; SAMPLER_PID=""
cleanup() {
    [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null && {
        kill -INT "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.5; done
        kill -KILL "$SERVER_PID" 2>/dev/null || true; }
    [[ -n "$SAMPLER_PID" ]] && kill -TERM "$SAMPLER_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM HUP

echo "[remote] serving ${ARCH} on ${BIND_ADDR}; log=${LOG}"
"$SERVER_BIN" --bind "$BIND_ADDR" "${SARGS[@]}" >"$LOG" 2>&1 &
SERVER_PID=$!
[[ -n "$PIDFILE" ]] && echo "$SERVER_PID" > "$PIDFILE"
mem_sampler "$SERVER_PID" "${LOG%.log}.mem.json" 1 & SAMPLER_PID=$!
wait_ready_log "$LOG" "$SERVER_PID" 600
echo "[remote] ready - blocking until stopped"
wait "$SERVER_PID"
