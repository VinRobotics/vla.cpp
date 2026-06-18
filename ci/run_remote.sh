#!/usr/bin/env bash
# Copyright 2026 VinRobotics - Apache-2.0
#
# Runs ON the orchestrator for a platform (rtx3090 | orin | m4). The server is
# SELF-MANAGED: each machine keeps its own git checkout and has already built
# vla-server at its commit (commit consistency is verified up front by
# ci/check_commits.sh). This script does NOT push code or build. Per model/suite
# it talks to the platform's vla-ci-agent (LAN, no SSH) via vla-ci-ctl: spawns
# remote_server.sh, drives the LIBERO client locally, stops the server, and
# fetches (get) the server log + mem.json into the local sweep dir so the gate
# sees one consistent tree.
#
# Control plane = vla-ci-ctl <-> vla-ci-agent over tcp://<SERVER_HOST>:<CTRL_PORT>.
# Data plane    = LIBERO client <-> vla-server over tcp://<SERVER_HOST>:<DATA_PORT>.
#
#   bash ci/run_remote.sh rtx3090|orin|m4
set -euo pipefail

PLATFORM="${1:?usage: run_remote.sh <rtx3090|orin|m4>}"
CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CI_DIR}/.." && pwd)"
source "${CI_DIR}/lib/common.sh"
source "${CI_DIR}/config/matrix.env"
[[ -f "${CI_DIR}/config/hosts.env" ]] && source "${CI_DIR}/config/hosts.env" \
    || { echo "ERROR: ci/config/hosts.env not found (copy from hosts.env.example and fill it)" >&2; exit 1; }

resolve_platform "${PLATFORM}" || exit 1

CTL="${VLA_CI_CTL:-${CI_DIR}/agent/build/vla-ci-ctl}"
[[ -x "${CTL}" ]] || { echo "ERROR: vla-ci-ctl not built at ${CTL}. Build it: cmake -S ci/agent -B ci/agent/build && cmake --build ci/agent/build" >&2; exit 1; }
export VLA_CI_TOKEN="${VLA_CI_TOKEN:-}"   # picked up by vla-ci-ctl; empty = no auth
ENDPOINT="tcp://${SRV_HOST}:${CTRL_PORT}"
ctl() { "${CTL}" --endpoint "${ENDPOINT}" "$@"; }

CLIENT="${REPO_ROOT}/eval/client/run_sim_client_direct.py"
CLIENT_ADDR="tcp://${SRV_HOST}:${SRV_PORT}"
OUT="${CI_OUTPUT_ROOT}/${PLATFORM}"
LOG_DIR="${OUT}/_server_logs"; mkdir -p "${LOG_DIR}"
RLOG_DIR="${RROOT}/outputs/ci/_server_logs"

[[ -x "${LIBERO_VENV_PY}" ]] || { echo "ERROR: LIBERO venv missing at ${LIBERO_VENV_PY}" >&2; exit 1; }

echo "[${PLATFORM}] agent ping ${ENDPOINT}"
ctl ping

# ── per-model: serve remote, client local, fetch logs ───────────────────────
SERVER_NAME="vla-${PLATFORM}"
stop_remote() { ctl stop --name "${SERVER_NAME}" >/dev/null 2>&1 || true; }
trap 'stop_remote' EXIT INT TERM

run_model() {
    local arch="$1"
    echo "==================== [${PLATFORM}/${arch}]"
    local nact; nact="$(nact_for "$arch")"
    local out_dir="${OUT}/${arch}"; mkdir -p "${out_dir}"

    # One server process per suite: bitvla / gr00t_n1_7 carry per-suite weights,
    # so the checkpoint (and gr00t stats) must be selected per suite. Single-suite
    # archs iterate once, so this is identical to a per-model spawn for them.
    for suite in $(suites_for "${PLATFORM}" "$arch"); do
        echo "-------------------- [${PLATFORM}/${arch}] suite=${suite}"
        stop_remote; sleep 1
        # Spawn the server detached on the target; the agent owns its lifecycle.
        ctl spawn --name "${SERVER_NAME}" --cwd "${RROOT}" --log "${RLOG_DIR}/${arch}.${suite}.launch.log" \
            -- bash ci/remote_server.sh --arch "${arch}" --suite "${suite}" --models-root "${RMODELS}" \
                --bind "${SRV_PORT}" --logdir "${RLOG_DIR}"
        wait_ready_tcp "${SRV_HOST}" "${SRV_PORT}" 600

        local stats; stats="$(stats_json_for "$arch" "${ORCH_MODELS_ROOT}" "$suite")"  # client-side, on the orchestrator
        local extra=(); [[ -n "$stats" && -f "$stats" ]] && extra+=(--stats-json "$stats")
        # Some archs (gr00t_n1_6) vendor their tokenizer in the model dir; point the client at it.
        local tok; tok="$(tokenizer_for "$arch" "${ORCH_MODELS_ROOT}")"
        [[ -n "$tok" && -d "$tok" ]] && extra+=(--tokenizer "$tok")

        for task_id in $(seq 0 9); do
            echo "[${PLATFORM}/${arch}] suite=${suite} task=${task_id}"
            "${LIBERO_VENV_PY}" "${CLIENT}" \
                --arch "${arch}" --vla-addr "${CLIENT_ADDR}" \
                --task "${suite}" --task-id "${task_id}" \
                --n-episodes "${N_EPISODES}" --n-action-steps "${nact}" \
                --output-dir "${out_dir}" "${extra[@]}"
        done

        stop_remote; sleep 2
        echo "[${PLATFORM}/${arch}] fetch server log + mem.json (${suite})"
        ctl get --remote "${RLOG_DIR}/${arch}.${suite}.log"      --local "${LOG_DIR}/${arch}.${suite}.log"      || echo "WARN: no ${arch}.${suite}.log fetched" >&2
        ctl get --remote "${RLOG_DIR}/${arch}.${suite}.mem.json" --local "${LOG_DIR}/${arch}.${suite}.mem.json" || echo "WARN: no ${arch}.${suite}.mem.json fetched" >&2
    done
    echo "[${PLATFORM}/${arch}] done"
}

MODELS="${MODELS_OVERRIDE:-$(models_for "${PLATFORM}")}"
echo "[config] ${PLATFORM} models: ${MODELS}  control=${ENDPOINT}  data=${CLIENT_ADDR}"
for m in ${MODELS}; do run_model "$m"; done
echo "==================== ${PLATFORM} sweep done -> ${OUT}"
