#!/usr/bin/env bash
# Copyright 2026 VinRobotics - Apache-2.0
#
# Runs ON the orchestrator for a platform (rtx3060 | orin | m4). Talks to the
# platform server's vla-ci-agent (LAN control plane, NO SSH) via vla-ci-ctl:
# uploads the PR checkout (put), (re)builds vla-server there (exec), then per
# model spawns remote_server.sh, drives the LIBERO client locally, stops the
# server, and fetches (get) the server log + mem.json into the local sweep dir
# so check_thresholds.py sees one consistent tree.
#
# Control plane = vla-ci-ctl <-> vla-ci-agent over tcp://<SERVER_HOST>:<CTRL_PORT>.
# Data plane    = LIBERO client <-> vla-server over tcp://<SERVER_HOST>:<SERVER_PORT>.
# Both ride the LAN IP (SERVER_HOST); only the ports differ.
#
#   bash ci/run_remote.sh rtx3060|orin|m4
set -euo pipefail

PLATFORM="${1:?usage: run_remote.sh <rtx3060|orin|m4>}"
CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CI_DIR}/.." && pwd)"
source "${CI_DIR}/lib/common.sh"
source "${CI_DIR}/config/matrix.env"
[[ -f "${CI_DIR}/config/hosts.env" ]] && source "${CI_DIR}/config/hosts.env" \
    || { echo "ERROR: ci/config/hosts.env not found (copy from hosts.env.example and fill it)" >&2; exit 1; }

case "${PLATFORM}" in
    rtx3060) RROOT="${RTX_REPO_ROOT}"; RMODELS="${RTX_MODELS_ROOT}"
             SRV_HOST="${RTX_SERVER_HOST}"; SRV_PORT="${RTX_SERVER_PORT}"
             CTRL_PORT="${RTX_CTRL_PORT}"; CMAKE_FLAGS="${RTX_CMAKE_FLAGS:-}" ;;
    orin)    RROOT="${ORIN_REPO_ROOT}"; RMODELS="${ORIN_MODELS_ROOT}"
             SRV_HOST="${ORIN_SERVER_HOST}"; SRV_PORT="${ORIN_SERVER_PORT}"
             CTRL_PORT="${ORIN_CTRL_PORT}"; CMAKE_FLAGS="${ORIN_CMAKE_FLAGS:-}" ;;
    m4)      RROOT="${M4_REPO_ROOT}"; RMODELS="${M4_MODELS_ROOT}"
             SRV_HOST="${M4_SERVER_HOST}"; SRV_PORT="${M4_SERVER_PORT}"
             CTRL_PORT="${M4_CTRL_PORT}"; CMAKE_FLAGS="${M4_CMAKE_FLAGS:-}" ;;
    *) echo "ERROR: platform must be rtx3060|orin|m4 (got ${PLATFORM})" >&2; exit 1 ;;
esac

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

# Excluded from the upload AND protected from prune on the server: generated /
# vendored / sim-only trees the server neither receives nor should lose.
PROTECT=(build outputs third_party
         eval/sim/libero/libero_uv eval/sim/libero/LIBERO
         eval/sim/simpler/simpler_uv eval/sim/simpler/SimplerEnv)
protect_args=(); for p in "${PROTECT[@]}"; do protect_args+=(--protect "${p}"); done

[[ -x "${LIBERO_VENV_PY}" ]] || { echo "ERROR: LIBERO venv missing at ${LIBERO_VENV_PY}" >&2; exit 1; }

# ── 1. reach the agent, push PR code, build on the target (once) ─────────────
echo "[${PLATFORM}] agent ping ${ENDPOINT}"
ctl ping
echo "[${PLATFORM}] put PR checkout -> ${RROOT}"
ctl put --src "${REPO_ROOT}" --dst "${RROOT}" --prune "${protect_args[@]}"

if [[ "${SKIP_REMOTE_BUILD:-0}" -ne 1 ]]; then
    echo "[${PLATFORM}] build vla-server  (flags: ${CMAKE_FLAGS:-<none, Metal auto>})"
    # patch.sh fetches/patches third_party/llama.cpp at the pinned tag. CUDA flags
    # are per-platform (sm_86 rtx3060 / sm_87 orin); macOS needs none (Metal auto).
    ctl exec --cwd "${RROOT}" -- bash -lc "set -e; bash patches/patch.sh; \
        cmake -B build -DCMAKE_BUILD_TYPE=Release ${CMAKE_FLAGS}; \
        cmake --build build -j\$(getconf _NPROCESSORS_ONLN)"
fi

# ── 2. per-model: serve remote, client local, fetch logs ────────────────────
SERVER_NAME="vla-${PLATFORM}"
stop_remote() { ctl stop --name "${SERVER_NAME}" >/dev/null 2>&1 || true; }
trap 'stop_remote' EXIT INT TERM

run_model() {
    local arch="$1"
    echo "==================== [${PLATFORM}/${arch}]"
    stop_remote; sleep 1
    # Spawn the server detached on the target; the agent owns its lifecycle.
    ctl spawn --name "${SERVER_NAME}" --cwd "${RROOT}" --log "${RLOG_DIR}/${arch}.launch.log" \
        -- bash ci/remote_server.sh --arch "${arch}" --models-root "${RMODELS}" \
            --bind "${SRV_PORT}" --logdir "${RLOG_DIR}"
    wait_ready_tcp "${SRV_HOST}" "${SRV_PORT}" 600

    local stats; stats="$(stats_json_for "$arch" "${ORCH_MODELS_ROOT}")"  # client-side, on the orchestrator
    local extra=(); [[ -n "$stats" && -f "$stats" ]] && extra+=(--stats-json "$stats")
    local nact; nact="$(nact_for "$arch")"
    local out_dir="${OUT}/${arch}"; mkdir -p "${out_dir}"

    for suite in $(suites_for "${PLATFORM}" "$arch"); do
        for task_id in $(seq 0 9); do
            echo "[${PLATFORM}/${arch}] suite=${suite} task=${task_id}"
            "${LIBERO_VENV_PY}" "${CLIENT}" \
                --arch "${arch}" --vla-addr "${CLIENT_ADDR}" \
                --task "${suite}" --task-id "${task_id}" \
                --n-episodes "${N_EPISODES}" --n-action-steps "${nact}" \
                --output-dir "${out_dir}" "${extra[@]}"
        done
    done

    stop_remote; sleep 2
    echo "[${PLATFORM}/${arch}] fetch server log + mem.json"
    ctl get --remote "${RLOG_DIR}/${arch}.log"      --local "${LOG_DIR}/${arch}.log"      || echo "WARN: no ${arch}.log fetched" >&2
    ctl get --remote "${RLOG_DIR}/${arch}.mem.json" --local "${LOG_DIR}/${arch}.mem.json" || echo "WARN: no ${arch}.mem.json fetched" >&2
    echo "[${PLATFORM}/${arch}] done"
}

MODELS="${MODELS_OVERRIDE:-$(models_for "${PLATFORM}")}"
echo "[config] ${PLATFORM} models: ${MODELS}  control=${ENDPOINT}  data=${CLIENT_ADDR}"
for m in ${MODELS}; do run_model "$m"; done
echo "==================== ${PLATFORM} sweep done -> ${OUT}"
