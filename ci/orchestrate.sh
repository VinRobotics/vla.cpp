#!/usr/bin/env bash
# Copyright 2026 VinRobotics - Apache-2.0
#
# CI entrypoint. Verify commit consistency across the tested machines, then run
# the sweep + gate for one platform, OR for every platform concurrently with
# `all`. Exit 0 only if the commit check passes (servers agree) AND all gates pass.
#
#   bash ci/orchestrate.sh rtx3090|orin|m4   # one platform
#   bash ci/orchestrate.sh all               # all platforms in parallel
set -euo pipefail

PLATFORM="${1:?usage: orchestrate.sh <rtx3090|orin|m4|all>}"
CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CI_DIR}/.." && pwd)"
source "${CI_DIR}/config/matrix.env"
if [[ ! -f "${CI_DIR}/config/hosts.env" ]]; then
    echo "ERROR: ci/config/hosts.env not found - copy ci/config/hosts.env.example to it and fill in your values." >&2
    exit 1
fi
source "${CI_DIR}/config/hosts.env"

pybin() { local p="${LIBERO_VENV_PY:-python3}"; [[ -x "$p" ]] || p="python3"; echo "$p"; }

# Run one platform's sweep (server self-managed) then gate it against its baseline.
sweep_and_gate() {
    local p="$1"
    bash "${CI_DIR}/run_remote.sh" "${p}"
    echo "==================== gate [${p}]"
    "$(pybin)" "${CI_DIR}/check_thresholds.py" \
        --platform "${p}" --sweep "${CI_OUTPUT_ROOT}/${p}" --baseline "${CI_DIR}/baselines/${p}.json"
}

# ── all: verify commits, sweep every platform concurrently, aggregate ───────
# One LIBERO client per platform runs on the orchestrator at once. Gated metrics
# are server-side, so orchestrator load cannot bias them.
if [[ "${PLATFORM}" == "all" ]]; then
    # Commit consistency across the tested machines; stops here if they disagree.
    bash "${CI_DIR}/check_commits.sh" || { echo "[all] commit check failed - aborting before sim." >&2; exit 1; }

    mkdir -p "${CI_OUTPUT_ROOT}"
    declare -A PID
    for p in ${ALL_PLATFORMS}; do
        ( sweep_and_gate "${p}" ) >"${CI_OUTPUT_ROOT}/${p}.run.log" 2>&1 &
        PID[$p]=$!
        echo "[all] launched ${p} (pid ${PID[$p]}) -> ${CI_OUTPUT_ROOT}/${p}.run.log"
    done
    fail=0
    for p in ${ALL_PLATFORMS}; do
        if wait "${PID[$p]}"; then echo "[all] ${p}: PASS"; else echo "[all] ${p}: FAIL"; fail=1; fi
    done
    # Aggregate per-platform server-side metrics + verdicts into one report.
    "$(pybin)" "${CI_DIR}/aggregate_report.py" --root "${CI_OUTPUT_ROOT}" \
        --platforms "${ALL_PLATFORMS}" \
        --commit "$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)" || true
    echo "==================== all platforms: $([[ ${fail} -eq 0 ]] && echo PASS || echo FAIL)"
    exit "${fail}"
fi

# ── one platform ────────────────────────────────────────────────────────────
case "${PLATFORM}" in
    rtx3090|orin|m4)
        bash "${CI_DIR}/check_commits.sh" "${PLATFORM}" \
            || { echo "[${PLATFORM}] commit check failed - aborting before sim." >&2; exit 1; }
        sweep_and_gate "${PLATFORM}" ;;
    *) echo "ERROR: platform must be rtx3090|orin|m4|all (got ${PLATFORM})" >&2; exit 1 ;;
esac
