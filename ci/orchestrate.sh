#!/usr/bin/env bash
# Copyright 2026 VinRobotics - Apache-2.0
#
# CI entrypoint for one platform: run the sweep, then gate it against the
# committed baseline. Exit code = the gate's (0 pass / 1 fail).
#
#   bash ci/orchestrate.sh rtx3060|orin|m4
set -euo pipefail

PLATFORM="${1:?usage: orchestrate.sh <rtx3060|orin|m4>}"
CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CI_DIR}/.." && pwd)"
# hosts.env (gitignored) holds the real hostnames/IPs/paths. It is required -
# the committed hosts.env.example is a placeholder-only template.
if [[ ! -f "${CI_DIR}/config/hosts.env" ]]; then
    echo "ERROR: ci/config/hosts.env not found - copy ci/config/hosts.env.example to it and fill in your values." >&2
    exit 1
fi
source "${CI_DIR}/config/hosts.env"

# Decoupled topology: every platform is a remote server (server on the target,
# client on the orchestrator). All three go through run_remote.sh.
case "${PLATFORM}" in
    rtx3060|orin|m4) bash "${CI_DIR}/run_remote.sh" "${PLATFORM}" ;;
    *) echo "ERROR: platform must be rtx3060|orin|m4 (got ${PLATFORM})" >&2; exit 1 ;;
esac

SWEEP="${CI_OUTPUT_ROOT}/${PLATFORM}"
BASELINE="${CI_DIR}/baselines/${PLATFORM}.json"
echo "==================== gate [${PLATFORM}]"
PYBIN="${LIBERO_VENV_PY:-python3}"
[[ -x "${PYBIN}" ]] || PYBIN="python3"
"${PYBIN}" "${CI_DIR}/check_thresholds.py" \
    --platform "${PLATFORM}" --sweep "${SWEEP}" --baseline "${BASELINE}"
