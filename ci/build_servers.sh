#!/usr/bin/env bash
# Copyright 2026 VinRobotics - Apache-2.0
#
# Build vla-server on the platform servers via the control agent (vla-ci-ctl
# build), in parallel. A convenience for the self-managed model: it builds each
# server's OWN checkout with that platform's CMake flags (from hosts.env) and does
# NOT change the server's commit. Run it after the servers are at the target
# commit, before `ci/orchestrate.sh all`.
#
#   bash ci/build_servers.sh                 # all ALL_PLATFORMS, in parallel
#   bash ci/build_servers.sh rtx3090 orin    # a subset
set -euo pipefail

CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CI_DIR}/.." && pwd)"
source "${CI_DIR}/lib/common.sh"
source "${CI_DIR}/config/matrix.env"
[[ -f "${CI_DIR}/config/hosts.env" ]] && source "${CI_DIR}/config/hosts.env" \
    || { echo "ERROR: ci/config/hosts.env not found (copy from hosts.env.example and fill it)" >&2; exit 1; }

CTL="${VLA_CI_CTL:-${CI_DIR}/agent/build/vla-ci-ctl}"
[[ -x "${CTL}" ]] || { echo "ERROR: vla-ci-ctl not built at ${CTL}" >&2; exit 1; }
export VLA_CI_TOKEN="${VLA_CI_TOKEN:-}"

PLATFORMS="${*:-${ALL_PLATFORMS}}"
mkdir -p "${CI_OUTPUT_ROOT}"

declare -A PID
for p in ${PLATFORMS}; do
    resolve_platform "${p}" || exit 1
    ep="tcp://${SRV_HOST}:${CTRL_PORT}"
    echo "[build] ${p} -> ${ep}  cwd=${RROOT}  flags=[${CMAKE_FLAGS:-<Metal auto>}]  log=${CI_OUTPUT_ROOT}/${p}.build.log"
    ( "${CTL}" --endpoint "${ep}" build --cwd "${RROOT}" --flags "${CMAKE_FLAGS}" ) \
        >"${CI_OUTPUT_ROOT}/${p}.build.log" 2>&1 &
    PID[$p]=$!
done

fail=0
for p in ${PLATFORMS}; do
    if wait "${PID[$p]}"; then echo "[build] ${p}: OK"
    else echo "[build] ${p}: FAILED - see ${CI_OUTPUT_ROOT}/${p}.build.log" >&2; fail=1; fi
done
echo "==================== build: $([[ ${fail} -eq 0 ]] && echo OK || echo FAIL)"
exit "${fail}"
