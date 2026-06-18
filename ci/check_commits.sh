#!/usr/bin/env bash
# Copyright 2026 VinRobotics - Apache-2.0
#
# Verify git-commit consistency across the tested platforms BEFORE running any
# sim. Each server self-manages its checkout; the agent's `rev` reports that
# server's real `git rev-parse HEAD`. The orchestrator's own commit need NOT
# match the servers. Rules (over the TESTED machines = the platform servers):
#
#   all four equal (orchestrator + servers)  -> INFO
#   servers equal, orchestrator differs      -> WARNING (proceed)
#   servers disagree (or a rev is unreadable) -> ERROR, exit 2 (stop the CI)
#
#   bash ci/check_commits.sh             # all ALL_PLATFORMS (the CI default)
#   bash ci/check_commits.sh rtx3090     # a subset (single server => never "differ")
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
ORCH_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
mkdir -p "${CI_OUTPUT_ROOT}"

declare -A COMMIT
unreadable=0 servers_same=1 first=""
json="{\"orchestrator\":\"${ORCH_COMMIT}\",\"servers\":{" ; sep=""
for p in ${PLATFORMS}; do
    resolve_platform "$p" || exit 2
    ep="tcp://${SRV_HOST}:${CTRL_PORT}"
    if c="$("${CTL}" --endpoint "${ep}" rev --path "${RROOT}" 2>/dev/null)"; then
        COMMIT[$p]="$c"
        mkdir -p "${CI_OUTPUT_ROOT}/${p}"; echo "$c" > "${CI_OUTPUT_ROOT}/${p}/commit.txt"
        [[ -z "$first" ]] && first="$c"
        [[ "$c" != "$first" ]] && servers_same=0
        echo "[commit] ${p}: ${c}"
        json="${json}${sep}\"${p}\":\"${c}\""
    else
        echo "ERROR: [${p}] could not read commit via ${ep} (agent down, or ${RROOT} is not a git repo)" >&2
        unreadable=1
        json="${json}${sep}\"${p}\":null"
    fi
    sep=","
done
json="${json}}}"
printf '%s\n' "${json}" > "${CI_OUTPUT_ROOT}/commits.json"
echo "[commit] orchestrator: ${ORCH_COMMIT}"

if [[ ${unreadable} -ne 0 ]]; then
    echo "ERROR: could not read the commit on every tested machine; stopping CI." >&2
    exit 2
fi
if [[ ${servers_same} -eq 0 ]]; then
    echo "ERROR: tested machines are at DIFFERENT commits; stopping CI:" >&2
    for p in ${PLATFORMS}; do echo "  ${p}: ${COMMIT[$p]}" >&2; done
    exit 2
fi
if [[ "${first}" == "${ORCH_COMMIT}" ]]; then
    echo "INFO: orchestrator and all tested machines are at the same commit (${first})."
else
    echo "WARNING: orchestrator (${ORCH_COMMIT}) differs from the tested machines (${first}); proceeding." >&2
fi
exit 0
