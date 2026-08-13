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

# Serial replacement for run_latency_compare_all.sh.
#
# The two-lane driver deals models across both GPUs and runs them concurrently.
# That is faster in wall-clock but it means every number is measured with a
# second full sweep — another server, another LIBERO sim — competing for the
# same 20 cores. For the PyTorch rows that matters a great deal: their
# host-side preprocessing is 10-70 ms of Python per call and it absorbs CPU
# contention directly (see docs/corl-paper/experiments/compile_compare.md).
#
# This driver runs exactly one model, one variant, one GPU at a time. Nothing
# else touches the machine, so the numbers are comparable across rows.
#
# Per-model env below reproduces the "fastest measured" configuration each row
# is quoted at, not the shipping default.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU="${GPU:-0}"
N_STEPS="${N_STEPS:-300}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/latency_postmerge}"
MODELS="${MODELS:-smolvla evo1 pi0 gr00t_n1_5 gr00t_n1_7 gr00t_n1_6 bitvla}"

mkdir -p "${OUT_ROOT}"
DRIVER_LOG="${OUT_ROOT}/serial.log"
: > "${DRIVER_LOG}"

say() { echo "$*" | tee -a "${DRIVER_LOG}"; }

say "=== serial latency sweep: GPU ${GPU}, n=${N_STEPS}, $(date +%F_%H:%M:%S) ==="
say "=== build: $(git -C "${REPO_ROOT}" rev-parse --short HEAD) ==="

for m in ${MODELS}; do
    # Switches that the report's headline figures were measured with. The
    # graph cache is on by default now, so it is not listed.
    env_args=()
    case "${m}" in
        smolvla) server_flags=(--flash-attn --mm-prec default) ;;
        pi0)     server_flags=(--act-dtype bf16 --flash-attn) ;;
        evo1)    server_flags=(--act-dtype bf16 --flash-attn) ;;
        *)       env_args=() ;;
    esac
    # bitvla has no PyTorch side on this harness (separate venv, see
    # run_bitvla_compile_compare.sh), so ask only for the vla.cpp variant.
    variants="vla.cpp,eager,compile-default,compile-reduce-overhead"
    [[ "${m}" == bitvla ]] && variants="vla.cpp"

    say ""
    say "--- ${m} start $(date +%H:%M:%S)  [${env_args[*]:-no switches}] ---"
    env "${env_args[@]}" \
        bash "${SCRIPT_DIR}/run_latency_compare.sh" \
            -m "${m}" -g "${GPU}" -n "${N_STEPS}" -v "${variants}" -o "${OUT_ROOT}" \
        >"${OUT_ROOT}/_${m}.log" 2>&1
    say "--- ${m} done rc=$? $(date +%H:%M:%S) ---"
    grep -E 'server ms' "${OUT_ROOT}/_${m}.log" | tee -a "${DRIVER_LOG}"
done

say ""
say "=== ALL DONE $(date +%F_%H:%M:%S) ==="
