#!/usr/bin/env bash
# Copyright 2026 VinRobotics - Apache-2.0
#
# Bit-exactness harness for the src/models -> layer/module/model refactor.
#
# Runs tests/predict_check over every arch's real GGUF with fixed images /
# language / state / noise and writes one action-chunk file per arch. Capture a
# baseline before touching the code, then re-run after each step and diff:
#
#   eval/refactor_verify.sh outputs/refactor/base
#   ...refactor...
#   eval/refactor_verify.sh outputs/refactor/new
#   diff -r outputs/refactor/base outputs/refactor/new && echo BIT-EXACT
#
# ARCHS=... restricts the sweep to a subset (space separated, names below).
# The square input side is probed rather than hardcoded: predict_check defaults
# to 224 and an arch whose tower wants another side returns action_len=0 on a
# mismatch instead of failing, so a wrong side would silently "pass" a diff.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN="${BIN:-${REPO_ROOT}/build/tests/vla_predict_check}"
HF="${HF:-/mnt/data/hf_data/vrfai}"
OUT="${1:-${REPO_ROOT}/outputs/refactor/baseline}"
SIDES="${SIDES:-224 256 448 512}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# arch|ckpt|mmproj|n_images|extra env
MODELS=(
    "smolvla|${HF}/smolvla-libero-gguf/smolvla-libero.gguf|${HF}/backup/mmproj-smolvla-libero.gguf|2|"
    "pi0|${HF}/pi0-libero-finetuned-v044-gguf/pi0-libero-finetuned-v044.gguf|${HF}/backup/mmproj-pi0-libero-finetuned-v044.gguf|2|"
    "pi05|${HF}/pi05-libero-gguf/pi05-libero.gguf|${HF}/backup/mmproj-pi05-libero.gguf|2|"
    "evo1|${HF}/evo1-libero-gguf/evo1-libero.gguf||2|"
    "gr00t_n1_5|${HF}/gr00tn1d5-libero-object-gguf/gr00tn1d5-libero-object.gguf||2|"
    "gr00t_n1_6|${HF}/gr00tn1d6-libero-gguf/gr00tn1d6-libero.gguf||2|"
    "gr00t_n1_7|${HF}/gr00tn1d7-libero-gguf/libero_object/gr00tn1d7-libero-object.gguf||2|"
    "bitvla|${HF}/bitvla-libero-gguf/libero_object/bitvla-libero-object.gguf||2|"
    "vla_adapter|${HF}/vla-adapter-libero-object-gguf/libero_object/vla-adapter-libero-object.gguf||2|"
    "openvla_oft|${HF}/openvla-oft-libero-gguf/openvla-oft-libero.gguf||2|"
    "vla_jepa|${HF}/vla-jepa-libero/vla-jepa.gguf||2|VLA_EXTRA_TOKEN=151697 VLA_EXTRA_COUNT=32"
)

[[ -x "${BIN}" ]] || { echo "ERROR: missing ${BIN} (cmake -DVLA_BUILD_TESTS=ON)" >&2; exit 1; }
mkdir -p "${OUT}"

fail=0
for row in "${MODELS[@]}"; do
    IFS='|' read -r arch ckpt mmproj nimg extra <<< "${row}"

    if [[ -n "${ARCHS:-}" && " ${ARCHS} " != *" ${arch} "* ]]; then
        continue
    fi
    if [[ ! -e "${ckpt}" ]]; then
        echo "[skip] ${arch}: no checkpoint at ${ckpt}"
        continue
    fi

    ok=0
    for side in ${SIDES}; do
        # shellcheck disable=SC2086
        if env ${extra} VLA_IMG_SIZE="${side}" "${BIN}" "${ckpt}" "${mmproj}" "${nimg}" \
               > "${OUT}/${arch}.actions.txt" 2> "${OUT}/${arch}.log"; then
            if ! grep -q '^action_len=0$' "${OUT}/${arch}.actions.txt"; then
                echo "[ok  ] ${arch}  side=${side}  $(head -1 "${OUT}/${arch}.actions.txt")"
                echo "${side}" > "${OUT}/${arch}.side"
                ok=1
                break
            fi
        fi
    done

    if [[ "${ok}" -eq 0 ]]; then
        echo "[FAIL] ${arch}: no input side in '${SIDES}' produced a chunk; see ${OUT}/${arch}.log" >&2
        fail=1
    fi
done

echo
echo "actions written to ${OUT}"
exit "${fail}"
