#!/usr/bin/env bash
# Copyright 2026 VinRobotics - Apache-2.0
#
# Bit-exactness and latency harness for the src/ layer/module/model refactor.
#
#   eval/refactor_verify.sh <outdir>            actions only
#   BENCH=20 eval/refactor_verify.sh <outdir>   actions + predict() timing
#
# Each arch runs twice: at its shipping defaults, and under the alternate
# precision. Both must stay byte-identical across a refactor, and neither may
# regress in latency.
#
#   eval/refactor_verify.sh outputs/refactor/before
#   ...change...
#   cmake --build build -j"$(nproc)" --target vla_predict_check
#   eval/refactor_verify.sh outputs/refactor/after
#   diff -r outputs/refactor/before outputs/refactor/after
#
# Never rebuild while a sweep is running: relinking libvla_core.so under it
# makes every remaining arch fail to load.
#
# ARCHS=... restricts the sweep. The square input side is probed rather than
# hardcoded, because a tower fed the wrong side returns action_len=0 instead of
# failing, and a wrong side would silently "pass" a diff.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN="${BIN:-${REPO_ROOT}/build/tests/vla_predict_check}"
HF="${HF:-/mnt/data/hf_data/vrfai}"
OUT="${1:-${REPO_ROOT}/outputs/refactor/baseline}"
SIDES="${SIDES:-224 256 448 512}"
BENCH="${BENCH:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# arch|ckpt|mmproj|n_images|env|alternate-config CLI flags.
# openvla_oft has no alternate: at f32 its weights need 30 GB.
MODELS=(
    "smolvla|${HF}/smolvla-libero-gguf/smolvla-libero.gguf|${HF}/backup/mmproj-smolvla-libero.gguf|2||--flash-attn --mm-prec default"
    "pi0|${HF}/pi0-libero-finetuned-v044-gguf/pi0-libero-finetuned-v044.gguf|${HF}/backup/mmproj-pi0-libero-finetuned-v044.gguf|2||--act-dtype bf16 --flash-attn"
    "pi05|${HF}/pi05-libero-gguf/pi05-libero.gguf|${HF}/backup/mmproj-pi05-libero.gguf|2||--weight-dtype f32"
    "evo1|${HF}/evo1-libero-gguf/evo1-libero.gguf||2||--act-dtype bf16 --flash-attn"
    "gr00t_n1_5|${HF}/gr00tn1d5-libero-object-gguf/gr00tn1d5-libero-object.gguf||2||--weight-dtype f32"
    "gr00t_n1_6|${HF}/gr00tn1d6-libero-gguf/gr00tn1d6-libero.gguf||2||--weight-dtype f32"
    "gr00t_n1_7|${HF}/gr00tn1d7-libero-gguf/libero_object/gr00tn1d7-libero-object.gguf||2||--weight-dtype f32"
    "bitvla|${HF}/bitvla-libero-gguf/libero_object/bitvla-libero-object.gguf||2||--weight-dtype bf16"
    "vla_adapter|${HF}/vla-adapter-libero-object-gguf/libero_object/vla-adapter-libero-object.gguf||2||--weight-dtype f32"
    "openvla_oft|${HF}/openvla-oft-libero-gguf/openvla-oft-libero.gguf||2||"
    "vla_jepa|${HF}/vla-jepa-libero/vla-jepa.gguf||2|VLA_EXTRA_TOKEN=151697 VLA_EXTRA_COUNT=32|--weight-dtype f32"
)

[[ -x "${BIN}" ]] || { echo "ERROR: missing ${BIN} (cmake -DVLA_BUILD_TESTS=ON)" >&2; exit 1; }
mkdir -p "${OUT}"

run_one() {
    local arch="$1" ckpt="$2" mmproj="$3" nimg="$4" env_str="$5" tag="$6" side="$7" cli="$8"
    # shellcheck disable=SC2086
    env ${env_str} VLA_IMG_SIZE="${side}" VLA_BENCH_ITERS="${BENCH}" \
        "${BIN}" "${ckpt}" "${mmproj}" "${nimg}" ${cli} \
        > "${OUT}/${arch}${tag}.actions.txt" 2> "${OUT}/${arch}${tag}.log"
}

fail=0
for row in "${MODELS[@]}"; do
    IFS='|' read -r arch ckpt mmproj nimg always fastest <<< "${row}"

    if [[ -n "${ARCHS:-}" && " ${ARCHS} " != *" ${arch} "* ]]; then
        continue
    fi
    if [[ ! -e "${ckpt}" ]]; then
        echo "[skip] ${arch}: no checkpoint at ${ckpt}"
        continue
    fi

    side=""
    for s in ${SIDES}; do
        if run_one "${arch}" "${ckpt}" "${mmproj}" "${nimg}" "${always}" "" "${s}" "" \
           && ! grep -q '^action_len=0$' "${OUT}/${arch}.actions.txt"; then
            side="${s}"
            echo "${s}" > "${OUT}/${arch}.side"
            break
        fi
    done
    if [[ -z "${side}" ]]; then
        echo "[FAIL] ${arch}: no input side in '${SIDES}' produced a chunk; see ${OUT}/${arch}.log" >&2
        fail=1
        continue
    fi

    line="[ok  ] ${arch}  side=${side}"
    [[ "${BENCH}" -gt 0 ]] && line+="  default=$(grep -oP 'min=\K[0-9.]+' "${OUT}/${arch}.log" | head -1)ms"

    if [[ -n "${fastest}" ]]; then
        if run_one "${arch}" "${ckpt}" "${mmproj}" "${nimg}" "${always}" ".alt" "${side}" "${fastest}" \
           && ! grep -q '^action_len=0$' "${OUT}/${arch}.alt.actions.txt"; then
            line+="  alt=ok"
            [[ "${BENCH}" -gt 0 ]] && line+=" $(grep -oP 'min=\K[0-9.]+' "${OUT}/${arch}.alt.log" | head -1)ms"
        else
            echo "[FAIL] ${arch}: alternate config produced no chunk; see ${OUT}/${arch}.fast.log" >&2
            fail=1
        fi
    fi
    echo "${line}"
done

echo
echo "written to ${OUT}"
exit "${fail}"
