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

# Per-model inference-latency comparison: vla.cpp against the upstream PyTorch
# policy in three variants (eager, torch.compile, torch.compile CUDA-graphs).
#
# What is measured
# ----------------
# SERVER-SIDE inference time, not client round-trip: the vla.cpp server reports
# its own latency in each response, and the PyTorch server times select_action
# in-process (see eval/pytorch_ref/utils/service.py) with CUDA synchronization
# on both sides of the call. Neither number includes ZMQ transport or image
# serialization, so the comparison isolates the model.
#
# Both stacks run with --n-action-steps 1. Chunk replay would otherwise make
# most get_action calls a cheap queue pop, and the reported mean would be a
# blend of "real forward" and "popped from a deque" rather than a latency.
#
# Warmup steps are timed but discarded (the PyTorch server's latency buffer is
# reset after warmup), which is what keeps torch.compile's first-call
# compilation — tens of seconds — out of the measured window.
#
# Runs ONE model per invocation so an outer driver can deal models across GPUs;
# see the -v flag to select which variants to run.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CKPT_ROOT="/mnt/data/hf_data"
GGUF_ROOT="/mnt/data/hf_data/vrfai"

usage() {
    cat <<EOF
Usage: $(basename "$0") -m <MODEL> [-v <VARIANTS>] [-g <GPU>] [-n <N_STEPS>]
                        [-o <OUTPUT_ROOT>] [-i <GGUF_ROOT>] [-c <CKPT_ROOT>]

  -m MODEL         smolvla | pi0 | evo1 | gr00t_n1_5 | gr00t_n1_6 | gr00t_n1_7
                   | bitvla (vla.cpp variant only)                    [required]
  -v VARIANTS      comma-separated subset of:
                     vla.cpp, eager, compile-default, compile-reduce-overhead
                   (default: all four)
  -g GPU           CUDA device index (default: 0)
  -n N_STEPS       timed inference calls per variant (default: 200)
  -w WARMUP        warmup calls before timing (default: 5 eager / 20 compiled)
  -t TASK_ID       libero_object task id used as the input stream (default: 0)
  -o OUTPUT_ROOT   default: ${REPO_ROOT}/outputs/latency_compare
  -i GGUF_ROOT     default: ${GGUF_ROOT}
  -c CKPT_ROOT     default: ${CKPT_ROOT}
  -h               show this help

Env overrides: PT_VENV, LIBERO_VENV, VLA_POLICY_DIR, HF_HOME
EOF
}

MODEL=""
VARIANTS="vla.cpp,eager,compile-default,compile-reduce-overhead"
GPU="0"
N_STEPS="200"
WARMUP=""
TASK_ID="0"
OUTPUT_ROOT=""

while getopts ":m:v:g:n:w:t:o:i:c:h" opt; do
    case "${opt}" in
        m) MODEL="${OPTARG}" ;;
        v) VARIANTS="${OPTARG}" ;;
        g) GPU="${OPTARG}" ;;
        n) N_STEPS="${OPTARG}" ;;
        w) WARMUP="${OPTARG}" ;;
        t) TASK_ID="${OPTARG}" ;;
        o) OUTPUT_ROOT="${OPTARG}" ;;
        i) GGUF_ROOT="${OPTARG}" ;;
        c) CKPT_ROOT="${OPTARG}" ;;
        h) usage; exit 0 ;;
        \?) echo "ERROR: unknown option -${OPTARG}" >&2; usage >&2; exit 1 ;;
        :)  echo "ERROR: option -${OPTARG} requires an argument" >&2; usage >&2; exit 1 ;;
    esac
done

case "${MODEL}" in
    smolvla|pi0|evo1|gr00t_n1_5|gr00t_n1_6|gr00t_n1_7|bitvla) ;;
    *)
        echo "ERROR: -m must be one of: smolvla | pi0 | evo1 | gr00t_n1_5 | gr00t_n1_6 | gr00t_n1_7 | bitvla (got '${MODEL}')" >&2
        usage >&2; exit 1 ;;
esac

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/latency_compare}"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"
OUT_DIR="${OUTPUT_ROOT}/${MODEL}"
LOG_DIR="${OUTPUT_ROOT}/_server_logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

PT_ROOT="${REPO_ROOT}/eval/pytorch_ref"
VENV_ROOT="${VENV_ROOT:-/mnt/data/vla_sr_compare/venvs}"
# GR00T-N1.7's Qwen3-VL backbone needs transformers >= 4.57.1, which conflicts
# with the openpi fork pi0 pulls in; N1.7 therefore has its own venv.
case "${MODEL}" in
    gr00t_n1_7) DEFAULT_VENV="${VENV_ROOT}/pt_ref_n17" ;;
    *)          DEFAULT_VENV="${VENV_ROOT}/pt_ref" ;;
esac
PT_VENV="${PT_VENV:-${DEFAULT_VENV}}"
PT_PY="${PT_VENV}/bin/python"
LIBERO_VENV="${LIBERO_VENV:-${REPO_ROOT}/eval/sim/libero/libero_uv/.venv}"
LIBERO_PY="${LIBERO_VENV}/bin/python"
# Overridable so two builds can be compared on the same harness, e.g. a
# scratch build at another llama.cpp tag (-DVLA_LLAMA_TAG=...).
SERVER_BIN="${SERVER_BIN:-${REPO_ROOT}/build/vla-server}"
BENCH="${REPO_ROOT}/eval/client/benchmark.py"
PORT="$((5700 + GPU))"

export VLA_POLICY_DIR="${VLA_POLICY_DIR:-/mnt/data/vla_sr_compare/weights}"
export HF_HOME="${HF_HOME:-/mnt/data/hf_data}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/mnt/data/hf_data/hub}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"
# Everything these policies need is on disk; left online, from_pretrained can
# wedge on a half-closed CDN socket with no timeout.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

for p in "${PT_PY}" "${LIBERO_PY}" "${SERVER_BIN}"; do
    if [[ ! -x "${p}" ]]; then
        echo "ERROR: not found or not executable: ${p}" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Per-model wiring: PyTorch checkpoint + server script on one side, GGUF +
# arch preset on the other.
# ---------------------------------------------------------------------------
STATS_JSON=""
TOKENIZER=""
case "${MODEL}" in
    smolvla)
        SERVER_SCRIPT="smolvla_server.py"
        MODEL_ID="${CKPT_ROOT}/HuggingFaceVLA/smolvla_libero"
        SERVER_EXTRA=()
        GGUF="${GGUF_ROOT}/smolvla-libero-gguf/smolvla-libero.gguf"
        ;;
    pi0)
        SERVER_SCRIPT="pi0_server.py"
        MODEL_ID="${CKPT_ROOT}/lerobot/pi0_libero_finetuned_v044"
        SERVER_EXTRA=()
        GGUF="${GGUF_ROOT}/pi0-libero-finetuned-v044-gguf/pi0-libero-finetuned-v044.gguf"
        ;;
    evo1)
        SERVER_SCRIPT="evo1_server.py"
        MODEL_ID="MINT-SJTU/Evo1_LIBERO"   # resolved under VLA_POLICY_DIR
        SERVER_EXTRA=()
        GGUF="${GGUF_ROOT}/evo1-libero-gguf/evo1-libero.gguf"
        ;;
    gr00t_n1_5)
        SERVER_SCRIPT="gr00t_n15_server.py"
        MODEL_ID="${CKPT_ROOT}/liorbenhorin-nv/groot-libero_object-64_40000"
        SERVER_EXTRA=()
        GGUF="${GGUF_ROOT}/gr00tn1d5-libero-object-gguf/gr00tn1d5-libero-object.gguf"
        STATS_JSON="${GGUF_ROOT}/gr00tn1d5-libero-object-gguf/dataset_statistics.json"
        ;;
    gr00t_n1_6)
        SERVER_SCRIPT="gr00t_server.py"
        MODEL_ID="${CKPT_ROOT}/0xAnkitSingh/GR00T-N1.6-LIBERO"
        SERVER_EXTRA=(--embodiment-tag libero_panda)
        GGUF="${GGUF_ROOT}/gr00tn1d6-libero-gguf/gr00tn1d6-libero.gguf"
        STATS_JSON="${GGUF_ROOT}/gr00tn1d6-libero-gguf/dataset_statistics.json"
        # N1.6 has no HF-default tokenizer; its Eagle tokenizer is vendored
        # alongside the GGUF.
        TOKENIZER="${GGUF_ROOT}/gr00tn1d6-libero-gguf"
        ;;
    gr00t_n1_7)
        SERVER_SCRIPT="gr00t_n17_server.py"
        MODEL_ID="${CKPT_ROOT}/nvidia/GR00T-N1.7-LIBERO/libero_object"
        SERVER_EXTRA=(--embodiment-tag libero_sim)
        GGUF="${GGUF_ROOT}/gr00tn1d7-libero-gguf/libero_object/gr00tn1d7-libero-object.gguf"
        STATS_JSON="${GGUF_ROOT}/gr00tn1d7-libero-gguf/libero_object/dataset_statistics.json"
        ;;
    bitvla)
        # vla.cpp side only. BitVLA's PyTorch reference needs OpenVLA-OFT's
        # prismatic package and BitVLA's own transformers fork, neither of which
        # can share the pt_ref venv, so its eager/compiled variants are measured
        # by eval/run_bitvla_compile_compare.sh instead. Listing it here keeps
        # the vla.cpp number on the same harness as every other row.
        SERVER_SCRIPT=""
        MODEL_ID=""
        SERVER_EXTRA=()
        GGUF="${GGUF_ROOT}/bitvla-libero-gguf/libero_object/bitvla-libero-object.gguf"
        STATS_JSON="${GGUF_ROOT}/bitvla-libero-gguf/libero_object/dataset_statistics.json"
        # Offline: the arch preset's tokenizer is a Hub id; the GGUF dir carries
        # the same Llama-3 BPE plus specials.
        TOKENIZER="${GGUF_ROOT}/bitvla-libero-gguf/libero_object"
        ;;
esac

echo "[config] MODEL=${MODEL}  GPU=${GPU}  PORT=${PORT}  TASK=libero_object/${TASK_ID}"
echo "[config] VARIANTS=${VARIANTS}"
echo "[config] N_STEPS=${N_STEPS}  n_action_steps=1 (every call is a real forward)"
echo "[config] OUT_DIR=${OUT_DIR}"

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[cleanup] stopping server (pid=${SERVER_PID})"
        kill -INT "${SERVER_PID}" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "${SERVER_PID}" 2>/dev/null || break
            sleep 0.5
        done
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap cleanup EXIT INT TERM

# Probe the ZMQ ping endpoint rather than grepping the log: the PyTorch
# server's "ready" print is not reliably flushed under a redirected stdout.
ping_zmq() {
    "${LIBERO_PY}" - "$1" <<'PYEOF' >/dev/null 2>&1
import sys, zmq, msgpack
ctx = zmq.Context()
s = ctx.socket(zmq.REQ)
s.setsockopt(zmq.RCVTIMEO, 3000)
s.setsockopt(zmq.LINGER, 0)
s.connect(f"tcp://localhost:{sys.argv[1]}")
s.send(msgpack.packb({"endpoint": "ping"}, use_bin_type=True))
sys.exit(0 if msgpack.unpackb(s.recv(), raw=False).get("status") == "ok" else 1)
PYEOF
}

wait_ready_zmq() {
    local log="$1"
    for _ in $(seq 1 360); do
        ping_zmq "${PORT}" && return 0
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "ERROR: server exited before becoming ready; see ${log}" >&2
            tail -n 40 "${log}" >&2 || true
            return 1
        fi
        sleep 5
    done
    echo "ERROR: server not ready within 1800s; see ${log}" >&2
    return 1
}

wait_ready_vlacpp() {
    local log="$1"
    for _ in $(seq 1 600); do
        if grep -q "bound to .* ready" "${log}" 2>/dev/null; then return 0; fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "ERROR: vla-server exited before becoming ready; see ${log}" >&2
            tail -n 40 "${log}" >&2 || true
            return 1
        fi
        sleep 1
    done
    echo "ERROR: vla-server not ready within 600s; see ${log}" >&2
    return 1
}

run_bench() {
    local variant="$1"; shift
    local out_json="${OUT_DIR}/${variant}.json"
    local warmup="$1"; shift
    local extra=("$@")

    echo "[${MODEL}/${variant}] benchmarking ${N_STEPS} steps (warmup ${warmup}) ..."
    ( cd "${REPO_ROOT}/eval" && \
      MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" \
      "${LIBERO_PY}" "${BENCH}" \
          --addr "tcp://localhost:${PORT}" \
          --server-pid "${SERVER_PID}" \
          --task libero_object \
          --task-id "${TASK_ID}" \
          --n-steps "${N_STEPS}" \
          --warmup-steps "${warmup}" \
          --n-action-steps 1 \
          --model "${MODEL}" \
          --variant "${variant}" \
          --output "${out_json}" \
          "${extra[@]}" )
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo "[${MODEL}/${variant}] WARNING: benchmark exited rc=${rc}" >&2
    fi
    return 0
}

# --- vla.cpp ---------------------------------------------------------------
run_variant_vlacpp() {
    if [[ ! -f "${GGUF}" ]]; then
        echo "[skip] vla.cpp: GGUF not found at ${GGUF}" >&2
        return 0
    fi
    local log="${LOG_DIR}/${MODEL}.vla-cpp.log"
    : > "${log}"

    case "${MODEL}" in
        gr00t_n1_5) export VLA_GR00T_EMBODIMENT="${VLA_GR00T_EMBODIMENT:-new_embodiment}" ;;
        gr00t_n1_6) export VLA_GR00T_EMBODIMENT="${VLA_GR00T_EMBODIMENT:-libero_panda}" ;;
        *)          unset VLA_GR00T_EMBODIMENT ;;
    esac

    echo "===================="
    echo "[${MODEL}/vla.cpp] starting vla-server -> ${log}"
    ( exec env CUDA_VISIBLE_DEVICES="${GPU}" \
      "${SERVER_BIN}" --bind "tcp://*:${PORT}" "${GGUF}" ) >"${log}" 2>&1 &
    SERVER_PID=$!
    wait_ready_vlacpp "${log}" || { cleanup; return 0; }
    echo "[${MODEL}/vla.cpp] server ready (pid=${SERVER_PID})"

    local extra=(--backend vla-cpp --arch "${MODEL}")
    [[ -n "${STATS_JSON}" ]] && extra+=(--stats-json "${STATS_JSON}")
    [[ -n "${TOKENIZER}"  ]] && extra+=(--tokenizer "${TOKENIZER}")
    run_bench "vla.cpp" 5 "${extra[@]}"
    cleanup
}

# --- PyTorch (eager / compiled) --------------------------------------------
run_variant_torch() {
    local variant="$1"
    local log="${LOG_DIR}/${MODEL}.${variant}.log"
    : > "${log}"

    local compile_env=()
    local warmup=5
    case "${variant}" in
        eager)
            compile_env=(VLA_TORCH_COMPILE=0) ;;
        compile-default)
            compile_env=(VLA_TORCH_COMPILE=1 VLA_TORCH_COMPILE_MODE=default)
            warmup=20 ;;
        compile-reduce-overhead)
            compile_env=(VLA_TORCH_COMPILE=1 VLA_TORCH_COMPILE_MODE=reduce-overhead)
            warmup=20 ;;
        *) echo "ERROR: unknown variant ${variant}" >&2; return 0 ;;
    esac
    [[ -n "${WARMUP}" ]] && warmup="${WARMUP}"

    echo "===================="
    echo "[${MODEL}/${variant}] starting policy server -> ${log}"
    # `exec` is load-bearing: without it $! is the subshell's pid, so cleanup
    # kills the shell and orphans python — which keeps its GPU memory.
    (
        cd "${PT_ROOT}" && \
        exec env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 \
        "${compile_env[@]}" \
        "${PT_PY}" "server/${SERVER_SCRIPT}" \
            --model-id "${MODEL_ID}" \
            --port "${PORT}" \
            --n-action-steps 1 \
            "${SERVER_EXTRA[@]}"
    ) >"${log}" 2>&1 &
    SERVER_PID=$!
    wait_ready_zmq "${log}" || { cleanup; return 0; }
    echo "[${MODEL}/${variant}] server ready (pid=${SERVER_PID})"

    run_bench "${variant}" "${warmup}" --backend lerobot
    cleanup
}

IFS=',' read -r -a WANTED <<< "${VARIANTS}"
for v in "${WANTED[@]}"; do
    case "${v}" in
        vla.cpp) run_variant_vlacpp ;;
        eager|compile-default|compile-reduce-overhead)
            if [[ -z "${SERVER_SCRIPT}" ]]; then
                echo "[skip] ${MODEL}/${v}: no PyTorch server in eval/pytorch_ref;" \
                     "use eval/run_bitvla_compile_compare.sh" >&2
                continue
            fi
            run_variant_torch "${v}" ;;
        *) echo "ERROR: unknown variant '${v}'" >&2 ;;
    esac
done

echo "===================="
echo "[${MODEL}] done -> ${OUT_DIR}"
