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

# PyTorch-reference counterpart of run_libero.sh: sweeps libero_object task-id
# 0..9 (N_EPISODES each) against the upstream PyTorch policies instead of the
# GGUF port, so the two stacks' success rates can be compared side by side.
#
# Same LIBERO env, same seed (42), same 500-step cap, and — critically — the
# same action-chunk replay: each policy server is launched with
# --n-action-steps set to the value eval/run_libero.sh gives the vla.cpp client
# for that arch. Without that, GR00T/Evo-1 would re-predict every env step here
# while replaying 8-16 actions there, and the SR delta would measure control
# cadence rather than port fidelity.
#
# Two venvs are involved, as in the vla.cpp path:
#   - policy server: PT_VENV (torch 2.8 + lerobot 0.4.4 + flash-attn)
#   - LIBERO client: eval/sim/libero/libero_uv/.venv
#
# Runs ONE model per invocation so an outer driver can schedule several across
# GPUs concurrently; see run_libero_compare.sh.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") -m <MODEL> [-i <CKPT_ROOT>] [-o <OUTPUT_ROOT>] [-n <N_EPISODES>]
                        [-g <GPU>] [-p <PORT>]

  -m MODEL         smolvla | pi0 | evo1 | gr00t_n1_5 | gr00t_n1_6 | gr00t_n1_7  [required]
  -i CKPT_ROOT     root holding the PyTorch checkpoint dirs
                   (default: ${CKPT_ROOT_DEFAULT:-/mnt/data/hf_data})
  -o OUTPUT_ROOT   destination for client outputs + server logs
                   (default: ${REPO_ROOT}/outputs/libero_object_pytorch)
  -n N_EPISODES    episodes per task-id (default: 10)
  -g GPU           CUDA device index for the policy server (default: 0)
  -p PORT          ZMQ port for the policy server (default: 5555 + GPU)
  -t TASK_IDS      space-separated task ids (default: "0 1 2 3 4 5 6 7 8 9")
  -h               show this help

Env overrides: PT_VENV, LIBERO_VENV, VLA_POLICY_DIR, HF_HOME
EOF
}

CKPT_ROOT="/mnt/data/hf_data"
OUTPUT_ROOT=""
N_EPISODES="10"
MODEL=""
GPU="0"
PORT=""
TASK_IDS="0 1 2 3 4 5 6 7 8 9"

while getopts ":m:i:o:n:g:p:t:h" opt; do
    case "${opt}" in
        m) MODEL="${OPTARG}" ;;
        i) CKPT_ROOT="${OPTARG}" ;;
        o) OUTPUT_ROOT="${OPTARG}" ;;
        n) N_EPISODES="${OPTARG}" ;;
        g) GPU="${OPTARG}" ;;
        p) PORT="${OPTARG}" ;;
        t) TASK_IDS="${OPTARG}" ;;
        h) usage; exit 0 ;;
        \?) echo "ERROR: unknown option -${OPTARG}" >&2; usage >&2; exit 1 ;;
        :)  echo "ERROR: option -${OPTARG} requires an argument" >&2; usage >&2; exit 1 ;;
    esac
done

case "${MODEL}" in
    smolvla|pi0|evo1|gr00t_n1_5|gr00t_n1_6|gr00t_n1_7) ;;
    *)
        echo "ERROR: -m must be one of: smolvla | pi0 | evo1 | gr00t_n1_5 | gr00t_n1_6 | gr00t_n1_7 (got '${MODEL}')" >&2
        usage >&2
        exit 1
        ;;
esac

if ! [[ "${N_EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: N_EPISODES must be a positive integer (got '${N_EPISODES}')" >&2
    exit 1
fi

PORT="${PORT:-$((5555 + GPU))}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/libero_object_pytorch}"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"
LOG_DIR="${OUTPUT_ROOT}/_server_logs"
mkdir -p "${LOG_DIR}"

PT_ROOT="${REPO_ROOT}/eval/pytorch_ref"
VENV_ROOT="${VENV_ROOT:-/mnt/data/vla_sr_compare/venvs}"
# GR00T-N1.7's Qwen3-VL backbone needs transformers >= 4.57.1, but pi0 pulls
# lerobot's `pi` extra which pins the 4.53-based openpi fork — the two cannot
# share an environment. N1.7 therefore gets its own venv; the other five share
# the base one. Override either with PT_VENV.
case "${MODEL}" in
    gr00t_n1_7) DEFAULT_VENV="${VENV_ROOT}/pt_ref_n17" ;;
    *)          DEFAULT_VENV="${VENV_ROOT}/pt_ref" ;;
esac
PT_VENV="${PT_VENV:-${DEFAULT_VENV}}"
PT_PY="${PT_VENV}/bin/python"
LIBERO_VENV="${LIBERO_VENV:-${REPO_ROOT}/eval/sim/libero/libero_uv/.venv}"
LIBERO_PY="${LIBERO_VENV}/bin/python"
CLIENT="${PT_ROOT}/client/run_libero_eval.py"
TASK_SUITE="libero_object"

export VLA_POLICY_DIR="${VLA_POLICY_DIR:-/mnt/data/vla_sr_compare/weights}"
export HF_HOME="${HF_HOME:-/mnt/data/hf_data}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/mnt/data/hf_data/hub}"
# Every weight/tokenizer/processor these policies need is already on disk (the
# finetunes under CKPT_ROOT, their backbones in the hub cache). Left online,
# lerobot's from_pretrained makes a Hub metadata call that can wedge on a
# half-closed CDN socket with no timeout — model load then hangs indefinitely
# instead of taking ~7 s. Offline turns any genuinely missing artifact into an
# immediate error; pre-warm the cache and re-run if that happens.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

for p in "${PT_PY}" "${LIBERO_PY}"; do
    if [[ ! -x "${p}" ]]; then
        echo "ERROR: interpreter not found: ${p}" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Per-model wiring. n_action_steps mirrors eval/run_libero.sh's per-arch
# defaults so both stacks replay the same number of actions per prediction.
# ---------------------------------------------------------------------------
case "${MODEL}" in
    smolvla)
        SERVER_SCRIPT="smolvla_server.py"
        MODEL_ID="${CKPT_ROOT}/HuggingFaceVLA/smolvla_libero"
        SERVER_EXTRA=()
        N_ACTION_STEPS="${N_ACTION_STEPS:-1}"
        ;;
    pi0)
        SERVER_SCRIPT="pi0_server.py"
        MODEL_ID="${CKPT_ROOT}/lerobot/pi0_libero_finetuned_v044"
        SERVER_EXTRA=()
        N_ACTION_STEPS="${N_ACTION_STEPS:-50}"
        ;;
    evo1)
        SERVER_SCRIPT="evo1_server.py"
        MODEL_ID="MINT-SJTU/Evo1_LIBERO"   # resolved under VLA_POLICY_DIR
        SERVER_EXTRA=()
        N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
        ;;
    gr00t_n1_5)
        SERVER_SCRIPT="gr00t_n15_server.py"
        MODEL_ID="${CKPT_ROOT}/liorbenhorin-nv/groot-libero_object-64_40000"
        SERVER_EXTRA=()
        N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
        ;;
    gr00t_n1_6)
        SERVER_SCRIPT="gr00t_server.py"
        MODEL_ID="${CKPT_ROOT}/0xAnkitSingh/GR00T-N1.6-LIBERO"
        SERVER_EXTRA=(--embodiment-tag libero_panda)
        N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
        ;;
    gr00t_n1_7)
        SERVER_SCRIPT="gr00t_n17_server.py"
        MODEL_ID="${CKPT_ROOT}/nvidia/GR00T-N1.7-LIBERO/libero_object"
        SERVER_EXTRA=(--embodiment-tag libero_sim)
        N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
        ;;
esac

echo "[config] MODEL=${MODEL}  GPU=${GPU}  PORT=${PORT}"
echo "[config] MODEL_ID=${MODEL_ID}"
echo "[config] N_EPISODES=${N_EPISODES}  N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "[config] OUTPUT_ROOT=${OUTPUT_ROOT}"

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[cleanup] stopping policy server (pid=${SERVER_PID})"
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

LOG="${LOG_DIR}/${MODEL}.log"
: > "${LOG}"
echo "===================="
echo "[${MODEL}] starting policy server -> ${LOG}"

# `exec` is load-bearing: without it the subshell stays alive as the parent and
# $! is the SUBSHELL's pid, so cleanup() kills the shell and orphans the python
# server — which keeps its GPU memory until manually killed. With exec the
# subshell is replaced by python, so $! is the server itself.
(
    cd "${PT_ROOT}" && \
    exec env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 \
    "${PT_PY}" "server/${SERVER_SCRIPT}" \
        --model-id "${MODEL_ID}" \
        --port "${PORT}" \
        --n-action-steps "${N_ACTION_STEPS}" \
        "${SERVER_EXTRA[@]}"
) >"${LOG}" 2>&1 &
SERVER_PID=$!
echo "[${MODEL}] server pid=${SERVER_PID}"

# Model load is minutes for the 3B checkpoints. Probe the ZMQ `ping` endpoint
# rather than grepping the log: RobotInferenceServer's "Server is ready" print
# is not flushed, so under a redirected stdout it can sit in the block buffer
# long after the server is answering requests (PYTHONUNBUFFERED=1 above fixes
# that too, but the probe is what actually proves readiness).
ping_server() {
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

ready=0
for _ in $(seq 1 360); do
    if ping_server "${PORT}"; then
        ready=1; break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: policy server exited before becoming ready; see ${LOG}" >&2
        tail -n 40 "${LOG}" >&2 || true
        exit 1
    fi
    sleep 5
done
if [[ "${ready}" != "1" ]]; then
    echo "ERROR: policy server not ready within 1800s (ping probe); see ${LOG}" >&2
    exit 1
fi
echo "[${MODEL}] server ready on port ${PORT}"

OUT_DIR="${OUTPUT_ROOT}/${MODEL}"
mkdir -p "${OUT_DIR}"

rc=0
for task_id in ${TASK_IDS}; do
    echo "[${MODEL}] task_id=${task_id}  episodes=${N_EPISODES}"
    # robosuite's EGL renderer needs a valid device index; the client's torch
    # stays on CPU, so it does not contend with the server for VRAM.
    ( cd "${PT_ROOT}" && \
      MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" \
      "${LIBERO_PY}" "${CLIENT}" \
          --task "${TASK_SUITE}/task_${task_id}" \
          --n-episodes "${N_EPISODES}" \
          --port "${PORT}" \
          --out-name "${MODEL}" \
          --n-action-steps "${N_ACTION_STEPS}" \
          --output-dir "${OUT_DIR}" ) || rc=$?
    if [[ "${rc}" != "0" ]]; then
        echo "[${MODEL}] WARNING: task_${task_id} exited rc=${rc}; continuing" >&2
        rc=0
    fi
done

cleanup
echo "[${MODEL}] done -> ${OUT_DIR}"
