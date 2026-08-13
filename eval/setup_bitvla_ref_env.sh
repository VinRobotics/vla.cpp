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

# Build the upstream PyTorch BitVLA reference environment.
#
# BitVLA's LIBERO driver (third_party/BitVLA/openvla-oft/experiments/robot/
# libero/run_libero_eval_bitnet.py) sits on top of OpenVLA-OFT's `prismatic`
# package AND on BitVLA's own transformers fork (4.51.0.dev0, which carries the
# W1.58-A8 BitLinear SigLIP and the `bit` LM). Upstream installs both with
# plain `pip install -e`, which drags in moojink's transformers fork first and
# then overwrites it. We install the local fork FIRST and pass --no-deps to the
# editable packages so that resolution never fetches the other fork, then add
# the runtime deps by hand (openvla-oft's tokenizers==0.19.1 pin is dropped
# because transformers 4.51 requires >= 0.21).
#
# LIBERO is installed from the checkout the rest of eval/ already uses, with
# the same robosuite/mujoco/numpy pins as eval/sim/libero/setup_libero.sh --
# robosuite 1.4.x calls the mujoco 2.3 mj_fullM signature.

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BITVLA_SRC="${BITVLA_SRC:-${REPO_ROOT}/third_party/BitVLA}"
LIBERO_REPO="${LIBERO_REPO:-${REPO_ROOT}/eval/sim/libero/LIBERO}"
VENV_ROOT="${VENV_ROOT:-/mnt/data/vla_sr_compare/venvs}"
VENV="${VENV:-${VENV_ROOT}/bitvla_ref}"

# egl-probe and friends declare cmake_minimum_required < 3.5.
export CMAKE_POLICY_VERSION_MINIMUM=3.5

mkdir -p "${VENV_ROOT}"
uv venv "${VENV}" --python 3.10
export VIRTUAL_ENV="${VENV}"
PY="${VENV}/bin/python"

uv pip install --python "${PY}" \
    torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# BitVLA's transformers fork first, so nothing later can pull the other one.
uv pip install --python "${PY}" -e "${BITVLA_SRC}/transformers"

# prismatic / experiments.robot live in openvla-oft; bitvla holds the model +
# constants. Neither's dependency metadata is usable here (see header).
uv pip install --python "${PY}" --no-deps -e "${BITVLA_SRC}/openvla-oft"
uv pip install --python "${PY}" --no-deps -e "${BITVLA_SRC}/openvla-oft/bitvla"

# openvla-oft runtime deps, minus the tokenizers pin.
uv pip install --python "${PY}" \
    accelerate draccus==0.8.0 einops huggingface_hub json-numpy jsonlines \
    matplotlib peft==0.11.1 protobuf rich sentencepiece==0.1.99 timm==0.9.10 \
    imageio uvicorn fastapi requests
# prismatic.models.action_heads imports diffusers, whose current releases
# require peft >= 0.17 -- openvla-oft pins peft at 0.11.1.
uv pip install --python "${PY}" diffusers==0.30.1
# The eval driver imports wandb unconditionally. TF 2.15 holds protobuf at
# 4.25, and wandb >= 0.18 generates its protos with a newer runtime, so the
# import dies on wandb.proto.wandb_telemetry_pb2.
uv pip install --python "${PY}" wandb==0.17.9

# experiments/robot/openvla_utils.py imports tensorflow at module scope, and
# importing prismatic drags in the RLDS training pipeline (dlimp + tfds) even
# though eval never touches a dataset.
uv pip install --python "${PY}" tensorflow==2.15.0 tensorflow_datasets==4.9.3 \
    "dlimp @ git+https://github.com/moojink/dlimp_openvla"
# tensorflow-metadata's current wheels are generated against protobuf 5; TF
# 2.15 caps protobuf below 5, and the mismatch surfaces as a missing
# google.protobuf.runtime_version. 1.14.0 is the last protobuf-3/4 build.
uv pip install --python "${PY}" tensorflow-metadata==1.14.0
# --no-deps: tensorflow_graphics pins an ancient tensorflow/keras pair that
# would clobber 2.15; only its geometry.transformation module is imported.
uv pip install --python "${PY}" --no-deps tensorflow_graphics==2021.12.3

# LIBERO + its sim stack.
# editable_mode=compat: LIBERO's setup.py leaves the modern editable finder
# with an empty package MAPPING, so `import libero` fails without it.
uv pip install --python "${PY}" --no-deps -e "${LIBERO_REPO}" \
    --config-settings editable_mode=compat
uv pip install --python "${PY}" \
    "imageio[ffmpeg]" robosuite==1.4.1 bddl easydict cloudpickle gym==0.25.2 \
    hydra-core==1.2.0 opencv-python robomimic==0.2.0 thop future \
    mujoco==2.3.2 numpy==1.26.4

# LIBERO prompts for a dataset path on first import when ~/.libero/config.yaml
# is missing, which would hang a headless eval.
echo "N" | MUJOCO_GL=egl "${PY}" -c "import libero.libero" || true

set +x
echo
echo "BitVLA reference env ready: ${VENV}"
"${PY}" - <<'EOF'
import torch, transformers
print("torch       ", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("transformers", transformers.__version__, transformers.__file__)
EOF
