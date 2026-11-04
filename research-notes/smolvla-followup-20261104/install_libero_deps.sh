#!/usr/bin/env bash
set -euo pipefail
export PATH="$PWD/research-notes/tools/uv/bin:$PATH"
export UV_CACHE_DIR="$PWD/research-notes/tools/uv-cache"
export UV_LINK_MODE=copy
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export VIRTUAL_ENV="$PWD/eval/sim/libero/libero_uv/.venv"
LIBERO_REPO="$PWD/eval/sim/libero/LIBERO"
uv pip install --requirements "$LIBERO_REPO/requirements.txt"
uv pip install -e "$LIBERO_REPO" --config-settings editable_mode=compat
uv pip install lerobot==0.4.3
uv pip install torch==2.5.1 torchvision==0.20.1 pydantic av tianshou==0.5.1 tyro pandas dm_tree einops==0.8.1 albumentations==1.4.18 zmq
uv pip install transformers==4.51.3 msgpack==1.1.0 msgpack-numpy==0.4.8 gymnasium==0.29.1
uv pip install pandas==2.0.3
uv pip install pyarrow==12.0.1
uv pip install diffusers==0.30.1
uv pip install numpy==1.26.4
# robosuite 1.4.0 calls the mujoco 2.3 mj_fullM(m, dst, M) signature; mujoco 3.x
# changed it, so a transitive dep pulling 3.x breaks the OSC controller at reset.
uv pip install mujoco==2.3.2
