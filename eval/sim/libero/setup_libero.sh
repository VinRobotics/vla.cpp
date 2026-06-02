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

set -euxo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set paths relative to script location
LIBERO_REPO="$SCRIPT_DIR/LIBERO"
LIBERO_UV_ENV="$SCRIPT_DIR/libero_uv"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GITMODULES_PATH="$REPO_ROOT/.gitmodules"
LIBERO_GIT_URL="$(git config -f "$GITMODULES_PATH" --get submodule.external_dependencies/LIBERO.url 2>/dev/null || true)"

if [ -z "$LIBERO_GIT_URL" ]; then
	LIBERO_GIT_URL="https://github.com/Lifelong-Robot-Learning/LIBERO.git"
fi

# Fallback for branches where LIBERO is listed in .gitmodules but not tracked as a gitlink.
mkdir -p "$(dirname "$LIBERO_REPO")"
if [ -d "$LIBERO_REPO/.git" ]; then
	echo "LIBERO repo already exists at $LIBERO_REPO, reusing existing checkout."
elif [ -d "$LIBERO_REPO" ] && [ -n "$(ls -A "$LIBERO_REPO" 2>/dev/null)" ]; then
	echo "Directory $LIBERO_REPO already exists and is not empty, skipping clone."
else
	git clone "$LIBERO_GIT_URL" "$LIBERO_REPO"
fi

rm -rf "$LIBERO_UV_ENV"deac
mkdir -p "$LIBERO_UV_ENV"
uv venv "$LIBERO_UV_ENV/.venv" --python 3.10
source "$LIBERO_UV_ENV/.venv/bin/activate"
uv pip install --requirements "$LIBERO_REPO/requirements.txt"
uv pip install -e "$LIBERO_REPO" --config-settings editable_mode=compat
uv pip install lerobot==0.4.3
uv pip install torch==2.5.1 torchvision==0.20.1 pydantic av tianshou==0.5.1 tyro pandas dm_tree einops==0.8.1 albumentations==1.4.18 zmq
uv pip install transformers==4.51.3 msgpack==1.1.0 msgpack-numpy==0.4.8 gymnasium==0.29.1
uv pip install pandas==2.0.3
uv pip install pyarrow==12.0.1
uv pip install diffusers==0.30.1
uv pip install numpy==1.26.4

rm -rf "$HOME/.libero"
