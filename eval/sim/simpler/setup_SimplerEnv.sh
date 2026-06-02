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
SIMPLER_REPO="$SCRIPT_DIR/SimplerEnv"
SIMPLER_ENV="$SCRIPT_DIR/simpler_uv"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GITMODULES_PATH="$REPO_ROOT/.gitmodules"
SIMPLER_GIT_URL="$(git config -f "$GITMODULES_PATH" --get submodule.external_dependencies/SimplerEnv.url 2>/dev/null || true)"

if [ -z "$SIMPLER_GIT_URL" ]; then
	SIMPLER_GIT_URL="https://github.com/squarefk/SimplerEnv.git"
fi

# Fallback for branches where SimplerEnv is listed in .gitmodules but not tracked as a gitlink.
mkdir -p "$(dirname "$SIMPLER_REPO")"
if [ -d "$SIMPLER_REPO/.git" ]; then
	echo "SimplerEnv repo already exists at $SIMPLER_REPO, reusing existing checkout."
elif [ -d "$SIMPLER_REPO" ] && [ -n "$(ls -A "$SIMPLER_REPO" 2>/dev/null)" ]; then
	echo "Directory $SIMPLER_REPO already exists and is not empty, skipping clone."
else
	git clone --recursive "$SIMPLER_GIT_URL" "$SIMPLER_REPO"
fi

# Numpy pin: cluster uses 1.26.4; SimplerEnv README mentions 1.24.4 for pinocchio IK.
# Override by exporting SIMPLER_NUMPY=1.24.4 if needed.
SIMPLER_NUMPY="${SIMPLER_NUMPY:-1.26.4}"

# python -m pip install -U uv
rm -rf "$SIMPLER_ENV"
mkdir -p "$SIMPLER_ENV"
uv venv "$SIMPLER_ENV/.venv" --python 3.10
source "$SIMPLER_ENV/.venv/bin/activate"
uv pip install "setuptools<70.0.0"

# Core deps (match cluster’s pyproject pattern)
uv pip install \
  gymnasium==0.29.1 \
  json-numpy>=2.1.1 \
  numpy=="$SIMPLER_NUMPY" \
  opencv-python-headless==4.10.0.84 \
  ray==2.48.0

# Install SimplerEnv sources (editable)
# uv pip install -e "$SIMPLER_REPO/ManiSkill2_real2sim" --config-settings editable_mode=compat
# uv pip install -e "$SIMPLER_REPO" --config-settings editable_mode=compat

uv pip install -e "$SIMPLER_REPO/ManiSkill2_real2sim"
uv pip install -e "$SIMPLER_REPO"
uv pip install pandas==2.0.3
uv pip install pyarrow==12.0.1
uv pip install diffusers==0.30.1
uv pip install tianshou==0.5.1 pydantic av zmq torchvision==0.22.0 transformers==4.51.3
