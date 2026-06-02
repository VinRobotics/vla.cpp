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

# Start vla-server with the GR00T-N1.6 ALOHA (new_embodiment) checkpoint.
#
# NOTE: build vla-server without CUDA graphs to avoid repeated graph-update
# failures caused by dynamic sequence lengths in GR00T-N1.6:
#
#   cmake -B build \
#       -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=OFF \
#       -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j$(nproc)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GGUF="$REPO/weights/vrfai/gr00t-n1d6-aloha-leftarm-gguf/gr00t-n1d6-aloha.gguf"
BIND="${VLA_BIND:-tcp://*:5555}"

if [[ ! -f "$GGUF" ]]; then
    echo "ERROR: GGUF not found: $GGUF" >&2
    exit 1
fi

export VLA_GR00T_EMBODIMENT=new_embodiment

echo "Starting vla-server"
echo "  GGUF:        $GGUF"
echo "  bind:        $BIND"
echo "  embodiment:  $VLA_GR00T_EMBODIMENT"
echo ""

exec "$REPO/build/vla-server" "$GGUF" --bind "$BIND"
