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

# Cross-build vla.cpp with the Hexagon NPU for a Snapdragon Linux board (e.g.
# IQ-10), in Qualcomm's Snapdragon Linux toolchain container. The Hexagon SDK
# has no arm64 Linux build, so the HTP skels cannot be built on the board.
#
#   scripts/build_linux_snapdragon.sh                   # with vla-server
#   scripts/build_linux_snapdragon.sh --no-server       # without it
#   scripts/build_linux_snapdragon.sh -- -DVLA_SPM=ON   # extra CMake arguments
#
# Needs Docker. The package lands in <build-dir>/pkg (bin/, lib/); copy it to
# the board and install the HTP skel as docs/backend/hexagon.md describes.
#
# With the server, every run installs Debian's arm64 protobuf and ZeroMQ dev
# packages in the container (multiarch); --no-server skips that.
#
# Environment:
#   HTP_SKEL_POST     command run as "$HTP_SKEL_POST <in> <out>" on each
#                     libggml-htp-v*.so, for boards that load only processed
#                     skels. Unset, the skels are packaged as built.
#   SNAPDRAGON_IMAGE  toolchain image (default below).

set -euo pipefail

IMAGE="${SNAPDRAGON_IMAGE:-ghcr.io/snapdragon-toolchain/arm64-linux:v0.7}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="build-lnx-htp"
SERVER=1
RECONFIGURE=0
EXTRA=()

usage() { sed -n '/^# Cross-build/,/^# *SNAPDRAGON_IMAGE/p' "$0" | sed 's/^# \{0,1\}//'; }

while (($#)); do
    case "$1" in
        --build-dir)   BUILD_DIR="${2:?--build-dir needs a value}"; shift 2 ;;
        --no-server)   SERVER=0; shift ;;
        --reconfigure) RECONFIGURE=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        --)            shift; EXTRA=("$@"); break ;;
        *)             echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

case "$BUILD_DIR" in
    /*|*..*) echo "--build-dir must be a path inside the repository" >&2; exit 2 ;;
esac
command -v docker >/dev/null || { echo "docker not found" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "cannot reach the Docker daemon (is this user in the docker group?)" >&2; exit 1; }
if [[ -n "${HTP_SKEL_POST:-}" ]] && ! command -v "${HTP_SKEL_POST%% *}" >/dev/null; then
    echo "HTP_SKEL_POST is not executable: ${HTP_SKEL_POST}" >&2; exit 1
fi

B="$REPO/$BUILD_DIR"
PKG="$B/pkg"
mkdir -p "$B"
if ((RECONFIGURE)); then rm -f "$B/CMakeCache.txt"; fi

CMAKE_ARGS=(
    -S . -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=arm64
    -DCMAKE_LIBRARY_ARCHITECTURE=aarch64-linux-gnu
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
    -DCMAKE_C_COMPILER_TARGET=aarch64-linux-gnu -DCMAKE_CXX_COMPILER_TARGET=aarch64-linux-gnu
    "-DCMAKE_C_FLAGS=-march=armv8.2a+fp16+dotprod"
    "-DCMAKE_CXX_FLAGS=-march=armv8.2a+fp16+dotprod"
    -DCMAKE_PREFIX_PATH=/usr/aarch64-linux-gnu -DPREBUILT_LIB_DIR=linux_aarch64
    -DGGML_HEXAGON=ON -DGGML_OPENMP=OFF -DLLAMA_OPENSSL=OFF
    -DVLA_SPM=OFF -DVLA_BUILD_TESTS=ON -DVLA_BUILD_SERVER="$( ((SERVER)) && echo ON || echo OFF )"
    "${EXTRA[@]}"
)

# The container runs as root only to install packages; the build runs as the
# calling user so the build tree stays theirs.
docker run --rm --platform linux/amd64 -v "$REPO":/src -w /src \
    -e SERVER="$SERVER" -e BUILD_DIR="$BUILD_DIR" -e UID_GID="$(id -u):$(id -g)" "$IMAGE" bash -c '
set -euo pipefail
if [[ $SERVER == 1 ]]; then
    dpkg --add-architecture arm64
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        pkg-config protobuf-compiler libprotobuf-dev:arm64 libzmq3-dev:arm64 cppzmq-dev:arm64 >/dev/null
fi
exec setpriv --reuid="${UID_GID%:*}" --regid="${UID_GID#*:}" --clear-groups \
    env HOME=/tmp PKG_CONFIG_LIBDIR=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig \
    bash -c "set -e
        cmake -DHEXAGON_SDK_ROOT=\$HEXAGON_SDK_ROOT -DHEXAGON_TOOLS_ROOT=\$HEXAGON_TOOLS_ROOT \"\$@\"
        cmake --build \"\$BUILD_DIR\" -j\$(nproc)
        cmake --build \"\$BUILD_DIR\" -j\$(nproc) --target htp-v73 htp-v75 htp-v79 htp-v81" _ "$@"
' _ "${CMAKE_ARGS[@]}"

rm -rf "$PKG"
mkdir -p "$PKG/bin" "$PKG/lib"
BINS=(vla-cli vla-bench tests/vla_predict_check)
if ((SERVER)); then BINS+=(vla-server); fi
for t in "${BINS[@]}"; do
    if [[ -f "$B/$t" ]]; then cp "$B/$t" "$PKG/bin/"; fi
done
cp -a "$B"/libvla_core.so* "$B"/bin/lib*.so* "$PKG/lib/"
for v in 73 75 79 81; do
    skel="$B/_deps/llama-build/ggml/src/ggml-hexagon/htp-v$v-prefix/src/htp-v$v-build/libggml-htp-v$v.so"
    if [[ -n "${HTP_SKEL_POST:-}" ]]; then
        $HTP_SKEL_POST "$skel" "$PKG/lib/libggml-htp-v$v.so"
    else
        cp "$skel" "$PKG/lib/"
    fi
done
if [[ -z "${HTP_SKEL_POST:-}" ]]; then
    echo "note: HTP_SKEL_POST unset, skels packaged as built; some boards load only processed skels"
fi

echo "package: $PKG"
ls "$PKG/bin"
