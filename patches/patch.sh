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

# patches/patch.sh - fetch the pinned llama.cpp and apply vla.cpp's local patch.
#
# vla.cpp builds against llama.cpp at tag b9016 with a small set of local fixes
# shipped as patches/llama.cpp-vla.patch. Run this once from the repo root after
# cloning, before configuring CMake:
#
#     bash ./patches/patch.sh
#
# Re-running is safe: an existing checkout is left in place and an
# already-applied patch is detected and skipped.

set -euo pipefail

# Repo root is one level up from this script (which lives in patches/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$ROOT/third_party/llama.cpp"
LLAMA_URL="https://github.com/ggml-org/llama.cpp.git"
LLAMA_TAG="b9016"
PATCH="$ROOT/patches/llama.cpp-vla.patch"
PATCH_SHA256_EXPECTED="6fc133c323caf30f9213d774fb8941f95b504a5a34c7c25397bfa03ef4817c67"

[[ -f "$PATCH" ]] || { echo "patch not found: $PATCH" >&2; exit 1; }

if command -v sha256sum >/dev/null 2>&1; then
  PATCH_SHA256_ACTUAL="$(sha256sum "$PATCH" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  PATCH_SHA256_ACTUAL="$(shasum -a 256 "$PATCH" | awk '{print $1}')"
else
  PATCH_SHA256_ACTUAL=""
  echo ">> warning: no sha256sum/shasum available - skipping patch checksum verification" >&2
fi
if [[ -n "$PATCH_SHA256_ACTUAL" && "$PATCH_SHA256_ACTUAL" != "$PATCH_SHA256_EXPECTED" ]]; then
  echo "ERROR: patch checksum mismatch" >&2
  echo "  file:     $PATCH" >&2
  echo "  expected: $PATCH_SHA256_EXPECTED" >&2
  echo "  actual:   $PATCH_SHA256_ACTUAL" >&2
  echo "  (If you intentionally edited the patch, update PATCH_SHA256_EXPECTED in $0.)" >&2
  exit 1
fi

if [[ -e "$LLAMA_DIR/.git" ]]; then
  echo ">> third_party/llama.cpp already present - skipping clone"
else
  echo ">> cloning llama.cpp @ $LLAMA_TAG"
  git clone "$LLAMA_URL" "$LLAMA_DIR"
  git -C "$LLAMA_DIR" checkout "$LLAMA_TAG"
fi

if git -C "$LLAMA_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo ">> patch already applied - nothing to do"
elif git -C "$LLAMA_DIR" apply --check "$PATCH" 2>/dev/null; then
  git -C "$LLAMA_DIR" apply "$PATCH"
  echo ">> patched - third_party/llama.cpp is ready to build"
else
  echo "ERROR: patch does not apply cleanly to $LLAMA_DIR (expected tag $LLAMA_TAG)" >&2
  exit 1
fi
