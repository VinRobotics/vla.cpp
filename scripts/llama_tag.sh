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

# Print the llama.cpp tag CMakeLists.txt pins. Two consumers read it - the CI
# cache key and print_versions.sh - and both used to grep for it themselves.
# That grep broke silently once when the tag moved behind ${VLA_LLAMA_TAG} and
# loudly again when the default moved behind another variable, so it lives here.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMAKE="${1:-${ROOT}/CMakeLists.txt}"

tag="$(grep -m1 -oE 'set\(_vla_llama_tag_default "[^"]+"' "${CMAKE}" | grep -oE '"[^"]+"' | tr -d '"' || true)"

if [[ -z "${tag}" ]]; then
    echo "llama_tag: no _vla_llama_tag_default in ${CMAKE}" >&2
    exit 1
fi
echo "${tag}"
