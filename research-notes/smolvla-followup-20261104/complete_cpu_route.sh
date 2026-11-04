#!/usr/bin/env bash
set -euo pipefail
export PATH="$PWD/research-notes/tools/uv/bin:$PATH"
export UV_CACHE_DIR="$HOME/.cache/uv-vla"
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=4
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export VIRTUAL_ENV="$PWD/eval/sim/libero/libero_uv/.venv"
uv pip install research-notes/tools/cpu-wheels/*.whl
uv pip install --requirements eval/sim/libero/LIBERO/requirements.txt
bash research-notes/smolvla-followup-20261104/finish_libero.sh
