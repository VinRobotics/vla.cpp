#!/usr/bin/env bash
# Copyright 2026 VinRobotics - Apache-2.0
# Repeatable, repo-local RoboCasa365 setup. No sudo or global Python changes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."
ROBOCASA_REV="${ROBOCASA_REV:-3443cdf3dd20f9ae8ed41adba442059b506891d0}"
ROBOSUITE_REV="${ROBOSUITE_REV:-5ce6643f3092639d08f7b0f90ed1c6a84f50552c}"
VENV="${SCRIPT_DIR}/robocasa_uv/.venv"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRIPT_DIR}/robocasa_uv/cache}"
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-10}"

usage() {
    cat <<'EOF'
Usage: bash eval/sim/robocasa/setup_robocasa.sh [--skip-assets] [--skip-smoke]

Creates/reuses a Python 3.11 uv venv and pinned RoboCasa/robosuite checkouts.
Downloads tex, tex_generative, fixtures_lw and objs_lw unless --skip-assets.
Runs a rendered reset/step check unless --skip-smoke.
Env: ROBOCASA_REV, ROBOSUITE_REV, UV_CACHE_DIR, MUJOCO_GL, PYOPENGL_PLATFORM.
System prerequisites: git, uv, protoc, and working EGL (or explicit OSMesa).
EOF
}
SKIP_ASSETS=0
SKIP_SMOKE=0
for arg in "$@"; do
    case "$arg" in
        --skip-assets) SKIP_ASSETS=1 ;;
        --skip-smoke) SKIP_SMOKE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $arg" >&2; exit 1 ;;
    esac
done
for tool in uv git protoc; do
    command -v "$tool" >/dev/null || { echo "ERROR: $tool is required" >&2; exit 1; }
done

checkout() {
    local name="$1" url="$2" rev="$3" path="${SCRIPT_DIR}/$1"
    if [[ ! -e "$path" ]]; then
        git clone --filter=blob:none --no-checkout "$url" "$path"
        git -C "$path" fetch --depth=1 origin "$rev"
        git -C "$path" checkout --detach "$rev"
    fi
    [[ -d "$path/.git" ]] || { echo "ERROR: $path is not a Git checkout" >&2; exit 1; }
    if [[ "$(git -C "$path" rev-parse HEAD)" != "$rev" ]]; then
        [[ -z "$(git -C "$path" status --porcelain --untracked-files=no)" ]] || {
            echo "ERROR: $name has local edits; refusing to change its revision" >&2; exit 1;
        }
        git -C "$path" fetch origin "$rev"
        git -C "$path" checkout --detach "$rev"
    fi
}
checkout robosuite https://github.com/ARISE-Initiative/robosuite.git "$ROBOSUITE_REV"
checkout robocasa https://github.com/robocasa/robocasa.git "$ROBOCASA_REV"
if [[ ! -x "$VENV/bin/python" ]]; then
    uv venv "$VENV" --python 3.11
fi
uv pip install --python "$VENV/bin/python" --torch-backend cpu -r "$SCRIPT_DIR/requirements.txt"
uv pip install --python "$VENV/bin/python" -e "$SCRIPT_DIR/robosuite"
# RoboCasa pins lerobot==0.3.3; the C++ eval client does not need LeRobot.
uv pip install --python "$VENV/bin/python" --no-deps -e "$SCRIPT_DIR/robocasa"

for package in robosuite robocasa; do
    if [[ ! -f "$SCRIPT_DIR/$package/$package/macros_private.py" ]]; then
        "$VENV/bin/python" -m "$package.scripts.setup_macros"
    fi
done

if [[ "$SKIP_ASSETS" == 0 ]]; then
    # Run as a module: executing a file beside the robocasa checkout can shadow
    # the editable package with the checkout root's namespace directory.
    PYTHONPATH="$SCRIPT_DIR/../..${PYTHONPATH:+:$PYTHONPATH}" \
        "$VENV/bin/python" -m sim.robocasa.download_assets
fi

uv pip freeze --python "$VENV/bin/python" > "$SCRIPT_DIR/robocasa_uv/installed.txt"
echo "[setup] Python: $VENV/bin/python"
if [[ "$SKIP_SMOKE" == 0 ]]; then
    "$VENV/bin/python" "$SCRIPT_DIR/../../client/run_robocasa_client_direct.py" \
        --check-env --max-episode-steps 2
fi
