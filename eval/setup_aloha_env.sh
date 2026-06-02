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

# Setup Python venv for the ALOHA real-robot inference client.
#
# Creates eval/aloha_uv/.venv with --system-site-packages so that ROS2
# and Interbotix packages (rclpy, cv_bridge, interbotix_xs_msgs) remain
# accessible without reinstalling them.
#
# Run once from the repo root:
#   bash eval/setup_aloha_env.sh
#
# To activate afterwards:
#   source eval/aloha_uv/.venv/bin/activate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/aloha_uv/.venv"

JETSON_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126"
JETSON_HOST="pypi.jetson-ai-lab.io"

# ---------------------------------------------------------------------------
# Sync system clock (certificate time-validation fails if clock is skewed)
# ---------------------------------------------------------------------------
echo ">> Syncing system clock (required for TLS certificate validation)..."
if command -v timedatectl &>/dev/null; then
    sudo timedatectl set-ntp true 2>/dev/null && sleep 2 \
        && echo "   timedatectl NTP enabled" \
        || echo "   timedatectl failed (no sudo?), skipping"
elif command -v ntpdate &>/dev/null; then
    sudo ntpdate -u pool.ntp.org 2>/dev/null \
        && echo "   ntpdate synced" \
        || echo "   ntpdate failed, skipping"
else
    echo "   no clock-sync tool found, skipping"
fi

# ---------------------------------------------------------------------------
# Create venv
# ---------------------------------------------------------------------------
echo ">> Creating venv at $VENV_DIR"
echo "   (--system-site-packages to inherit ROS2 + Interbotix packages)"
mkdir -p "$(dirname "$VENV_DIR")"
uv venv "$VENV_DIR" --python 3.10 --system-site-packages
source "$VENV_DIR/bin/activate"

# ---------------------------------------------------------------------------
# Install PyTorch + torchvision from Jetson AI Lab index
# ---------------------------------------------------------------------------
echo ""
echo ">> Installing torch + torchvision from $JETSON_INDEX"
UV_HTTP_TIMEOUT=600 UV_INSECURE_HOST="$JETSON_HOST" \
    uv pip install "torch==2.8.0" "torchvision" \
        --index-url "$JETSON_INDEX" \
        --allow-insecure-host "$JETSON_HOST"

# ---------------------------------------------------------------------------
# Install remaining dependencies
# ---------------------------------------------------------------------------
echo ""
echo ">> Installing transformers, Pillow, pyzmq, protobuf, opencv"
uv pip install \
    "transformers==4.51.3" \
    "Pillow>=10.0" \
    "pyzmq>=25.0" \
    "protobuf>=4.21" \
    "numpy>=1.24,<2.0" \
    "opencv-python-headless>=4.5"

# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
echo ""
echo ">> Smoke-testing key imports..."
python - <<'PYEOF'
import sys
ok = True

def chk(mod, label=None):
    global ok
    try:
        __import__(mod)
        print(f"  OK  {label or mod}")
    except ImportError as e:
        print(f"  FAIL {label or mod}: {e}")
        ok = False

chk("torch")
chk("torchvision")
chk("transformers")
chk("zmq")
chk("cv2")
chk("PIL", "Pillow")
chk("google.protobuf", "protobuf")
chk("rclpy")
chk("cv_bridge")
chk("interbotix_xs_msgs.msg", "interbotix_xs_msgs")

import torch
print(f"\n  torch version : {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device   : {torch.cuda.get_device_name(0)}")

if not ok:
    print("\nWARNING: some imports failed (see above).")
    sys.exit(1)
else:
    print("\nAll imports OK.")
PYEOF

echo ""
echo "================================================================"
echo "  Setup complete."
echo ""
echo "  Activate with:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  Then run the server + client:"
echo "    bash eval/run_aloha_server.sh"
echo "    bash eval/run_aloha_client.sh --task \"pick up the cup\""
echo "================================================================"
