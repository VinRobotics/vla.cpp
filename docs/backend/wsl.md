# `vla.cpp` on Windows (WSL2 + CUDA)

`vla.cpp` targets Linux and macOS. On Windows the supported path is **WSL2**
with an Ubuntu distribution: the toolchain (`libzmq`, `protobuf`, `pkg-config`)
and the CUDA build all run natively inside the Linux environment, while still
using the host NVIDIA GPU through the WSL CUDA driver.

## Prerequisites

A WSL2 distribution and a recent **NVIDIA Windows driver** that exposes the GPU to WSL.
Confirm the GPU is visible from inside WSL before building:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# e.g. NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB, 556.29
```

> Note: the Windows driver ships the WSL CUDA *driver*, but not the CUDA
> *toolkit* (`nvcc`). The toolkit is installed inside WSL, below.

Install the build dependencies and a CUDA toolkit that matches your driver
(e.g. driver 556.29 supports CUDA ≤ 12.6; CUDA ≥ 12.4 is required for the GCC 13
shipped on Ubuntu 24.04):

```bash
sudo apt-get update
sudo apt-get install -y libzmq3-dev libprotobuf-dev protobuf-compiler \
    pkg-config build-essential cmake git wget

# CUDA toolkit (nvcc) via the WSL-specific repo
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6

export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
nvcc --version   # confirm the toolkit is on PATH
```

### Windows `PATH` translation errors

WSL imports the Windows `PATH` into the Linux environment. A stale Windows
`PATH` entry that points at a drive no longer present produces, on every
command:

```
<3>WSL (NNN) ERROR: UtilTranslatePathList:2852: Failed to translate E:\Some\Dir
```

It is harmless, but to remove it delete the dead entry from the Windows user
`PATH` (PowerShell), then restart WSL:

```powershell
$p = [Environment]::GetEnvironmentVariable('Path','User') -split ';'
$p = $p | Where-Object { $_ -and $_ -ne 'E:\Some\Dir' }
[Environment]::SetEnvironmentVariable('Path', ($p -join ';'), 'User')
```

```powershell
wsl --shutdown   # reopen WSL afterwards so the new PATH is picked up
```

## Configure & build

```bash
# CUDA build. Set CMAKE_CUDA_ARCHITECTURES for your GPU (see the table in the
# top-level README; e.g. RTX 40-series / Ada = 89).
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURE
cmake --build build -j"$(nproc)"
```

## GPU offload

The VLA core selects its compute backend at load time. With a CUDA build it
picks the CUDA backend; confirm it from the SmolVLA startup banner:

```
vla: backend = CUDA (device 0)
```

If you instead see `vla: backend = CPU (4 threads)`, the build did not pick up
CUDA - rebuild from a clean `build/` and check `GGML_CUDA` is `ON` in the CMake
cache and that `nvcc` was on `PATH` at configure time.

## Run a model

SmolVLA ships a combined GGUF plus a separate `mmproj` vision tower (see
[Models](../../README.md#models); GGUF published
[here](https://huggingface.co/collections/vrfai/vlacpp-model-bundles)).

```bash
./build/vla-server "$VLA_GGUF"
# vla-server: bound to tcp://*:5555. ready.
```

## End-to-end evaluation (LIBERO client on WSL)

The eval client (LIBERO simulator + preprocessing) can run in the *same* WSL
distribution as `vla-server`, so the full serving loop - sim reset/render →
client preprocess → `vla-server` inference → action → sim step → video - runs on
one Windows machine with no separate Linux host.

Two WSL-specific gotchas:

- **Work from the Linux filesystem, not `/mnt/c` or `/mnt/d`.** `uv`/venv installs
  are slow and unreliable on the Windows-mounted drives (drvfs); `git clone` the
  repo into your home directory (e.g. `~/vla.cpp`) and run from there. This also
  sidesteps CRLF issues in the shell scripts.
- **Add your user to the `render` group** so MuJoCo's EGL renderer can open the
  GPU render node (`/dev/dri/renderD128`, owned `root:render`). Without it, EGL
  fails with `libEGL warning: failed to open /dev/dri/renderD128: Permission
  denied` and falls back to software rendering:

  ```bash
  sudo usermod -aG render "$USER"
  # then, from PowerShell, restart WSL so the new group takes effect:
  #   wsl --shutdown
  ```

Install the LIBERO client into an isolated `uv` venv. Set `UV_LINK_MODE=copy` and
a home-directory cache so `uv` does not try to hardlink across filesystems:

```bash
export UV_LINK_MODE=copy
export UV_CACHE_DIR=$HOME/.cache/uv-vla
bash eval/sim/libero/setup_libero.sh
```

Then serve SmolVLA and drive one episode. The client renders with EGL and talks
to `vla-server` over ZeroMQ:

```bash
# terminal 1 - server (GPU inference)
./build/vla-server --bind 'tcp://*:5555' "$VLA_GGUF"

# terminal 2 - client (LIBERO sim + preprocessing)
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0
eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py \
    --arch smolvla --vla-addr tcp://localhost:5555 \
    --task libero_object --task-id 0 --n-episodes 1 --n-action-steps 1 \
    --output-dir outputs/wsl_smoke
```

The client writes `episode_000000.mp4` and `summary.txt` under
`outputs/wsl_smoke/smolvla/libero_object/task_0/`.

### One-command runner

`eval/run_libero_wsl.sh` wraps the two steps above - it starts `vla-server`,
waits for it to become ready, drives one client run, and stops the server on
exit. Pass the task-id and episode count as arguments, or run with none and it
prompts for them:

```bash
bash eval/run_libero_wsl.sh          # prompts for task-id + episodes
bash eval/run_libero_wsl.sh 3 5      # task-id 3, 5 episodes, no prompt
```

Override the served model, suite, or output dir via environment
(`VLA_GGUF`, `TASK_SUITE`, `OUTPUT_DIR`); see the header of the script.

## Results

Evo1 (libero, 1.20 GiB BF16 weights incl. InternViT vision tower),
**NVIDIA GeForce RTX 3050 Ti Laptop GPU** (4 GB VRAM, ~2.8 GB allocated),
steady state:

| Stage        | CUDA (WSL2) |
|--------------|------------:|
| vision       |   ~1,000 ms |
| inference    |     ~400 ms |
| other        |     ~350 ms |
| **total/req**| **~1,750 ms** |

On libero_object task 0 (pick up the alphabet soup and place it in the basket),
the episode succeeded after 171 steps at ~2114 ms/step client-side.

SmolVLA (libero, 532 MB GGUF), **NVIDIA GeForce RTX 4050 Laptop GPU** (6 GB VRAM,
~1.07 GB allocated), with the full server + LIBERO client stack co-resident on a
single WSL2 (Ubuntu 24.04) machine, steady state:

| Stage         | CUDA (WSL2) |
|---------------|------------:|
| vision        |      ~84 ms |
| inference     |      ~46 ms |
| other         |       ~1 ms |
| **total/req** | **~131 ms** |

On libero_object task 0 the episode succeeded end-to-end (`is_success: True`,
149 requests served) at ~155 ms/step client-side (including software EGL
rendering), producing a rendered MP4.
