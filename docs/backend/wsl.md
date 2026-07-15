# `vla.cpp` on Windows (WSL2 + CUDA)

`vla.cpp` targets Linux and macOS. On Windows the supported path is **WSL2**
with an Ubuntu distribution: the toolchain (`libzmq`, `protobuf`, `pkg-config`,
the bash `patches/patch.sh` script) and the CUDA build all run natively inside
the Linux environment, while still using the host NVIDIA GPU through the
WSL CUDA driver.

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
