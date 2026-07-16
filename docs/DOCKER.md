# Docker evaluation workflow for vla.cpp

This document describes the Docker Compose evaluation stack, which runs the
vla.cpp inference server and the Python simulation client in separate
containers. The stack supports both **CUDA GPU** and **CPU-only** deployments.

| Container | Image | Purpose |
|-----------|-------|---------|
| `server` | root `Dockerfile` | C++ `vla-server` daemon (CUDA or CPU) |
| `client` | `eval/Dockerfile.client` | Python simulation environment (MuJoCo, LIBERO / SimplerEnv) |

> **Source of truth**: For model details, supported architectures, and
> benchmark numbers, refer to the top-level [README.md](../README.md).
> This document covers the Docker-specific evaluation workflow only.

---

## Quick start (CUDA GPU)

### Prerequisites

- [Docker Compose](https://docs.docker.com/compose/) v2.24+
- NVIDIA GPU with proprietary driver ≥ 535. GPU access uses **CDI**
  (`devices: - nvidia.com/gpu=all`) — no `nvidia-container-toolkit` needed.
  See [CUDA GPU access](#cuda-gpu-access) for details.

### 1. Download model GGUF files

```bash
docker compose -f eval/docker-compose.yml build client
docker compose -f eval/docker-compose.yml run --no-deps --rm client \
    hf download vrfai/smolvla-libero-gguf --local-dir /models
```

> `--no-deps` skips building the server image, which isn't needed for downloads.
> Each model ships as a single self-contained GGUF — no separate `mmproj` file.

Models are mounted into both containers at `/models`.

### 2. Build the images

```bash
docker compose -f eval/docker-compose.yml build
```

Build args accepted by the server `Dockerfile`:

| Arg | Default | Notes |
|-----|---------|-------|
| `BACKEND` | `cuda` | `cuda` or `cpu` |
| `CUDA_ARCH` | `120` | Blackwell; `89` for RTX40, `87` for Orin, `86` for RTX30 |
| `BASE_IMAGE` | `nvidia/cuda:12.9.1-devel-ubuntu24.04` | Overridden automatically for CPU |
| `JOBS` | `nproc` | Lower if nvcc segfaults on flash-attn kernels |

Override via e.g. `docker compose -f eval/docker-compose.yml build --build-arg CUDA_ARCH=89 server`.

### 3. Start the server

```bash
docker compose -f eval/docker-compose.yml up -d server
docker compose -f eval/docker-compose.yml logs server
# … vla-server: bound to tcp://*:5555. ready.
```

The default `command` in `eval/docker-compose.yml` starts SmolVLA for LIBERO.
For other models (e.g. GR00T-N1.7):

```bash
docker compose -f eval/docker-compose.yml run --rm server \
    --bind tcp://*:5555 /models/gr00tn1d7-libero.gguf
```

> **π0 note**: π0 needs a separate `mmproj` vision GGUF. Pass both files:
> `--bind tcp://*:5555 /models/mmproj-....gguf /models/ckpt.gguf`.
> See the [README model table](../README.md#models) for details.

### 4. Run a LIBERO evaluation episode

```bash
docker compose -f eval/docker-compose.yml run --rm client \
    python eval/client/run_sim_client_direct.py \
        --task libero_object --task-id 0 --n-episodes 1 \
        --output-dir /tmp/libero_outputs --arch smolvla \
        --vla-addr tcp://server:5555
```

Or drop into an interactive shell:

```bash
docker compose -f eval/docker-compose.yml run --rm client
root@...:/workspace/vla.cpp# python eval/client/run_sim_client_direct.py \
    --task libero_object --task-id 0 --n-episodes 1 \
    --output-dir /tmp/libero_outputs --arch smolvla \
    --vla-addr tcp://server:5555
```

Results (videos, summary) are written to `/tmp/libero_outputs` on the host.

**Example output (RTX 5060 Ti, CUDA arch 120):**

```
vla-cpp-direct[arch=smolvla]: connected to tcp://server:5555
- Step 220: reward=1.00, done=True, truncated=False
- Episode finished after 220 steps.  Final reward: 1.00
- Success rate: 100.00%  (1/1)
- Average inference time per step: 116.45 ms
```

---

## Quick start (CPU-only, no GPU)

On machines without an NVIDIA GPU,
build and run the server image with `BACKEND=cpu`:

### 1. Build the server image for CPU

```bash
docker build -t vla-cpp-cpu \
    --build-arg BACKEND=cpu \
    --build-arg BASE_IMAGE=ubuntu:24.04 .
```

### 2. Download the model

```bash
docker build -t vla-cpp-client -f eval/Dockerfile.client .
docker run --rm -v /tmp/smolvla-models:/models vla-cpp-client \
    hf download vrfai/smolvla-libero-gguf --local-dir /models
```

### 3. Start the server

```bash
docker run -d --name vla-cpp-server -p 5555:5555 \
    -v /tmp/smolvla-models:/models:ro \
    vla-cpp-cpu --bind tcp://*:5555 /models/smolvla-libero.gguf
```

Verify with `docker logs vla-cpp-server` — look for `vla-server: bound to tcp://*:5555. ready.`

### 4. Run a LIBERO evaluation episode

```bash
docker run --rm --network host \
    -v /tmp/smolvla-models:/models \
    -v /tmp/libero_outputs:/tmp/libero_outputs \
    vla-cpp-client \
    python eval/client/run_sim_client_direct.py \
        --task libero_object --task-id 0 --n-episodes 1 \
        --output-dir /tmp/libero_outputs --arch smolvla \
        --vla-addr tcp://localhost:5555
```

> CPU inference is significantly slower than GPU (e.g. ~888 ms/step on Apple M4
> vs ~113 ms/step on RTX 3090 for SmolVLA). Expect multi-minute episodes.

---

## Supported simulators

The Docker client image supports both simulators wired through the eval scaffold:

| Simulator | Supported arches | Setup script |
|-----------|-----------------|-------------|
| **LIBERO** | smolvla, pi0, pi05, gr00t_n1_5, gr00t_n1_6, gr00t_n1_7, bitvla, evo1, openvla_oft, vla_adapter, vla_jepa | `eval/sim/libero/setup_libero.sh` |
| **SimplerEnv** | gr00t_n1_6 | `eval/sim/simpler/setup_SimplerEnv.sh` |

### SimplerEnv example

```bash
docker compose -f eval/docker-compose.yml run --rm client \
    python eval/client/run_simpler_client_direct.py \
        --arch gr00t_n1_6 \
        --task-id oxe_widowx/widowx_spoon_on_towel --n-episodes 1 \
        --embodiment oxe_widowx --image-size 252 \
        --stats-json /models/dataset_statistics.json
```

---

## Configuration reference

### Volumes

| Host / Volume | Container mount | Purpose |
|---------------|----------------|---------|
| `/tmp/smolvla-models` | `client:/models` (rw), `server:/models:ro` | GGUF model files |
| `/tmp/libero_outputs` | `client:/tmp/libero_outputs` | Eval videos & summaries |
| `hf-cache` (named) | `client:/root/.cache/huggingface` | HuggingFace tokenizer cache |

### Ports

| Service | Host | Container |
|---------|------|-----------|
| server  | `5555` | `5555` |

### Network

Both services share the default Compose network. The client reaches the server
via hostname `server`.

### CUDA GPU access

The server uses CDI (`devices: - nvidia.com/gpu=all`). This works without
`nvidia-container-toolkit` as long as:
1. The NVIDIA proprietary driver is installed (≥ 535).
2. A CDI-enabled container runtime is available (containerd ≥ 1.7,
   cri-o ≥ 1.29, or Docker with `nvidia-ctk` from `nvidia-container-toolkit`
   ≥ 1.15 to generate `/etc/cdi/nvidia.yaml`).

---

## Running without Docker Compose

### Server only (GPU)

```bash
docker build -t vla-cpp-server \
    --build-arg BACKEND=cuda --build-arg CUDA_ARCH=120 .

# CDI
docker run --rm --device nvidia.com/gpu=all -p5555:5555 \
    -v /tmp/smolvla-models:/models:ro \
    vla-cpp-server --bind tcp://*:5555 /models/model.gguf

# nvidia-container-toolkit
docker run --rm --gpus all -p5555:5555 \
    -v /tmp/smolvla-models:/models:ro \
    vla-cpp-server --bind tcp://*:5555 /models/model.gguf
```

### Server only (CPU)

```bash
docker build -t vla-cpp-cpu \
    --build-arg BACKEND=cpu \
    --build-arg BASE_IMAGE=ubuntu:24.04 .

docker run --rm -p5555:5555 \
    -v /tmp/smolvla-models:/models:ro \
    vla-cpp-cpu --bind tcp://*:5555 /models/model.gguf
```

### Client only

```bash
docker build -t vla-cpp-client -f eval/Dockerfile.client .
docker run --rm -it --network host \
    -v /tmp/smolvla-models:/models \
    -v /tmp/libero_outputs:/tmp/libero_outputs \
    vla-cpp-client
# Inside: connect to server at localhost:5555
```

---

## Known issues

| Issue | Workaround |
|-------|-----------|
| `Unsupported gpu architecture 'compute_120'` with CUDA < 12.8 | Use CUDA 12.8+ for `sm_120`, or set `CUDA_ARCH=89` for RTX40-series compatibility |
| NumPy 2.x: `module 'numpy' has no attribute 'core'` | `Dockerfile.client` pins `numpy==1.26.4` and patches accelerate |
| `lerobot` pulls GPU torch | `Dockerfile.client` re-pins `torch==2.5.1` (CPU) after installing lerobot |
| LIBERO data files not found | Editable install (`-e`) keeps `bddl_files/` / `init_files/` / `assets/` accessible at runtime |
| LIBERO hangs on first import (dataset path prompt) | `echo "N" \| python3 -c "import libero.libero"` pre-seeds `~/.libero/config.yaml` |
| `pandas` segfaults on import | Pin `pandas==2.0.3` (last NumPy 1.x-compatible release) |
| MuJoCo 3.x: robosuite init fails | Pin `mujoco<3.0` (2.3.7 known-good) |
| `nvidia-container-toolkit` not installed | Use CDI (`devices: - nvidia.com/gpu=all`) instead of `runtime: nvidia` |
| CPU-only: no GPU available | Use `BACKEND=cpu` build arg and `BASE_IMAGE=ubuntu:24.04` instead of CUDA base |

---

## Summary

The Docker Compose evaluation stack provides a reproducible two-container
workflow for vla.cpp:

1. **Server** — upstream `Dockerfile`, compiles `vla-server` with GPU or CPU backend.
2. **Client** — `eval/Dockerfile.client`, Python simulation stack with pinned
   dependency versions (NumPy 1.x, MuJoCo 2.x, Pandas 2.0.x).
3. **CDI** eliminates the `nvidia-container-toolkit` dependency for GPU access.
4. **CPU-only** mode works without any GPU — use `BACKEND=cpu` build arg.
5. **First-step overhead** (~35 s CUDA graph warmup) occurs once per process (GPU only).