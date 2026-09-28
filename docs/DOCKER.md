# Docker evaluation workflow for vla.cpp

This document describes the Docker Compose evaluation stack, which runs the
vla.cpp inference server and the Python simulation client in separate
containers. The current `eval/docker-compose.yml` is a **CUDA GPU** stack that
requests an NVIDIA CDI device; CPU-only Docker commands are provided separately
below.

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
- NVIDIA driver 525 or newer. The default image is CUDA 12.9, which runs on
  older 12.x drivers under CDI. With `--gpus all` on a GeForce card the
  driver has to be 575 or newer; see [Known issues](#known-issues).
- CDI GPU access for Docker (`devices: - nvidia.com/gpu=all`). See
  [CUDA GPU access](#cuda-gpu-access) for runtime setup details.

### 1. Download model GGUF files

```bash
docker compose -f eval/docker-compose.yml build client
docker compose -f eval/docker-compose.yml run --no-deps --rm client \
    hf download vrfai/smolvla-libero-gguf --local-dir /models
```

> `--no-deps` skips building the server image, which isn't needed for downloads.
> The Compose file mounts the host directory `/tmp/smolvla-models` into both
> containers as `/models`; the default server command expects
> `/models/smolvla-libero.gguf`.

Models are mounted into both containers at `/models`.

### 2. Build the images

```bash
docker compose -f eval/docker-compose.yml build
```

Build args accepted by the server `Dockerfile`:

| Arg | Default | Notes |
|-----|---------|-------|
| `BACKEND` | `cuda` | `cuda` or `cpu`. `cpu` builds and runs on `ubuntu:24.04` |
| `CUDA_VERSION` | `12.9.1` | Picks the `nvidia/cuda` `-devel` build and `-runtime` run images. `13.4.1` needs driver 580 or newer |
| `CUDA_ARCH` | `120` in Compose, `75-real;80-real;86-real;89-real;90;120-real;121-real` in the `Dockerfile` | One arch builds much faster: `86` RTX30, `89` RTX40, `90` H100, `87` Orin, `120` RTX50 |
| `GGML_NATIVE` | `ON` | Tunes the CPU code for the build machine. `OFF` gives a portable image |
| `JOBS` | `nproc` | Lower if nvcc segfaults on flash-attn kernels |

The build runs in a `-devel` stage. The final image holds only `vla-server`,
`vla-cli` and their libraries in `/app`, on the matching `-runtime` base.
A local build is tuned for the CPU it was built on. The published image is
built with `GGML_NATIVE=OFF` and runs on any x86-64 CPU with AVX2.

Override the arch from the environment, `CUDA_ARCH=89 docker compose -f eval/docker-compose.yml build server`,
or per build, `docker compose -f eval/docker-compose.yml build --build-arg CUDA_ARCH=89 server`.

### 3. Start the server

```bash
docker compose -f eval/docker-compose.yml up -d server
docker compose -f eval/docker-compose.yml logs server
# … vla-server: bound to tcp://*:5555. ready.
```

The default `command` in `eval/docker-compose.yml` starts SmolVLA for LIBERO:
`--bind tcp://*:5555 /models/smolvla-libero.gguf`. To serve another model,
create a Compose override that replaces only `server.command`; for example:

```bash
cat >/tmp/vla-compose.override.yml <<'YAML'
services:
  server:
    command:
      - --bind
      - tcp://*:5555
      - /models/gr00tn1d7-libero.gguf
YAML

docker compose -f eval/docker-compose.yml -f /tmp/vla-compose.override.yml up -d server
```

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

The checked-in Compose file requests `devices: - nvidia.com/gpu=all`, so use
plain `docker build` / `docker run` for a CPU-only host unless you also maintain
a local Compose override that removes the GPU device request. Build and run the
server image with `BACKEND=cpu`:

### 1. Build the server image for CPU

```bash
docker build -t vla-cpp-cpu --build-arg BACKEND=cpu .
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

The server service in `eval/docker-compose.yml` uses CDI
(`devices: - nvidia.com/gpu=all`). This works when:
1. The NVIDIA proprietary driver is installed (525 or newer for the default
   CUDA 12.9 image, 580 or newer for `CUDA_VERSION=13.4.1`).
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

Each release also publishes this image, built for CUDA 12.9 and sm_75 to sm_120:
`ghcr.io/vinrobotics/vla.cpp:<tag>` or `:latest`.

### Server only (CPU)

```bash
docker build -t vla-cpp-cpu --build-arg BACKEND=cpu .

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
| `Unsupported gpu architecture 'compute_121'` (or `compute_120`) | `CUDA_VERSION` is too old for the arch list: 121 needs 12.9, 120 needs 12.8. Use the default `12.9.1`, or pass a `CUDA_ARCH` without them |
| `unsatisfied condition: cuda>=12.9` (or `cuda>=13.4`) with `--gpus all` | The container toolkit waives this check only for datacenter and workstation cards. On GeForce, update the driver, use CDI, or add `-e NVIDIA_DISABLE_REQUIRE=1` (driver 525+ for the default image, 580+ for `CUDA_VERSION=13.4.1`) |
| NumPy 2.x: `module 'numpy' has no attribute 'core'` | `Dockerfile.client` pins `numpy==1.26.4` and patches accelerate |
| `lerobot` pulls GPU torch | `Dockerfile.client` re-pins `torch==2.5.1` (CPU) after installing lerobot |
| LIBERO data files not found | Editable install (`-e`) keeps `bddl_files/` / `init_files/` / `assets/` accessible at runtime |
| LIBERO hangs on first import (dataset path prompt) | `echo "N" \| python3 -c "import libero.libero"` pre-seeds `~/.libero/config.yaml` |
| `pandas` segfaults on import | Pin `pandas==2.0.3` (last NumPy 1.x-compatible release) |
| MuJoCo 3.x: robosuite init fails | Pin `mujoco<3.0` (2.3.7 known-good) |
| `nvidia-container-toolkit` not installed | Use CDI (`devices: - nvidia.com/gpu=all`) instead of `runtime: nvidia` |
| CPU-only: no GPU available | Use the CPU-only `docker build` / `docker run` flow above, or maintain a Compose override that removes `devices: - nvidia.com/gpu=all` and builds with `BACKEND=cpu` |

---

## Summary

The Docker evaluation stack provides a reproducible two-container workflow for
vla.cpp:

1. **Server**: root `Dockerfile`, builds `vla-server` for CUDA by default or
   for CPU with `BACKEND=cpu`, and ships it on a runtime-only base.
2. **Client**: `eval/Dockerfile.client`, Python simulation stack with pinned
   dependency versions (NumPy 1.x, MuJoCo 2.x, Pandas 2.0.x).
3. **CDI** is the GPU access path used by the checked-in Compose file.
4. **CPU-only** mode works without any GPU through the standalone Docker commands
   above.
5. **First-step overhead** (~35 s CUDA graph warmup) occurs once per process (GPU only).
