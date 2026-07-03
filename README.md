
# vla.cpp

![logo](assets/logo_vlacpp_white.png)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.md)
[![Built on llama.cpp](https://img.shields.io/badge/built%20on-llama.cpp-lightgrey)](https://github.com/ggml-org/llama.cpp)
[![Models on HF](https://img.shields.io/badge/%F0%9F%A4%97%20models-Hugging%20Face-yellow)](https://huggingface.co/vrfai)
[![arXiv](https://img.shields.io/badge/arXiv-2606.08094-b31b1b.svg)](http://arxiv.org/abs/2606.08094)

An efficient C++ inference engine for **Vision-Language-Action (VLA) models**, built on top of [`llama.cpp`](https://github.com/ggml-org/llama.cpp).
It brings today's open VLA policies - SmolVLA, π0, BitVLA, Evo-1, and GR00T N1.5/1.6/1.7
and more - under one runtime, packaging each as a single self-contained GGUF that needs no Python or PyTorch at inference time.
The binary can drive robots across **CPU**, **Apple Silicon**, or **CUDA**, scaling from consumer GPUs down to the Jetson-class boards.

## Build the server

### Prerequisites

- CMake ≥ 3.22
- A C++17 compiler (GCC 11+ or Clang 14+)
- CUDA 12.x (optional - required only for GPU builds)
- `libzmq3-dev`, `libprotobuf-dev`, `protobuf-compiler`

```bash
sudo apt-get install -y libzmq3-dev libprotobuf-dev protobuf-compiler
```

### From source

Identify your machine CUDA architecture:

| GPU family | Example cards | `CUDA_ARCHITECTURE` |
|---|---|---|
| Ampere (Jetson) | Orin Nano, Orin NX | `87` |
| Ampere (consumer) | RTX 30-series, A40 | `86` |
| Ada Lovelace | RTX 40-series, L40 | `89` |
| Hopper | H100, H200 | `90` |
| Blackwell (consumer) | RTX 50-series | `120` |
| Blackwell (datacenter) | B100, B200, GB200 | `100` |


Configure and build the source:

CMake fetches and pins `llama.cpp` automatically (no patch, no submodule):

```bash
# CPU build:
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# CUDA build (set CMAKE_CUDA_ARCHITECTURES for your GPU):
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURE
cmake --build build -j$(nproc)
```

If system cannot detect CUDA, declare CUDA explicitly in environment variables

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

On Apple Silicon (e.g. Mac Mini M4), Metal is enabled by default and runs both the transformer and vision tower on the GPU. See [docs/backend/metal.md](docs/backend/metal.md) for building `vla.cpp` on macOS.

## Install simulators

The eval scaffold under [`eval/`](eval/) supports two simulators end-to-end. Each setup script bootstraps an isolated Python 3.10 `uv` venv next to itself and clones the upstream sim repo. Both require [`uv`](https://github.com/astral-sh/uv) on `PATH`.

### LIBERO

```bash
bash eval/sim/libero/setup_libero.sh
```

Clones LIBERO into [`eval/sim/libero/LIBERO/`](eval/sim/libero/LIBERO), creates `eval/sim/libero/libero_uv/.venv/`, and pins compatible versions of torch, lerobot, transformers, and gymnasium.

### SimplerEnv

```bash
bash eval/sim/simpler/setup_SimplerEnv.sh
```

Clones SimplerEnv (and its nested `ManiSkill2_real2sim`) into [`eval/sim/simpler/SimplerEnv/`](eval/sim/simpler/SimplerEnv), creates `eval/sim/simpler/simpler_uv/.venv/`, and pins gymnasium 0.29.1, numpy 1.26.4, transformers 4.51.3, plus the ManiSkill2 + SimplerEnv editable installs.

## Running the server

`vla-server` loads the model once at startup and answers ZeroMQ REQ/REP requests synchronously.

```bash
# SmolVLA / π0 (mmproj + ckpt):
./build/vla-server "$VLA_MMPROJ" "$VLA_GGUF"

# Evo-1 / BitVLA / GR00T-N1.{5,6,7} (vision baked into the ckpt - omit mmproj):
./build/vla-server "$VLA_GGUF"
```

When ready, the server prints:

```
vla-server: bound to tcp://*:5555. ready.
```

Bound address and port can be configured by `--bind` flag. Stop server with `Ctrl-C`.


## Running the client

[`eval/client/`](eval/client/) ships an end-to-end LIBERO benchmark runner that drives `vla-server` directly over the protobuf protocol. Make sure the LIBERO venv from [Install simulators](#install-simulators) is set up first.

### Run an episode (LIBERO)

With `vla-server` already running:

```bash
source eval/sim/libero/libero_uv/.venv/bin/activate
python eval/client/run_sim_client_direct.py \
    --task libero_object --task-id 0 --n-episodes 1 \
    --output-dir /tmp/libero_outputs \
    --arch "$VLA_ARCH"
```

Note:

- Use proper `--arch` flag (see [Models](#models)) to match the GGUF that `vla-server` is serving.
- Pi0 uses the gated `google/paligemma-3b-pt-224` tokenizer (`huggingface-cli login` + accept the licence, or point `--tokenizer` at a local copy)
- The GR00T arches need `--stats-json <ckpt>/dataset_statistics.json` (action/state un-normalisation) and an embodiment selected server-side via `VLA_GR00T_EMBODIMENT` (`new_embodiment` for N1.5, `libero_panda` for N1.6, `libero_sim` for N1.7), with `VLA_GR00T_BF16_WEIGHTS=1` to fit the 8 GB card.

### Run an episode (SimplerEnv)

So far only **GR00T-N1.6** is wired (the `gr00t-n1d6-bridge` checkpoint with the `oxe_widowx` embodiment). Start `vla-server` on port 5566 with `oxe_widowx` embodiment:

```bash
VLA_GR00T_BF16_WEIGHTS=1 VLA_GR00T_EMBODIMENT=oxe_widowx \
    ./build/vla-server "$GR00T_N1D6_GGUF"
```

Then drive it from the SimplerEnv venv (set up via [Install simulators](#install-simulators)):

```bash
source eval/sim/simpler/simpler_uv/.venv/bin/activate
python eval/client/run_simpler_client_direct.py \
    --arch gr00t_n1_6 \
    --task-id oxe_widowx/widowx_spoon_on_towel --n-episodes 1 \
    --embodiment oxe_widowx --image-size 252 \
    --stats-json "$VLA_STATS_JSON"
```

`$VLA_STATS_JSON` is the `statistics.json` shipped beside the bridge GGUF. The default 224-px GGUF mis-localises on WidowX (≈20% success) - the 252-px build is required.

## Models

Each model ships a combined VLA GGUF (LM + action expert + dataset stats + arch config) and, where applicable, a matching mmproj GGUF (vision tower). SmolVLA, π0 and π0.5 ship a separate mmproj; BitVLA, Evo-1, VLA-Adapter and OpenVLA-OFT bake their vision tower into the combined GGUF, so no mmproj file is needed.

| Model       | Converted GGUF | Source ckpt | Client `--arch` flag |
|---|---|---|---|
| SmolVLA      | [`smolvla-libero`](https://huggingface.co/vrfai/smolvla-libero-gguf) | [link](https://huggingface.co/HuggingFaceVLA/smolvla_libero) | `smolvla` |
| π0           | [`pi0-libero`](https://huggingface.co/vrfai/pi0-libero-finetuned-v044-gguf) | [link](https://huggingface.co/lerobot/pi0_libero_finetuned_v044) | `pi0` |
| BitVLA       | [`bitvla-libero`](https://huggingface.co/vrfai/bitvla-libero-gguf) | [link](https://huggingface.co/hongyuw/ft-bitvla-bitsiglipL-224px-libero_object-bf16) | `bitvla` |
| OpenVLA-OFT  | [`openvla-oft-libero`](https://huggingface.co/vrfai/openvla-oft-libero-gguf) | [link](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10) | `openvla_oft` |
| Groot-N1.7   | [`gr00tn1d7-libero`](https://huggingface.co/vrfai/gr00tn1d7-libero-gguf) | [link](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) | `gr00t_n1_7` |

`vla.cpp` also supports `gr00t_n1_5`, `gr00t_n1_6`, `evo1`, `vla_adapter`, `vla_jepa` and `pi05`.
If you would rather convert a HuggingFace safetensors checkpoint yourself, [`scripts/`](scripts/) provides per-arch GGUF converters.
Set up a venv for converter by:

```bash
python3 -m venv .venv-converter
source .venv-converter/bin/activate
pip install -e ".[convert]"   # torch, safetensors, gguf, numpy (see pyproject.toml)
```

Then run any of the per-arch converters (`--help` for the full flag list):

```bash
python scripts/convert_smolvla_to_gguf.py \
    --ckpt /path/to/smolvla-libero \
    --out  /path/to/smolvla-libero-bf16.gguf
```

## Benchmarks

Full `libero_object` sweep - all 10 tasks × 20 episodes (200 episodes per arch),
run via `vla-server` + `eval/client/run_sim_client_direct.py` across three deployment
targets: an **RTX 3060** (sm_86), an **NVIDIA Jetson AGX Orin** (sm_87, Jetson-class
deployment hardware), and an **NVIDIA Jetson Orin Nano (8 GB)** (sm_87, the cheapest
Jetson and the project's primary deployment target). On the Orin Nano, the 8 GB budget
forces the ~6 GB servers (`gr00t_n1_5`, `pi0`) to run split - server on the Nano, client
(sim) on the RTX 3060 - while lighter models run co-located; GR00T-N1.6 and GR00T-N1.7
could not be loaded on 8 GB and are omitted there.

Columns report `client/call (ms)` and `Peak RAM (MiB)` for each hardware target.

| Model | 3060 call (ms) | 3060 RAM (MiB) | AGX Orin call (ms) | AGX Orin RAM (MiB) | Orin Nano call (ms) | Orin Nano RAM (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| `smolvla`    |  113 | 1410 |  262 |  689.4 |  567 | 2031.2 |
| `bitvla`     |  303 | 1312 |  809 | 1148.8 | 2845 | 2199.0 |
| `evo1`       |  509 | 1564 | 1048 |  637.5 | 3671 | 2135.0 |
| `pi0`        |  312 | 5548 |  893 |  640.4 | 1955 | 6067.7 |
| `gr00t_n1_5` |  227 | 4866 |  461 | 1331.3 | 1356 | 5974.9 |
| `gr00t_n1_6` |  165 | 6048 |  427 | 1340.5 |    - |      - |
| `gr00t_n1_7` |  164 | 6302 |  429 | 1316.5 |    - |      - |

## Roadmap

Support matrix of models (rows) against platforms (columns). Legend: `Y` =
supported (released and benchmarked), `~` = in progress, `-` = planned.

| Model | CPU (x86-64 / ARM) | CUDA | Metal | OpenVINO | Hexagon |
|---|:--:|:--:|:--:|:--:|:--:|
| SmolVLA     | Y | Y | Y | - | - |
| π0          | Y | Y | Y | - | - |
| π0.5        | Y | Y | ~ | - | - |
| GR00T N1.5  | Y | Y | ~ | - | - |
| GR00T N1.6  | Y | Y | ~ | - | - |
| GR00T N1.7  | Y | Y | Y | - | - |
| BitVLA      | Y | Y | ~ | - | - |
| Evo-1       | Y | Y | ~ | - | - |
| VLA-Adapter | Y | Y | ~ | - | - |
| OpenVLA-OFT | Y | Y | ~ | - | - |
| VLA-JEPA    | Y | Y | ~ | - | - |

Looking ahead, we will support more models, more platforms, and continue to
optimize the framework.

## Contributors

- [Khanh Dang Nguyen](https://github.com/khanhnd61-vr)
- [Hung Thinh Ho](https://github.com/hungho77)
- [Chinh Truong Nguyen](https://github.com/nguyentruongchinh04z)
- [An Thai Le](https://github.com/anindex)

## License

Licensed under the [Apache License, Version 2.0](LICENSE.md).

## Acknowledgements

Supported VLA models:

- [SmolVLA](https://huggingface.co/lerobot/smolvla_base) - Hugging Face LeRobot team.
- [π0,π0.5](https://github.com/Physical-Intelligence/openpi) - Physical Intelligence.
- [BitVLA](https://github.com/ustcwhy/BitVLA) - Hongyu Wang et al.
- [Evo-1](https://github.com/MINT-SJTU/Evo-1/tree/main) - Tao Lin et al.
- [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) - Yihao Wang et al.
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft) - Moo Jin Kim et al.
- [GR00T N1.x](https://github.com/NVIDIA/Isaac-GR00T) - NVIDIA Isaac.
- [VLA-JEPA](https://github.com/ginwind/VLA-JEPA) - Jingwen Sun et al.

Behavioural evaluation is built on:

- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) - LLM inference engine in C/C++.
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) - the lifelong-robot-learning benchmark suite our success-rate sweeps run on.
- [SimplerEnv](https://github.com/simpler-env/SimplerEnv) - the second simulator wired through the eval scaffold.
