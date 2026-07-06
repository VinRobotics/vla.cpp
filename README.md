
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

## Quickstart

Once the binaries are built, run one CPU prediction without a server or simulator:

```bash
pip install -U "huggingface_hub[cli]" gguf
hf download vrfai/smolvla-libero-gguf --local-dir models/smolvla

# front.jpg must already be the model input size (512x512 for this checkpoint).
./build/vla-cli --ckpt models/smolvla/smolvla-libero.gguf \
    --image front.jpg --tokens 1,100,200,2 --pretty
```

`--tokens` are language token ids from the client tokenizer. For the design overview
see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); for the server path and other models
see [Roadmap](#roadmap).

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

Clones SimplerEnv (and its nested `ManiSkill2_real2sim`) into [`eval/sim/simpler/SimplerEnv/`](eval/sim/simpler/SimplerEnv), creates `eval/sim/simpler/simpler_uv/.venv/`.

## Running the server

`vla-server` loads the model once at startup and answers ZeroMQ REQ/REP requests synchronously.

```bash
./build/vla-server "$VLA_GGUF"
```

When ready, the server prints:

```
vla-server: bound to tcp://*:5555. ready.
```

Bound address and port can be configured by `--bind` flag. Stop server with `Ctrl-C`.

## One-shot CLI

`vla-cli` runs a single prediction without a server or simulator: give it a model,
an image, and the tokenized instruction, and it prints the action chunk. Handy for
smoke-testing a GGUF or scripting a quick inference.

```bash
./build/vla-cli --ckpt "$VLA_GGUF" \
    --image front.jpg --image wrist.jpg --tokens 1,100,200,2 --pretty
```

Tokenization stays in the Python client, so the instruction is passed as token ids.
`--pretty` prints one action row per line; `--state` sets proprioception (defaults to zeros).

## Running the client

[`eval/client/`](eval/client/) ships an end-to-end LIBERO benchmark runner that drives `vla-server` directly over the protobuf protocol. Make sure the LIBERO venv from [Install simulators](#install-simulators) is set up first.

### LIBERO

With `vla-server` already running:

```bash
source eval/sim/libero/libero_uv/.venv/bin/activate
python eval/client/run_sim_client_direct.py \
    --task libero_object --task-id 0 --n-episodes 1 \
    --output-dir /tmp/libero_outputs \
    --arch "$VLA_ARCH"
```

The GR00T models needs:
  -  `--stats-json /path/to/dataset_statistics.json` in client side. 
  - `VLA_GR00T_EMBODIMENT` (`new_embodiment` for N1.5, `libero_panda` for N1.6, `libero_sim` for N1.7) and `VLA_GR00T_BF16_WEIGHTS=1` (to fit the 8 GB card) in server side.

### SimplerEnv

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

## Models

### Conversion

Each model ships as a single self-contained GGUF.
If you would rather convert a HuggingFace safetensors checkpoint yourself, [`scripts/`](scripts/) provides per-arch GGUF converters.
Set up a venv for converter by:

```bash
python3 -m venv .venv-converter
source .venv-converter/bin/activate
pip install -e ".[convert]"
```

Then run any of the per-arch converters (`--help` for the full flag list):

```bash
python scripts/convert_smolvla_to_gguf.py \
    --ckpt /path/to/smolvla-libero \
    --out  /path/to/smolvla-libero-bf16.gguf
```

### Quantization

The shipped GGUFs are bf16. `scripts/quantize_gguf.py` repacks the LM-backbone weight
matrices to a smaller type and copies everything else unchanged; the loader keeps the
packed weights and lets `ggml_mul_mat` dequantize at compute, so the file just loads and
runs like the bf16 one.

```bash
python scripts/quantize_gguf.py --in model-bf16.gguf --out model-q8_0.gguf --type Q8_0
```

`Q8_0` is near-lossless and roughly halves the LM. `Q4_0` is 4-bit for a bigger cut.
Embeddings, the output head, norms and the action expert stay float; pass `--vision` to
pack the vision tower too (smaller, but more accuracy loss).

## Benchmarks

Latency (ms, inference time + transport overhead) measured at client sides
across four deployment targets: an **RTX 3090**, an **NVIDIA Jetson AGX
Orin**, an **NVIDIA Jetson Orin Nano (8 GB)**, and an **Apple M4**.


| Model | 3090 call (ms) | AGX Orin call (ms) | Orin Nano call (ms) | M4 call (ms) |
|---|---:|---:|---:|---:|
| `smolvla`     |  113 |  262 |  567 |  888 |
| `pi0`         |  312 |  893 | 1955 | 1135 |
| `gr00t_n1_5`  |  227 |  461 | 1356 |    - |
| `gr00t_n1_7`  |  164 |  429 |    - |  755 |
| `bitvla`      |  303 |  809 | 2845 |    - |
| `evo1`        |  509 | 1048 | 3671 |    - |

## Roadmap

Support matrix of models (rows) against platforms (columns). Legend: `Y` =
supported (released and benchmarked), `~` = in progress, `-` = planned.

| Model | CPU (x86-64 / ARM) | CUDA | Metal | OpenVINO | Hexagon |
|---|:--:|:--:|:--:|:--:|:--:|
| [SmolVLA](https://hf.co/vrfai/smolvla-libero-gguf)             | Y | Y | Y | - | - |
| [π0](https://hf.co/vrfai/pi0-libero-finetuned-v044-gguf)       | Y | Y | Y | - | - |
| [π0.5](https://hf.co/vrfai/pi05-libero-gguf)                   | Y | Y | ~ | - | - |
| [GR00T N1.5](https://hf.co/vrfai/gr00tn1d5-libero-object-gguf) | Y | Y | ~ | - | - |
| [GR00T N1.6](https://hf.co/vrfai/gr00tn1d6-libero-gguf)        | Y | Y | ~ | - | - |
| [GR00T N1.7](https://hf.co/vrfai/gr00tn1d7-libero-gguf)        | Y | Y | Y | - | - |
| [BitVLA](https://hf.co/vrfai/bitvla-libero-gguf)               | Y | Y | ~ | - | - |
| [Evo-1](https://hf.co/vrfai/evo1-libero-gguf)                  | Y | Y | ~ | - | - |
| [VLA-Adapter](https://hf.co/vrfai/vla-adapter-libero-gguf)     | Y | Y | ~ | - | - |
| [OpenVLA-OFT](https://hf.co/vrfai/openvla-oft-libero-gguf)     | Y | Y | ~ | - | - |
| [VLA-JEPA](https://hf.co/vrfai/vla-jepa-libero)                | Y | Y | ~ | - | - |

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
