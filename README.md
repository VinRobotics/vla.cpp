# vla.cpp

![logo](assets/logo_vlacpp_white.png)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.md)
[![Built on llama.cpp](https://img.shields.io/badge/built%20on-llama.cpp-lightgrey)](https://github.com/ggml-org/llama.cpp)
[![Models on HF](https://img.shields.io/badge/%F0%9F%A4%97%20models-Hugging%20Face-yellow)](https://huggingface.co/vrfai)
[![arXiv](https://img.shields.io/badge/arXiv-2606.08094-b31b1b.svg)](http://arxiv.org/abs/2606.08094)
[![Docs](https://img.shields.io/badge/docs-Learn%20vla.cpp-brightgreen)](https://fai-modelopt-tech.github.io/learn-vla-cpp/)

A C++ inference engine for **Vision-Language-Action (VLA) models**, built on [`llama.cpp`](https://github.com/ggml-org/llama.cpp).
It runs the open VLA policies - SmolVLA, π0, BitVLA, Evo-1, GR00T N1.5/1.6/1.7 and more -
under one runtime, each packaged as a single self-contained GGUF that needs no Python or
PyTorch at inference time. The binaries drive robots on **CPU**, **Apple Silicon**, **CUDA** -
from consumer GPUs down to Jetson-class boards - or **Intel GPUs** via SYCL.

[**Learn vla.cpp**](https://fai-modelopt-tech.github.io/learn-vla-cpp/) walks through the engine design and how each policy is implemented on ggml.

---

## Build the server

### Prerequisites

- CMake ≥ 3.22
- A C++17 compiler (GCC 11+ or Clang 14+)
- CUDA 12.x (optional - required only for CUDA GPU builds)
- Intel oneAPI 2025.x + GPU compute runtime (optional - only for Intel GPU
  builds, see [docs/backend/sycl.md](docs/backend/sycl.md))
- `libzmq3-dev`, `cppzmq-dev`, `libprotobuf-dev`, `protobuf-compiler`

```bash
sudo apt-get install -y libzmq3-dev cppzmq-dev libprotobuf-dev protobuf-compiler
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

Then configure and build. CMake fetches and pins `llama.cpp` automatically (no patch, no submodule):

```bash
# CPU build:
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# CUDA build (set CMAKE_CUDA_ARCHITECTURES for your GPU):
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURE
cmake --build build -j$(nproc)
```

```bash
# Intel GPU build (Arc / Flex / Max / Xe iGPU). ggml's SYCL sources need the
# oneAPI DPC++ driver, so the whole project is compiled by icpx:
source /opt/intel/oneapi/setvars.sh
cmake -B build \
    -DGGML_SYCL=ON \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The driver and oneAPI setup that this needs is in
[docs/backend/sycl.md](docs/backend/sycl.md).

If CMake cannot find CUDA, point the environment at it explicitly:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Check [docs/backend](docs/backend) for compiling `vla.cpp` on other platforms.
WSL2 and Apple Silicon are both tested.

---

## Quickstart

Once the binaries are built, run one CPU prediction without a server or simulator:

```bash
pip install -U "huggingface_hub[cli]" transformers

# -hf fetches and caches the checkpoint (under $VLA_CACHE, default ~/.cache/vla)
./build/vla-cli -hf vrfai/smolvla-libero-gguf \
    --image assets/front.jpg --text "pick up the black bowl" --pretty

# or point at a file you already have
./build/vla-cli --ckpt models/smolvla/smolvla-libero.gguf \
    --image assets/front.jpg --text "pick up the black bowl" --pretty
```

`vla-cli` runs a single prediction without a server or simulator: give it a model,
an image, and an instruction, and it prints the action chunk. Handy for
smoke-testing a GGUF or scripting a quick inference.

There is no tokenizer in the C++ core, so `--text` calls
`scripts/tokenize_prompt.py` with the tokenizer the architecture was trained on
(`VLA_PYTHON` picks the interpreter, `VLA_TOKENIZE_SCRIPT` the script). Pass
`--tokens 1,100,200,2` instead if you already have ids.
`--pretty` prints one action row per line;
`--state` sets proprioception (defaults to zeros).

For the design overview see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), for the long-running path see
[Running the server](#running-the-server), and for the other checkpoints see [Roadmap](#roadmap).

The rest of this README refers to a few shell variables:

```bash
export VLA_GGUF=models/smolvla/smolvla-libero.gguf   # the checkpoint to serve
export VLA_ARCH=smolvla                              # client-side arch preset, see --help
```

---

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

---

## Running the server

`vla-server` loads the model once at startup and answers ZeroMQ REQ/REP requests synchronously.

```bash
./build/vla-server "$VLA_GGUF"
```

When ready, the server prints:

```
vla-server: bound to tcp://*:5555. ready.
```

Use `--bind` to change the address and port. Stop the server with `Ctrl-C`.

`vla-server` also takes `-hf user/repo[:file.gguf]` in place of a checkpoint path.

Environment knobs that apply to every arch:

- `VLA_N_THREADS` - CPU backend thread count, default core count capped at 16.
- `VLA_DEVICE` - GPU ordinal for CUDA and SYCL builds, default 0.
- `VLA_CACHE` - where `-hf` stores checkpoints, default `~/.cache/vla`.

---

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

The GR00T models need two extras:

- client side: `--stats-json /path/to/dataset_statistics.json`
- server side: `VLA_GR00T_EMBODIMENT` (`new_embodiment` for N1.5, `libero_panda` for N1.6, `libero_sim` for N1.7) and `VLA_GR00T_BF16_WEIGHTS=1` (to fit the 8 GB card).

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

---

## Models

### Conversion

Each model ships as a single self-contained GGUF. To convert a HuggingFace safetensors
checkpoint yourself, [`scripts/`](scripts/) has a converter per arch. Set up its venv:

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

---

## Benchmarks

`vla-bench` times `predict()` in-process on synthetic inputs: engine only, no
transport, no simulator, no claim about task success.

```bash
./build/vla-bench -hf vrfai/smolvla-libero-gguf --images 2 --size 512 --markdown
```

RTX 5090, driver 595.84, CUDA 13.2, 24-core host, weights as shipped, 20 reps
after 3 warmups, best of three sweeps, each model at its native input size and
view count.

| Model | Views | Input | min ms | p50 ms | p90 ms | vision ms |
|---|--:|--:|--:|--:|--:|--:|
| VLA-Adapter | 1 | 224 | 18.2 | 19.8 | 21.1 |  9.4 |
| VLA-JEPA    | 1 | 256 | 19.9 | 21.5 | 22.9 |  6.3 |
| BitVLA      | 1 | 224 | 23.6 | 25.3 | 26.4 |  5.4 |
| GR00T N1.5  | 1 | 224 | 28.2 | 29.4 | 30.5 |  5.9 |
| GR00T N1.7  | 1 | 256 | 31.0 | 33.4 | 34.6 |  6.2 |
| GR00T N1.6  | 1 | 224 | 33.4 | 35.7 | 37.3 |  6.3 |
| OpenVLA-OFT | 1 | 224 | 47.4 | 49.2 | 50.2 | 10.3 |
| SmolVLA     | 2 | 512 | 47.8 | 49.6 | 54.0 | 16.1 |
| pi0         | 2 | 224 | 48.9 | 52.1 | 55.0 | 11.6 |
| Evo-1       | 1 | 448 | 52.2 | 55.2 | 57.3 | 17.8 |
| pi0.5       | 2 | 224 | 53.4 | 56.1 | 59.3 | 11.4 |

Jetson and Apple targets are absent: they have not been re-measured with
`vla-bench`.

### Task success

Latency says nothing about whether a policy works. LIBERO-Object, 10 tasks and 20
episodes per model, terminated episodes counted as failures:

| Model | Chunk replay | Success rate |
|---|--:|--:|
| BitVLA     |  8 | 100.0% |
| GR00T N1.7 | 16 |  98.0% |
| GR00T N1.5 | 16 |  96.0% |
| Evo-1      |  8 |  94.5% |
| SmolVLA    |  4 |  90.5% |
| π0         | 32 |  87.5% |
| GR00T N1.6 | 16 |  86.5% |

From [eval/reports/report-rtx-3060.md](eval/reports/report-rtx-3060.md), swept on
an RTX 3060 at commit `dcc29a3` (2026-05-24). It predates π0.5, VLA-Adapter,
OpenVLA-OFT and VLA-JEPA, which have not been swept. Jetson AGX Orin and Orin
Nano runs are in the same directory. Success rate belongs to the checkpoint, not
the engine; `vla_predict_check` in [CONTRIBUTING.md](CONTRIBUTING.md) is how a
change is shown to leave it alone.

---

## Roadmap

Support matrix of models (rows) against platforms (columns). Legend: `Y` =
supported (released and benchmarked), `~` = in progress, `-` = planned.

| Model | CPU (x86-64 / ARM) | CUDA | SYCL (Intel) | Metal | OpenVINO | Hexagon |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| [SmolVLA](https://hf.co/vrfai/smolvla-libero-gguf)             | Y | Y | Y | Y | - | - |
| [π0](https://hf.co/vrfai/pi0-libero-finetuned-v044-gguf)       | Y | Y | - | Y | - | - |
| [π0.5](https://hf.co/vrfai/pi05-libero-gguf)                   | Y | Y | - | ~ | - | - |
| [GR00T N1.5](https://hf.co/vrfai/gr00tn1d5-libero-object-gguf) | Y | Y | - | ~ | - | - |
| [GR00T N1.6](https://hf.co/vrfai/gr00tn1d6-libero-gguf)        | Y | Y | - | ~ | - | - |
| [GR00T N1.7](https://hf.co/vrfai/gr00tn1d7-libero-gguf)        | Y | Y | - | Y | - | - |
| [BitVLA](https://hf.co/vrfai/bitvla-libero-gguf)               | Y | Y | - | ~ | - | - |
| [Evo-1](https://hf.co/vrfai/evo1-libero-gguf)                  | Y | Y | Y | ~ | - | - |
| [VLA-Adapter](https://hf.co/vrfai/vla-adapter-libero-gguf)     | Y | Y | ~ | ~ | - | - |
| [OpenVLA-OFT](https://hf.co/vrfai/openvla-oft-libero-gguf)     | Y | Y | - | ~ | - | - |
| [VLA-JEPA](https://hf.co/vrfai/vla-jepa-libero)                | Y | Y | - | ~ | - | - |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to prove a change is numerically
neutral, and the six sites you touch to add an architecture.

---

## Contributors

- [Khanh Dang Nguyen](https://github.com/khanhnd61-vr)
- [Hung Thinh Ho](https://github.com/hungho77)
- [Chinh Truong Nguyen](https://github.com/nguyentruongchinh04z)
- [An Thai Le](https://github.com/anindex)

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE.md).

---

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

Built on:

- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) - LLM inference engine in C/C++.
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) - benchmark suite for the success-rate sweeps.
- [SimplerEnv](https://github.com/simpler-env/SimplerEnv) - the second simulator in the eval scaffold.
