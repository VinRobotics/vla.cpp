# Architecture

vla.cpp runs Vision-Language-Action (VLA) policies on the ggml/llama.cpp runtime.
Every model is a self-contained GGUF that the engine loads, detects, and drives on
CPU, CUDA, Metal, SYCL, OpenVINO, OpenCL or Hexagon. This page is the map; the
source is the detail.

## Layers

- `src/model.h` - public API: `model_load`, `predict`, `model_config`, `last_stats`.
- `src/arch.h` - the `Arch` enum, the `ModelArchBase` interface, and one `*_create`
  factory per architecture.
- `src/model.cpp` - loads a checkpoint, detects the architecture from its GGUF keys
  (or safetensors namespace), and dispatches to the matching factory.
- `src/models/*.cpp` - one translation unit per architecture. Each owns its ggml
  contexts, vision tower, weights, and compute graph.
- `src/gguf_reader.h` - the shared GGUF reader (metadata, tensor bytes,
  on-demand embedding rows). `src/loader.h` uploads weights at the resident dtype.
- `src/layers/` - ggml building blocks: linear, attention, FFN, norms, RoPE,
  time embeddings.
- `src/modules/` - parts shared between archs: the SigLIP, Qwen3-VL and
  DINOv2+SigLIP vision towers, the Gemma action expert, the Qwen3 LM, the DiT
  head, and image preprocessing (`preprocess.h`: view checks, CHW normalization).
- `src/scratch_ctx.h` - compute contexts and `graph_cache`, reused across calls.
- `src/backend.h` - backend selection; `src/backend_fallback.cpp` runs the ops an
  accelerator rejects on the CPU.
- `src/tokenizer.h` - SentencePiece tokenizers stored in the GGUF, for
  `vla-cli --text`.
- `include/vla.h`, `src/vla_c_api.cpp` - the C ABI (`libvla`).
- `src/serving/` - `vla-server` (ZeroMQ + protobuf, action prediction), `vlm-server`
  (chat), `vla-cli` (one-shot inference) and `vla-bench` (timing).
- `src/kernels/bitvla/` - custom 1.58-bit ternary CUDA kernels for BitVLA.
- `src/foldquant.h`, `src/kernels/foldquant/`, `src/cuda/` - FoldQuant INT8 linears
  and the in-tree CUDA kernels behind the ggml extension hook.

## The prediction path

A forward pass has two stages that most architectures share.

1. **Prefix.** Camera views go through a vision tower, language tokens through the
   embedding table, and proprioception through a small projection. Concatenated, they
   form the prefix that the language backbone attends over (bidirectionally, minus any
   padded language tokens).

2. **Action head.** A smaller expert reads the prefix and produces an action chunk of
   shape `[num_steps, max_action_dim]`, where only the first `real_action_dim` columns
   carry values and the rest are zero padding. The head comes in four flavours:
   - **Flow-matching expert** (SmolVLA, pi0, pi0.5): integrates a velocity field with
     Euler steps from noise at `t=1` to the action at `t=0`. SmolVLA alternates
     self-attention among action tokens with cross-attention to the prefix.
   - **DiT** (GR00T N1.5/1.6/1.7, Evo-1, VLA-JEPA): a diffusion transformer head,
     also integrated with flow-matching Euler steps.
   - **Diffusion** (Octo): DDPM with a small MLP denoiser over the transformer readout.
   - **Parallel decode** (BitVLA, OpenVLA-OFT, VLA-Adapter): OpenVLA-OFT-style
     bidirectional decode of the action tokens in a single pass. TurboVLA is also
     one pass: an ACT decoder whose learned queries read the fused tokens.

Actions leave `predict` in world units when `Config::denormalized` is true (the
default). GR00T N1.5/1.6/1.7 and VLA-JEPA return normalised actions and the caller
un-normalises.

## Vision deployment

Every architecture ships its vision tower inside the one GGUF. `model_load` and the
binaries still take an mmproj path so older command lines work, but every arch
ignores it.

## Backends and packaging

llama.cpp is fetched by CMake `FetchContent` and pinned by `VLA_LLAMA_TAG` in
`CMakeLists.txt`. Each arch picks a resident dtype for its GEMM weights
(`--weight-dtype` overrides it), and a GGUF can be repacked to Q8_0/Q4_0 with
`scripts/quantize_gguf.py`.
The loader keeps packed weights packed. On CPU and CUDA, ggml quantizes the
activations to 8 bits and runs int8 dot products on the blocks.

A FoldQuant GGUF (see [QUANTIZATION.md](QUANTIZATION.md))
carries INT8 or INT4 codes plus sidecar scales instead. `src/foldquant.h` declares those
sites, `src/layers/fq_linear.h` turns each into two `GGML_OP_CUSTOM` nodes, the
CPU backend runs the reference in `src/foldquant_ref.cpp`, and on CUDA the
`src/kernels/foldquant/` integer kernels claim the same nodes through the ggml
extension hook (`src/cuda/`), and on OpenVINO `src/openvino/foldquant_ov.cpp`
translates them into OpenVINO ops. Every other backend reads the sites back as
float weights through `WeightLoader::as_float` and runs the arch's float path. CPU thread count scales to the machine core count;
the GPU backends run the towers and the transformer on the device.

## Adding an architecture

Extend the `Arch` enum, declare a `*_create` factory in `arch.h`, implement it under
`src/models/`, wire detection and dispatch in `src/model.cpp`, and add a converter in
`scripts/`. Reuse `gguf_reader.h`, `loader.h`, `layers/` and `modules/` rather than
copying them. [CONTRIBUTING.md](../CONTRIBUTING.md) lists the exact sites.
