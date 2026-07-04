# Architecture

vla.cpp runs Vision-Language-Action (VLA) policies on the ggml/llama.cpp runtime.
Every model is a self-contained GGUF (or a checkpoint plus a vision mmproj) that the
engine loads, detects, and drives on CPU, CUDA, or Metal. This page is the map; the
source is the detail.

## Layers

- `src/model.h` - public API: `model_load`, `predict`, `model_config`, `last_stats`.
- `src/arch.h` - the `Arch` enum, the `ModelArchBase` interface, and one `*_create`
  factory per architecture.
- `src/model.cpp` - loads a checkpoint, detects the architecture from its GGUF keys
  (or safetensors namespace), and dispatches to the matching factory.
- `src/models/*.cpp` - one translation unit per architecture. Each owns its ggml
  contexts, vision tower, weights, and compute graph.
- `src/models/gguf_reader.h` - the shared GGUF reader (metadata, tensor bytes,
  on-demand embedding rows).
- `src/models/vision_common.h` - small pure vision helpers (pixel-shuffle, view checks).
- `src/serving/` - `vla-server` (ZeroMQ + protobuf, action prediction), `vlm-server`
  (chat), and `vla-cli` (one-shot inference).
- `src/kernels/bitvla/` - custom 1.58-bit ternary CUDA kernels for BitVLA.

## The prediction path

A forward pass has two stages that most architectures share.

1. **Prefix.** Camera views go through a vision tower, language tokens through the
   embedding table, and proprioception through a small projection. Concatenated, they
   form the prefix that the language backbone attends over (bidirectionally, minus any
   padded language tokens).

2. **Action head.** A smaller expert reads the prefix and produces an action chunk of
   shape `[num_steps, max_action_dim]`, where only the first `real_action_dim` columns
   carry values and the rest are zero padding. The head comes in three flavours:
   - **Flow-matching expert** (SmolVLA, pi0, pi0.5): integrates a velocity field with
     Euler steps from noise at `t=1` to the action at `t=0`. SmolVLA alternates
     self-attention among action tokens with cross-attention to the prefix.
   - **DiT** (GR00T N1.5/1.6/1.7, Evo-1, VLA-JEPA): a diffusion transformer head.
   - **Parallel decode** (BitVLA, OpenVLA-OFT, VLA-Adapter): OpenVLA-OFT-style
     bidirectional decode of the action tokens in a single pass.

Actions leave `predict` normalised to the training statistics; the caller
un-normalises into world units.

## Vision deployment

Two patterns, chosen per architecture:

- **Baked-in tower** (BitVLA, Evo-1, GR00T, OpenVLA-OFT, VLA-Adapter, VLA-JEPA): the
  vision tower ships inside the combined GGUF, so a single file is enough.
- **Separate mmproj** (SmolVLA, pi0, pi0.5): the SigLIP tower ships as a second GGUF;
  merge it into the checkpoint with `scripts/merge_*_mmproj_to_gguf.py` before loading.

## Backends and packaging

llama.cpp is fetched and pinned by CMake `FetchContent`; a bump is a one-line
`GIT_TAG` change. Weights are bf16 by default and can be repacked to Q8_0/Q4_0 with
`scripts/quantize_gguf.py`; the loader runs quantized GGUFs directly and lets
`ggml_mul_mat` dequantize at compute. CPU thread count scales to the machine core
count; CUDA and Metal run the towers and the transformer on the GPU.

## Adding an architecture

Extend the `Arch` enum, declare a `*_create` factory in `arch.h`, implement it under
`src/models/`, wire detection and dispatch in `src/model.cpp`, and add a converter in
`scripts/`. Reuse `gguf_reader.h` and `vision_common.h` rather than copying them.
