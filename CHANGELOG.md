# Changelog

Notable changes to vla.cpp. Format loosely follows [Keep a Changelog](https://keepachangelog.com).

## [0.2.0] - 2026-08-09

### Added
- SYCL backend for Intel GPUs (Arc, Flex, Data Center Max, Xe iGPU). `VLA_DEVICE` picks the ordinal on CUDA and SYCL alike. See `docs/backend/sycl.md`.
- Stable C ABI (`include/vla.h`, `libvla`) and Python bindings over it (`bindings/python`).
- Four more architectures: π0.5, VLA-Adapter, OpenVLA-OFT and VLA-JEPA.
- `vla-bench` for engine-only latency, and `-hf user/repo[:file.gguf]` to fetch a checkpoint on first use.
- `vla-cli --text`, tokenized by `scripts/tokenize_prompt.py` with the tokenizer the architecture was trained on.
- Release workflow publishing Linux x86-64 (CPU and CUDA), Linux aarch64 (CPU), macOS Metal and a GHCR image.

### Changed
- One shared backend ladder (`src/backend.h`) instead of a copy per arch. CMake rejects two accelerators in one build directory.
- Shared headers for the Qwen3-VL tower, the DINOv2+SigLIP dual tower, the DiT time embeddings, the causal mask and CHW image preprocessing.
- `vla::graph_cache` keeps the compute graph across `predict` calls in nine architectures, not just GR00T N1.7. Output is unchanged.
- llama.cpp pinned at b10331. GR00T N1.5 and N1.6 shift by up to 4.6e-4 on actions peaking near 0.87, from an upstream ggml kernel change in the SigLIP tower they share. The other nine architectures are bit-identical.

### Fixed
- Reject checkpoint geometry that contradicts itself before it sizes a buffer, in smolvla, bitvla, gr00tn1d6, vla_adapter and the Qwen3-VL position resample.
- A peer that stalls mid-message no longer parks either server.
- Treat a missing state vector as zeros in every architecture rather than dereferencing it.
- Build every registered test before `ctest`, so the four that were never built stop reporting as not run.

## [0.1.1] - 2026-07-04

### Added
- `vla-cli`: one-shot inference from the command line (image + tokens to action), no server needed.
- `scripts/quantize_gguf.py`: repack LM weights to Q8_0/Q4_0. The loader runs quantized GGUFs directly (Q8_0 roughly halves the LM, near-lossless).

### Changed
- One shared GGUF reader across the model loaders, replacing the per-arch copies.
- Cap inbound message size (256 MiB) and image count (16) on `vla-server` and `vlm-server`.
- Scale CPU threads to the machine core count across all loaders instead of a fixed 4.
- Read GGUF file offsets as 64-bit and reject non-float embedding tensors in row-fetch.

### Fixed
- Reject out-of-range language tokens in OpenVLA-OFT and VLA-Adapter.
- Zero the padded action dimensions so only real action dims carry values.
- Reject images that do not match the model input size in VLA-Adapter and OpenVLA-OFT (out-of-bounds read on a smaller view).
- Validate Evo-1 action dims at load so a client-supplied noise buffer cannot underrun.
- Only enable the BitVLA CUDA path once every device buffer allocates.

## [0.1.0] - 2026-07-03

First tagged release. One self-contained GGUF per model (vision tower + LM + action
expert + dataset stats), CPU or CUDA, no external mmproj and no patch to llama.cpp.

### Added
- Seven VLA policies auto-detected from the GGUF: SmolVLA, pi0, BitVLA, Evo-1, GR00T N1.5/N1.6/N1.7.
- In-tree vision towers (SigLIP, BitSigLIP, InternViT, RADIO) on stable public ggml/llama APIs.
- ZeroMQ + protobuf `vla-server`; a separate `vlm-server` for VLM chat.
- BitVLA 1.58-bit custom ternary CUDA kernels.
- Per-arch HuggingFace -> GGUF converters and mmproj-merge helpers (`scripts/`).
- Robot eval harness for LIBERO, SimplerEnv, and ALOHA (`eval/`), with device benchmark reports.
- Minimal CI: pixel-shuffle unit test, converter-remap test, CPU build gate.
- `pyproject.toml` for the Python tooling and a CUDA `Dockerfile` for `vla-server`.

### Changed
- llama.cpp is fetched + pinned via CMake `FetchContent` (tag `b9866`); bumping is a
  one-line `GIT_TAG` change. Removed the `patches/` fetch script.

[0.2.0]: https://github.com/VinRobotics/vla.cpp/releases/tag/v0.2.0
[0.1.1]: https://github.com/VinRobotics/vla.cpp/releases/tag/v0.1.1
[0.1.0]: https://github.com/VinRobotics/vla.cpp/releases/tag/v0.1.0
