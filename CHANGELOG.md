# Changelog

Notable changes to vla.cpp. Format loosely follows [Keep a Changelog](https://keepachangelog.com).

## [0.3.0] - 2026-08-14

Every architecture is byte-identical to 0.2.0 at matching settings.
`libero_object`, 100 episodes per model, on one RTX 3090:

| Model | SR | Latency, fastest | Fastest flags |
|---|---:|---:|---|
| `bitvla` | 99/100 | 48.0 ms | `--weight-dtype bf16` |
| `gr00t_n1_5` | 99/100 | 67.9 ms | *(none)* |
| `gr00t_n1_7` | 98/100 | 55.4 ms | *(none)* |
| `openvla_oft` | 97/100 | 219.5 ms | *(none)* |
| `pi05` | 96/100 | 112.3 ms | *(none)* |
| `vla_adapter` | 96/100 | 69.7 ms | *(none)* |
| `evo1` | 91/100 | 114.1 ms | `--act-dtype bf16 --flash-attn` |
| `smolvla` | 90/100 | 50.5 ms | `--flash-attn --mm-prec default` |
| `gr00t_n1_6` | 84/100 | 55.5 ms | *(none)* |
| `pi0` | 81/100 | 94.1 ms | `--act-dtype bf16 --flash-attn` |
| `vla_jepa` | not evaluated | 44.0 ms | *(none)* |

SR is measured at each model's defaults, so it does not carry over to the four
rows whose fastest flags change numerics. Full detail in `refactor-report.md`.

### Added
- Model code split into three levels: `src/layers/` (stateless graph fragments), `src/modules/` (weights plus the graph consuming them), `src/models/` (config, composition, `predict`).
- `vla::WeightLoader`: declares weights by name, reports a miss once, allocates and uploads in one call. Replaces the `mk`/`mk_mm`/`mk_f32` lambdas and `ok &= a&&b&&c` chain each of the eleven architectures carried.
- `vla-server` flags `--weight-dtype f32|bf16`, `--act-dtype f32|bf16`, `--flash-attn [0|1]`, `--mm-prec default|f32`, also readable from a `"runtime"` object in the `--config` JSON.
- `eval/refactor_verify.sh`: diffs every architecture's action chunk at two precisions against a reference run, with `BENCH=N` for per-config `predict()` timing.

### Changed
- GR00T N1.5/N1.6/N1.7 and VLA-JEPA default to BF16 weights. `--weight-dtype f32` restores the old default of v0.2.0, bit-identically.
- The per-architecture precision switches (`VLA_GR00T_BF16_WEIGHTS`, `VLA_*_FA`, `VLA_*_BF16_ACT`, `VLA_*_F32_WEIGHTS`, `VLA_MM_PREC`, `VLA_WEIGHT_DTYPE`) are retired. Setting one now fails the load naming its replacement instead of being ignored.
- Deduplicated: `build_dit_block` 4 copies to 1, `SigLipLayerW` 5 to 1, `Qwen3LayerW` 4 to 1, the DINOv2+SigLIP declaration 2 to 1. 1,817 lines of shared code now serve all eleven architectures.
- BF16 elementwise kernels address rows by block index instead of a per-element 64-bit divide, and move eight values per thread on contiguous rows. `VLA_BF16_FLAT=1` selects the scalar path.

### Fixed
- BF16 activations aborted on any fused elementwise run: ggml fuses upstream of the extension hook, and its fused path handles F32/F16 only. The hook now gets first refusal on fused add/mul.
- Boolean environment switches read their value, not their presence, so `VLA_EVO1_FA=0` no longer enabled flash attention.
- SmolVLA ignored its runtime options, leaving its weight dtype unsettable once `VLA_WEIGHT_DTYPE` retired.


### Known limitations
- Thread count, solver steps, GR00T embodiment and un-normalisation key remain environment-only (`VLA_N_THREADS`, `VLA_NUM_STEPS`, `VLA_GR00T_EMBODIMENT`, `VLA_*_UNNORM_KEY`).
- VLA-JEPA has no LIBERO success rate; the client cannot emit its `<embodied>` tokens.

## [0.2.0] - 2026-08-12

### Added
- SYCL backend for Intel GPUs (Arc, Flex, Data Center Max, Xe iGPU). `VLA_DEVICE` picks the ordinal on CUDA and SYCL alike. See `docs/backend/sycl.md`.
- Stable C ABI (`include/vla.h`, `libvla`) and Python bindings over it (`bindings/python`).
- Four more architectures: π0.5, VLA-Adapter, OpenVLA-OFT and VLA-JEPA.
- `vla-bench` for engine-only latency, and `-hf user/repo[:file.gguf]` to fetch a checkpoint on first use.
- `vla-cli --text`, tokenized by `scripts/tokenize_prompt.py` with the tokenizer the architecture was trained on.
- Release workflow publishing Linux x86-64 (CPU and CUDA), Linux aarch64 (CPU), macOS Metal and a GHCR image.
- Opt-in BF16 activations for Evo-1 and pi0 (`VLA_EVO1_BF16_ACT`, `VLA_PI0_BF16_ACT`). Weight GEMMs, bias adds, residuals, norms and activations carry BF16; attention scores, softmax, RoPE and the flow-matching integrator stay F32. Needs CUDA and a BF16 checkpoint, and is ignored otherwise.
- Opt-in fused attention for Evo-1, pi0 and SmolVLA (`VLA_EVO1_FA`, `VLA_PI0_FA`, `VLA_SMOLVLA_FA`). Off by default: ggml's CUDA flash attention computes K/V at F16 whatever the input type, which measured 4 to 5 points lower on libero_object.

### Changed
- One shared backend ladder (`src/backend.h`) instead of a copy per arch. CMake rejects two accelerators in one build directory.
- Shared headers for the Qwen3-VL tower, the DINOv2+SigLIP dual tower, the DiT time embeddings, the causal mask and CHW image preprocessing.
- `vla::graph_cache` keeps the compute graph across `predict` calls in nine architectures, not just GR00T N1.7. Output is unchanged.
- llama.cpp pinned at b10331. GR00T N1.5 and N1.6 shift by up to 4.6e-4 on actions peaking near 0.87, from an upstream ggml kernel change in the SigLIP tower they share. The other nine architectures are bit-identical.
- Evo-1 encodes every camera view in one vision graph and one compute, rather than a `graph_compute` per view. The arithmetic per view is unchanged, only the submission pattern.
- BitVLA's ternary GEMM feeds four column tiles per CTA from one shared activation block, and pads its shared-memory row stride to 144 B to break 16-way bank conflicts. `VLA_BITVLA_NARROW_GEMM=1` selects the previous one-tile kernel.
- The CUDA BF16 kernels live in `src/cuda/` and depend only on the public ggml header. The fetched ggml gets one addition to carry them: a function-pointer hook at the top of its CUDA op dispatch (`scripts/patch_ggml_cuda_ext_hook.py`). Left unregistered the pointer is null and ggml behaves exactly as shipped. This is the first llama.cpp patch since 0.1.0 removed the old `patches/` script, and it makes Python 3 a requirement for configuring any build.
- llama.cpp's tool binaries are no longer part of the default build. Ask for one by name when you want it: `cmake --build build --target llama-mtmd-cli`.
- `vla_core` links ggml alone; the VLA architectures call no `llama_*` API. Only the VLM path links llama.
- Build snippets no longer pass `-DGGML_CUDA_GRAPHS=ON`, which llama.cpp already defaults on.

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
