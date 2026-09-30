# Changelog

Notable changes to vla.cpp. Format loosely follows [Keep a Changelog](https://keepachangelog.com).

## [Unreleased]

### Added

- Release tarballs that run off the CI runner: every shared library ships next
  to the binaries with an `$ORIGIN` (`@loader_path`) rpath, builds use
  `GGML_NATIVE=OFF`, and a smoke step checks `ldd`/`otool -L` and runs
  `vla-cli --help` with the build tree moved away. New
  `linux-x86_64-cuda-13.4` and `linux-aarch64-cuda-13.4` (sm_87, sm_110,
  sm_121) tarballs, each with a `cudart-*.tar.gz`; the x86 CUDA list gains sm_80
  and sm_90. `vla-server` still needs `libzmq5` from the system; the aarch64 CUDA
  build needs glibc 2.39 and a CUDA 13 driver, so not JetPack 6.
- `cmake --install` installs the binaries and libraries with a relocatable
  `$ORIGIN/../lib` rpath, and `bindings/python` builds a self-contained wheel
  with scikit-build-core (`pip install ./bindings/python`).
- `vla-cli` takes the precision flags and `--config`, like `vla-server`.
- The tokenizer can live in the GGUF, as in llama.cpp:
  `scripts/add_tokenizer_to_gguf.py` embeds the SentencePiece model for π0,
  π0.5 and OpenVLA-OFT, and `vla-cli --text` then builds the prompt in-process
  with no Python. Otherwise it falls back to `tokenize_prompt.py`, which it now
  finds next to its own binary or under `share/vla`.
- `--num-steps N` (or `runtime.num_steps`) overrides the solver step count at
  load for π0, π0.5, SmolVLA, Evo-1, GR00T N1.5/N1.6/N1.7 and VLA-JEPA; archs
  that do not read it refuse it. `VLA_NUM_STEPS` still works as a fallback.
- `-hf user/repo:sub/dir/file.gguf` and llama.cpp-style `:Q8_0` tags. A repo
  with several GGUFs lists them instead of picking the largest, and only GGUFs
  are downloaded.
- `vla-cli --text` builds each arch's real prompt (pi0.5 digitizes `--state`
  into it, OpenVLA-OFT appends its empty token, VLA-Adapter uses its chat
  template), matching the eval client token for token.
- **Snapdragon X on Windows on Arm: Hexagon NPU, Adreno GPU and CPU.**
  - `-DGGML_HEXAGON=ON` and `-DGGML_OPENCL=ON` build natively with Visual
    Studio's Clang; `scripts/build_windows_snapdragon.ps1` drives the build,
    including skel signing.
  - Ops either accelerator rejects run on the CPU through a wrapper backend
    (`src/backend_fallback.cpp`), with no change to any arch.
  - SmolVLA runs in 1.23 s on the NPU (2.57 s CPU), within 1.5e-3 of the CPU
    reference; eleven checkpoints run on both accelerators.
  - Five ggml-hexagon kernels that give wrong answers for VLA shapes are routed
    around. See `docs/backend/hexagon-windows.md`.
- `--weight-dtype f16`. It is the default on Hexagon and OpenCL, and 2.5-3.5x
  faster than BF16 on CPUs without BF16 matmul.
- `VLA_BUILD_SERVER=OFF` builds `vla-cli` and `vla-bench` without protobuf or
  ZeroMQ.
- **OpenVINO backend.** `-DGGML_OPENVINO=ON` runs the archs on Intel CPUs, iGPUs
  and NPUs through ggml's OpenVINO backend. SmolVLA, π0.5, Evo-1 and VLA-Adapter
  match an F32 CPU reference to 1e-3; on an Arc B390 iGPU that is 3.0x to 9.6x
  the native CPU backend. Every arch that can reach this backend - ten of the
  eleven - is inside the accuracy bar on the OpenVINO CPU plugin, and nine of the
  ten on the iGPU. BitVLA is the eleventh and pins to the CPU backend by design.
  See `docs/backend/ov.md`.
- `scripts/install_ov.sh` installs the OpenVINO runtime and the Intel GPU/NPU
  driver stack on Ubuntu 22.04 and 24.04, with the runtime archive checksummed
  against a digest pinned in the script.
- `scripts/patch_ggml_openvino.py` applies thirteen fixes to the fetched
  ggml-openvino sources at configure time. Each hunk is checked on its own, so a
  `build/_deps` patched by an older checkout fails loudly instead of building
  something quietly wrong.
- `tests/test_graph_names.cpp` pins `vla::graph_unique_names`.
- CI now checks that both llama.cpp patch scripts still apply, on a copy of
  the fetched tree. Neither ran on a CPU build, so their anchors could rot
  unnoticed until someone configured a CUDA or OpenVINO tree.
- `docs/UPSTREAMING.md` and `scripts/upstream_split.py` regroup the thirteen
  ggml-openvino fixes into one llama.cpp branch per PR. They are generic
  backend defects, not vla.cpp workarounds; landing them upstream removes the
  configure-time patch step entirely.

### Fixed

- π0 fed its image tokens to the language model scaled by 1/sqrt(2048). Every
  reference, including the lerobot v0.4.4 code that trained the shipped
  checkpoint, feeds the raw projector output. On 100 paired LIBERO-Object
  episodes π0 goes from 83 to 90 successes (McNemar p=0.17, not significant).
  π0.5 dropped the same scale and its undo, a rounding-level change.
- TurboVLA skipped DINOv3's final LayerNorm, which the checkpoint was trained
  through, so its actions were off by up to 0.47. It now matches the PyTorch
  reference to 9e-6.
- Octo now matches the JAX reference it was trained with: tanh GELU (flax's
  default) instead of erf, JAX GroupNorm and StdConv epsilons, F32 im2col in
  the stem, and discretized proprio bins that were off by one. Readout is within
  2e-6 of JAX on CPU.
- The Qwen3-VL patch mergers used tanh GELU where the reference uses erf, which
  moved GR00T N1.7 and VLA-JEPA vision features by up to 4e-3. GR00T N1.7 goes
  from 97 to 99 on 100 paired LIBERO-Object episodes (p=0.5, not significant).
- SmolVLA rounded its F32 cross-attention k/v projections to BF16. With them
  kept in F32 it goes from 90 to 92 on 100 paired episodes (p=0.63, not
  significant).
- The eval client sent VLA-Adapter raw proprio where the reference normalizes it
  with q01/q99 bounds, and mapped constant GR00T state dims to -1 where the
  reference uses 0. VLA-Adapter is unchanged on LIBERO-Object (298 vs 295 of 300
  paired episodes, not significant) but now sees the inputs it was trained on.
- Evo-1's fallback prompt lacked the `Image-N:` prefixes, and its constant state
  dims were normalized differently from the reference. BitVLA truncated f32 to
  bf16 instead of rounding to nearest even, and its legacy unpacked path
  under-scaled every BitLinear.
- `--flash-attn` never reached the GR00T N1.5 and N1.6 towers, and the shared
  encoder's flash path aborted with more than one view. Both are wired now; on
  CUDA the flash path is within 5e-3 of the default for both archs. The default
  path is unchanged.
- On OpenVINO, SmolVLA, π0 and π0.5 failed shape inference on every device at
  llama.cpp b11223. ggml-openvino caches each RoPE sin/cos table under the op's
  parameters alone, so the action suffix picked up the prefix's table.
  `scripts/patch_ggml_openvino.py` now keys that cache on the position input too.
- vla-server no longer dies on a bad request. An out-of-vocab Octo token, 9 to
  16 OpenVLA-OFT or VLA-Adapter views, a BitVLA prompt past 1024 tokens, or Octo
  stats without a mask each aborted the process; they now get an error
  reply. Images are decoded as JPEG or PNG only, a request is capped at 64
  megapixels in total, precomputed embeddings at 16 views, and a port that is
  already in use exits with a message instead of SIGABRT after the model load.
- vlm-server read an uninitialized prompt length, died on a chat-template
  exception, corrupted multi-turn prompts, streamed invalid UTF-8, and handed
  network bytes to the ffmpeg image fallback.
  All fixed; a full context now ends with `finish_reason="length"`.
- Graph sizes follow the real node count in π0, π0.5, SmolVLA, OpenVLA-OFT,
  VLA-Adapter, VLA-JEPA and GR00T N1.7, so more views or `VLA_NUM_STEPS` no
  longer trip a fixed 16384/65536-node assert.
- Malformed GGUF metadata (zero heads or patch size, non-square position tables,
  bad RoPE theta, mismatched patch shapes) is rejected at load for every arch
  instead of dividing by zero or reading past a buffer.
- BitVLA: BF16/F16/Q8_0 weights were uploaded to CUDA as float (NaN or garbage
  actions), `VLA_DEVICE` was ignored, CUDA allocation and launch errors were
  dropped, three kernels had shared-memory races, and failed inits leaked up to
  250 MiB.
- The BF16 CUDA hook could write F32 into a BF16 buffer on a declined matmul,
  lost launch errors, shared one cuBLAS handle across threads, and missed an
  alignment check on batched views.
- Evo-1's Q8_0 file aborted on a view assert; its attention split now uses row
  views and no longer copies 24 weight slices per call.
- `WeightLoader::fuse` read quantized sources as float, so a quantized TurboVLA
  failed to load.
- A `--config` runtime block overrode flags given on the command line and
  silently dropped bad values. The command line now wins, bad values are an
  error, and `libvla` and the Python bindings apply the block too.
- Two models in one process shared the flash-attention and matmul-precision
  flags of whichever loaded last.
- The C API and Python bindings are safe to call from several threads on one
  handle, and the bindings reject an image whose dtype does not match its pixel
  format instead of reading past it.
- The CPU fallback wrapper reuses one threadpool instead of spawning threads for
  every split.
- `scripts/quantize_gguf.py` wrote array metadata as INT32, which broke Octo and
  TurboVLA, and packed the action experts and vision towers its skip list meant
  to keep float.
- Converters: both lerobot key layouts for π0/π0.5, legacy SmolVLA stats, head
  counts from config and `torch.load(weights_only=True)` on third-party
  checkpoints (torch >= 2.6). Unsupported config variants are refused, including
  a π0.5 normalization mapping other than QUANTILES.
- The eval client's REQ socket stayed stuck after one timeout, and the ALOHA
  prefetch could replay a chunk from an earlier observation.
- A `scripts/quantize_gguf.py` file did not load for SmolVLA on any platform:
  its loader read every weight as float and refused the packed connector. It now
  keeps packed GEMM weights packed, like the other archs, and dequantizes the
  rest.
- Two elementwise adds stacked on a GEMM came out wrong on the Intel iGPU. The
  GPU plugin folds elementwise ops into the preceding GEMM as post-ops, and given
  `ADD(ADD(residual, GEMM), graph_input)` it folds both and silently drops the
  second operand - the result equals the inner add. A llama.cpp graph never builds
  that chain; a VLA does, wherever a vision tower's features are added on top of an
  FFN residual. VLA-JEPA (5.4e-1) and GR00T N1.7 (1.9e0) were wrong on the iGPU
  while matching the CPU plugin to 1e-4. Re-associating the two adds so the GEMM
  keeps one post-op puts both at 2.6e-3. Bisected with `GGML_OPENVINO_DEBUG_NODE`.
- π0's action dims drifted 4e-2 on the iGPU and its gripper flipped a step late,
  because the GPU plugin computes in F16 and π0 unrolls its whole denoise loop
  inside one graph. `GGML_OPENVINO_GPU_PRECISION` now exposes the plugin's
  inference precision; `backend_init` defaults it to f32 for π0 alone, which costs
  about 3x on that arch and puts it at 6.5e-5.
- `scripts/patch_ggml_openvino.py` now fails if `EDITS` has a duplicate key. Python
  keeps the last one silently, and a duplicate briefly removed the whole Intel
  OpenCL platform fix from the patch without any error.
- The position-input fix stopped running when llama.cpp moved to `b10729`. That
  release relocated the naming out of `GgmlOvDecoder::get_graph_input_ov_name()`,
  which the patch guards, into a new free `get_tensor_graph_input_ov_name()`, and
  left the member behind with no callers. The hunk still applied cleanly, so
  nothing failed loudly - SmolVLA and π0.5 simply stopped returning actions
  ("Argument shapes are inconsistent", a 113-token prefix ROPE reading the
  50-token suffix's table). Both functions are guarded now, and the patch script
  says to check for a live caller, not just a matching anchor, on every tag bump.
- `scripts/upstream_split.py` addressed hunks by position in the patch script's
  edit list. Adding a hunk to the front of a file's list silently handed every
  later hunk to the wrong branch, and its own coverage count still read 29/29
  because each index was still used exactly once. Two branches had been swapped
  this way. Hunks are now addressed by a unique substring of their anchor, which
  fails loudly instead. The PERMUTE `op_case` fix, which had no branch at all,
  now has one.
- `graph_unique_names` renamed through `ggml_format_name`, which passes the
  tensor's own name to `vsnprintf` as both destination and `%s` source. glibc
  empties it, so every duplicate node became the bare string `#<index>`.
- The OpenVINO naive-path compiled-model cache was keyed on node count plus the
  first and last node name. Two graphs of the same size collided and the second
  ran the first's compiled model. It now also keys on every node's op and shape,
  and the map is bounded.
- `GGML_OPENVINO_NAIVE_GRAPH_SIZE` went through `atoi`, so junk parsed to 0 and
  sent every graph down the decoder-only-LLM path with nothing said. Empty
  environment values no longer count as a setting either.
- `GGML_OPENVINO_CACHE_DIR` is cleared rather than warned about: a warm cache
  returns wrong actions, and stderr is not always read. `VLA_ALLOW_OV_CACHE=1`
  keeps it.
- `scripts/print_versions.sh` printed `?` for the llama.cpp pin ever since the
  tag moved behind `VLA_LLAMA_TAG`.
- The OpenVINO `find_package` failure message was unreachable, sitting after the
  fetch whose own `find_package(REQUIRED)` fired first.
- BitVLA indexed its action slots as `seq-2-n_action+i` with no check that the
  sequence is long enough. Neither `ggml_get_rows` nor the CUDA gather
  bound-checks, so a short prompt read out of bounds and returned it as hidden
  states. One guard now covers both LM paths.
- pi0 and pi0.5 fell back to identity normalisation stats on a dimension mismatch
  or a short read, and said so on stdout. That returns un-denormalised actions
  from a checkpoint that looked fine. Both now fail the load, and the message
  goes to stderr - stdout is the action stream `predict_check` diffs.
- `scratch_ctx::reset` ignored an arena larger than the first call's, which would
  abort in `ggml_new_tensor` if any call site ever sized one from the input.
- The safetensors arch probe would allocate up to 256 MB for a header it only
  substring-searches. Capped at 16 MB.
- The two CUDA targets were the only first-party code built without
  `-Wall -Wextra`.
- `tests/bitvla_gemm_check.cu` had no build target and a comment claiming it was
  never committed. It builds now, under `GGML_CUDA`.
- Stale references to `vision_common.h` (now `modules/preprocess.h`) and to the
  retired `VLA_EVO1_BF16_ACT` switch.

### Changed

- Faster predict, measured on an RTX 5090 at default flags against `7abe1b4`
  (min over 5 interleaved rounds): SmolVLA -18%, π0.5 -16%, GR00T N1.6 and
  TurboVLA -15%, VLA-Adapter -12%, π0, OpenVLA-OFT, GR00T N1.5 and N1.7 -9%,
  VLA-JEPA -8%, Octo -7%, Evo-1 -6%. BitVLA is about 3% slower (20.2 ms to
  20.8 ms), the cost of correct bf16 rounding. The speedups themselves leave
  actions byte-identical. π0.5's adaRMS and the DiT heads' timestep
  conditioning are computed once at load, GR00T keeps only the selected
  embodiment's projectors resident (1.2 GiB less VRAM), SmolVLA widens its
  weights to F32 at load instead of on every call (0.7 GiB more VRAM), TurboVLA
  caches the encoded instruction, and vision outputs stay on the device.
- About 1300 lines of per-arch copies now use the shared `src/layers` and
  `src/modules` code, byte-identical for every arch.
- The Docker image is multi-stage (devel to runtime) and the published one
  covers sm_75 to sm_120 with `GGML_NATIVE=OFF`, instead of sm_89 only. CUDA 13.4
  is a build-arg for drivers 580 and newer.
- A missing `--config` file is an error instead of being ignored.
- Octo is always built. `VLA_OCTO` is renamed `VLA_SPM` and only controls the
  SentencePiece tokenizer; the old name still works with a warning.
- Release workflow tokens are least-privilege, and a manual run builds without
  publishing.
- The x86 CUDA 12.8 release tarball is now `linux-x86_64-cuda-12.8` (was
  `linux-x86_64-cuda`); scripts that download it by name need the new suffix.
- The macOS release tarball ships only `vla-cli` and `vla-bench`, without the
  servers or the in-GGUF SentencePiece tokenizer, so it no longer needs Homebrew
  protobuf or zeromq. `--text` there goes through `tokenize_prompt.py`.
- llama.cpp pinned at `b11223`, up from `b10331` (via `b10729`). Brings the
  IM2COL+MatMul to native-convolution fusion, the `RELU`/`NEG`/`SQR` translators
  the local patch no longer adds, CUDA RMS_NORM+SCALE fusion, fixes for a
  divergent `__syncthreads` in the f16 flash-attention kernel and races in
  mmf/mmid, and Metal and OpenCL correctness fixes. All thirteen archs are
  byte-identical on CUDA (sm_120) against `b10729`. The
  `ggml_mul_mat_set_prec`/`ggml_flash_attn_ext_set_prec` calls moved to
  `ggml_prec_set_acc`, which writes the same op params. The build.yml cache key
  now reads the tag out of `CMakeLists.txt` instead of repeating it.
- The default CUDA architecture list is set before ggml is configured, so
  ggml-cuda and the in-tree kernels build the same set, and it gains sm_110
  (Jetson Thor, CUDA 13) and sm_121 (DGX Spark, CUDA 12.9). Only the f16 flash
  attention vector kernels are built (`GGML_CUDA_FA_QUANTS=f16-f16`).
- SentencePiece `v0.2.1`, which fixes a heap overflow on a malformed
  normalization model. `vla-cli --text` loads that model from GGUF bytes
  (`src/tokenizer.cpp`).
- OpenVINO 2026.4 in `scripts/install_ov.sh`, with newer Intel GPU and NPU
  drivers on Ubuntu 24.04. Every downloaded archive and package is now
  checksummed, including the NPU driver, level-zero and IGC packages.
- The release macOS job runs on `macos-15`; `macos-14` runners are retired.
- Python tooling: transformers 5.x is supported (the client needs `>=5.4` for
  the VLA-JEPA processor), the client extra gains the modules it imports
  (`torchvision`, `protobuf`, `opencv-python-headless`), and both pyprojects use
  an SPDX license string.
- `src/models/dit_common.h` is gone. It redefined six `vla::` functions that
  `src/layers/` already had, with both copies linked into `vla_core`. Every
  includer used only `sinusoidal_time_emb` or `build_causal_mask`, so they now
  include `layers/embed.h`. Byte-identical across all 11 archs.

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
