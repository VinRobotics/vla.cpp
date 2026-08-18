# `vla.cpp` on macOS (Metal backend)

Short notes for getting `vla.cpp` to compile and link on macOS. The Metal
backend is auto-detected by the vendored `llama.cpp`/`ggml` and needs no special
CMake flag.

## Prerequisites

```bash
brew install protobuf zeromq cppzmq pkg-config
```

All four are required at configure time, not optional: `find_package(Protobuf)`
and `pkg_check_modules(libzmq)` are unconditional, so a missing `zeromq` or
`cppzmq` fails CMake before anything builds.

To let the binaries fetch checkpoints with `-hf`, also install the Hugging Face
CLI - the fetch shells out to `hf` and stops with `hf: command not found`
without it:

```bash
pip install -U "huggingface_hub[cli]"   # or: uv tool install huggingface_hub
```

## Configure & build

On MacOS, Metal is enabled by default. Using Metal makes the computation run on the GPU.
To disable the Metal build at compile time use the `-DGGML_METAL=OFF` cmake option.

```bash
# cmake fetches llama.cpp at the pinned tag; no patch step.
# On MacOS, Metal is enabled by default
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
```

The VLA binaries pick their backend at load time and take no flag for it:
`-DGGML_METAL=OFF` at configure time is the only way to put `vla-cli`,
`vla-server` and `vla-bench` on the CPU. `VLA_DEVICE` only chooses an ordinal
for CUDA and SYCL, so it does nothing here, and `--n-gpu-layers` belongs to
`vlm-server` on the VLM path, not to the VLA binaries.

## GPU offload

Each arch calls `backend_init` (`src/backend.h`) exactly once at load time and
runs everything on what it returns, vision tower included. On macOS that is
Metal (`ggml_backend_metal_init`). Confirm it from the startup banner, which is
tagged with the arch:

```
vla(pi0): backend = Metal
```

SmolVLA is the one that logs under a bare `vla` tag, so for it the line reads:

```
vla: backend = Metal
```

There is no second banner to look for: the VLA path prints no `clip_ctx: CLIP
using GPU backend` line, because there is no separate CLIP context to bring up -
the single backend already covers both towers. A Metal build that reports only
the line above is working.

If you instead see `vla(<arch>): backend = CPU (N threads)`, the build didn't
pick up Metal - rebuild from a clean `build/` and check `GGML_METAL` is `ON` in
the CMake cache (`grep GGML_METAL build/CMakeCache.txt`).

> Single-backend, no per-op CPU fallback: the core uses one backend + `gallocr`,
> not a scheduler. SmolVLA's ops are all Metal-supported; an arch that hits an
> unimplemented op would assert at predict time rather than silently fall back.

BitVLA is the exception and does not run on Metal at all. It calls
`ggml_backend_cpu_init()` directly (`src/models/bitvla.cpp:568`) because its
graph stays on CPU and the LM offloads through CUDA, so it reports `vla(bitvla):
ggml backend = CPU (N threads)` even on a Metal build - that banner is expected,
not a broken build. The published GGUFs are also int2-packed, which `model_load`
rejects outside a CUDA build (`VLA_BITVLA_CUDA_KERNELS`), so on macOS it fails
to load rather than running slowly.

## Results

`vla-bench` times `predict()` in-process on synthetic inputs: engine only, no
transport, no simulator, no claim about task success. Apple M5 Max (18-core CPU,
40-core GPU, 64 GB unified memory), macOS 26.6.1, High Power Mode, llama.cpp
`b10331`, weights as shipped, 20 reps after 3 warmups, best of three sweeps (the
sweep with the lowest p50), each model at its native input size and view count -
the same counts the RTX 5090 table in
[the README](../../README.md#benchmarks) uses, so the two compare cell for cell.

| Model | Views | Input | min ms | p50 ms | p90 ms | vision ms |
|---|--:|--:|--:|--:|--:|--:|
| VLA-Adapter | 1 | 224 |  64.5 |  64.8 |  65.1 | 33.0 |
| VLA-JEPA    | 1 | 256 |  74.9 |  75.1 |  75.4 | 17.2 |
| GR00T N1.5  | 1 | 224 | 104.0 | 104.3 | 104.6 | 20.7 |
| SmolVLA     | 2 | 512 | 114.8 | 115.2 | 115.9 | 29.0 |
| GR00T N1.7  | 1 | 256 | 128.0 | 128.4 | 128.9 | 17.3 |
| GR00T N1.6  | 1 | 224 | 133.0 | 133.3 | 134.4 | 21.3 |
| OpenVLA-OFT | 1 | 224 | 183.4 | 184.2 | 184.6 | 33.7 |
| Evo-1       | 1 | 448 | 214.5 | 215.0 | 216.2 | 50.3 |
| pi0         | 2 | 224 | 220.2 | 220.8 | 221.3 | 39.3 |
| pi0.5       | 2 | 224 | 237.1 | 237.5 | 237.7 | 39.1 |

Every run came up on Metal (`backend = Metal` in the load banner); none fell
back to CPU. The three sweeps agree to within 0.6% per model, and `min` to `p90`
spans no more than 2 ms, so these settle rather than scatter.

BitVLA has no row, for the reason in the section above: there is no Metal path
to time.

### Agreement with the CPU backend

Outputs were checked against the CPU backend on all ten models above, on an M5
Max, using `vla_predict_check` on fixed images, tokens, state and noise - the
same build twice, once as configured and once with `-DGGML_METAL=OFF`. It is a
test target, so add `-DVLA_BUILD_TESTS=ON` to the configure line above to get
it.

The loosest model is π0: 1.9e-2 absolute on actions peaking near 1.0, RMS 5.2e-4
for that model, with a worst per-model RMS of 1.7e-3 across the set. Eight of
the ten stay under 7e-3 absolute. The two outliers are π0 and GR00T N1.7
(1.4e-2), which run multi-step denoise loops where per-step rounding compounds.
Metal output is deterministic run to run - repeated runs are bit-identical - so
these are BF16/F32 kernel rounding, not instability.

### Older M4 figures

These predate the table above and are not the same experiment - they were taken
end-to-end through `vla-server` on an M4, so they include transport and
preprocessing that `vla-bench` excludes, and they come from an older revision.
Read them as evidence that GPU offload is worth having, not as current numbers.

SmolVLA (libero, `mmproj` + 878 MiB BF16 weights), **Apple M4**, steady state:

| Stage        | CPU (before) | Metal GPU (after) |
|--------------|-------------:|------------------:|
| vision       |   22,367 ms  |          ~178 ms  |
| inference    |   12,878 ms  |          ~144 ms  |
| **total/req**|  ~35,250 ms  |         **~324 ms** |

≈ **108× faster** end-to-end. First request is ~671 ms (Metal pipeline warmup),
then it settles to ~321–328 ms/req.

### libero_object (10 episodes, Apple M4)

| Model       | SR  | Client/step | Server/step                          |
|-------------|----:|------------:|--------------------------------------|
| SmolVLA     | 0.7 |     888 ms  | 324 ms (181 vision + 141 inf + 2)    |
| Pi0         | 0.8 |    1135 ms  | 1129 ms (922 vision + 200 inf + 7)   |
| Gr00t-n1.7  | 1.0 |     755 ms  | 600 ms (185 vision + 405 inf + 10)   |
