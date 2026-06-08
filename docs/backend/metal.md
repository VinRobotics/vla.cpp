# `vla.cpp` on macOS (Metal backend)

Short notes for getting `vla.cpp` to compile and link on macOS. The Metal
backend is auto-detected by the vendored `llama.cpp`/`ggml` and needs no special
CMake flag.

## Prerequisites

```bash
brew install protobuf zeromq cppzmq pkg-config
```
## Configure & build

On MacOS, Metal is enabled by default. Using Metal makes the computation run on the GPU.
To disable the Metal build at compile time use the `-DGGML_METAL=OFF` cmake option.

When built with Metal support, you can explicitly disable GPU inference with the `--n-gpu-layers 0` command-line argument.

```bash
# Fetch llama.cpp at pinned tag and apply local patch
bash patches/patch.sh

# On MacOS, Metal is enabled by default
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
```

## GPU offload

The VLA core selects its compute backend at load time. On macOS it picks Metal
(`ggml_backend_metal_init`); the CLIP/vision encoder mirrors that choice, so both
the transformer and the vision tower run on the GPU. Confirm it from the startup
banner:

```
vla: backend = Metal
clip_ctx: CLIP using GPU backend
```

If you instead see `vla: backend = CPU (4 threads)` / `CLIP using CPU backend`,
the build didn't pick up Metal — rebuild from a clean `build/` and check
`GGML_METAL` is `ON` in the CMake cache.

> Single-backend, no per-op CPU fallback: the core uses one backend + `gallocr`,
> not a scheduler. SmolVLA's ops are all Metal-supported; an arch that hits an
> unimplemented op would assert at predict time rather than silently fall back.

## Results

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
