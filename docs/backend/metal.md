# `vla.cpp` on macOS (Metal backend)

Short notes for getting `vla.cpp` to compile and link on macOS. The Metal
backend is auto-detected by the vendored `llama.cpp`/`ggml` and needs no special
CMake flag.

> **Status:** macOS/Metal is **build-and-link validated only**. All VLA inference
> paths in this repo (parity tests, LIBERO/SIMPLER sweeps, latency numbers) are
> CUDA-validated — see `CLAUDE.md`/`NOTES.md`. Treat Metal as a way to compile
> and develop on a Mac, not as a benchmarked inference target.

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