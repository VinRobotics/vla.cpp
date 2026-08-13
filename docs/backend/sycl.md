# `vla.cpp` on Intel GPUs (SYCL backend)

Notes for building and running `vla.cpp` on Intel discrete and integrated GPUs
through oneAPI SYCL. Unlike Metal, SYCL is **not** auto-detected: it needs the
oneAPI DPC++ compiler and an explicit `-DGGML_SYCL=ON`.

Verified on an **Intel Arc A380** (DG2 / Xe-HPG, `8086:56a5`, 6 GB) on Ubuntu
22.04, kernel 6.8, with an AMD Ryzen host CPU. The same path covers the rest of
the Arc A/B series, Flex, Data Center Max, and the Xe iGPUs.

## Prerequisites

### 1. GPU compute runtime

The kernel driver (`i915`, in-tree since 6.2 for DG2) is not enough - you also
need the userspace compute stack: Level Zero plus the NEO OpenCL runtime.

```bash
wget -qO- https://repositories.intel.com/gpu/intel-graphics.key \
  | sudo gpg --yes --dearmor -o /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" \
  | sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list
sudo apt-get update
sudo apt-get install -y intel-opencl-icd libze-intel-gpu1 libze1 libze-dev intel-ocloc clinfo
```

Substitute your distro codename for `jammy`. Install the userspace packages
only - do **not** add `intel-i915-dkms` on a 6.8+ kernel, whose in-tree `i915`
already drives DG2.

Then give your user access to the render node and re-login:

```bash
sudo usermod -aG render,video "$USER"
```

Check it before going further - `clinfo -l` must name your GPU:

```
Platform #0: Intel(R) OpenCL Graphics
 `-- Device #0: Intel(R) Arc(TM) A380 Graphics
```

### 2. oneAPI

The SYCL backend needs the DPC++ compiler, oneMKL and oneDNN. **Deep Learning
Essentials** carries exactly those and is much smaller than the full Base
Toolkit.

```bash
wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | sudo gpg --yes --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt-get update
sudo apt-get install -y intel-deep-learning-essentials-2025.3
```

2025.3 is the newest release verified by llama.cpp's own SYCL docs that still
supports Ubuntu 22.04; the 2026.x series dropped jammy. Confirm the toolchain
sees the GPU over Level Zero:

```bash
source /opt/intel/oneapi/setvars.sh
sycl-ls
```

```
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Arc(TM) A380 Graphics 12.56.5 [1.6.31294+20]
[opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A380 Graphics OpenCL 3.0 NEO  [24.39.31294]
```

Plus the usual host dependencies. Ubuntu 22.04 has no `cppzmq` package, so drop
its two headers in by hand:

```bash
sudo apt-get install -y cmake ninja-build pkg-config \
    protobuf-compiler libprotobuf-dev libzmq3-dev
wget -q https://github.com/zeromq/cppzmq/archive/refs/tags/v4.10.0.tar.gz -O - | tar xz
sudo install -m644 cppzmq-4.10.0/zmq.hpp cppzmq-4.10.0/zmq_addon.hpp /usr/local/include/
```

## Configure & build

ggml's SYCL sources only compile under the oneAPI DPC++ driver, and
`CMAKE_CXX_COMPILER` is global, so the whole project - `vla_core`, the servers,
the CLI - is built by `icpx`. Configure a **fresh** build directory; switching
compilers in an existing one does not work.

```bash
source /opt/intel/oneapi/setvars.sh

cmake -B build-sycl -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_SYCL=ON \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl -j$(nproc)
```

`setvars.sh` must be sourced in every shell that builds *or runs* the binaries -
`libsycl.so`, `libdnnl.so` and the oneMKL libraries live under `/opt/intel`.

## GPU offload

The core picks its backend at load time. Confirm from the startup banner:

```
vla: backend = SYCL (device 0: Intel(R) Arc(TM) A380 Graphics)
```

If you see `vla: backend = CPU (8 threads)` instead, the build did not pick up
SYCL, or no SYCL device was visible - re-check `sycl-ls` and your `render` group
membership.

On a multi-GPU box, `VLA_DEVICE=<n>` selects the ordinal (the same variable
selects the CUDA device). The index is range-checked against the SYCL device
count; an out-of-range value logs and falls back to CPU rather than running off
the end of the device array.

> Single-backend, no per-op CPU fallback: the core drives one backend through
> `gallocr`, not a scheduler. An arch that hits an op the SYCL backend does not
> implement asserts at predict time rather than silently falling back.

BitVLA is the one exception: it pins its ggml graph to the CPU backend by design
and offloads its LM through separate hand-written CUDA kernels, so a SYCL build
leaves it on the CPU. There is no SYCL port of those kernels.

## Known issue: the SYCL VMM pool and oneDNN

ggml-sycl's VMM pool hands out virtual-memory-backed pointers that oneDNN cannot
wrap in a `dnnl::memory`. When it happens the GEMM aborts the process:

```
could not create a memory object
SYCL error: ... in function ggml_sycl_op_mul_mat at .../ggml-sycl.cpp:3055
```

It fires whenever `src0` is not already F32 - BF16, F16 and every quantized type
are converted into that pool before the GEMM - which is most checkpoints,
including the default BF16 weights of SmolVLA, π0, π0.5, Evo-1, VLA-Adapter and
OpenVLA-OFT.

`vla.cpp` defaults `GGML_SYCL_ENABLE_VMM=0` when it brings up SYCL, which avoids
it and is the faster of the two workarounds (disabling oneDNN with
`GGML_SYCL_ENABLE_DNN=0` also clears the crash, but costs ~8%). It is only a
default: set `GGML_SYCL_ENABLE_VMM=1` explicitly to keep the pool on hardware
where it pays off.

## Fixed upstream: `bf16 -> f32` copies

ggml-sycl's copy table used to have `f16 -> f32` but no `bf16 -> f32`, so an arch
whose graph contained that copy aborted at predict time. VLA-Adapter hit it with
its default BF16 weights, and the workaround was `VLA_ADAPTER_F32_WEIGHTS=1`.

llama.cpp b10326 adds the missing kernel (`cpy_1_bf16_f32` in
`ggml/src/ggml-sycl/cpy.cpp`), so VLA-Adapter should run on stock BF16 weights
now. Not yet re-tested on the A380 - if you hit the old abort, fall back to
`VLA_ADAPTER_F32_WEIGHTS=1` and file an issue.

## Performance note: F32 weights

BF16 has no native DPAS path on Xe-HPG, so BF16 weights are slower there than
plain F32 despite the extra bandwidth. Each arch exposes a switch
(`VLA_WEIGHT_DTYPE=f32` for SmolVLA, `VLA_PI0_F32_WEIGHTS=1` for π0, and so on),
and on the A380 it is worth ~16%:

| SmolVLA weights | vision | inference | total |
|---|---:|---:|---:|
| BF16 (default) | 158 ms | 474 ms | **630 ms** |
| F32 (`VLA_WEIGHT_DTYPE=f32`) | 173 ms | 355 ms | **528 ms** |

The tradeoff is memory - F32 doubles the resident weights (1.07 GiB -> 2.09 GiB
for SmolVLA), which matters on a 6 GB A380 for the larger checkpoints. The
default stays BF16 for that reason.

## Results

Measured with `vla_predict_check`, which is a test target - add
`-DVLA_BUILD_TESTS=ON` to the configure line above to get it. Fixed noise, so
runs are comparable; best of 5-10 iterations after 3 warmups. Host is an AMD Ryzen 5 5500 (CPU backend uses 8
threads); GPU is the Arc A380.

| Model | input | CPU | Arc A380 | speedup |
|---|---|---:|---:|---:|
| SmolVLA     | 512 | 1,920 ms | **630 ms** | 3.0x |
| Evo-1       | 448 | 7,695 ms | **1,176 ms** | 6.5x |
| VLA-Adapter | 224 | 2,994 ms | **517 ms** | 5.8x |

VLA-Adapter is measured with `VLA_ADAPTER_F32_WEIGHTS=1` on both sides, which was
required at the time (see the `bf16 -> f32` section above); the others run their
stock defaults.

Per-stage for SmolVLA:

| Stage | CPU | Arc A380 (SYCL) |
|--------------|-------------:|------------------:|
| vision       |    1,119 ms  |          158 ms  |
| inference    |      804 ms  |          474 ms  |
| **total/req**|  **1,920 ms**|      **630 ms**  |

SmolVLA gains least because its flow-matching denoise loop is a long chain of
small GEMMs that cannot fill 128 EUs; its vision tower alone is 7.1x. With
`VLA_WEIGHT_DTYPE=f32` it reaches 528 ms (3.6x).

Outputs were checked against the CPU backend on every model above: max absolute
deviation 2.9e-3 on actions peaking at 0.99 (2.9e-6 for the all-F32
VLA-Adapter run), RMS 2.4e-4 - BF16/F32 kernel rounding, not a numerical
regression.

### Memory ceiling

The A380 has 6 GB, of which ~5.7 GB is addressable. GR00T N1.7 (6.3 GB of F32
weights) does not fit and dies in the allocator:

```
level_zero backend failed with error: 38 (UR_RESULT_ERROR_OUT_OF_HOST_MEMORY)
```

`--weight-dtype bf16` (now the default) halves the weights but its activations still overflow
the card. There is no host-memory spill path - the core is single-backend - so
the larger checkpoints need an A770/B580-class card or better.
