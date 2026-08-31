# `vla.cpp` on Intel CPUs, GPUs and NPUs (OpenVINO backend)

Notes for building `vla.cpp` against ggml's OpenVINO backend, and an honest
account of how far it currently runs. Like SYCL, OpenVINO is **not**
auto-detected: it needs an explicit `-DGGML_OPENVINO=ON` and the OpenVINO
runtime on the configure line.

> **Status: builds and runs, no arch completes a prediction yet.** The backend
> comes up, weights fold in, and the vision towers translate and execute - on
> the CPU plugin and on the GPU plugin alike. The language model and action
> expert do not: ggml's OpenVINO backend models a decoder-only LLM with one
> position input and an F16 KV cache, and every vla.cpp arch has several
> position inputs and no KV cache. See
> [What still blocks it](#what-still-blocks-it). The OpenVINO column of the
> README support matrix stays `-` until an arch passes end to end.

Checked on an **Intel Core Ultra X7 358H** (Panther Lake) with the Arc B390
iGPU, Ubuntu 24.04, OpenVINO 2026.2.1, against `vrfai/smolvla-libero-gguf` and
`vrfai/pi05-libero-gguf`.

OpenVINO is Intel's inference toolkit; ggml's backend translates a ggml compute
graph into an OpenVINO model and hands it to the CPU, GPU or NPU plugin, which
compiles and fuses it for the device. Unlike SYCL it needs no separate compiler:
the stock GCC/Clang build links `libopenvino` and everything else is ordinary
C++.

## Supported devices

- Intel CPUs
- Intel GPUs (integrated Xe / Arc, and discrete)
- Intel NPUs (Core Ultra)

## Prerequisites

Linux (Ubuntu 22.04 or 24.04) on Intel hardware.

### 1. Device access

The CPU plugin needs nothing. The GPU and NPU plugins reach the hardware through
`/dev/dri/renderD*` and `/dev/accel/accel0`, both owned by the `render` group:

```bash
sudo usermod -aG render,video "$USER"
```

Re-login, then check that the OpenCL runtime actually enumerates the GPU:

```bash
clinfo -l
```

`Number of platforms 0` with `/etc/OpenCL/vendors/intel.icd` present almost
always means the render group has not taken effect yet. Without it,
`GGML_OPENVINO_DEVICE=GPU` warns and silently falls back to the CPU plugin.

For the GPU compute runtime and NPU driver packages themselves, follow
[llama.cpp's OpenVINO notes](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md).

The NPU additionally needs the **Level Zero loader**, which the NPU driver
packages do not pull in:

```bash
sudo apt-get install -y libze1          # provides libze_loader.so.1
```

`intel-level-zero-npu` ships `libze_intel_npu.so.1`, the *driver*; OpenVINO's NPU
plugin reaches it through the loader and enumerates nothing without one. The
symptom is not an error - the device simply does not appear:

```text
GGML OpenVINO Backend: device NPU is not available, fallback to CPU
OpenVINO: using device CPU
```

`ldconfig -p | grep ze_loader` is the check.

### 2. OpenVINO runtime + OpenCL headers

```bash
sudo apt-get install -y opencl-clhpp-headers ocl-icd-opencl-dev opencl-headers
```

Then either install OpenVINO
[from the archive](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-archive-linux.html)
by hand, or run the bundled installer, which also pulls the GPU driver stack and
adds you to `render`:

```bash
bash scripts/install_ov.sh
```

Plus the usual host dependencies:

```bash
sudo apt-get install -y cmake ninja-build pkg-config \
    protobuf-compiler libprotobuf-dev libzmq3-dev cppzmq-dev
```

## Configure & build

```bash
source /opt/intel/openvino/setupvars.sh

cmake -B build-ov -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENVINO=ON
cmake --build build-ov -j$(nproc)
```

`setvars`-style sourcing is needed in every shell that builds *or* runs the
binaries: `libopenvino.so` and its TBB live under `/opt/intel`. Configure fails
early with a pointer back here if the runtime is not on `CMAKE_PREFIX_PATH`.

`scripts/patch_ggml_openvino.py` runs as the FetchContent patch step, so the
four ggml fixes described in its docstring are applied automatically and
re-applied on a clean reconfigure. There is no manual `git apply`.

## Run

`GGML_OPENVINO_DEVICE` picks the target by name. Do not type the placeholder
`<DEVICE_TYPE>` literally - in a shell the angle brackets are input redirection.

```bash
GGML_OPENVINO_DEVICE=GPU ./build-ov/vla-server ./weights/smolvla-libero.gguf
```

Two lines identify the selection at startup:

```text
OpenVINO: using device GPU
vla: backend = OPENVINO (requested device GPU)
```

The first comes from ggml and is authoritative: an unavailable device logs a
warning there and falls back to `CPU`, which is still the OpenVINO CPU plugin,
not ggml's native CPU backend. The second line echoes what was requested, so the
pair tells you whether you got the device you asked for. `VLA_DEVICE` does *not*
apply - ggml exposes OpenVINO as a single device and the target is chosen by
name.

OpenVINO compiles each graph on first use, which is slow (minutes for a vision
tower). Set `GGML_OPENVINO_CACHE_DIR=<dir>` to keep compiled graphs across
restarts, and give any client a receive timeout well above the first request.

## What vla.cpp had to change

Three of these are ordinary correctness fixes that happen to be invisible on the
other backends:

- **Weight buffers are tagged.** `ggml_backend_alloc_ctx_tensors` leaves a
  buffer on `GGML_BACKEND_BUFFER_USAGE_ANY`, and ggml-openvino reads ANY as "KV
  cache", giving every weight a dynamic sequence dimension. `vla::alloc_weights`
  in [`src/backend.h`](../../src/backend.h) tags it `..._WEIGHTS`, which is what
  llama.cpp does with its own weights and what lets the frontend fold them in as
  constants.
- **Graph tensors get unique names.** ggml derives a result's name from its
  source, so `ggml_reshape_2d` of an unnamed tensor is called `" (reshaped)"` -
  and a graph whose intermediates were never named ends up with many tensors
  sharing one name. ggml-openvino keys its translation map on those names, so
  duplicates silently collapse into one node and the graph wires up the wrong
  tensor. `vla::graph_unique_names` relabels duplicates before compute. It
  compiles to nothing outside an OpenVINO build.
- **SmolVLA's time tiles moved out of the weight buffer.** They are precomputed
  once but they are graph inputs, not checkpoint parameters. As weights they
  became 2-D constants that could not be concatenated with the 4-D activation
  beside them.
- **`GGML_OPENVINO_NAIVE_GRAPH_SIZE` defaults high.** ggml-openvino translates a
  graph under 20 nodes literally and sends anything larger through an LLM model
  builder. The literal path is the one that fits a vision tower; the threshold is
  raised in `backend_init`, and an explicit setting still wins.

None of it changes what the other backends compute: `vla_predict_check` on a CPU
build of this branch is byte-identical to the same build of `main` for SmolVLA
and π0.5, apart from the `weight_buf` line, which drops by the size of the time
tiles that moved.

## What still blocks it

With the above in place, SmolVLA's SigLIP tower translates and runs, and the
prefix/expert graph reaches OpenVINO's shape inference before failing:

```text
opset1::Multiply (Split[1]:f32[1,113,5,32], Multiply[0]:f32[1,50,1,32])
Argument shapes are inconsistent.
```

The two operands are RoPE tables of different lengths. `GgmlOvDecoder` maps
*every* tensor feeding a `GGML_OP_ROPE`'s second input to one graph parameter
named `inp_pos`, because an llama.cpp graph has exactly one position input. Every
vla.cpp arch has several - SmolVLA alone passes a prefill, a full and a rebased
position tensor - and they collapse onto each other.

π0.5 fails on the same node with the same message (`[1,50,1,128]` against
`[1,262,1,128]`), so this is the shared blocker rather than a SmolVLA quirk.

That is not something vla.cpp can work around from the outside: the fix belongs
in ggml-openvino, which needs to key position inputs per tensor rather than by a
fixed name. The same class of assumption shows up in the KV-cache-shaped dynamic
sequence dimension and in the `compute_op_case` pattern tables, two of which
already needed narrowing (see `scripts/patch_ggml_openvino.py`).

Separately, several archs use ops the backend has no translator for at all -
`GGML_UNARY_OP_RELU` (every GR00T, Evo-1, VLA-Adapter, OpenVLA-OFT, BitVLA,
VLA-JEPA), `GGML_UNARY_OP_GELU_ERF`, `GGML_OP_NEG`, `GGML_OP_SQR` - and the core
drives a single backend through `gallocr` rather than a scheduler, so there is no
per-op CPU fallback to absorb them. SmolVLA, π0 and π0.5 are the three archs
whose op sets are fully covered today, which is why SmolVLA is the one to retest
first when the position-input handling lands upstream.
