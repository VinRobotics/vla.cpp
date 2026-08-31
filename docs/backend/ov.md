# `vla.cpp` on Intel CPUs, GPUs and NPUs (OpenVINO backend)

Notes for building `vla.cpp` against ggml's OpenVINO backend, and an honest
account of how far it currently runs. Like SYCL, OpenVINO is **not**
auto-detected: it needs an explicit `-DGGML_OPENVINO=ON` and the OpenVINO
runtime on the configure line.

> **Status: SmolVLA and π0.5 run end to end on CPU, GPU and NPU.** The Arc iGPU
> is 2.9x faster than the native CPU backend on SmolVLA and 4.4x on π0.5. Six fixes
> were needed, four of them inside ggml's OpenVINO backend, which is written
> against llama.cpp's graphs and had never seen a vision tower or an action
> expert - see [What had to change](#what-had-to-change). The other archs are
> blocked on ops the backend has no translator for, listed under
> [What is left](#what-is-left).

Measured on an **Intel Core Ultra X7 358H** (Panther Lake) with the Arc B390
iGPU and the AI Boost NPU, Ubuntu 24.04, OpenVINO 2026.2.1, against
`vrfai/smolvla-libero-gguf` and `vrfai/pi05-libero-gguf`.

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

The NPU needs two more things that its driver packages do not pull in. Neither
failure is reported as an error - the device simply does not appear, and every
`GGML_OPENVINO_DEVICE=NPU` run lands on the CPU plugin instead:

```text
GGML OpenVINO Backend: device NPU is not available, fallback to CPU
OpenVINO: using device CPU
```

**1. The Level Zero loader.** `intel-level-zero-npu` ships `libze_intel_npu.so.1`,
the *driver*; OpenVINO's NPU plugin only reaches it through the loader.

```bash
sudo apt-get install -y libze1          # provides libze_loader.so.1
```

`ldconfig -p | grep ze_loader` is the check.

**2. Point the loader at the NPU driver.** Ubuntu's loader (1.16.1 in noble)
does not discover `libze_intel_npu.so.1` on its own, so installing it is not
enough by itself. Name the driver explicitly:

```bash
export ZE_ENABLE_ALT_DRIVERS=/lib/x86_64-linux-gnu/libze_intel_npu.so.1
```

With that set the device enumerates as `NPU  Intel(R) AI Boost` and the startup
banner reads `OpenVINO: using device NPU`. A loader from Intel's own graphics
repository, version-matched to the NPU driver, should discover it without the
override - untested here.

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

`scripts/patch_ggml_openvino.py` runs as the FetchContent patch step, so the six
ggml fixes described in its docstring are applied automatically and re-applied on
a clean reconfigure. There is no manual `git apply`.

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

OpenVINO compiles each graph on first use, which is slow - a minute or two for a
vision tower on the GPU. Compiled graphs are then cached in-process for the life
of the model, so only the first prediction pays that; give any client a receive
timeout well above the first request.

Do **not** set `GGML_OPENVINO_CACHE_DIR` to carry them across restarts. It
produces silently wrong actions here - see
[Known issues](#known-issues). The backend warns at startup if it is set.

## Results

`vla_predict_check` (a test target - add `-DVLA_BUILD_TESTS=ON`), fixed noise, one
camera view, best of 6 iterations (4 for π0.5) after 3 warmups. "CPU backend" is
ggml's own CPU backend on the same 16-core host; the other columns are this build
with `GGML_OPENVINO_DEVICE` set. No `GGML_OPENVINO_CACHE_DIR`, for the reason in
[Known issues](#known-issues).

| Model | CPU backend | OpenVINO CPU | OpenVINO GPU | OpenVINO NPU |
|---|---:|---:|---:|---:|
| SmolVLA (512px) | 1,312 ms | 1,357 ms | **448 ms** (2.9x) | 1,107 ms (1.2x) |
| π0.5 (224px)    | 2,775 ms | 4,278 ms | **633 ms** (4.4x) | 931 ms (3.0x) |

The iGPU is the reason to use this backend. The OpenVINO CPU plugin is at best
parity with ggml's own CPU backend and on π0.5 well behind it, so it is only
worth running to debug a translation. The NPU beats the CPU backend on both
models while drawing far less power, which is the interesting result for a robot.

Checked against the CPU backend on the same inputs:

| Run | max abs deviation | RMS | peak action |
|---|---:|---:|---:|
| SmolVLA, OpenVINO CPU | 1.2e-3 | 1.7e-4 | 0.995 |
| SmolVLA, OpenVINO GPU | 1.2e-3 | 1.9e-4 | 0.995 |
| SmolVLA, OpenVINO NPU | 1.6e-2 | 2.1e-3 | 0.995 |
| π0.5, OpenVINO CPU    | 8.9e-4 | 8.7e-5 | 0.904 |
| π0.5, OpenVINO GPU    | 6.9e-4 | 1.1e-4 | 0.904 |
| π0.5, OpenVINO NPU    | 1.6e-3 | 1.9e-4 | 0.904 |

CPU and GPU sit in the same band as the SYCL backend's numbers - kernel rounding,
plus the F16 K/V conversion the SDPA fix introduces. SmolVLA on the NPU is an
order of magnitude looser because the NPU compile config turns on dynamic
quantization; π0.5 is not, so treat SmolVLA's NPU deviation as a property of that
model on that device rather than of the backend.

## What had to change

Two of these are ordinary correctness fixes on the vla.cpp side that happen to be
invisible on the other backends:

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

One is a judgement call about what a weight is:

- **SmolVLA's time tiles moved out of the weight buffer.** They are precomputed
  once but they are graph inputs, not checkpoint parameters. As weights they
  became 2-D constants that could not be concatenated with the 4-D activation
  beside them.

And `backend_init` sets one default, the way the SYCL rung already sets
`GGML_SYCL_ENABLE_VMM=0`:

- **`GGML_OPENVINO_NAIVE_GRAPH_SIZE` defaults high.** ggml-openvino translates a
  graph under 20 nodes literally and sends anything larger through a model
  builder that infers a decoder-only LLM. The literal path is the one that fits a
  vision tower and an action expert; the threshold is raised in `backend_init`,
  and an explicit setting still wins.

The remaining four are in ggml's OpenVINO backend itself, applied by
`scripts/patch_ggml_openvino.py` at configure time. Its docstring carries the
detail; in short they narrow an llama.cpp-shaped assumption that is stricter than
the ggml contract:

| Fix | Assumption it relaxes |
|---|---|
| Intel OpenCL platform selection | the first OpenCL platform is Intel's |
| RESHAPE `op_case` guard | a reshape flattening dims 0-2 is the KV-cache flatten |
| SDPA K/V converted with Q | K/V arrive as F16 because the KV cache is |
| **Position inputs keyed per tensor** | **a graph has exactly one position input** |

The last one is what carries an arch through to a full prediction, and it is the
one worth upstreaming. Every tensor feeding a `GGML_OP_ROPE`'s second input was
renamed to a single parameter called `inp_pos`, and a shared sin/cos table was
built from it. SmolVLA passes three position tensors - prefill, full and rebased -
so they aliased each other and every RoPE took the table built from whichever
won:

```text
opset1::Multiply (Split[1]:f32[1,113,5,32], Multiply[0]:f32[1,50,1,32])
Argument shapes are inconsistent.
```

When the graph has more than one, each keeps its own name. Nothing is then called
`inp_pos`, the shared-table precompute returns early, and `translate_rope()`
falls back to building sin/cos per op from its own position input - a path that
already existed for mixed RoPE parameters. Graphs with a single position input
are untouched and keep the shared table.

The fourth fix is about speed rather than correctness: the naive path had no
`graph_key` cache, so it re-converted and re-compiled the whole OpenVINO model on
*every* `ggml_backend_graph_compute`. SmolVLA on the CPU plugin ran at 22.7 s per
prediction before that was fixed and 1.8 s after.

None of the vla.cpp-side changes alter what the other backends compute:
`vla_predict_check` on a CPU build of this branch is byte-identical to the same
build of `main` for SmolVLA and π0.5, apart from the `weight_buf` line, which
drops by the size of the time tiles that moved.

## Known issues

**Do not set `GGML_OPENVINO_CACHE_DIR`.** OpenVINO's on-disk blob cache reloads a
compiled graph that computes the wrong thing. A cold run against a fresh cache
directory is correct; the very next run, reading back the blobs it just wrote, is
not:

```text
GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_CACHE_DIR=$dir   # cold: max |delta| 1.2e-3
GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_CACHE_DIR=$dir   # warm: max |delta| 2.9e0
```

Nothing is logged - the actions are just wrong, which for a policy server is the
worst possible failure mode. `backend_init` warns at startup when the variable is
set. Unverified guess at the cause: the blob key does not capture something that
differs between vla.cpp's several graphs, so one graph gets another's blob. In
practice, pay the compile once per process and leave it unset.

**SmolVLA's `VLA_TIMING=phase` path is wrong under OpenVINO.** SmolVLA has a
second graph builder used when a caller asks for per-phase timings, and it does
not survive translation - max |delta| 1.9 on every device, with or without the
in-process cache. The default `TimingDetail::NONE` path, which is what
`vla-server` and `vla-cli` use, is correct. On the native CPU backend the two
paths agree exactly, so this is specific to the OpenVINO translation of that
second graph and is not yet diagnosed. π0.5's phase path is unaffected. Per-stage
timings for SmolVLA are therefore omitted from the table above.

## What is left

**Op coverage.** The core drives a single backend through `gallocr` rather than a
scheduler, so there is no per-op CPU fallback. An arch that uses an op
ggml-openvino has no translator for cannot run at all:

| Op | Archs that need it |
|---|---|
| `GGML_UNARY_OP_RELU` | GR00T N1.5/1.6/1.7, Evo-1, VLA-Adapter, OpenVLA-OFT, BitVLA, VLA-JEPA |
| `GGML_UNARY_OP_GELU_ERF` | Evo-1, VLA-Adapter, OpenVLA-OFT, GR00T N1.6, BitVLA |
| `GGML_OP_NEG`, `GGML_OP_SQR` | VLA-JEPA, GR00T N1.7, BitVLA |

SmolVLA, π0 and π0.5 are the three archs fully covered today. π0 is untested here
only because there was no checkpoint on the machine; its op set matches π0.5's.

**BitVLA** is a separate case: it pins its ggml graph to the CPU backend by
design and offloads its LM through hand-written CUDA kernels, so an OpenVINO
build leaves it on the CPU regardless.

**Splitting across devices.** Intel's own
[π0.5 write-up](https://docs.openedgeplatform.intel.com/2026.1/OEP-articles/publications/optimizing-pi0.5-lva-model.html)
puts the vision encoder and language model on the iGPU and the action expert on
the NPU, with the KV cache as the only cross-device handoff. That is a different
toolchain - PyTorch exported to OpenVINO IR as three separate models, no ggml -
so none of it drops into this backend. What does carry over is the shape of the
answer: the two devices are good at different stages, and π0.5 is already within
1.5x of the iGPU on the NPU alone at a fraction of the power.

vla.cpp cannot make that split today because the core drives one backend for a
whole prediction. It would need a per-*stage* backend rather than a per-op
scheduler - the vision tower, the prefix and the action expert already hand off
through host memory, so the seam is in the right place - but that is an engine
change, not a backend one.
