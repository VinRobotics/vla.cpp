# `vla.cpp` on Intel CPUs, GPUs and NPUs (OpenVINO backend)

Notes for building `vla.cpp` against ggml's OpenVINO backend, and an honest
account of how far it currently runs. Like SYCL, OpenVINO is **not**
auto-detected: it needs an explicit `-DGGML_OPENVINO=ON` and the OpenVINO
runtime on the configure line.

> **Status: five architectures run end to end.** SmolVLA, π0.5, Evo-1,
> VLA-Adapter and VLA-JEPA all produce actions matching the CPU backend, on the
> CPU and GPU plugins; the NPU takes two of the five. On the Arc B390 iGPU the
> speedup over the native CPU backend runs from 3.1x to 8.2x. Nine fixes were
> needed, all but two of them inside ggml's OpenVINO backend, which is written
> against llama.cpp's graphs and had never seen a vision tower or an action
> expert - see [What had to change](#what-had-to-change).

Measured on an **Intel Core Ultra X7 358H** (Panther Lake) with the Arc B390
iGPU and the AI Boost NPU, Ubuntu 24.04, OpenVINO 2026.2.1, llama.cpp `b10331`
(the tag `CMakeLists.txt` pins), on the checkpoints under `vrfai/` on the Hub.

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

`vla_predict_check` (a test target - add `-DVLA_BUILD_TESTS=ON`), fixed noise,
one camera view, best of 4-6 iterations after 3 warmups. "CPU backend" is ggml's
own CPU backend on the same 16-core host. No `GGML_OPENVINO_CACHE_DIR`, for the
reason in [Known issues](#known-issues).

| Model | input | CPU backend | OpenVINO CPU | OpenVINO GPU | OpenVINO NPU |
|---|---|---:|---:|---:|---:|
| SmolVLA     | 512 | 1,364 ms | 1,340 ms | **446 ms** (3.1x) | 1,162 ms |
| π0.5        | 224 | 2,802 ms | 4,285 ms | **641 ms** (4.4x) |   916 ms |
| Evo-1       | 448 | 3,114 ms | 4,523 ms | **574 ms** (5.4x) | not supported |
| VLA-Adapter | 224 | 1,228 ms | 1,603 ms | **162 ms** (7.6x) | not supported |
| VLA-JEPA    | 256 | 1,046 ms | 1,265 ms | **128 ms** (8.2x) | not supported |

The iGPU is the reason to use this backend, and it pays off most where the model
is most vision-heavy. The OpenVINO CPU plugin is at best parity with ggml's own
CPU backend and often well behind it, so it is only worth running to debug a
translation. The NPU beats the CPU backend on the two models it accepts while
drawing far less power, which is the interesting result for a robot - see the
NPU limits under [Known issues](#known-issues).

Actions checked against the CPU backend on identical inputs. Both sides are
deterministic run to run, so these are exact, not sampled:

| Model | device | max abs deviation | RMS | peak action |
|---|---|---:|---:|---:|
| SmolVLA     | CPU | 1.2e-3 | 1.7e-4 | 0.995 |
| SmolVLA     | GPU | 1.2e-3 | 1.9e-4 | 0.995 |
| SmolVLA     | NPU | 1.6e-2 | 2.1e-3 | 0.995 |
| π0.5        | CPU | 8.9e-4 | 8.7e-5 | 0.904 |
| π0.5        | GPU | 6.9e-4 | 1.1e-4 | 0.904 |
| π0.5        | NPU | 1.6e-3 | 1.9e-4 | 0.904 |
| Evo-1       | CPU | 2.7e-3 | 3.8e-4 | 0.899 |
| Evo-1       | GPU | 2.9e-3 | 4.2e-4 | 0.899 |
| VLA-Adapter | CPU | 2.1e-3 | 7.1e-4 | 0.662 |
| VLA-Adapter | GPU | 2.9e-3 | 1.1e-3 | 0.662 |
| VLA-JEPA    | CPU | 1.2e-2 | 4.1e-3 | 1.145 |
| VLA-JEPA    | GPU | 9.1e-3 | 4.5e-3 | 1.145 |

Most of these sit in the same band as the SYCL backend's numbers - kernel
rounding, plus the F16 K/V conversion the SDPA fix introduces.

Two rows are looser and worth naming rather than burying. SmolVLA on the NPU is
an order of magnitude off because the NPU compile config turns on dynamic
quantization; π0.5 on the same device is not, so treat it as a property of that
model on that device. **VLA-JEPA is the loosest CPU/GPU result at ~1% relative,
and it is not diagnosed** - the error is spread evenly across the action vector
rather than sitting in one element, and VLA-JEPA does not use flash attention, so
the SDPA conversion is not the cause. Verify it against your own policy before
trusting VLA-JEPA on this backend.

## What had to change

Two fixes on the vla.cpp side. Both are ordinary correctness fixes that happen to
be invisible on the other backends:

- **Weight buffers are tagged.** `ggml_backend_alloc_ctx_tensors` leaves a buffer
  on `GGML_BACKEND_BUFFER_USAGE_ANY`, and ggml-openvino reads ANY as "KV cache",
  giving every weight a dynamic sequence dimension. `vla::alloc_weights` in
  [`src/backend.h`](../../src/backend.h) tags it `..._WEIGHTS`, which is what
  llama.cpp does with its own weights and what lets the frontend fold them in as
  constants. Since 0.3.0 every arch allocates through `vla::WeightLoader`, so
  this is one call site in [`src/loader.cpp`](../../src/loader.cpp).
- **Graph tensors get unique names.** ggml derives a result's name from its
  source, so `ggml_reshape_2d` of an unnamed tensor is called `" (reshaped)"` -
  and a graph whose intermediates were never named ends up with many tensors
  sharing one name. ggml-openvino keys its translation map on those names, so
  duplicates silently collapse into one node and the graph wires the wrong tensor
  into the next op. `vla::graph_unique_names` relabels duplicates before compute,
  at each of the 29 `ggml_backend_graph_compute` call sites. It compiles to
  nothing outside an OpenVINO build.

`backend_init` also sets one default, the way the SYCL rung already sets
`GGML_SYCL_ENABLE_VMM=0`: **`GGML_OPENVINO_NAIVE_GRAPH_SIZE` defaults high.**
ggml-openvino translates a graph under 20 nodes literally and sends anything
larger through a model builder that assumes a decoder-only LLM. The literal path
is the one that fits a vision tower and an action expert. An explicit setting
still wins.

The other seven are in ggml's OpenVINO backend itself, applied by
`scripts/patch_ggml_openvino.py` at configure time. Its docstring carries the
detail; in short each narrows an llama.cpp-shaped assumption that is stricter
than the ggml contract, or fills a gap:

| Fix | What it addresses |
|---|---|
| Intel OpenCL platform selection | assumes the first OpenCL platform is Intel's |
| RESHAPE `op_case` guard | assumes a reshape flattening dims 0-2 is the KV-cache flatten |
| SDPA K/V converted with Q | assumes K/V arrive as F16 because the KV cache is |
| **Position inputs keyed per tensor** | **assumes a graph has exactly one position input** |
| Folded weights padded to full rank | a 2-D weight becomes a rank-2 constant, but views index it at ggml rank |
| CONCAT input ranks aligned | same rank-2 constants, and concat cannot broadcast rank |
| Missing op translators | RELU, GELU_ERF, NEG, SQR had no table entry |
| Naive-path graph cache | that path re-compiled the whole model on every graph_compute |

Three are worth expanding.

**Position inputs** is what carries an arch through to a full prediction. Every
tensor feeding a `GGML_OP_ROPE`'s second input was renamed to a single parameter
called `inp_pos`, and one shared sin/cos table was built from it. SmolVLA passes
three position tensors - prefill, full and rebased - so they aliased each other
and every RoPE took the table built from whichever won:

```text
opset1::Multiply (Split[1]:f32[1,113,5,32], Multiply[0]:f32[1,50,1,32])
Argument shapes are inconsistent.
```

When the graph has more than one, each keeps its own name. Nothing is then called
`inp_pos`, the shared-table precompute returns early, and `translate_rope()`
falls back to building sin/cos per op from its own position input - a path that
already existed for mixed RoPE parameters. Single-position graphs are untouched.

**Rank padding** is the other structural one. A ggml tensor that is 2-D folds in
as a rank-2 constant, which is what a GEMM operand wants, but the graph indexes
it at full ggml rank. Evo-1 views Q, K and V out of one fused `attn_in` weight
and got `Axis 2 out of the tensor rank range [-2, 1]`. Padding in
`process_view_input_new` fixed that class generally, and padding in the concat
translator let vla.cpp **delete** an arch-specific workaround that had moved
SmolVLA's time tiles into their own buffer.

**The naive-path cache** is about speed, not correctness. The dynamic and static
paths keep a `graph_key`-indexed cache; the naive path had none, so it rebuilt
the decoder, re-converted the model and called `compile_model()` on every single
`ggml_backend_graph_compute`. SmolVLA on the CPU plugin ran at 22.7 s per
prediction before, 1.4 s after.

None of the vla.cpp-side changes alter what the other backends compute:
`vla_predict_check` on a CPU build of this branch is byte-identical to the same
build of the base commit, for every model tested.

## Known issues

**Do not set `GGML_OPENVINO_CACHE_DIR`.** OpenVINO's on-disk blob cache reloads a
compiled graph that computes the wrong thing. A cold run against a fresh cache
directory is correct; the very next run, reading back the blobs it just wrote, is
not:

```text
GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_CACHE_DIR=$dir   # cold: max |delta| 1.2e-3
GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_CACHE_DIR=$dir   # warm: max |delta| 2.9e0
```

Nothing is logged - the actions are simply wrong, which for a policy server is
the worst possible failure mode. `backend_init` warns at startup when the
variable is set. Unverified guess at the cause: the blob key does not capture
something that differs between vla.cpp's several graphs, so one graph gets
another's blob. In practice, pay the compile once per process and leave it unset.

**The NPU takes two of the five archs, and fails three different ways.** SmolVLA
and π0.5 run. The others do not:

| Model | NPU outcome |
|---|---|
| Evo-1 | compiler rejects: `Input channels '1025' is not aligned by '16'` |
| VLA-Adapter | compiler rejects: `Input channels '261' is not aligned by '16'` |
| VLA-JEPA | compiles and runs, returns all `NaN` |

The two rejections are Intel's NPU compiler, not vla.cpp: 1025 is Evo-1's 1024
patches plus a CLS token, 261 is VLA-Adapter's 256 plus 5, and neither is a
multiple of 16. SmolVLA and π0.5 happen to have 16-aligned sequence lengths. The
VLA-JEPA NaN is a third failure mode and is not diagnosed. Note that a
partially-failing NPU run still reports a wall-clock time, so do not read a
latency number off a run whose actions did not come out.

**SmolVLA's `VLA_TIMING=phase` path is wrong under OpenVINO.** SmolVLA has a
second graph builder used when a caller asks for per-phase timings, and it does
not survive translation - max |delta| 1.9 on every device, with or without the
in-process cache. The default `TimingDetail::NONE` path, which is what
`vla-server` and `vla-cli` use, is correct, and on the native CPU backend the two
paths agree exactly. Evo-1 and π0.5 are unaffected on the same path, so this is
specific to SmolVLA's second graph. One hypothesis - that the split-graph guard
sends it down the LLM path - was tested and is wrong: forcing the naive path on
split graphs returns zeros. Per-stage timings for SmolVLA are therefore omitted
from the tables above.

## What is left

**Op coverage is no longer the blocker.** With RELU, GELU_ERF, NEG and SQR added
to the table, every ggml op the eleven in-tree archs build is translatable. The
one exception is `ggml_map_custom1`, used only by BitVLA - and BitVLA pins its
ggml graph to the CPU backend by design and offloads its LM through hand-written
CUDA kernels, so an OpenVINO build leaves it on the CPU regardless. `GGML_OP_SQR`
is therefore **untested**: only BitVLA emits it, and BitVLA never reaches this
backend. It is in the table because it is a real gap in ggml-openvino, not
because anything here exercises it.

**Untested archs.** π0 and OpenVLA-OFT are untested here for want of a local
checkpoint, not because anything is known to block them; both use op sets already
covered by tested archs (π0 matches π0.5, OpenVLA-OFT matches VLA-Adapter). The
GR00T family is being brought up separately. Treat any untested arch as `-` in
the README matrix until it has actually produced actions.

**Splitting across devices.** Intel's own
[π0.5 write-up](https://docs.openedgeplatform.intel.com/2026.1/OEP-articles/publications/optimizing-pi0.5-lva-model.html)
puts the vision encoder and language model on the iGPU and the action expert on
the NPU, with the KV cache as the only cross-device handoff. That is a different
toolchain - PyTorch exported to OpenVINO IR as three separate models, no ggml -
so none of it drops into this backend. What carries over is the shape of the
answer: the two devices suit different stages, and π0.5 on the NPU alone is
already within 1.5x of the iGPU at a fraction of the power.

vla.cpp cannot make that split today because the core drives one backend for a
whole prediction. It would need a per-*stage* backend rather than a per-op
scheduler - the vision tower, the prefix and the action expert already hand off
through host memory, so the seam is in the right place - but that is an engine
change, not a backend one.
