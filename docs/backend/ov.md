# `vla.cpp` on Intel CPUs, GPUs and NPUs (OpenVINO backend)

Notes for building `vla.cpp` against ggml's OpenVINO backend, and an honest
account of how far it currently runs. Like SYCL, OpenVINO is **not**
auto-detected: it needs an explicit `-DGGML_OPENVINO=ON` and the OpenVINO
runtime on the configure line.

> **Status: seven architectures translate faithfully, one is wrong.** SmolVLA,
> π0.5, Evo-1, VLA-Adapter, GR00T N1.5, GR00T N1.6 and VLA-JEPA all agree with a
> CPU-backend reference to 1.3e-3 or better on the OpenVINO CPU plugin - Evo-1 and
> VLA-Adapter to about 3e-6. On the Arc B390 iGPU the speedup over the native CPU
> backend runs from 3.0x to 9.6x. GR00T N1.7 translates and returns
> plausible-looking actions that are simply wrong; it is the one open failure.
> Twelve fixes were needed, ten of them inside ggml's OpenVINO backend, which is
> written against llama.cpp's graphs and had never seen a vision tower or an
> action expert - see [What had to change](#what-had-to-change).
>
> Note the baseline: OpenVINO executes the checkpoint's BF16 weights at F32, so
> compare against `--weight-dtype f32` or you will charge the backend for a
> precision upgrade. See [Picking the right baseline](#picking-the-right-baseline).

Measured on an **Intel Core Ultra X7 358H** (Panther Lake) with the Arc B390
iGPU and the AI Boost NPU, Ubuntu 24.04, OpenVINO 2026.2.1, llama.cpp `b10331`,
on the checkpoints under `vrfai/` on the Hub. `CMakeLists.txt` has since moved to
`b10729`, which is byte-identical on the CPU backend for all eleven archs; the
OpenVINO numbers below have not been re-measured on it.

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
| VLA-JEPA    | 256 | 1,046 ms | 1,265 ms | **127 ms** (8.2x) | returns NaN |
| GR00T N1.5  | 224 | 1,420 ms | 2,199 ms | **148 ms** (9.6x) | plugin throws |
| VLA-Adapter | 224 | 1,228 ms | 1,603 ms | **161 ms** (7.6x) | not supported |
| GR00T N1.6  | 224 | 1,276 ms | 2,256 ms | **323 ms** (3.9x) | plugin throws |
| SmolVLA     | 512 | 1,364 ms | 1,340 ms | **451 ms** (3.0x) | 1,162 ms |
| Evo-1       | 448 | 3,114 ms | 4,523 ms | **563 ms** (5.5x) | not supported |
| π0.5        | 224 | 2,802 ms | 4,285 ms | **683 ms** (4.1x) |   916 ms |
| GR00T N1.7  | 256 | not timed | not timed | not timed | not attempted |

GR00T N1.7 is not timed because it computes the wrong answer - a latency for work
that is not the same work would be misleading.

The iGPU is the reason to use this backend, and it pays off most where the model
is most vision-heavy. The OpenVINO CPU plugin is at best parity with ggml's own
CPU backend and often well behind it, so it is only worth running to debug a
translation. The NPU beats the CPU backend on the two models it accepts while
drawing far less power, which is the interesting result for a robot - see the
NPU limits under [Known issues](#known-issues).

### Picking the right baseline

Actions are checked against the CPU backend on identical inputs; both sides are
deterministic, so the numbers are exact rather than sampled. But **which** CPU run
you compare against matters more than it looks.

OpenVINO folds the checkpoint's BF16 weights in as constants and its CPU plugin
executes them at F32. ggml's CPU backend, on the same checkpoint, keeps them BF16.
So a naive comparison charges the OpenVINO backend for a precision *upgrade*.
Running the reference with `--weight-dtype f32` removes that term.

The two references bracket the answer, and which one is tighter is arch-dependent
- it turns on how much of a given checkpoint is BF16 in the first place. Report
both and take the smaller as the fidelity figure:

| Model | vs BF16 reference | vs F32 reference | tighter reference |
|---|---:|---:|---|
| VLA-Adapter | 2.1e-3 | **2.4e-6** | F32 |
| Evo-1       | 2.7e-3 | **3.5e-6** | F32 |
| π0.5        | 7.7e-4 | **3.5e-5** | F32 |
| VLA-JEPA    | 1.1e-2 | **1.1e-4** | F32 |
| GR00T N1.5  | 5.5e-3 | **6.0e-4** | F32 |
| GR00T N1.6  | 5.0e-3 | **1.0e-3** | F32 |
| SmolVLA     | **8.9e-4** | 1.3e-3 | BF16 |
| GR00T N1.7  | 1.5e0 | 1.5e0 | neither - it is wrong |

Evo-1 and VLA-Adapter agree with an F32 reference to six decimal places, which is
as close to "the translation is exact" as this harness can show - for those two,
OpenVINO is doing F32 arithmetic and the BF16 comparison was measuring nothing but
the dtype. SmolVLA is the counterexample that stops this being a universal rule: it lands
closer to the BF16 reference, so its checkpoint evidently is not uniformly BF16
where it matters. For context, the
CPU backend's own output moves by 2.0e-3 (SmolVLA), 2.7e-3 (Evo-1) or 1.1e-2
(VLA-JEPA) when you flip that one flag, so the model's intrinsic sensitivity to
precision is the same size as the numbers being reported.

### Full results

Against the F32 reference, which is the fidelity number:

| Model | OpenVINO CPU | OpenVINO GPU | OpenVINO NPU |
|---|---:|---:|---:|
| VLA-Adapter | 2.4e-6 | 3.2e-3 | not supported |
| Evo-1       | 3.5e-6 | 1.2e-3 | not supported |
| π0.5        | 3.5e-5 | 4.7e-4 | 9.9e-4 |
| VLA-JEPA    | 1.1e-4 | 8.7e-3 | returns NaN |
| GR00T N1.5  | 6.0e-4 | 4.7e-3 | plugin throws |
| GR00T N1.6  | 1.0e-3 | 2.0e-3 | plugin throws |
| SmolVLA     | 8.9e-4 | 1.3e-3 | 1.7e-2 |
| GR00T N1.7  | 1.5e0  | 1.5e0  | not attempted |

Every arch except GR00T N1.7 is inside the 2.9e-3 bar on the CPU plugin, most by
one to three orders of magnitude. On the GPU the picture is looser because that
plugin computes in F16: VLA-JEPA (8.7e-3), GR00T N1.5 (4.7e-3) and VLA-Adapter (3.2e-3) sit outside the bar there
even though all three are far inside it on the CPU plugin. Judge translation
fidelity on the CPU plugin; treat the GPU as a separate precision target.

Two effects explain the residuals that remain. The GPU plugin's F16 arithmetic is
one. The other is SmolVLA on the NPU (1.7e-2), whose compile config turns on
dynamic quantization - π0.5 on the same device stays at 9.9e-4, so that is a
property of the model on that device rather than of the backend.

GR00T N1.7 is wrong outright - see [What is left](#what-is-left).

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

The other ten are in ggml's OpenVINO backend itself, applied by
`scripts/patch_ggml_openvino.py` at configure time. Its docstring carries the
detail; in short each narrows an llama.cpp-shaped assumption that is stricter
than the ggml contract, or fills a gap:

| Fix | What it addresses |
|---|---|
| **GELU translated as tanh, not erf** | **assumes ggml's GELU is the exact erf form** |
| Intel OpenCL platform selection | assumes the first OpenCL platform is Intel's |
| RESHAPE `op_case` guard | assumes a reshape flattening dims 0-2 is the KV-cache flatten |
| SDPA K/V converted with Q | assumes K/V arrive as F16 because the KV cache is |
| **Position inputs keyed per tensor** | **assumes a graph has exactly one position input** |
| Folded weights padded to full rank | a 2-D weight becomes a rank-2 constant, but views index it at ggml rank |
| CONCAT input ranks aligned | same rank-2 constants, and concat cannot broadcast rank |
| Missing op translators | RELU, GELU_ERF, NEG, SQR had no table entry |
| Naive-path graph cache | that path re-compiled the whole model on every graph_compute |

Four are worth expanding.

**The GELU mode** is the highest-yield single fix in the list. ggml's
`GGML_UNARY_OP_GELU` is the *tanh* approximation - its CPU kernel additionally
reads an fp16 lookup table - while `ov::op::v7::Gelu` defaults to the exact erf
formulation, and both ggml GELU ops were mapped onto that default. The error per
node is small, but a Qwen3-VL vision tower contains dozens of them and it
compounds through the encoder. Setting the mode explicitly moved VLA-JEPA from
5.5e-3 to 1.1e-4 (48x) and GR00T N1.5 from 2.7e-2 to 6.0e-4 (45x), turning both
from "runs but drifts" into supported, and improved GR00T N1.6 and π0.5 too. It
is worth upstreaming alongside the position-input fix.

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

**The NPU takes two of the eight archs, and fails four different ways.** SmolVLA
and π0.5 run. The others do not:

| Model | NPU outcome |
|---|---|
| Evo-1 | compiler rejects: `Input channels '1025' is not aligned by '16'` |
| VLA-Adapter | compiler rejects: `Input channels '261' is not aligned by '16'` |
| VLA-JEPA | compiles and runs, returns all `NaN` |
| GR00T N1.5 | NPUW partitioning throws (`partitioning.cpp:1350`) |
| GR00T N1.6 | plugin throws (`core.cpp:117`) |
| GR00T N1.7 | not attempted - wrong on CPU and GPU already |

Five archs, four distinct failures, none of them vla.cpp's. The two alignment rejections are
Intel's NPU compiler: 1025 is Evo-1's 1024
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

**GR00T N1.7 translates but computes the wrong answer.** It runs to completion
and returns a full action chunk, so nothing errors, but 86% of the 5,280 values
are off by more than 1e-2 and the worst is 1.48 against a peak of 0.948. The
result is deterministic and identical on the CPU and GPU plugins, so this is a
translation bug rather than a device or precision effect. It is **not** the mrope
handling: fixes 9 and 10 in `scripts/patch_ggml_openvino.py` both target that
path and neither moved the number by a digit. VLA-JEPA shares the same Qwen3-VL
vision tower and the same interleaved mrope and is only ~1% off, which points at
GR00T N1.7's own action expert rather than anything shared. Not yet diagnosed;
the arch stays `-` in the README matrix.

**GR00T N1.7 is the one open failure.** It translates, returns a full action
chunk, and the values are wrong: max|delta| 1.477, rms 8.3e-2 against a peak of
0.948, with 86% of 5280 values off by more than 1e-2. Deterministic, and
identical on the CPU and GPU plugins. Ruled out so far, each with evidence:

- *Precision.* Identical against BF16 and F32 references. OpenVINO is 1530x less
  weight-dtype-sensitive than ggml CPU here, and the model's own bf16/f32
  sensitivity is rms 7.2e-4 against the error's 8.3e-2.
- *Translation-path selection.* The naive path is taken by default and
  `is_model_splitted` returns false for every graph of this arch; forcing all
  graphs down the decoder-only-LLM path instead changes the answer by 1e-5 while
  both remain 1.477 from the reference.
- *Sequence length and token composition.* Swept 3.7x (SEQ 70 to 262) by two
  independent routes; relative error stayed within 0.2545-0.2841 and the fraction
  off by >1e-2 within 84.7-86.6%. Nothing accumulates.
- *Matmul precision and the compiled-model cache.* Both bitwise no-ops.
- *The GELU mode*, which fixed VLA-JEPA and GR00T N1.5, moves N1.7 by nothing.
- *A missing or unsupported op.* GR00T N1.6 (works) and N1.7 (broken) have the
  same 17-op vocabulary, and LM layer 0 is node-for-node identical except that
  N1.7's position input is `[4*SEQ]` for IMROPE where N1.6's is `[SEQ]` for NEOX.
  VLA-JEPA also uses IMROPE and is now clean, which weakens that lead.

Three earlier claims about N1.7 turned out to be **measurement artifacts** and
should not be reused. That its first LM block is 73-95% wrong: OpenVINO's
`lm_h_00..03` dumps are bit-exact copies of the *input* arrays and `lm_h_04..15`
are zeros, because the main graph has exactly one real output, `action_pred`.
That `--flash-attn 1` yields 0.948: it actually fails with "Got less inputs than
expected" and returns no actions, and 0.948 is `max|reference|` - the number you
get comparing against nothing. And that honouring `GGML_TENSOR_FLAG_OUTPUT`
changes the answer: same artifact.

The next step follows from the first artifact. The stage dump cannot see inside
the graph because ggml-openvino writes back only true graph outputs, so either
add a debug mode that materialises selected intermediates as `ov::Result`s, or
bisect with cut-down graphs.

**Untested archs.** π0 and OpenVLA-OFT are untested here for want of a local
checkpoint, not because anything is known to block them; both use op sets already
covered by tested archs (π0 matches π0.5, OpenVLA-OFT matches VLA-Adapter). Both
are `~` in the README matrix; treat any untested arch that way until it has
actually produced actions.

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
