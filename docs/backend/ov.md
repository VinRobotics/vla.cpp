# `vla.cpp` on Intel CPUs, GPUs and NPUs (OpenVINO backend)

Notes for building `vla.cpp` against ggml's OpenVINO backend, and an honest
account of how far it currently runs. Like SYCL, OpenVINO is **not**
auto-detected: it needs an explicit `-DGGML_OPENVINO=ON` and the OpenVINO
runtime on the configure line.

> **Status: every architecture that can reach this backend translates faithfully
> and runs on all three plugins: CPU, iGPU and NPU.** That is eleven archs -
> SmolVLA, π0, π0.5, Evo-1, VLA-Adapter, GR00T N1.5, GR00T N1.6, GR00T N1.7,
> VLA-JEPA, OpenVLA-OFT and TurboVLA. On the OpenVINO CPU plugin each agrees with
> ggml's CPU backend at F32 weights to 2.5e-3 or better. The Arc B390 iGPU stays
> inside the error ggml's own BF16 default carries on every BF16 arch, at 7.3x to 26x
> the native CPU backend's speed. The NPU runs them all and is the fastest
> OpenVINO device for six; it lands 2-14x outside that BF16 error on four, and π0
> is wrong there. Octo-Small has no GroupNorm translation yet, ACT is in progress,
> and BitVLA pins its ggml graph to the CPU backend by design.
>
> Twenty fixes were needed, eighteen of them inside ggml's OpenVINO backend, which
> is written against llama.cpp's graphs and had never seen a vision tower or an
> action expert - see [What had to change](#what-had-to-change).
>
> Note the baseline: OpenVINO executes the checkpoint's BF16 weights at F32, so
> compare against `--weight-dtype f32` or you will charge the backend for a
> precision upgrade. See [Picking the right baseline](#picking-the-right-baseline).

Measured on an **Intel Core Ultra X7 358H** (Panther Lake) with the Arc B390
iGPU and the AI Boost NPU, Ubuntu 24.04, OpenVINO 2026.4, NPU driver 1.33.0,
llama.cpp `b11223`, at main `a83bcad` (November 2026), on the checkpoints under
`vrfai/` on the Hub. The latencies come from the
[Core Ultra X7 358H report](../benchmark/core-ultra-x7-358h.md); the fidelity
numbers compare fixed inputs against ggml's own CPU backend.

ggml's backend translates a ggml compute graph into an OpenVINO model and hands
it to the CPU, GPU or NPU plugin, which compiles and fuses it for the device.
Unlike SYCL it needs no separate compiler: the stock GCC/Clang build links
`libopenvino`.

## Supported devices

Intel CPUs, Intel GPUs (integrated Xe / Arc, and discrete), and Intel NPUs
(Core Ultra). Linux only here - Ubuntu 22.04 or 24.04.

## Prerequisites

### 1. Device access

The CPU plugin needs nothing. The GPU and NPU plugins reach the hardware through
`/dev/dri/renderD*` and `/dev/accel/accel0`, both owned by the `render` group:

```bash
sudo usermod -aG render,video "$USER"   # re-login afterwards
clinfo -l                               # must enumerate the GPU
```

`Number of platforms 0` with `/etc/OpenCL/vendors/intel.icd` present almost
always means the render group has not taken effect yet. Without it,
`GGML_OPENVINO_DEVICE=GPU` warns and silently falls back to the CPU plugin.

For the GPU compute runtime and NPU driver packages themselves, follow
[llama.cpp's OpenVINO notes](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md).

The NPU needs two more things its driver packages do not pull in. Neither failure
is reported as an error - the device simply does not appear and every
`GGML_OPENVINO_DEVICE=NPU` run lands on the CPU plugin instead
(`device NPU is not available, fallback to CPU`).

```bash
sudo apt-get install -y libze1   # the Level Zero loader; the driver alone is not enough
                                 # check with: ldconfig -p | grep ze_loader
export ZE_ENABLE_ALT_DRIVERS=/lib/x86_64-linux-gnu/libze_intel_npu.so.1
```

Ubuntu's loader (1.16.1 in noble) does not discover `libze_intel_npu.so.1` on its
own, hence the override; a loader from Intel's own graphics repository,
version-matched to the driver, should not need it - untested here. With both in
place the device enumerates as `NPU  Intel(R) AI Boost`.

### 2. OpenVINO runtime + OpenCL headers

```bash
sudo apt-get install -y opencl-clhpp-headers ocl-icd-opencl-dev opencl-headers \
    cmake ninja-build pkg-config protobuf-compiler libprotobuf-dev \
    libzmq3-dev cppzmq-dev
```

Then either install OpenVINO
[from the archive](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-archive-linux.html)
by hand, or run `bash scripts/install_ov.sh`, which also pulls the GPU driver
stack and adds you to `render`.

## Configure & build

```bash
source /opt/intel/openvino/setupvars.sh

cmake -B build-ov -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON
cmake --build build-ov -j$(nproc)
```

`setupvars.sh` must be sourced in every shell that builds *or* runs the binaries:
`libopenvino.so` and its TBB live under `/opt/intel`. Configure fails early with
a pointer back here if the runtime is not on `CMAKE_PREFIX_PATH`.

`scripts/patch_ggml_openvino.py` runs as the FetchContent patch step, so the
thirteen ggml fixes are applied automatically - there is no manual `git apply`.
The step only runs when FetchContent populates the source dir, so a `build/_deps`
left over from an older checkout keeps the hunks it was patched with: delete it
after pulling rather than trusting a reconfigure. The script checks each hunk on
its own and fails loudly on a tree it cannot bring up to date.

## Run

`GGML_OPENVINO_DEVICE` picks the target by name (`VLA_DEVICE` does *not* apply -
ggml exposes OpenVINO as a single device):

```bash
GGML_OPENVINO_DEVICE=GPU ./build-ov/vla-server ./weights/smolvla-libero.gguf
```

Two lines identify the selection at startup:

```text
OpenVINO: using device GPU
vla: backend = OPENVINO (asked for GPU, see ggml's "using device" line)
```

The first comes from ggml and is authoritative: an unavailable device logs a
warning there and falls back to `CPU`, which is still the OpenVINO CPU plugin,
not ggml's native CPU backend. The second echoes what was requested, so the pair
tells you whether you got the device you asked for.

OpenVINO compiles each graph on first use, which is slow: up to about 80 s per
process on the GPU, and on the NPU 1-3 minutes for most archs, 7 for SmolVLA, 9
for π0 and π0.5 and 14 for OpenVLA-OFT
([per arch](../benchmark/core-ultra-x7-358h.md#notes)). Compiled graphs are then cached in-process for the life
of the model, so only the first prediction pays that; give any client a receive
timeout well above the first request. Do **not** set `GGML_OPENVINO_CACHE_DIR` to
carry them across restarts; it produces silently wrong actions here - see
[Known issues](#known-issues).

## Results

`vla-bench` on synthetic inputs, each arch at its native view count: 3 warmups
and 20 timed calls per process, 3 processes, the best process's mean, at the
defaults. "CPU backend" is ggml's own CPU backend on the same 16 cores. The
[Core Ultra X7 358H report](../benchmark/core-ultra-x7-358h.md) adds min, p50,
p90, memory and the fastest flags for each device. No `GGML_OPENVINO_CACHE_DIR`.

| Model | views | input | CPU backend | OpenVINO CPU | OpenVINO GPU | OpenVINO NPU |
|---|--:|--:|--:|--:|--:|--:|
| TurboVLA | 2 | 256 | 232 ms | 281 ms | 30 ms (7.8x) | **25 ms** (9.2x) |
| VLA-JEPA | 1 | 256 | 1,265 ms | 1,413 ms | 117 ms (10.8x) | **91 ms** (13.9x) |
| GR00T N1.5 | 1 | 224 | 1,986 ms | 2,467 ms | **107 ms** (18.5x) | 135 ms (14.7x) |
| VLA-Adapter | 1 | 224 | 1,666 ms | 1,838 ms | 185 ms (9.0x) | **128 ms** (13.0x) |
| GR00T N1.6 | 1 | 224 | 1,705 ms | 2,312 ms | **131 ms** (13.0x) | 162 ms (10.5x) |
| GR00T N1.7 | 1 | 256 | 1,539 ms | 2,084 ms | **132 ms** (11.7x) | 146 ms (10.6x) |
| SmolVLA | 2 | 512 | 1,763 ms | 1,914 ms | 215 ms (8.2x) | **165 ms** (10.7x) |
| π0.5 | 2 | 224 | 6,134 ms | 7,551 ms | **234 ms** (26.2x) | 438 ms (14.0x) |
| Evo-1 | 1 | 448 | 4,122 ms | 4,651 ms | 568 ms (7.3x) | **258 ms** (16.0x) |
| OpenVLA-OFT | 1 | 224 | 9,990 ms | 11,721 ms | 1,335 ms (7.5x) | **408 ms** (24.5x) |
| π0 | 2 | 224 | 6,160 ms | 7,584 ms | 625 ms (9.9x) | 445 ms, [wrong](#known-issues) |

The NPU is the fastest OpenVINO device for TurboVLA, VLA-JEPA, VLA-Adapter,
SmolVLA, Evo-1 and OpenVLA-OFT, and the GPU plugin for the GR00T models and π0.5.
π0's NPU number is a latency only; see [Known issues](#known-issues). The OpenVINO
CPU plugin is 9-36% behind ggml's own CPU backend on every arch, so it is only
worth running to debug a translation. On the same iGPU, ggml's SYCL backend beats
both OpenVINO devices on SmolVLA, VLA-Adapter, GR00T N1.5, GR00T N1.6 and π0
([sycl.md](sycl.md)); the report has all five ways to run on this chip side by
side.

### Picking the right baseline

Actions are checked against the CPU backend on identical inputs; both sides are
deterministic, so the numbers are exact rather than sampled. But **which** CPU run
you compare against matters. OpenVINO folds the checkpoint's BF16 weights in as
constants and its CPU plugin executes them at F32, while ggml's CPU backend keeps
them BF16, so a naive comparison charges the OpenVINO backend for a precision
*upgrade*. Running the reference with `--weight-dtype f32` removes that term.

The two references bracket the answer. On the OpenVINO CPU plugin, max|delta| on
the continuous action dims (0-5), six fixed inputs per arch:

| Model | vs BF16 reference | vs F32 reference |
|---|---:|---:|
| TurboVLA    | 3.1e-6 | **3.1e-6** |
| OpenVLA-OFT | 1.4e-3 | **4.0e-5** |
| VLA-Adapter | 3.5e-3 | **4.7e-5** |
| Evo-1       | 2.7e-3 | **5.7e-5** |
| π0.5        | 1.2e-3 | **8.2e-5** |
| π0          | 5.5e-3 | **1.3e-4** |
| GR00T N1.5  | 2.8e-3 | **4.9e-4** |
| GR00T N1.7  | 2.4e-2 | **6.6e-4** |
| SmolVLA     | 2.9e-2 | **7.6e-4** |
| VLA-JEPA    | 2.4e-2 | **1.4e-3** |
| GR00T N1.6  | 1.4e-2 | **2.5e-3** |

At `b11223` the F32 reference is the tighter one for every arch, SmolVLA included,
which was the one exception at `b10729`. TurboVLA's GGUF is F32, so its two
references are the same run. OpenVLA-OFT, VLA-Adapter, Evo-1 and π0.5 agree with
the F32 reference to 1e-4 or better, which is as close to "the translation is
exact" as this harness can show; against BF16 they measure nothing but the dtype.
The BF16 column is also about what the CPU backend's own output moves when you
flip that one flag, so the models' sensitivity to precision is the same size as
the numbers being reported, or larger.

### Full results

Against the F32 reference, which is the fidelity number: max|delta| on the
continuous action dims (0-5), six fixed inputs per arch (2 views for SmolVLA, π0,
π0.5 and TurboVLA, 1 for the rest, 16-token prompts; π0 on the GPU and NPU: 48
inputs). A bracket counts the chunks whose gripper (dim 6) flips at a different
step from the reference. The first column is ggml's own CPU backend at its default
BF16 weights, the yardstick for what precision alone does:

| Model | CPU backend, BF16 | OpenVINO CPU | OpenVINO GPU | OpenVINO NPU |
|---|---:|---:|---:|---:|
| TurboVLA    | F32 GGUF | 3.1e-6 | 2.7e-6 | 8.1e-3 |
| OpenVLA-OFT | 1.4e-3 | 4.0e-5 | 4.1e-5 | 8.1e-4 |
| VLA-Adapter | 3.5e-3 | 4.7e-5 | 4.8e-5 | 7.6e-3 |
| Evo-1       | 2.7e-3 | 5.7e-5 | 5.8e-5 | 3.8e-2 |
| π0.5        | 1.2e-3 | 8.2e-5 | 4.2e-4 | 5.9e-3 |
| π0          | 5.4e-3 | 1.3e-4 | 7.3e-4 | **3.7e-2** [1/48] |
| GR00T N1.5  | 2.8e-3 | 4.9e-4 | 8.1e-4 | 8.8e-3 |
| GR00T N1.7  | 2.5e-2 [1/6] | 6.6e-4 | 1.7e-2 [1/6] | 2.2e-2 [1/6] |
| SmolVLA     | 2.8e-2 [2/6] | 7.6e-4 | 1.2e-2 [1/6] | 1.8e-2 [1/6] |
| VLA-JEPA    | 2.4e-2 | 1.4e-3 | 1.4e-2 | 1.4e-2 |
| GR00T N1.6  | 1.3e-2 | 2.5e-3 | 2.2e-3 | 4.9e-3 |

Every arch is within 2.5e-3 of the F32 reference on the CPU plugin, most by one to
two orders of magnitude: the translation is faithful. Judge translation fidelity
there, and treat the GPU and NPU as separate precision targets. The GPU plugin
lands below the CPU backend's own BF16 error on every BF16 arch, and VLA-Adapter,
Evo-1, OpenVLA-OFT and TurboVLA come out as close as on the CPU plugin. The NPU
computes in F16 only. It lands below the BF16 yardstick on SmolVLA, the GR00T
N1.6 and N1.7 models, VLA-JEPA and OpenVLA-OFT, and 2-14x above it on VLA-Adapter,
GR00T N1.5, π0.5 and Evo-1. The gripper flips in the GPU and NPU columns also
appear between ggml's own BF16 and F32 runs on these inputs: a saturating ±1
channel near its threshold flips a step early or late on any precision change,
and reads as a delta near 2. π0 is the exception; see
[Known issues](#known-issues).

## What had to change

Twenty fixes: two in vla.cpp, eighteen in ggml's OpenVINO backend. Both
vla.cpp-side ones are ordinary correctness fixes that happen to be invisible on
the other backends - `vla_predict_check` on a CPU build of this branch is
byte-identical to the same build of the base commit, for every model tested.

- **Weight buffers are tagged.** `ggml_backend_alloc_ctx_tensors` leaves a buffer
  on `GGML_BACKEND_BUFFER_USAGE_ANY`, and ggml-openvino reads ANY as "KV cache",
  giving every weight a dynamic sequence dimension. `vla::alloc_weights` in
  [`src/backend.h`](../../src/backend.h) tags it `..._WEIGHTS`, which is what
  lets the frontend fold weights in as constants. Since 0.3.0 every arch
  allocates through `vla::WeightLoader`, so this is one call site in
  [`src/loader.cpp`](../../src/loader.cpp).
- **Graph tensors get unique names.** ggml derives a result's name from its
  source, so a graph whose intermediates were never named ends up with many
  tensors sharing one name (`" (reshaped)"` and friends). ggml-openvino keys its
  translation map on those names, so duplicates silently collapse into one node
  and the graph wires the wrong tensor into the next op. `vla::graph_unique_names`
  relabels duplicates before compute, at each of the 29
  `ggml_backend_graph_compute` call sites. It compiles to nothing outside an
  OpenVINO build.

`backend_init` also sets one default, the way the SYCL rung already sets
`GGML_SYCL_ENABLE_VMM=0`: **`GGML_OPENVINO_NAIVE_GRAPH_SIZE` defaults high.**
ggml-openvino translates a graph under 20 nodes literally and sends anything
larger through a model builder that assumes a decoder-only LLM. The literal path
is the one that fits a vision tower and an action expert. An explicit setting
still wins.

The other eighteen are applied to ggml's OpenVINO backend by
`scripts/patch_ggml_openvino.py` at configure time; its docstring carries the
per-fix detail. Each narrows an llama.cpp-shaped assumption that is stricter than
the ggml contract, or fills a gap:

| Fix | What it addresses |
|---|---|
| **PERMUTE op_case 2 requires a ROPE** | **assumes any permute of a view is a rope'd query** |
| **Two elementwise adds never stacked on a GEMM** | **the GPU plugin folds both in as post-ops and drops the second operand** |
| GPU inference precision exposed | the plugin's F16 default compounds through a denoise loop unrolled in one graph |
| **GELU translated as tanh, not erf** | **assumes ggml's GELU is the exact erf form** |
| Intel OpenCL platform selection | assumes the first OpenCL platform is Intel's |
| RESHAPE `op_case` guard | assumes a reshape flattening dims 0-2 is the KV-cache flatten |
| SDPA K/V converted with Q | assumes K/V arrive as F16 because the KV cache is |
| **Position inputs keyed per tensor** | **assumes a graph has exactly one position input** |
| Folded weights padded to full rank | a 2-D weight becomes a rank-2 constant, but views index it at ggml rank |
| CONCAT input ranks aligned | same rank-2 constants, and concat cannot broadcast rank |
| Missing `GELU_ERF` translator | the exact-erf GELU op had no table entry at all, so a graph using it could not run |
| Naive-path graph cache | that path re-compiled the whole model on every graph_compute, and its `graph_key` is a node count plus tensor names, which two graphs of different shapes can share |
| Interleaved-mrope sectors bounded | the sector cycle ignored `sections`, so the last few took the wrong stream |
| Naive-path threshold settable | the 20-node constant is what picks the literal path |
| **No NPUW for naive graphs on the NPU** | **the NPU config turns on NPUW for llama.cpp's decoder layers, and its partitioning hands the compiler dynamic Reshape inputs** |
| **An RMSNorm that cannot overflow f16, on the NPU** | **the NPU computes in f16, and a 2048-wide sum of squares overflows once activations reach about 6** |
| 2D im2col folded into one Convolution | the matcher only knew the 1D chain, so the NPU ran a vision tower's patch embed on its SHAVE cores |
| NPU compile properties settable | `GGML_OPENVINO_NPU_PROPS="K=V,..."` reaches the compile config, for example `NPU_TURBO=YES` |

The bolded rows are the ones that turned a wrong arch into a correct one:
PERMUTE op_case 2 for GR00T N1.7, the double-elementwise guard for the Eagle-VLM
archs on the iGPU, the GELU mode for VLA-JEPA and GR00T N1.5, per-tensor position
inputs for SmolVLA (which passes three position tensors, so they aliased), no
NPUW for π0.5, Evo-1, the GR00T models and VLA-JEPA on the NPU, which did not
compile before, and the f16-safe RMSNorm for the GR00T models and VLA-JEPA,
which returned NaN there. The 2D im2col fold is a speedup: SmolVLA's 512 px
views spent 350 ms each in the NPU's patch extraction. The GELU mode and the
position-input fix are the two worth upstreaming.

**Debugging a mistranslation.** The backend writes back true graph outputs and
nothing else, so an interior tensor is observable only two ways: truncate the
graph at a stage, which makes that stage the terminal node, or set
`GGML_OPENVINO_DEBUG_NODE` to materialise one node as an extra `ov::Result`. Both
of the subtlest fixes above were found by bisecting that way, comparing each
stage against a CPU-backend reference; neither logs anything when it goes wrong.

## FoldQuant (W8A8 / W4A4) checkpoints

A FoldQuant GGUF runs on the CPU and GPU plugins through
`src/openvino/foldquant_ov.cpp`, which translates FoldQuant's two custom nodes
into OpenVINO ops with the weights kept as `i8` / `i4` constants (hunk 14 of
`scripts/patch_ggml_openvino.py` registers it). The NPU, and `VLA_FQ_DEQUANT=1`,
read the sites back as float weights instead. See
[QUANTIZATION.md](../QUANTIZATION.md#openvino) for the arithmetic and accuracy.

π0.5 LIBERO, 2 views at 224 px, 48 tokens, Intel Core Ultra X7 358H with an Arc
B390 iGPU, OpenVINO 2026.4, llama.cpp `b11223`, `vla-bench` p50 over 10 calls
after 3 warmups:

| Device | Checkpoint | p50 ms | Peak RSS |
|---|---|--:|--:|
| CPU | bf16 | 6601 | 16.6 GB |
| CPU | FoldQuant W4A4 | 3751 | 6.5 GB |
| CPU | FoldQuant W4A4, o/down INT8 | 3682 | 7.1 GB |
| CPU | FoldQuant W8A8 | 3536 | 8.6 GB |
| GPU | bf16 | 243 | 11.5 GB |
| GPU | FoldQuant W4A4 | 227 | 3.7 GB |
| GPU | FoldQuant W4A4, o/down INT8 | 229 | 4.1 GB |
| GPU | FoldQuant W8A8 | 241 | 4.8 GB |

On the GPU the per-site activation ops (reduction, rounding, the rotation
MatMul) run on top of the GEMMs, so FoldQuant is only 7% faster than bf16 there,
at a third of the memory. On an Intel GPU the SYCL backend is the faster
FoldQuant path: built with sycl-tla, the same W4A4 file runs in 92 ms on this
iGPU, against 263 ms for bf16 on SYCL
([sycl.md](sycl.md#foldquant-w8a8--w4a4-checkpoints)). The NPU runs π0.5's bf16
GGUF in 438 ms; FoldQuant files read back as float weights there and were not
timed.

The FoldQuant files are FoldQuantVLA quantized checkpoints converted with
`scripts/convert_quantized_model_to_gguf.py`: W4A4 and W4A4 with INT8 o/down are
[vrfai/pi05-libero-w4a4](https://huggingface.co/vrfai/pi05-libero-w4a4) and
[vrfai/pi05-libero-w4a4-res8](https://huggingface.co/vrfai/pi05-libero-w4a4-res8);
W8A8 is FoldQuantVLA's W8A8 arm (`--llm-scheme w8a8_sr --expert-scheme w8a8_sh`)
calibrated on the same 128 LIBERO frames, seed 0.

## Known issues

**π0 needs F32 on the GPU, and gets it by default.** The GPU plugin computes in
F16, which is most of why it is fast. π0 unrolls its whole 10-step denoise loop
inside a single graph, so that error compounds with nothing to reset it. On 48
fixed inputs against the CPU backend at F32 weights, F16 puts its continuous
action dims up to 2.8e-2 off, and in one chunk in 48 its gripper - a saturating
±1 channel - crosses its threshold one step early. That reads as max|delta| 1.8;
on a robot it is an early grasp or release. `GGML_OPENVINO_GPU_PRECISION=f32`
brings the continuous dims to 7.3e-4 with every flip on time, and `backend_init`
defaults it for π0 alone because it costs 2.5x (255 ms -> 625 ms). Set `=f16` to
override. No other arch needs it.

**The 2D im2col fold costs VLA-Adapter on the CPU plugin.** Folding
`ggml_conv_2d`'s im2col chain into one Convolution is what made SmolVLA fast on
the NPU, and on the GPU it is 5-7% faster for VLA-Adapter and OpenVLA-OFT. On the
CPU plugin it is slower: VLA-Adapter's vision takes 1,100 ms with it and 974 ms
without (1,828 against 1,704 ms end to end), measured alternating on the same
binary. `GGML_OPENVINO_FUSE_CONV2D=0` turns it off.

**Do not set `GGML_OPENVINO_CACHE_DIR`.** OpenVINO's on-disk blob cache reloads a
compiled graph that computes the wrong thing. A cold run against a fresh cache
directory is correct; the very next run, reading back the blobs it just wrote, is
not - max|delta| 1.2e-3 cold, 2.9e0 warm, with nothing logged, which for a policy
server is the worst possible failure mode. `backend_init` therefore clears the
variable and says so; `VLA_ALLOW_OV_CACHE=1` keeps it. Unverified guess at the
cause: the blob key does not capture something that differs between vla.cpp's
several graphs, so one graph gets another's blob - the same class of bug as the
in-process `graph_key` above, which is now keyed on shapes. In practice, pay the
compile once per process and leave it unset.

**Every arch now compiles on the NPU, but check every NPU run against the startup
banner.** An unavailable NPU falls back to the CPU plugin silently and would
otherwise report excellent numbers that are not NPU numbers at all. A
partially-failing run still reports a wall-clock time, so do not read a latency
off a run whose actions did not come out.

At `b10729` the NPU accepted three of the ten archs, and only two of those were
correct. At `b11223`, with OpenVINO 2026.4 and NPU driver 1.33, all eleven compile
and run, and the [Full results](#full-results) give their error. Three fixes did
most of it: no NPUW for naive graphs (π0.5, Evo-1, the GR00T models and VLA-JEPA
failed on dynamic Reshape inputs or an LLVM abort), the f16-safe RMSNorm (the GR00T
models and VLA-JEPA returned NaN), and the 2D im2col fold (SmolVLA's vision went
from 1,121 to 55 ms). Evo-1 (1025 input channels) and VLA-Adapter and
OpenVLA-OFT (261), which the NPU compiler used to reject as not aligned to 16,
compile at this pin as well.

The NPU is slow to compile: 1-3 minutes per process for most archs, 7 for
SmolVLA, 9 for π0 and π0.5 and 14 for OpenVLA-OFT, all before the first action
comes out. It also drops its clock between requests, which costs a policy loop
with 50-500 ms gaps 50-80 ms per request (`vla-bench --gap-ms` shows it);
`GGML_OPENVINO_NPU_PROPS=NPU_TURBO=YES` keeps it up.

**π0 on the NPU is the same bug as π0 on the GPU at F16, and here there is no
remedy.** On the same 48 inputs, continuous dims 0-5 land within 3.7e-2 (median
8.3e-3), and in one chunk in 48 the gripper flips one step early (max|delta| 2.0),
as at F16 on the GPU. The GPU fix does not transfer: at `b10729`, setting the
inference precision to F32 made the NPU refuse to compile at all
(`core.cpp:117`), because the hint conflicted with the NPUW and
dynamic-quantization config the NPU path set up then. So `GGML_OPENVINO_GPU_PRECISION` is GPU-only by necessity,
and π0 should not be run on the NPU.

**SmolVLA's `VLA_TIMING=phase` path is wrong under OpenVINO** (found at `b10729`,
not re-checked since). SmolVLA has a
second graph builder used when a caller asks for per-phase timings, and it does
not survive translation - max|delta| 1.9 on every device. The default
`TimingDetail::NONE` path, which is what `vla-server` and `vla-cli` use, is
correct, and on the native CPU backend the two paths agree exactly. Evo-1 and
π0.5 are unaffected on the same path, so this is specific to SmolVLA's second
graph; one hypothesis - that the split-graph guard sends it down the LLM path -
was tested and is wrong. Per-stage timings for SmolVLA are omitted from the
tables above.

## TODO

**Splitting across devices.** Intel's own
[π0.5 write-up](https://docs.openedgeplatform.intel.com/2026.1/OEP-articles/publications/optimizing-pi0.5-lva-model.html)
puts the vision encoder and language model on the iGPU and the action expert on
the NPU, with the KV cache as the only cross-device handoff. The toolchain does
not carry over - PyTorch exported to OpenVINO IR as three separate models, no
ggml - but the shape of the answer does: the two devices suit different stages,
and π0.5 on the NPU alone is within 1.9x of the iGPU (438 ms against 234 ms) at a
fraction of the power. vla.cpp cannot make that split today because the core
drives one backend for a whole prediction. The one exception is a prototype:
SmolVLA can send its last camera view to an NPU worker process while SYCL runs
the rest ([sycl.md](sycl.md#prototype-one-smolvla-view-on-the-npu)). It would need a per-*stage* backend rather than a per-op
scheduler - the vision tower, the prefix and the action expert already hand off
through host memory, so the seam is in the right place - but that is an engine
change, not a backend one.
