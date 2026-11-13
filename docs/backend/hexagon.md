# `vla.cpp` on Snapdragon (Hexagon NPU)

What running `vla.cpp` on Qualcomm's Hexagon NPU takes, and what it turned out
to take once it ran.

> **Status: running on Windows on Arm and on a Dragonwing IQ-10 under Linux.**
> vla.cpp builds with `GGML_HEXAGON=ON` and runs eleven checkpoints on a
> Snapdragon X laptop's v73 NPU and thirteen on an IQ-10's v81 NPU under Linux,
> each checked against a CPU reference. The laptop build, its measurements and
> the upstream bugs found along the way are in
> [hexagon-windows.md](hexagon-windows.md); the Linux board is in [Linux
> (IQ-10)](#linux-iq-10). What follows is the backend in general: the parts that
> hold on any Snapdragon part, and the IQ-9 and multi-NPU notes from before
> anything ran, which are still unverified.
>
> The earlier version of this file was written without hardware and projected
> 0.8-2.0 s per SmolVLA action chunk on a v79 part. On a v73 part it measured
> **1.23 s**, inside the band. Its op-gap prediction was half right: `GELU_ERF`
> is still missing, and `RELU` has landed upstream. What it did not foresee was
> ops the NPU claims and then computes wrongly.

## What "IQ9" and "IQ10" refer to

Two commits, both inside the pinned `b11223` (`4da63377`, 2026-09-27):

| Commit | Date | What it actually did |
|---|---|---|
| `d81e63dc` `CI : support IOT device (IQ9) (#22987)` | 2026-05-14 | Added a third device to the Snapdragon CI matrix: `QCS9075M`, alongside the `SM8750` / `SM8850` phone parts. It is the only entry that is **Linux**, not Android - a separate build job (`linux-iot-snapdragon`, with `-DGGML_OPENCL=ON` on top of the preset), a BASH test framework instead of Appium/pytest, and `--retries 2 --retry-delay 300` because the IoT device in Qualcomm's Device Cloud is scarce. |
| `192067b7` `hexagon: support for multi-NPU devices (IQ9, IQ10) and fully asynchronous backend (#26501)` | 2026-08-26 | 5,314 insertions over 44 files, by Qualcomm. Physical vs virtual NPU sessions, an `ALLREDUCE` HTP kernel and sync tokens so a tensor split can span NPUs, a fully async backend with events, `GET_ROWS`/`SET_ROWS`, Q8_0 flash attention, op fusion, and the `build.py` / `run.py` scripts that replaced the old `adb/*.sh` and `windows/*.ps1` wrappers. |

What the names do **not** tell you: neither `IQ9` nor `IQ10` appears anywhere in
the source, the docs, or the CI config as an identifier. The only board named in
code is `QCS9075M`, which places IQ-9 as the Dragonwing-class industrial/robotics
part rather than a phone SoC. IQ-10 exists in that commit title and nowhere else.
So "IQ9/IQ10" is upstream shorthand; what is real is the *capability* the commit
added, which is what the rest of this document reads against.

The generational split visible in the code, rather than in the marketing:

- **One physical NPU, N virtual sessions** (`HTP0:0`, `HTP0:1`, ...) - the IQ-9
  shape, and what every phone part does today.
- **N physical NPUs** (`HTP0:0`, `HTP1:0`) - discovered via `FASTRPC_GET_DOMAINS`,
  with a static fallback that hardcodes physical 0 → CDSP domain 3 (`cdsp`) and
  physical 1 → domain 4 (`cdsp1`). Ask for a third physical core without dynamic
  discovery and the backend refuses by name: *"physical CDSP core %d not
  supported without dynamic discovery"*. Two is what the fallback knows about;
  the discovery path is open-ended.

## What the backend gives you at `b11223`

- **Two libraries.** `libggml-hexagon.so` on the CPU side, `libggml-htp-vNN.so`
  on the NPU side. Skels are built for v73, v75, v79 and v81 and the right one
  is picked at runtime from `htpdrv_get_arch`; a failed query falls back to v73
  and older parts are capped to it. `GGML_HEXAGON_ARCH` overrides.
- **Sessions are devices.** Each Hexagon process domain shows up to ggml as one
  device and behaves like a GPU for offload and model splitting.
  `GGML_HEXAGON_DEVICES` takes either a count or an explicit
  `HTP<physical>:<virtual>` list. `GGML_HEXAGON_NDEV` is the deprecated spelling.
- **~3.5 GB per session.** The backend now maps and unmaps execution buffers
  during graph execution to fit larger models into one session, and layer- or
  tensor-splitting across sessions is the alternative.
- **Repack buffers.** Q4_0, Q4_1, Q8_0, IQ4_NL, MXFP4, Q4_K, Q5_K and Q6_K
  weights are repacked into non-host buffers; since #26501 non-host is the
  default and `GGML_HEXAGON_HOSTBUF=1` is the opt-out (needed to exercise
  `MUL_MAT` in `test-backend-ops`).
- **VTCM is the real budget.** `supports_op` precomputes kernel params for
  `MUL_MAT`, `FLASH_ATTN_EXT` and friends and returns false when the tile does
  not fit VTCM. An op is not rejected by shape rules so much as by whether it
  fits - which means coverage is a function of your tensor sizes, and has to be
  measured on the board, not predicted from a table.
- **Fusion**, controlled by `GGML_HEXAGON_OPFUSION`: `RMS_NORM+MUL`,
  `MUL_MAT+ADD`, N-way `MUL_MAT` and `MUL_MAT_ID`, `ALLREDUCE+ADD`,
  `GATED_DELTA_NET+CPY`.

The knobs worth knowing on day one:

| Variable | Use |
|---|---|
| `GGML_HEXAGON_DEVICES=HTP0:0,HTP1:0` | which NPUs/sessions to open |
| `GGML_HEXAGON_VERBOSE=1` | log every op the NPU accepted, with dtypes and buffers |
| `GGML_HEXAGON_OPFILTER=<regex>` | force matching ops off the NPU - the bisection tool |
| `GGML_HEXAGON_PROFILE=1\|2` | per-op usecs/cycles (+PMU), pipe into `scripts/snapdragon/ggml-hexagon-profile.py -` |
| `GGML_HEXAGON_OPTRACE` | fine-grained HVX/HMX/DMA event trace |
| `GGML_HEXAGON_OPPOLL=1` | poll for op completion instead of sleeping - matters for many small graphs |
| `GGML_HEXAGON_HOSTBUF=1` | disable repack buffers (op testing) |

## Op coverage, measured

The previous version read `supports_op` at `b10729` and found two gaps, `RELU`
and `GELU_ERF`, and warned that in this engine an absent op is fatal. Running it
changed all three conclusions:

- **`RELU` landed upstream**, and the local checkout used here (build 11201) has
  it.
- **`GELU_ERF` is still missing.** It now runs on the CPU (next point) in every
  DINOv2 and SigLIP-so400m tower, and costs what that implies: Evo-1 spends more
  on copies than the NPU saves. An HTP kernel was written and checked on the
  IQ-10 (b11223): more accurate than the CPU's (relative error 1.5e-6 against
  5.1e-5), and Evo-1 7% faster. It was left out: GR00T N1.7 sits close to the
  bar, and the kernel moved its fixed-input chunk from 2.0e-3 to 3.0e-3.
  VLA-Adapter's and OpenVLA-OFT's DINOv2 MLP and projector build it from ops the
  NPU runs instead (`vla::gelu_erf()`: x*sigmoid(p(x)), p an odd cubic in x
  clamped to [-5, 5], within 2.5e-5 of the exact GELU): VLA-Adapter 740 to
  705 ms, OpenVLA-OFT 1771 to 1740 ms.
- **Absent is no longer fatal.** `src/backend_fallback.cpp` wraps the NPU in a
  backend that runs whatever it rejects on the CPU, without touching an arch.
  That also covers ops refused for shape rather than type: softmax rows that are
  not a multiple of 32, `GROUP_NORM`, `SIN` and `COS`.

The larger finding was not a missing op but **wrong ones**. `supports_op` claims
these, and they compute something else:

| Op | What goes wrong | Handled by |
|---|---|---|
| `GELU` | Runs `x*sigmoid(1.702x)` (GELU_QUICK) | `vla::gelu()` builds the tanh form from correct ops |
| `ADD`/`MUL`/... | `src1` broadcast over one dim but not a higher one | Refused, runs on the CPU |
| `IM2COL` | Wrong unless kernel = stride > 1 and no padding | Refused, runs on the CPU |
| `CPY` F32→F16 | Reads back wrong on the host | Copies feeding a CPU op run on the CPU |
| `REPEAT` | Reads stale cache lines of a source written by DMA earlier in the same op batch | Refused unless the source is a weight, runs on the CPU |
| fused `MUL_MAT+ADD`, `RMS_NORM+MUL` | ~10x the error of the unfused ops | `GGML_HEXAGON_OPFUSION=0` by default |

Each one flips the gripper channel, or comes close, on some checkpoint. The
earlier warning that a wrong op would be the worse failure than a missing one
was right: a missing op fails at load, and these only show in the actions.
[hexagon-windows.md](hexagon-windows.md#hexagon-issues-found-and-worked-around)
has the numbers and how each was bisected. They belong upstream.

Two more ops are claimed and then refused at run time. On buffers mapped above
4 GB of DSP address space, which the IQ-10 under Linux hands out, the NPU runs
`CONCAT` only as the dim-1 contiguous DMA copy and answers any other concat with
`NO_SUPPORT`, and the host aborts on the first predict. The wrapper sends those
concats to the CPU. A `CPY` or a `PAD` that reads a weight gets the same answer
(VLA-Adapter casts its F16 action queries to F32 in the graph, ACT zero-pads its
stem kernel), and also runs on the CPU.

The experts' KV concats (π0, π0.5, SmolVLA) were such concats: along the token
dim of contiguous [head dim, heads, tokens] tensors, once per layer per denoise
step. Appending along that dim appends whole rows, so on Hexagon
`concat_dim2()` in `src/backend.h` does them as a dim-1 concat of the
[head dim x heads, tokens] views, which writes the same bytes and stays on the
NPU: π0.5 1379 to 1133 ms, SmolVLA 810 to 638 ms, outputs unchanged. π0 takes
the same path.

Flash attention on the HMX (v81) keeps scores and probabilities in F16. That is
a precision limit rather than a wrong op, but in GR00T N1.7's LM it moves the
action chunk 6.2e-3 from the CPU reference. N1.7's LM therefore takes the
explicit path unless `--flash-attn` is given: 2.0e-3, at the same latency.
VLA-JEPA's LM does the same: with flash attention a served LIBERO observation
lands 6.2e-3 from the reference, without it 2.2e-3, at a cost of 683 to 810 ms
with two views.

The softmax takes rows of up to 32 elements or a multiple of 32. Where a
sequence can grow without changing its result, vla.cpp pads it to
`gemm_width()` on Hexagon rather than send every layer's softmax to the CPU:
VLA-Adapter's causal LM and Octo's block transformer get zero tokens appended,
masked out as keys, and each VLA-Adapter action-head block stacks its three
key/value sets as rows, padded and masked, instead of concatenating scores
along their columns. VLA-Adapter goes from 859 to 747 ms, Octo-Small from 126.5
to 122.6 ms. OpenVLA-OFT's LM gains nothing that way: its 22 padding tokens add
as much 7B matmul as the softmax saves.

`MUL_MAT` with a 2D weight against a batch of activations (weight `ne[2..3]`
of 1, activations above 1) is claimed and comes out wrong. GR00T N1.6's SigLIP2
tower takes two camera views that way: its action chunk lands 0.16 from the CPU
reference. The wrapper runs those matmuls on the CPU. On Hexagon `linear()` in
`src/layers/linear.h` multiplies a contiguous batch as one 2D matrix of rows
instead, the same product, so most of them never reach the rule: GR00T N1.6
with two views lands 7.5e-4 from the reference, and the matmuls of TurboVLA's
DINOv3, which takes its two views as a batch, run on the NPU.

A copy that transposes the innermost dim, such as the `CONT` of a permuted V
for explicit attention, is a strided gather on the NPU: 48 ms for OpenVLA-OFT's
[330,128,32], 32 ms for Evo-1's [1025,112,8], 18 ms for TurboVLA's
[512,256,4]. The wrapper runs those of 256K elements or more on the CPU, which
writes the same bytes; the smaller ones take tens of microseconds on the NPU,
less than the trip to the CPU.

The matrix unit (HMX) takes a matmul only when its inner dim K and its output
dim N are multiples of 32, and with a batch only an F16 src0. Anything else runs
on the vector units, at about 1/20 of the speed. `GGML_HEXAGON_PROFILE=1` on the
IQ-10 showed such matmuls dominating, and vla.cpp now avoids them in
Hexagon builds:

- **SigLIP-so400m's MLP is 4304 wide**: 440 ms of GR00T N1.5's 565 ms tower.
  The loader zero-pads fc1, its bias and fc2 to 4320
  (`WeightLoader::pad_mlp`, `gemm_width()` in `src/backend.h`), which leaves
  the output unchanged. VLA-Adapter's and OpenVLA-OFT's SigLIP
  (`src/modules/dual_tower.h`) take the same padding.
- **π0.5's attention ran explicit scores** in SigLIP (head dim 72, F32 x F32),
  the VLM and the action expert, because its flash attention was tied to SYCL's
  view-taking kernel. The VLM's and the expert's single KV head also sent their
  matmuls to the CPU (the batched `MUL_MAT` rule above). All three now take the
  plain flash attention on Hexagon (`use_flash_attention()` in `pi05.cpp`).
  VLA-Adapter's and OpenVLA-OFT's DINOv2 and SigLIP towers did the same and now
  take flash attention on Hexagon too: their vision stage goes from 790 to
  311 ms.
- **Evo-1's LM attention multiplied F32 K and V against 14 query heads.** K and
  V now go to F16 (`fa_kv()`); the scores still accumulate in F32.
- **Explicit attention elsewhere multiplied F32 K and V too**, and its key
  counts and head dims are often not multiples of 32. OpenVLA-OFT's LM (330
  keys), Octo's block transformer, Evo-1's DiT cross-attention (head dim 112,
  1025 keys) and GR00T's DiT (head dim 48) now take F16 K/V with the head dim
  and the keys zero-padded to `gemm_width()` and the padded keys masked out
  (`hmx_kv()` in `dit_head.cpp`): OpenVLA-OFT 1725 to 1120 ms, Evo-1 2348 to
  1661 ms, Octo-Small 119 to 85 ms, GR00T N1.5 595 to 532 ms. Two LMs keep part
  of it in F32: in VLA-Adapter's LM only V goes to F16, since F16 scores put the
  chunk 3.9e-3 off, and VLA-JEPA's cross-attention K/V overflow F16 (NaN
  actions), so only its DiT self-attention takes the HMX.
- **ACT's ResNet convs are im2col and one matmul** with the patches as its
  first operand. The stem's K of 147 and layer3's and layer4's 1200 and 300
  output positions (640x480) are not multiples of 32: 80 ms of the 101 ms spent
  on the NPU. On Hexagon `conv_2d_padded()` in `act.cpp` zero-pads the patch
  matrix and the stem's kernel to `gemm_width()` and drops the extra rows:
  228.5 to 155.6 ms (mean, 5 reps). Its im2col (padded and overlapping), bias
  adds (a gapped broadcast) and max pool still run on the CPU; under the
  benchmark's protocol ACT takes 152.9 ms on the NPU against 170.8 ms on the
  CPU.

What is left on the vector units: VLA-Adapter's QK scores and VLA-JEPA's LM and
DiT cross-attention scores, which stay F32 for the reasons above, and Evo-1's
InternViT attention, which stays on the CPU because F16 K/V there measured
3.9e-2.

## What changed in `vla.cpp`

The shape of the diff was predicted correctly, apart from the scheduler:

1. **Build flags.** `GGML_HEXAGON` (and `GGML_OPENCL`) joined the accelerator
   list and define `GGML_USE_HEXAGON` / `GGML_USE_OPENCL`.
2. **A rung in [src/backend.h](../../src/backend.h).**
   - It goes through the registry, because `ggml_backend_hexagon_init` is
     declared but not defined upstream.
   - `VLA_DEVICE` indexes HTP sessions, and `VLA_DEVICE=cpu` skips them. That
     run is not a fidelity reference (see [Fidelity](#fidelity)).
   - It sets the defaults above and points `ADSP_LIBRARY_PATH` at the
     executable's folder.
3. **Weights.**
   - The NPU takes no BF16. Both accelerator builds default to F16 resident
     weights, which hold BF16 exactly.
   - Q8_0 works, but moves actions by 1e-2 or more on most checkpoints, and does
     so on the CPU too.
   - SmolVLA's loader learned to keep packed weights.
4. **Serving dependencies** are optional (`VLA_BUILD_SERVER`), and on Windows
   come from vcpkg.
5. **No scheduler.**
   - The fallback wraps one backend instead, so the ~84 gallocr/compute sites
     across 13 archs stay as they were.
   - CUDA, Metal, SYCL and OpenVINO do not compile it.
   - K/V for flash attention are cast to F16 on Hexagon only (`vla::fa_kv`),
     because the NPU kernel takes nothing else.

## Bring-up checklist, for a new part

In order. Each step is cheap, and steps 4 and 5 are where every bug above was
found.

1. **Confirm the arch and session lines.**
   - Expect `ggml-hex: Hexagon Arch version v73` (or v79/v81) and
     `allocating new session`.
   - A v73 line on a part you believe is newer means `htpdrv_get_arch` failed,
     and you are on the fallback skel.
2. **`llama-bench` on a small model**, to prove the driver, signing and
   `ADSP_LIBRARY_PATH` before any VLA is involved.
3. **Run a VLA and read the fallback lines.** Each process prints which ops went
   to the CPU and how much was copied (`VLA_FALLBACK_STATS=1` for per-graph
   detail).
4. **Fidelity before latency.**
   - Compare `vla_predict_check` output against a CPU-only build
     (`-DGGML_HEXAGON=OFF`) of the same commit, run with `--weight-dtype f32
     --flash-attn 0`. The Hexagon binary with `VLA_DEVICE=cpu` is not a
     reference: its NPU changes (flash attention, F16 K/V, padding to
     `gemm_width()`) run on the CPU too.
   - The bar is 2.9e-3.
   - A run with a wrong kernel still prints a latency, and it is usually a good
     one.
5. **Bisect anything off the bar** with `GGML_HEXAGON_OPFILTER=<regex>`, which
   moves matching ops to the CPU fallback. Families first, then single ops.
   `GGML_HEXAGON_OPFILTER=.*` must reproduce the CPU run exactly; if it does
   not, the fallback is at fault, not the NPU.
6. **Then profile.** Use `GGML_HEXAGON_PROFILE=1 ... |& scripts/snapdragon/ggml-hexagon-profile.py -`.

## Building llama.cpp for the board

The upstream path, unchanged, and the thing to get working first - if
`llama-bench` will not run on the device, nothing downstream matters. Requires
only Docker on the host; `build.py` pulls
`ghcr.io/snapdragon-toolchain/arm64-linux:v0.7` itself.

```bash
# cross-compile and deploy over SSH in one step
./scripts/snapdragon/build.py --target lnx:user@host --push

# or by hand, inside the container
docker run -it --rm -u $(id -u):$(id -g) --volume $(pwd):/workspace \
    --platform linux/amd64 ghcr.io/snapdragon-toolchain/arm64-linux:v0.7
[d]/workspace> cp docs/backend/snapdragon/CMakeUserPresets.json .
[d]/workspace> cmake --preset arm64-linux-snapdragon-release -B build-snapdragon
[d]/workspace> cmake --build build-snapdragon -j $(nproc)
[d]/workspace> cmake --install build-snapdragon --prefix pkg-linux
```

On the device, both library paths must be set - `ADSP_LIBRARY_PATH` is how the
FastRPC loader finds the HTP skel (not on every board: the IQ-10 ignores it, see
[Linux (IQ-10)](#linux-iq-10)):

```bash
export LD_LIBRARY_PATH=./lib
export ADSP_LIBRARY_PATH=./lib
./bin/llama-cli -m Llama-3.2-3B-Instruct-Q4_0.gguf --device HTP0 -ngl 99 -p "..."
```

`run.py` maps flags to those environment variables and can drive the device over
SSH or ADB from the host:

```bash
./scripts/snapdragon/run.py --target lnx:user@host --devices HTP0 -- \
    llama-bench -m Llama-3.2-1B-Instruct-Q4_0.gguf -p 128 -n 64

# multi-NPU tensor split - the IQ-10 shape
./scripts/snapdragon/run.py --target lnx:user@host --devices HTP0:0,HTP1:0 -- \
    llama-completion -m gemma-2b-it-Q4_0.gguf -f prompt.txt --split-mode tensor
```

## Linux (IQ-10)

Run on a Qualcomm Dragonwing IQ-10 board (IQX EVK, QCS8797) under Linux
(Ubuntu 24.04): Hexagon v81, four CDSP domains, llama.cpp `b11223`.

### Building

The Hexagon SDK has no arm64 Linux build, so vla.cpp is cross-built on an
x86-64 host, in the same toolchain container as llama.cpp above:

```bash
scripts/build_linux_hexagon.sh                 # with vla-server
scripts/build_linux_hexagon.sh --no-server     # without it
```

The package lands in `build-lnx-htp/pkg` (`bin/`, `lib/`). For the server the
script installs Debian's arm64 protobuf and ZeroMQ dev packages in the
container; the board needs the matching runtime libraries
(`libprotobuf32t64`, `libzmq5`). See `scripts/build_linux_hexagon.sh --help`
for the options.

### Installing the skel

On this board FastRPC ignores `ADSP_LIBRARY_PATH`. It loads
`libggml-htp-v81.so` from `/vendor/dsp/cdspN/`, one copy per CDSP domain, and
falls back to `/usr/share/fastrpc/`. It also loads only skels processed by the
board vendor's tooling: an unprocessed one fails with `failed to open session
... error 0x80000406`. Set `HTP_SKEL_POST` to that tool when building, then
install the skel on the board as root:

```bash
for d in /vendor/dsp/cdsp*; do cp pkg/lib/libggml-htp-v81.so "$d"/; done
```

Reinstall it whenever the build changes. A skel from another llama.cpp build,
such as one left in `/usr/share/fastrpc/`, can open a session and then hang on
the first graph.

### Running

```bash
export LD_LIBRARY_PATH=$PWD/pkg/lib
./pkg/bin/vla-server --bind 'tcp://*:5555' smolvla-libero.gguf
```

### Fidelity

`vla_predict_check` at the Hexagon defaults, against a CPU-only build
(`-DGGML_HEXAGON=OFF`) of the same commit with `--weight-dtype f32
--flash-attn 0`. The Hexagon build's own CPU run is not a reference: what it
changes for the NPU (F16 K/V, padding to `gemm_width()`) is compiled in and runs
on the CPU too. Views are the ones the LIBERO client sends, so they
differ from the benchmark's for VLA-JEPA, VLA-Adapter, OpenVLA-OFT and Octo:

| Model | Views | max\|Δ\| |
|---|--:|--:|
| SmolVLA | 2 | 7.4e-4 |
| π0 | 2 | 7.7e-4 |
| π0.5 | 2 | 4.8e-4 |
| Evo-1 | 1 | 6.2e-4 |
| GR00T N1.5 | 1 | 1.9e-3 |
| GR00T N1.6 | 1 | 1.3e-3 |
| GR00T N1.7 | 1 | 2.0e-3 |
| VLA-JEPA | 2 | 2.1e-3 |
| VLA-Adapter | 2 | 3.9e-4 |
| OpenVLA-OFT | 2 | 3.1e-4 |
| Octo-Small | 1 | 5.0e-4 |
| TurboVLA | 2 | 3.7e-4 |
| ACT | 1 | 6.1e-4 |

GR00T N1.6 with two views is 7.5e-4. At the benchmark's single view VLA-JEPA is
1.8e-3, VLA-Adapter 4.8e-4 and OpenVLA-OFT 1.8e-4. ACT takes its one 640x480
view, the benchmark's input.

### Serving

`vla-server` on the board, driven from the host by
`eval/client/vla_cpp_client.py` and the adapters `run_sim_client_direct.py`
uses, on one fixed LIBERO observation (two views, the task prompt, a fixed robot
state), with fixed noise for the archs that draw it. The same request goes to a
Hexagon server and to a server from the CPU-only build with `--weight-dtype f32
--flash-attn 0`.
Every model returns a finite chunk, the same chunk for a repeated request, an
error reply to a malformed or an invalid request, and keeps serving afterwards.
ACT has no `eval/client` preset; it gets the same checks from raw `vla.proto`
requests carrying one fixed 640x480 view and a fixed 14-dim state.

| Model | max\|Δ\| vs CPU F32 | CPU F16 vs CPU F32 |
|---|--:|--:|
| SmolVLA | 1.9e-3 | |
| π0.5 | 3.7e-4 | |
| Evo-1 | 1.1e-3 | |
| GR00T N1.5 | 1.2e-2 | 2.8e-2 |
| GR00T N1.6 | 1.0e-2 | 9.0e-3 |
| GR00T N1.7 | 2.0e-3 | |
| π0 | 5.8e-4 | |
| VLA-JEPA | 2.2e-3 | |
| VLA-Adapter | 3.9e-4 | |
| OpenVLA-OFT | 7.5e-4 | |
| Octo-Small | 1.4e-4 | |
| TurboVLA | 8.5e-4 | |
| ACT | 1.0e-3 | |

GR00T N1.5 and N1.6 are above the bar on this input, and so is the CPU itself
with F16 weights: N1.5 lands closer to the reference than the CPU at F16, N1.6
slightly further (1.0e-2 against 9.0e-3). They are sensitive to 16-bit precision
rather than miscomputed. Running them with F32 weights on the NPU has not been
tried yet.

### Q8_0 weights

`scripts/quantize_gguf.py --type Q8_0` packs a checkpoint's weights to Q8_0.
Three flags shape a file for the NPU:

- `--pad-m32` zero-pads in the file the FFN widths that are not a multiple of
  32. The HMX takes no other width for a packed weight, and the loader pads an
  FFN at load only when its weights are float. The zeros leave the product
  unchanged, but `fc2`'s rows then split into blocks of 32, so it is packed too.
- `--dit` also packs the DiT's weight matrices, except adaLN and the time
  embedding.
- `--keep-float REGEX` leaves the weights whose name matches REGEX float.

Q8_0 moves most models' actions past the 2.9e-3 bar. Within it on both fixed
and served inputs are Octo-Small and ACT, π0.5 with or without `--vision`, and
π0, OpenVLA-OFT and VLA-Adapter only with some weights left float. π0.5 with
`--vision --pad-m32` is 15% faster than F16 and takes 38% less system RAM:

```bash
python scripts/quantize_gguf.py --in pi05-libero.gguf --out pi05-libero-q8_0.gguf \
    --type Q8_0 --vision --pad-m32
```

Latency, memory and fidelity per configuration:
[benchmark/qualcomm-iq10.md](../benchmark/qualcomm-iq10.md#q8_0-weights).

## IQ-10, imagined

Nothing about a second-generation part is knowable from this repository, so what
follows is a bet, not a forecast. The code in #26501 tells you what Qualcomm
built *for*: two or more physical NPUs, an `ALLREDUCE` kernel with a DMA solver
and fused `ALLREDUCE+ADD`, sync tokens to order work across devices, and a static
domain table that stops at two.

If a dual-NPU IQ-10 lands, the obvious use - `--split-mode tensor` across
`HTP0:0,HTP1:0`, paying an allreduce per layer - is the *worse* fit for a VLA.
An action chunk has three stages that already hand off through host memory: the
vision tower, the VLM prefix, and the action expert. Putting the tower on one NPU
and the expert on the other pipelines across timesteps with **one** handoff per
stage instead of one collective per layer, and a policy server at 10 Hz has a
steady stream of chunks to pipeline.

`vla.cpp` cannot do that today, for the same reason it cannot split across an
Intel iGPU and NPU ([ov.md](ov.md) reaches this conclusion from the other
direction): the core drives one backend for a whole prediction. (The CPU fallback
splits a graph by op, not by stage, so it does not change this.) It would need a
per-*stage* backend, not a per-op scheduler. The seam is already in the right
place. That is the one engine change this whole document argues for, and it pays
off on more than Hexagon.

## What the measurements settled

The previous version listed four ways it could be wrong. Here is how each came out:

- **"`RELU` and `GELU_ERF` are a weekend each."** `RELU` came from upstream;
  `GELU_ERF` has not. The CPU fallback made both moot for correctness.
- **"`IM2COL`'s partial support might not cover our patch shapes."** It covers
  SigLIP's 16x16/16 patch embed, and SmolVLA's vision tower runs 4x faster than
  on the CPU. It gets every other geometry wrong while claiming it.
- **"Q8_0 might move an arch past 2.9e-3."** It does, for most archs and on the
  CPU as much as the NPU, so F16 is the default.
- **"VTCM might reject our GEMM shapes."** Nothing was rejected for VTCM. The
  NPU loses where the fallback copies dominate (GELU_ERF towers, 1025-wide
  softmaxes), and on SigLIP-so400m, whose tower is slower on the NPU than on the
  CPU with nothing falling back.
