# `vla.cpp` on Snapdragon (Hexagon NPU)

Notes on what it would take to run `vla.cpp` on Qualcomm's Hexagon NPU, read out
of llama.cpp's own Hexagon history up to `b10729` - the tag this tree pins.

> **Status: read and planned, never run.** There is no Hexagon rung in
> `src/backend.h`, no `GGML_HEXAGON` in `CMakeLists.txt`, and no IQ-9 or IQ-10
> board on any desk here. Every claim below is either checked against llama.cpp
> at `b10729` (op tables, env vars, commit hashes - go verify them) or is
> explicitly a **projection**. Nothing in this file is a measurement on Qualcomm
> hardware, and the projected latencies are arithmetic, not results.
>
> Two things upstream did in 2026 make this worth writing down now: the NPU
> backend learned `IM2COL` for patch-embedding convolutions - which is exactly
> and only what a VLA's vision tower needs - and it learned multi-NPU. Both are
> already inside the tag we pin.

The previous version of this file was a verbatim copy of upstream's
[`docs/backend/snapdragon/linux.md`](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/linux.md).
That build recipe is still the right starting point and is kept below, but it
builds *llama.cpp*, not this engine; the gap between the two is the subject here.

## What "IQ9" and "IQ10" refer to

Two commits, both inside `b10729` (`458681e1`, 2026-09-01), and no Hexagon commit
lands after it as of this writing:

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

## What the backend gives you at `b10729`

- **Two libraries.** `libggml-hexagon.so` on the CPU side, `libggml-htp-vNN.so`
  on the NPU side. Skels are built for v68, v69, v73, v75, v79 and v81 and the
  right one is picked at runtime from `htpdrv_get_arch`; a failed query falls
  back to v73. `GGML_HEXAGON_ARCH` overrides.
- **Sessions are devices.** Each Hexagon process domain shows up to ggml as one
  device and behaves like a GPU for offload and model splitting.
  `GGML_HEXAGON_DEVICES` takes either a count or an explicit
  `HTP<physical>:<virtual>` list. `GGML_HEXAGON_NDEV` is the deprecated spelling.
- **~3.5 GB per session.** The backend now maps and unmaps execution buffers
  during graph execution to fit larger models into one session, and layer- or
  tensor-splitting across sessions is the alternative.
- **Repack buffers.** Q4_0, Q4_1, Q8_0, IQ4_NL and MXFP4 weights are repacked
  into non-host buffers; since #26501 non-host is the default and
  `GGML_HEXAGON_HOSTBUF=1` is the opt-out (needed to exercise `MUL_MAT` in
  `test-backend-ops`).
- **VTCM is the real budget.** `supports_op` precomputes kernel params for
  `MUL_MAT`, `FLASH_ATTN_EXT` and friends and returns false when the tile does
  not fit VTCM. An op is not rejected by shape rules so much as by whether it
  fits - which means coverage is a function of your tensor sizes, and has to be
  measured on the board, not predicted from a table.
- **Fusion**, controlled by `GGML_HEXAGON_OPFUSION`: `RMS_NORM+MUL`,
  `MUL_MAT+ADD`, N-way `MUL_MAT`, `ALLREDUCE+ADD`.

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

## Op coverage against what `vla.cpp` actually builds

This is the part that decides whether the port is a weekend or a month. The
column on the right is `ggml_backend_hexagon_device_supports_op` at `b10729`;
the middle column is a grep of `src/`.

| ggml op | Where we build it | Hexagon at `b10729` |
|---|---|---|
| `MUL_MAT` / `MUL_MAT_ID` | 154 sites, everywhere | yes - Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4 (repacked), or F16/F32 direct. **No BF16.** Refuses `src0->ne[1] > 32768` (lm-heads), and anything whose tile misses VTCM |
| `IM2COL` | 6 `ggml_conv_2d`, every SigLIP/DINOv2 patch embed | yes, **partial** - added by `355303ed` (#26007) "targeting only patch-embedding convolutions": 2D only, F32 source, contiguous, **zero padding**. Our conv is kernel = stride = patch, pad 0, dilation 1. On paper it is precisely the supported case |
| `FLASH_ATTN_EXT` | 8 sites | yes - F16 or Q8_0 K/V, F16 mask, F32 sinks |
| `ROPE` / `ROPE_MULTI` | 14 sites | yes, including MROPE/IMROPE (`17d22a35`) and vision RoPE (`f2d1c2f3`) |
| `NORM`, `RMS_NORM`, `L2_NORM`, `SOFT_MAX`, `SCALE`, `CONCAT`, `SQR`, `GET_ROWS`, `ADD/SUB/MUL/DIV` | all over | yes |
| `CPY` / `CONT` (`ggml_cast`, 14 sites) | dtype hops between stages | yes, but **F32↔F16 only, and never a reshape and a conversion in the same op** |
| `SILU`, `GELU` (tanh), `GELU_QUICK`, `TANH`, `SIGMOID`, `NEG`, `ABS`, `EXP`, `SOFTPLUS` | 22 `ggml_silu`, 10 `ggml_gelu` | yes |
| **`RELU`** | **18 sites** - `action_expert.cpp` state encoder and action decoder, plus Evo-1, VLA-Adapter, VLA-JEPA, GR00T N1.7, OpenVLA-OFT, BitVLA | **not implemented.** `grep -i relu` over `ggml-hexagon/` returns nothing |
| **`GELU_ERF`** | **13 sites** - `ffn_gelu_erf` in `src/layers/ffn.h` and `dual_tower.h`, i.e. every tower trained with the exact erf form (DINOv2, SigLIP-so400m) | **not implemented.** `ggml_gelu` (tanh) is a *different op*, and substituting it silently changes numerics |
| `ggml_map_custom1` | BitVLA only | no, and BitVLA pins itself to CPU by design anyway |

Two missing unary kernels, and one of them is in the vision tower of half the
model zoo. That is the whole gap. Note the shape of the mistake it invites: on
OpenVINO, "GELU is GELU" cost us a wrong arch and a day of bisection
([ov.md](ov.md)); here the op is simply absent, which is the *better* failure -
`supports_op` returns false and you find out at load time.

**Except that in this engine, absent is fatal.** `src/backend.h` brings up one
backend and drives it through `gallocr` - there is no scheduler and no per-op CPU
fallback, by design, so an unsupported op asserts at predict time instead of
quietly limping. Every other backend we support (CUDA, Metal, SYCL, OpenVINO)
covers our op set completely, so this constraint has never bitten. Hexagon would
be the first backend where it does.

So the port is gated on one of:

1. **Two HVX kernels upstream.** `e70802a0` (#27786) added `ABS` and `LOG` as HTP
   unary ops; `RELU` and `GELU_ERF` are the same shape of change, and `RELU` in
   particular is a `vmax` against zero. This is the honest path and it is small.
2. **A scheduler in `vla.cpp`.** `ggml_backend_sched` with a CPU fallback, which
   is an engine change with consequences for every backend, not a Hexagon one.
3. **Graph surgery** - swapping `gelu_erf` for `gelu` and re-checking fidelity.
   Cheap, wrong, and it will cost more than it saves. Do not.

## What would have to change in `vla.cpp`

Nothing here has been written; this is the shape of the diff.

**1. Build flags.** `GGML_HEXAGON` joins the mutually-exclusive accelerator list
at [CMakeLists.txt:15](../../CMakeLists.txt#L15) and defines `GGML_USE_HEXAGON`
the way the other four do.

**2. A rung in the ladder** at [src/backend.h:176](../../src/backend.h#L176).
The Hexagon init API takes no ordinal - `ggml_backend_hexagon_init(void)` - so
device selection is environment-driven, and `VLA_DEVICE` (an integer ordinal for
CUDA and SYCL) does not map onto it. The right binding is `VLA_DEVICE=HTP0:0`
passed through to `GGML_HEXAGON_DEVICES`, which means the rung needs its own
parse rather than `backend_device_index()`.

**3. Weights.** Our default resident dtype is BF16, which the NPU does not
accept in any op. Every Hexagon run needs a requantized checkpoint -
`python scripts/quantize_gguf.py --in model-bf16.gguf --out model-q8_0.gguf
--type Q8_0` (or `Q4_0`) - and the fidelity of *that* against the CPU reference
has to be established before any NPU number means anything. `--weight-dtype f32`
is the other option and doubles resident memory.

**4. The build that is actually the most work: `vla-server`'s dependencies.**
`find_package(Protobuf REQUIRED)` and `pkg_check_modules(libzmq REQUIRED)` are
unconditional in our `CMakeLists.txt`, and the Snapdragon toolchain image carries
the Hexagon SDK and the OpenCL SDK but not protobuf, libzmq or cppzmq for arm64.
Configure fails before a single object compiles - including for `vla-cli` and
`vla-bench`, which link neither. Options, cheapest first:

- Make the serving deps conditional so `vla-cli` / `vla-bench` configure without
  them. Worth doing regardless of Hexagon; it is a dozen lines.
- Add the three to the cross sysroot.
- Build natively on the board, which for an IQ-9-class Linux part is plausible
  and is how the first bring-up will probably actually happen.

**5. Presets.** We have no `CMakeUserPresets.json`; upstream's
`arm64-linux-snapdragon-release` is the base to copy, with `GGML_HEXAGON=ON` and
`-march=armv8.2a+fp16+dotprod`. Note it sets `GGML_OPENCL=OFF` - upstream CI adds
`-DGGML_OPENCL=ON` on the command line for the IoT job.

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
FastRPC loader finds the HTP skel:

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

## Bring-up checklist, for the day a board arrives

In order. Each step is cheap and each one has failed for somebody upstream.

1. **Confirm the arch and session lines.** `ggml-hex: Hexagon Arch version v79`
   and `allocating new session: HTP0:0`. A v73 line on a part you believe is
   newer means `htpdrv_get_arch` failed and you are on the fallback skel.
2. **`test-backend-ops` before anything of ours.**
   `run.py --hex-hostbuf 0 --devices HTP0:0 -- test-backend-ops -b HTP0:0 -o MUL_MAT`,
   then `-o IM2COL`, `-o FLASH_ATTN_EXT`, `-o ROPE`. Filter to the dtypes we
   would actually ship (`q8_0`, `q4_0`, `f16`).
3. **`llama-bench` on a 1B Q4_0** to get a throughput anchor on *this* board that
   is comparable with upstream's published numbers, before any VLA is involved.
4. **Load a requantized VLA with `GGML_HEXAGON_VERBOSE=1`** and read which ops the
   NPU accepted. This is where the two missing unaries will announce themselves.
5. **Fidelity before latency.** max|delta| against a CPU-backend reference on the
   same checkpoint, the same 2.9e-3 bar SYCL and OpenVINO are held to
   ([ov-progress.md](ov-progress.md)). A partially-failing run still prints a
   wall-clock time; do not read a latency off a run whose actions did not come out.
6. **Then profile.** `GGML_HEXAGON_PROFILE=1 ... |& scripts/snapdragon/ggml-hexagon-profile.py -`,
   and `GGML_HEXAGON_OPFILTER` to bisect anything that looks wrong.

## Projections

Everything in this section is arithmetic over numbers measured elsewhere. It is
here to set expectations and to be *falsified* by the first real run, not to be
quoted.

Measured reference points, all ours, all `server_total_ms` per action chunk
(`ci/baselines/`, [sycl.md](sycl.md)):

| Device | SmolVLA | π0 | GR00T N1.5 | Evo-1 |
|---|---:|---:|---:|---:|
| Ryzen 5 5500, CPU backend, 8 threads | 1,920 | - | - | 7,695 |
| Jetson Orin Nano | 510 | 1,485 | 1,183 | 3,552 |
| Apple M4 (Metal) | 324 | 1,129 | - | - |
| Arc A380 (SYCL) | 630 | - | - | 1,176 |
| RTX 3090 | 99 | 262 | 204 | 487 |

The only Hexagon anchor that exists is upstream's own, on a v79 phone part:
Llama-3.2-1B Q4_0, **pp128 = 169 t/s**, **tg64 = 51.5 t/s** (single `HTP0`
session). Read as work rather than tokens: prefill at 169 t/s over 1.24 B params
is ≈ 0.42 TFLOP/s effective; decode at 51.5 t/s over 730 MiB of weights is
≈ 37 GB/s of weight traffic, which is a believable LPDDR5 read rate and a good
sign the published number is not cherry-picked.

A VLA action chunk is **all prefill and no decode**: fixed shapes, a tiny KV
cache, no sampling loop. That is the regime the 0.42 TFLOP/s figure comes from,
which makes the extrapolation less dishonest than usual. SmolVLA is 1.07 GiB of
BF16 weights (≈ 0.54 B params), spent on three stages per prediction - a vision
tower over 2-3 views, a ~113-token prefix through the VLM, and 10 denoise steps
over a ~50-token suffix through the much smaller action expert. Multiplying that
out lands in the region of 300-400 GFLOP per chunk, so at the anchor rate:

**SmolVLA, single v79-class HTP, Q4_0: 0.8-2.0 s per action chunk.** The band is
wide on purpose. The bottom needs the vision tower to land on the NPU via the new
`IM2COL` and the denoise loop to stay fed; the top is what you get if the loop is
dispatch-bound and the towers fall back. Either way that is somewhere between the
Orin Nano (510 ms) and a desktop CPU (1,920 ms) - useful for a 5-10 Hz policy on
a part that draws a few watts, not competitive with a discrete GPU. Expect the
first honest number to be nearer the top of the band and to improve with repack,
fusion and `OPPOLL`, exactly as upstream's own numbers did over the 107 commits
that have touched `ggml/src/ggml-hexagon/` since the backend landed in October
2025 (`63d2fc46`, "experimental").

Where the interesting risk is:

- **The vision tower is the prize.** It is 58% of SmolVLA's CPU-backend time
  (1,119 ms of 1,920 ms) and 7.1x faster on an A380. `IM2COL` landing for
  patch-embed convolutions specifically is the single most encouraging thing in
  this entire history, and it landed in July 2026 for reasons that had nothing to
  do with us.
- **The denoise loop is the risk.** Ten sequential steps of small GEMMs is the
  worst case for any offload engine - it is why SmolVLA gains least on SYCL
  (3.0x when its own vision tower gains 7.1x). The fully-async backend, op
  batching and graph reuse from #26501 are aimed at exactly this, which is
  encouraging, but "many small ops over FastRPC" is a dispatch-latency problem
  and NPUs do not usually win those.
- **The lm-head refusal does not touch us.** Hexagon bails on `src0->ne[1] >
  32768`; VLA action heads are small and no in-tree arch decodes a vocabulary on
  the action path.

### IQ-10, imagined

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
direction): the core drives one backend for a whole prediction. It would need a
per-*stage* backend, not a per-op scheduler. The seam is already in the right
place. That is the one engine change this whole document argues for, and it pays
off on more than Hexagon.

## What would falsify all of this

- `RELU` and `GELU_ERF` turn out to be a 200-line HVX patch each → the port is a
  weekend once the board exists, and everything above about schedulers is moot.
- `IM2COL`'s "partial" support turns out not to cover our patch shapes after all
  (multi-view batching, a non-contiguous pixel buffer) → the vision tower stays
  on CPU, the projection's lower bound disappears, and the whole exercise gets
  much less interesting.
- Q8_0 requantization moves any arch past 2.9e-3 → the dtype story needs work
  before the backend story does.
- VTCM budgets reject our GEMM shapes (which are wider and shorter than an LLM's)
  → `supports_op` starts returning false for reasons no table here predicts, and
  the answer is measurement, not more reading.
