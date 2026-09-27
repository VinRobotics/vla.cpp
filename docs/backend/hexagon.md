# `vla.cpp` on Snapdragon (Hexagon NPU)

What running `vla.cpp` on Qualcomm's Hexagon NPU takes, and what it turned out
to take once it ran.

> **Status: running on Windows on Arm, not yet on a Linux board.** vla.cpp builds
> with `GGML_HEXAGON=ON` and runs eleven checkpoints on a Snapdragon X laptop's
> v73 NPU, each checked against a CPU reference. The build, the measurements and
> the upstream bugs found along the way are in
> [hexagon-windows.md](hexagon-windows.md). What follows is the backend in
> general: the parts that hold on any Snapdragon part, and the Linux / IQ-9 notes
> from before anything ran, which are still unverified.
>
> The earlier version of this file was written without hardware and projected
> 0.8-2.0 s per SmolVLA action chunk on a v79 part. On a v73 part it measured
> **1.23 s**, inside the band. Its op-gap prediction was half right: `GELU_ERF`
> is still missing, and `RELU` has landed upstream. What it did not foresee was
> ops the NPU claims and then computes wrongly.

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

## Op coverage, measured

The previous version read `supports_op` at `b10729` and found two gaps, `RELU`
and `GELU_ERF`, and warned that in this engine an absent op is fatal. Running it
changed all three conclusions:

- **`RELU` landed upstream**, and the local checkout used here (build 11201) has
  it.
- **`GELU_ERF` is still missing.** It now runs on the CPU (next point) in every
  DINOv2 and SigLIP-so400m tower, and costs what that implies: Evo-1 spends more
  on copies than the NPU saves.
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
| fused `MUL_MAT+ADD`, `RMS_NORM+MUL` | ~10x the error of the unfused ops | `GGML_HEXAGON_OPFUSION=0` by default |

Each one flips the gripper channel, or comes close, on some checkpoint. The
earlier warning that a wrong op would be the worse failure than a missing one
was right: a missing op fails at load, and these only show in the actions.
[hexagon-windows.md](hexagon-windows.md#hexagon-issues-found-and-worked-around)
has the numbers and how each was bisected. They belong upstream.

## What changed in `vla.cpp`

The shape of the diff was predicted correctly, apart from the scheduler:

1. **Build flags.** `GGML_HEXAGON` (and `GGML_OPENCL`) joined the accelerator
   list and define `GGML_USE_HEXAGON` / `GGML_USE_OPENCL`.
2. **A rung in [src/backend.h](../../src/backend.h).**
   - It goes through the registry, because `ggml_backend_hexagon_init` is
     declared but not defined upstream.
   - `VLA_DEVICE` indexes HTP sessions, and `VLA_DEVICE=cpu` gives the reference.
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
   - Compare `vla_predict_check` output against the same binary with
     `VLA_DEVICE=cpu`, and against `--weight-dtype f32`.
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
