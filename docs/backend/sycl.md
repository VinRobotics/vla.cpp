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

## SYCL defaults: fused graphs and flash attention

On SYCL, several archs swap their per-op ggml graph for fused nodes
(`src/layers/fused_ew.h`, `src/modules/fused_lm.h`, each node with a CPU
reference and a SYCL kernel), and flash attention runs on oneDNN's fused SDPA. A
SYCL build turns these on without any flag:

| Arch | On by default | Needs | Off switch |
|---|---|---|---|
| SmolVLA | flash attention; SigLIP as fused blocks, views batched; LM and expert layers as fused nodes, K/V in place | BF16 weights, flash attention | `VLA_SMOLVLA_NO_VIT_FUSE=1` (tower), `VLA_SMOLVLA_NO_FUSE=1` (LM and expert) |
| VLA-Adapter | DINOv2 + SigLIP towers and Qwen2 LM layers as fused nodes | BF16 weights, flash attention | `VLA_ADAPTER_NO_FUSE=1` or `--flash-attn 0` |
| GR00T N1.5, N1.6 | DiT blocks and SigLIP tower as fused nodes | BF16 weights, flash attention | `VLA_GR00T_NO_FUSE=1` |
| GR00T N1.7 | per-op graph; `VLA_GR00T_N17_FUSE=1` fuses the DiT and the Qwen3-VL tower | BF16 weights, flash attention | - |
| Evo-1 | flash attention; with `--act-dtype bf16`, InternViT, Qwen2 LM and DiT layers as fused nodes | BF16 activations, flash attention | `VLA_EVO1_NO_FUSE=1`, `--flash-attn 0` |
| TurboVLA | F16 weights and flash attention | - | `--weight-dtype`, `--flash-attn 0` |
| π0.5 | flash attention; fused expert elementwise ops; with `--act-dtype bf16`, SigLIP as fused blocks | - | `VLA_PI05_NO_FUSE=1`, `VLA_PI05_NO_VIT_FUSE=1`, `--flash-attn 0` |

The load log has one line per switch that took effect, naming its off switch, for
example `vla(smolvla): fused LM and expert layers, K/V in place
(VLA_SMOLVLA_NO_FUSE=1 to disable)`. The fused graphs need BF16 weights, so
`--weight-dtype f16` or `f32` drops an arch back to the per-op graph, which is
much slower: on the Arc B390, SmolVLA takes 222 ms with F16 weights against 62 ms
at the defaults. SmolVLA has one more opt-in on its fused graph:
`VLA_SMOLVLA_W8A8=all`, or any of `vit`, `vlm`, `expert`, runs those GEMMs as
W8A8 (`src/layers/qlinear.h`), with weights quantized at load.

Arc B390 iGPU (OpenCL adapter), 2 views per model, `vla-bench` median of 20
calls after 3 warmups, one process. "Per-op" is main just before the fused
graphs (`4a542ed`); the escape hatches above restore it bit for bit:

| Model | per-op | default | speedup |
|---|--:|--:|--:|
| SmolVLA (512 px) | 120.4 ms | 62.7 ms | 1.92x |
| SmolVLA, `VLA_SMOLVLA_W8A8=all` | 120.4 ms | 52.5 ms | 2.29x |
| VLA-Adapter | 304.4 ms | 86.5 ms | 3.52x |
| GR00T N1.5 | 283.1 ms | 175.7 ms | 1.61x |
| GR00T N1.6 | 226.3 ms | 115.7 ms | 1.96x |
| GR00T N1.7, `VLA_GR00T_N17_FUSE=1` | 228.8 ms | 126.7 ms | 1.81x |
| Evo-1 | 825.0 ms | 348.5 ms | 2.37x |
| Evo-1, `--act-dtype bf16` | 825.0 ms | 168.9 ms | 4.88x |
| TurboVLA | 47.1 ms | 25.5 ms | 1.84x |

The [Core Ultra X7 358H report](../benchmark/core-ultra-x7-358h.md) has every
arch at its native view count under the full benchmark protocol.

The fused graphs are not bit-identical to the per-op graph. Against the ggml CPU
build at F32 weights on 48 paired inputs, the median error grows from 1.7e-3 to
2.8e-3 for SmolVLA, 2.5e-3 to 3.5e-3 for VLA-Adapter, 4.3e-3 to 7.7e-3 for
GR00T N1.6 and 2.2e-3 to 6.3e-3 for GR00T N1.5, whose worst input reaches 4.4e-2
against 8.0e-3. GR00T N1.7's fused graph moves its actions by up to 1.3e-2,
which is why it stays opt-in until a LIBERO run validates it. TurboVLA's F16
weights keep its actions within 3.4e-4 of the F32 graph, by the author's
measurement on a Core Ultra 7 356H. The author also reports
LIBERO-object success on the fused defaults of 99/100 for Evo-1 and 100/100 for
VLA-Adapter and TurboVLA.

### Prototype: one SmolVLA view on the NPU

On a Core Ultra with an NPU, SmolVLA can encode its last camera view on the NPU
while the GPU encodes the others, so vision costs the slower of the two rather
than their sum. A Python worker serves the vision tower through OpenVINO, built
straight from the GGUF:

```bash
python3 scripts/build_vision_ir.py smolvla-libero.gguf smolvla_vision.xml   # once
python3 scripts/npu_vision_worker.py --ir smolvla_vision.xml --sock /tmp/smolvla_npu.sock --turbo
VLA_SMOLVLA_NPU_VISION=/tmp/smolvla_npu.sock ./build-sycl/vla-server smolvla-libero.gguf
```

Run the worker in a shell without oneAPI's `setvars.sh`: OpenVINO and oneAPI ship
conflicting TBB and OpenCL libraries. `--turbo` keeps the NPU clocked between
requests. It needs 2 or more views, and Linux: on Windows the socket client is
compiled out. If the worker fails, dies or stalls past a 2 s socket timeout, that
view is encoded on the GPU and the log says `NPU view failed, encoding it on the
GPU`. With the variable unset, the graph is unchanged. It has not been
benchmarked in this repository.

## FoldQuant (W8A8 / W4A4) checkpoints

A FoldQuant GGUF runs its integer path on SYCL: `src/sycl/vla_sycl_foldquant.cpp`
claims FoldQuant's two custom nodes through an extension hook that
`scripts/patch_ggml_sycl_ext_hook.py` adds to ggml-sycl at configure time. The
activation quantization runs in its own kernels and the GEMMs in oneDNN's int8
matmul, bit-identical to the CPU reference. `VLA_FQ_DEQUANT=1` reads the sites back
as float weights instead. See [QUANTIZATION.md](../QUANTIZATION.md#sycl) for the
arithmetic and accuracy.

### The fast path: sycl-tla

π0.5's FoldQuant path has faster kernels built from
[sycl-tla](https://github.com/intel/sycl-tla) (tested at v0.9.3). They use Xe 2D
block loads and DPAS, and have only been run on the Arc B390 below. Point
`VLA_SYCL_TLA_DIR` at a checkout when configuring:

```bash
git clone --depth 1 --branch v0.9.3 https://github.com/intel/sycl-tla.git
cmake -B build-sycl -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL=ON \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DVLA_SYCL_TLA_DIR=$PWD/sycl-tla
```

That adds INT4 x INT4 DPAS GEMMs for the W4A4 sites
(`src/sycl/vla_sycl_s4gemm.cpp`) and the attention nodes of
`src/sycl/vla_sycl_fmha.cpp`: q/k/v + RoPE fused into one node that packs the
prefix's K/V in F16, which oneDNN's SDPA reads in place (`VLA_FQ_TLA_PREFIX=1`:
sycl-tla's FMHA instead). The action expert writes its own K/V into spare rows
of that pack and runs the same SDPA over all of it (`VLA_FQ_NO_KV_SDPA=1`:
sycl-tla's cached-KV FMHA). The expert's adaRMS rides in FoldQuant's activation
nodes. Configure runs `scripts/patch_sycl_tla_fmha.py` on the checkout, in
place: sycl-tla's FMHA drops or mis-masks a partial last key tile without it.
The templates add a minute or two to the build. Without `VLA_SYCL_TLA_DIR`, π0.5
runs the plain FoldQuant path above.

oneDNN fuses the SDPA only when its output is F16 like its inputs (an F32 output
silently becomes GEMM + softmax + GEMM), so the attention output, which is the o
projection's input, is rounded to F16 before FoldQuant quantizes it, where the
CPU reference keeps it F32. That moves the odd activation code across a rounding
tie, as the float layers between the sites do, and the reference check below
includes it.

The INT4 GEMMs apply the reference's epilogue to their accumulators in
registers, so a site is one kernel, with 16-row tiles for the action expert's
10 rows. The expert's INT8 sites (W8A8, and o/down in the INT8 o/down file) run
the same DPAS loop on `s8` at up to 16 rows; larger INT8 sites stay on oneDNN,
which beat CuTe tiles on the prefix. All of it is bit-identical to the oneDNN
path, and each step has an opt-out: `VLA_FQ_NO_S4EPI=1`,
`VLA_FQ_S4_NO_SMALL_M=1`, `VLA_FQ_NO_S8DPAS=1`, `VLA_FQ_NO_KV_SDPA=1`. The INT8
sites are bound by weight bandwidth: VTune traces their GPU time at 18.8 ms per
call on DPAS (13.06 + 5.73) against 20.9 ms for oneDNN's matmul plus the
epilogue (12.46 + 2.00 + 5.39 + 1.08), but the wall clock barely moves (W8A8
110.2 -> 110.0 ms in an earlier sitting).

Small Xe parts (up to 64 compute units, such as the 4 Xe cores of a Core Ultra
7 356H) take INT4 tiles sized for them: 192 x 128 for the long-K down
projection, whose epilogue then runs as its own kernel because it would spill
registers there, and 48 x 64 for the fused gate/up. `VLA_S4_TILES=small|large`
forces either set, with the same output. The Arc B390 reports 96 compute units
and takes the large set; forced small, it measured 79.4 ms for W4A4 and 89.1 ms
for W4A4 with INT8 o/down in the sitting of the table below, not yet attributed
to a site.

π0.5 LIBERO, 2 views at 224 px, 48 tokens, Intel Core Ultra X7 358H with an Arc
B390 iGPU (Level Zero, compute runtime 26.35), oneAPI 2026.1, built with
`VLA_SYCL_TLA_DIR`, `vla-bench` p50 over 10 calls after 3 warmups, median of 3
interleaved rounds:

| Checkpoint | GGUF | p50 ms | vision ms |
|---|--:|--:|--:|
| bf16 | 6.71 GB | 269 | 57 |
| bf16, `--act-dtype bf16` | 6.71 GB | 160 | 16 |
| bf16, `--act-dtype bf16`, `VLA_PI05_W8A8=all` | 6.71 GB | 99 | 12 |
| bf16, `--act-dtype bf16`, `VLA_PI05_W8A8=all VLA_PI05_W4=expert` | 6.71 GB | 87 | 12 |
| FoldQuant W4A4, `VLA_FQ_DEQUANT=1` | 3.27 GB | 210 | 16 |
| FoldQuant W4A4 | 3.27 GB | **82** | 16 |
| FoldQuant W4A4, `VLA_PI05_W8A8=vit` | 3.27 GB | 76 | 12 |
| FoldQuant W4A4, o/down INT8 | 3.61 GB | 92 | 16 |
| FoldQuant W4A4, o/down INT8, `VLA_PI05_W8A8=vit` | 3.61 GB | 86 | 12 |
| FoldQuant W8A8 | 4.42 GB | 115 | 16 |
| FoldQuant W8A8, `VLA_PI05_W8A8=vit` | 4.42 GB | 106 | 12 |

Re-measured on 5 November at main `a83bcad` on another Core Ultra X7 358H, built
without `VLA_SYCL_TLA_DIR`, on the OpenCL adapter with compute runtime 26.22,
under the [benchmark protocol](../benchmark/core-ultra-x7-358h.md) (3 warmups +
20 calls, 3 processes, the best mean), the bf16 rows at 48 tokens give 282.2 ms
(bf16), 173.5 ms (`--act-dtype bf16`), 101.2 ms (`+ VLA_PI05_W8A8=all`) and
92.6 ms (`+ VLA_PI05_W4=expert`), and at the report's 16-token prompts 276.4,
165.3, 99.6 and 91.6 ms. Their p50s are 2-8% above the table's. That machine
differs from the table's in adapter, runtime and sycl-tla; the gap was not split
between them. The FoldQuant rows need sycl-tla and the FoldQuant files, and were
not re-measured.

The FoldQuant rows run vision as the fused bf16 SigLIP blocks, as
`--act-dtype bf16` does; plain bf16 keeps it F32. W4A4 is 3.3x faster than bf16
and 1.9x faster than bf16 activations, at half the file size. The two bf16 opt-ins
quantize the weights at load time instead (W8A8 GEMMs in SigLIP, the VLM and the
expert; `VLA_PI05_W4=expert` makes the expert's weights INT4): W4A8 comes within
5 ms of FoldQuant W4A4 without a quantized file, and W8A8 beats FoldQuant W8A8,
whose rotated per-row activation quantization costs more than the opt-in's
per-token scales.

`VLA_PI05_W8A8=vit` on a FoldQuant file runs SigLIP, which the file carries in
BF16, on the same W8A8 GEMMs: its weights are quantized at load to symmetric
per-output-channel INT8 by round-to-nearest, with no calibration, and its
activations per token (`src/layers/qlinear.h`). That is not FoldQuant's
arithmetic, so it stays an opt-in; it takes 6-9 ms off a call.

LIBERO success rate of FoldQuant W4A4 with and without it, same server and
client as below, 10 tasks x 20 episodes per suite:

| Suite | W4A4 | W4A4, `VLA_PI05_W8A8=vit` |
|---|--:|--:|
| libero_object | 99.5% (199/200) | 99.0% (198/200) |
| libero_spatial | 99.0% (198/200) | 97.0% (194/200) |

Most of the libero_spatial gap is task 4 (16/20 with `vit`, 20/20 without); 50
more episodes of that task scored 50/50 for both, so it does not reproduce, but
the combined 66/70 against 70/70 is not level either, and SigLIP's INT8 is not
calibrated. These ran at an earlier commit whose actions match the table's
build bit for bit.

LIBERO success rate of the two opt-ins on the same machine: the bf16 π0.5 LIBERO
GGUF with `--act-dtype bf16`, served by `vla-server` (`eval/run_libero.sh`), 10 tasks x 50
episodes per suite, `n_action_steps` 10:

| Suite | `VLA_PI05_W8A8=all` | `+ VLA_PI05_W4=expert` |
|---|--:|--:|
| libero_object | 99.2% (496/500) | 99.4% (497/500) |
| libero_spatial | 99.0% (495/500) | 98.4% (492/500) |
| libero_goal | 98.4% (492/500) | 99.2% (496/500) |
| libero_10 | 92.4% (462/500) | 93.4% (467/500) |
| **Average** | **97.3%** | **97.6%** |

The two are within the noise of 500 episodes on every suite; one step of
inference took 35-43 ms in both.

LIBERO success rate of the FoldQuant files and the bf16 GGUF on the same Arc B390
at the latency table's build, same server and client, 10 tasks x 50 episodes per suite,
environment seed 42, 10 denoise steps, `n_action_steps` 10. With openpi's prompt
(`VLA_PI05_PROMPT=openpi`), the setting to use for the FoldQuant π0.5 files:

| Suite | FoldQuant W4A4 | bf16, `--act-dtype bf16` |
|---|--:|--:|
| libero_object | 99.8% (499/500) | 99.4% (497/500) |
| libero_spatial | 99.0% (495/500) | 98.8% (494/500) |
| libero_goal | 99.0% (495/500) | 99.2% (496/500) |
| libero_10 | 95.0% (475/500) | 93.8% (469/500) |
| **Average** | **98.2%** | **97.8%** |

With the client's default LeRobot prompt (`Task: <task>, State: <bins>;\nAction: `),
which the opt-in table above also used, successes out of 500 per suite:

| Checkpoint | libero_object | libero_spatial | libero_goal | libero_10 | Average |
|---|--:|--:|--:|--:|--:|
| bf16 | 493 | 489 | 490 | 461 | 96.7% |
| bf16, `--act-dtype bf16` | 497 | 495 | 490 | 457 | 97.0% |
| FoldQuant W4A4 | 495 | 486 | 471 | 458 | 95.5% |
| FoldQuant W4A4, `VLA_PI05_W8A8=vit` | 496 | 491 | 474 | 450 | 95.6% |
| FoldQuant W4A4, o/down INT8 | 492 | 495 | 484 | 471 | 97.1% |
| FoldQuant W8A8 | 498 | 496 | 490 | 462 | 97.3% |

These checkpoints come from openpi's π0.5 (`vrfai/pi05_libero_pytorch`), and
FoldQuant calibrated the activation scales of its files on the plain task prompt.
On the State prompt W4A4 loses 2.7 points, most of them on libero_goal (471
against 495), and INT8 o/down and W8A8 win most of that back; `vit` gains nothing
on success rate. The client's default prompt stays LeRobot's.

The FoldQuant files are FoldQuantVLA quantized checkpoints converted with
`scripts/convert_quantized_model_to_gguf.py`: W4A4 and W4A4 with INT8 o/down are
[vrfai/pi05-libero-w4a4](https://huggingface.co/vrfai/pi05-libero-w4a4) and
[vrfai/pi05-libero-w4a4-res8](https://huggingface.co/vrfai/pi05-libero-w4a4-res8);
W8A8 is FoldQuantVLA's W8A8 arm (`--llm-scheme w8a8_sr --expert-scheme w8a8_sh`)
calibrated on the same 128 LIBERO frames, seed 0.

Against FoldQuantVLA's own fake-quant π0.5 (openpi PyTorch, `vrfai/pi05_libero_pytorch`
with the W4A4 and INT8 o/down states) on 8 real LIBERO frames, with the same
images, tokens and noise, 10 steps: worst-frame cosine of the 10 x 7 action chunk,
after the output transforms and in the model's normalized output space:

| Checkpoint | actions | normalized |
|---|--:|--:|
| bf16 (against the reference's bf16) | 0.99999 | 0.99999 |
| FoldQuant W4A4 | 0.99916 | 0.99844 |
| FoldQuant W4A4, `VLA_PI05_W8A8=vit` | 0.99921 | 0.99876 |
| FoldQuant W4A4, o/down INT8 | 0.99965 | 0.99923 |
| FoldQuant W4A4, o/down INT8, `VLA_PI05_W8A8=vit` | 0.99958 | 0.99914 |

The quantized rows are not bit-exact to the reference by construction: its
attention runs in bf16 and the F16 rounding above sits before the o projection,
so a few activation codes land on the other side of a rounding tie. Without the
gripper dimension the worst normalized cosine is 0.9955 (W4A4).

## Known issue: the Level Zero loader on Ubuntu 24.04

oneAPI 2026.1's Level Zero adapters crashed (`sycl-ls` segfaults) against Ubuntu
24.04's `libze1` 1.16 on the machine above with an older GPU compute runtime.
With the runtime at 26.35 (`libze-intel-gpu1`, `intel-opencl-icd`) Level Zero
works against the same `libze1`, and the FoldQuant table above uses it. With
runtime 26.22 it still crashes, which is why the
[Core Ultra X7 358H report](../benchmark/core-ultra-x7-358h.md) ran on OpenCL. On
an older runtime, pointing the runtime at the OpenCL adapter works:

```bash
export UR_ADAPTERS_FORCE_LOAD=/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_opencl.so.0
```

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
its default BF16 weights, and the workaround was F32 weights (then
`VLA_ADAPTER_F32_WEIGHTS=1`, now `--weight-dtype f32`).

llama.cpp b10326 adds the missing kernel (`cpy_1_bf16_f32` in
`ggml/src/ggml-sycl/cpy.cpp`), so VLA-Adapter should run on stock BF16 weights
now. Not yet re-tested on the A380 - if you hit the old abort, fall back to
`--weight-dtype f32` and file an issue.

## Performance note: F32 weights

BF16 has no native DPAS path on Xe-HPG, so BF16 weights are slower there than
plain F32 despite the extra bandwidth. `--weight-dtype f32` switches any arch
to F32 weights, and on the A380 it is worth ~16%:

| SmolVLA weights | vision | inference | total |
|---|---:|---:|---:|
| BF16 (default) | 158 ms | 474 ms | **630 ms** |
| F32 (`--weight-dtype f32`) | 173 ms | 355 ms | **528 ms** |

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

VLA-Adapter is measured with F32 weights on both sides, which was
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
`--weight-dtype f32` it reaches 528 ms (3.6x).

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
