# Intel Core Ultra X7 358H: Arc B390 iGPU, AI Boost NPU and CPU

`vla-bench` times `predict()` in-process on synthetic inputs. It measures the
engine only: no transport, no simulator, and no claim about task success.

Five ways to run on this chip, from three builds. The OpenVINO build selects its device with
`GGML_OPENVINO_DEVICE`: `GPU` for the Arc B390 iGPU, `NPU` for the AI Boost NPU and `CPU` for
OpenVINO's CPU plugin. Every OpenVINO run was checked against ggml's `OpenVINO: using device` line,
so a silent fallback to another device would have been counted as a failure. A SYCL build runs
ggml's SYCL backend on the Arc B390, and a build with neither gives ggml's own CPU backend. All
five sections were measured in one round, at one commit, under one power profile.

## Test setup

| | |
|---|---|
| Device | Intel Core Ultra X7 358H (Panther Lake): 16 cores, 16 threads; Arc B390 iGPU; AI Boost NPU; 62 GiB of shared RAM |
| Threads | 16 (the default: all hardware threads, capped at 16), for both CPU sections |
| Power | platform profile `performance`; PL1 25 W over a 28 s window, PL2 65 W; `intel_pstate` `powersave` governor. `thermald --adaptive` was running and can change PL1; see the [notes](#notes) |
| OS | Ubuntu 24.04.4 LTS, kernel 7.0 |
| Drivers | GPU compute runtime 26.22.38646.4, NPU driver 1.33.0, Level Zero loader 1.16.1 |
| Toolkit | OpenVINO 2026.4.0, the version `scripts/install_ov.sh` pins, unpacked into a private directory; the system's 2026.2.1 was not used. oneAPI 2026.1 (DPC++ 2026.1.1). GCC 13.3 |
| SYCL adapter | OpenCL (`UR_ADAPTERS_FORCE_LOAD=.../libur_adapter_opencl.so.0`, `ONEAPI_DEVICE_SELECTOR=opencl:gpu`). The Level Zero adapter crashes with this machine's compute runtime (26.22) and loader (1.16.1), so every SYCL run went through OpenCL. The A380 report ran on Level Zero. |
| Commit | `a83bcad` (main: PR #73 and two fixes to SmolVLA's opt-in NPU vision client, which the timed paths do not use) |
| llama.cpp | `b11223` |
| Date | 2026-11-04 13:12 to 2026-11-05 06:05 (UTC+07) |
| Build | OpenVINO: `-DGGML_OPENVINO=ON -DCMAKE_BUILD_TYPE=Release -G Ninja`, after sourcing OpenVINO's `setupvars.sh`. SYCL: `-DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release -G Ninja`, after sourcing oneAPI's `setvars.sh`. ggml CPU backend: `-DCMAKE_BUILD_TYPE=Release -G Ninja`. All three with `GGML_NATIVE=ON` |
| Runtime flags | defaults; the fastest flags per model are in each section's second table |
| Method | 3 warmups + 20 timed reps per process, 3 processes. The process with the lowest mean is reported, and memory is the peak over all three. |

Memory is shared with the CPU. **Peak device buffers** is the process's GPU or NPU buffer memory from the DRM `fdinfo` counters, sampled every 0.5 s: `drm-total-gtt`, `-system` and `-stolen` for the xe GPU driver, and `drm-total-memory` for the intel_vpu NPU driver. **Peak RSS** is the process's peak resident set (`getrusage`). The two are separate counters, so adding them can double-count a buffer the CPU also maps. The two CPU sections report peak RSS only.

## Model configuration

Each model runs at its native view count and input size, with weights as
shipped. Checkpoints are the `vrfai/*` GGUFs; where a repo ships several, we use
the `libero_object` variant. Settings not listed are the defaults.

| Model | Checkpoint | Views | Input | Lang tokens | Action chunk | Action dim | Action head | Steps |
|---|---|--:|--:|--:|--:|--:|---|--:|
| SmolVLA | `smolvla-libero-gguf` | 2 | 512 | 16 | 50 | 7 | flow matching | 10 |
| π0 | `pi0-libero-finetuned-v044-gguf` | 2 | 224 | 16 | 50 | 7 | flow matching | 10 |
| π0.5 | `pi05-libero-gguf` | 2 | 224 | 16 | 50 | 7 | flow matching | 10 |
| GR00T N1.5 | `gr00tn1d5-libero-object-gguf` | 1 | 224 | 16 | 16 | 32 | flow matching (DiT) | 4 |
| GR00T N1.6 | `gr00tn1d6-libero-gguf` | 1 | 224 | 16 | 50 | 128 | flow matching (DiT) | 4 |
| GR00T N1.7 | `gr00tn1d7-libero-gguf` | 1 | 256 | 16 | 40 | 132 | flow matching (DiT) | 4 |
| VLA-JEPA | `vla-jepa-libero` | 1 | 256 | 16 + 32 | 7 | 7 | flow matching (DiT) | 4 |
| Evo-1 | `evo1-libero-gguf` | 1 | 448 | 16 | 50 | 7 | flow matching | 32 |
| BitVLA | `bitvla-libero-gguf` (int2) | 1 | 224 | 16 | 8 | 7 | parallel decoding | 1 |
| VLA-Adapter | `vla-adapter-libero-gguf` | 1 | 224 | 16 | 8 | 7 | bridge-attention policy | 1 |
| OpenVLA-OFT | `openvla-oft-libero-gguf` | 1 | 224 | 16 | 8 | 7 | L1 regression head | 1 |
| Octo-Small | `octo-small-libero-gguf` | 2 | 256 + 128 | 16 | 4 | 7 | DDPM diffusion | 20 |
| TurboVLA | `turbovla-libero-gguf` | 2 | 256 | 16 | 12 | 7 | single-pass decoder | 1 |

- **Lang tokens** is the synthetic prompt length. VLA-JEPA adds its 32
  `<embodied>` tokens with `--extra-token 151697 --extra-count 32`.
- **Action dim** is the checkpoint's padded width. LIBERO uses the first 7 dims.
- **GR00T** needs its embodiment set in the environment:
  `VLA_GR00T_EMBODIMENT=new_embodiment` for N1.5, `libero_panda` for N1.6 and
  `libero_sim` for N1.7. **Octo** needs `VLA_OCTO_UNNORM_DATASET=libero_object`.
- **Octo** takes a 256 px primary view and a 128 px wrist view.

### How the fastest configuration is picked

We screen each model over these runtime flag sets, then re-run the fastest set
under the full protocol:

- the defaults;
- `--flash-attn` and `--mm-prec default`, alone and together;
- `--weight-dtype bf16` and `--weight-dtype f16`, each alone and with
  `--flash-attn`;
- `--act-dtype bf16 --flash-attn`, for π0, π0.5 and Evo-1 on SYCL only. These
  are the only archs that take BF16 activations.

The screen runs 2 warmups and 5 reps in one process per flag set. A flag set
replaces the defaults only if its mean is more than 3% lower, both in the screen
and in the full re-run.

Flash attention, bf16 activations and lower-precision weights can move the
action chunk. Success rates measured at the defaults therefore do not carry over
to these rows.

## Latency and memory: Arc B390 iGPU (OpenVINO GPU plugin)

π0 runs at F32 inference precision on the GPU by default
(`GGML_OPENVINO_GPU_PRECISION=f32`, which vla.cpp sets for π0 alone). π0.5 pads
its prompt to a 16-token bucket, which the 16-token synthetic prompt already
fills, and runs with `GGML_OPENVINO_EXEC_MODE=performance`; both are vla.cpp's
defaults for π0.5 on OpenVINO.

Rows are sorted by latency. `vision ms` is `-` for archs that do not time their
vision stage separately.

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak device buffers MiB | Peak RSS MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| TurboVLA | 2 | 256 | 28.5 | 29.7 | 29.6 | 30.4 | - | 1080 | 1540 |
| GR00T N1.5 | 1 | 224 | 106.0 | 107.1 | 107.2 | 107.4 | 38.1 | 10575 | 6630 |
| VLA-JEPA | 1 | 256 | 116.3 | 117.0 | 117.1 | 117.8 | 19.9 | 9387 | 5010 |
| GR00T N1.6 | 1 | 224 | 122.8 | 131.2 | 132.6 | 134.5 | 15.6 | 14698 | 9922 |
| GR00T N1.7 | 1 | 256 | 131.4 | 132.0 | 131.8 | 132.8 | 20.1 | 15035 | 10999 |
| VLA-Adapter | 1 | 224 | 170.2 | 185.4 | 172.4 | 207.7 | 97.8 | 7431 | 3621 |
| SmolVLA | 2 | 512 | 213.1 | 214.9 | 215.0 | 216.0 | 111.7 | 4211 | 3616 |
| π0.5 | 2 | 224 | 232.5 | 233.7 | 233.5 | 234.5 | 36.5 | 15964 | 11464 |
| Evo-1 | 1 | 448 | 566.0 | 568.2 | 568.2 | 569.5 | 227.9 | 4982 | 3279 |
| π0 | 2 | 224 | 620.5 | 625.1 | 624.6 | 630.1 | 78.7 | 25056 | 16936 |
| OpenVLA-OFT | 1 | 224 | 1262.5 | 1334.9 | 1326.8 | 1419.9 | 139.5 | 41761 | 15264 |

### Fastest configuration: Arc B390 iGPU (OpenVINO GPU plugin)

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak device buffers MiB | Peak RSS MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| TurboVLA | `--weight-dtype bf16 --flash-attn` | 25.7 | 26.6 | 26.6 | 27.2 | - | 1330 | 1175 | -10% |
| GR00T N1.5 | *(defaults)* | 106.0 | 107.1 | 107.2 | 107.4 | 38.1 | 10575 | 6630 | - |
| VLA-JEPA | *(defaults)* | 116.3 | 117.0 | 117.1 | 117.8 | 19.9 | 9387 | 5010 | - |
| GR00T N1.6 | `--weight-dtype f16 --flash-attn` | 121.9 | 123.4 | 123.5 | 124.1 | 14.6 | 4965 | 5025 | -6% |
| GR00T N1.7 | *(defaults)* | 131.4 | 132.0 | 131.8 | 132.8 | 20.1 | 15035 | 10999 | - |
| SmolVLA | `--weight-dtype bf16 --flash-attn` | 169.6 | 170.2 | 170.2 | 170.6 | 65.3 | 4132 | 3580 | -21% |
| VLA-Adapter | *(defaults)* | 170.2 | 185.4 | 172.4 | 207.7 | 97.8 | 7431 | 3621 | - |
| π0.5 | *(defaults)* | 232.5 | 233.7 | 233.5 | 234.5 | 36.5 | 15964 | 11464 | - |
| Evo-1 | `--weight-dtype f16 --flash-attn` | 467.4 | 472.7 | 472.5 | 477.0 | 135.3 | 4871 | 3094 | -17% |
| π0 | *(defaults)* | 620.5 | 625.1 | 624.6 | 630.1 | 78.7 | 25056 | 16936 | - |
| OpenVLA-OFT | *(defaults)* | 1262.5 | 1334.9 | 1326.8 | 1419.9 | 139.5 | 41761 | 15264 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| TurboVLA | 29.8 | 26.5 | 29.3 | 27.5 | 29.4 | 26.4 | 29.2 | 28.2 |
| GR00T N1.5 | 107.2 | 109.7 | 107.5 | 107.9 | 107.3 | 109.7 | 107.3 | 108.1 |
| VLA-JEPA | 116.8 | 114.9 | 116.8 | 115.2 | 115.7 | 114.6 | 116.7 | 114.2 |
| GR00T N1.6 | 129.9 | 124.2 | 133.0 | 124.6 | 130.6 | 124.1 | 133.9 | 123.8 |
| GR00T N1.7 | 134.0 | 130.9 | 132.6 | 131.3 | 132.1 | 131.0 | 132.4 | 130.7 |
| VLA-Adapter | 232.8 | 210.0 | 259.5 | 220.9 | 202.2 | 187.6 | 189.6 | 183.3 |
| SmolVLA | 215.5 | 171.5 | 217.9 | 170.7 | 223.6 | 169.6 | 217.1 | 172.1 |
| π0.5 | 233.4 | 234.1 | 235.0 | 232.2 | 234.4 | 233.3 | 229.6 | 229.1 |
| Evo-1 | 567.9 | 477.1 | 576.2 | 478.6 | 568.0 | 478.0 | 568.2 | 477.0 |
| π0 | 609.4 | 623.9 | 611.4 | 603.7 | 609.4 | 610.5 | 610.8 | 607.7 |
| OpenVLA-OFT | 1205.1 | 1507.0 | 1232.8 | 1439.8 | 1298.3 | 1298.6 | 1355.9 | 2154.5 |

</details>

## Latency and memory: Arc B390 iGPU (SYCL)

SYCL builds run fused graphs by default for SmolVLA, VLA-Adapter, GR00T N1.5
and GR00T N1.6 (PR #73), and fused elementwise ops in π0.5's action expert
(PR #42). Evo-1 and π0.5 run their fused graphs only with `--act-dtype bf16`,
which is why that flag set is their fastest row. SmolVLA, Evo-1, π0.5 and
TurboVLA also default to flash attention on SYCL, and TurboVLA to F16 weights,
unless `--flash-attn` or `--weight-dtype` says otherwise. The fused graphs need
BF16 weights, so the screen's `--weight-dtype f16` columns run the per-op graph.
Each fused path has an escape hatch that restores the per-op graph, such as
`VLA_SMOLVLA_NO_FUSE=1`; the model's load log names it, and
[sycl.md](../backend/sycl.md#sycl-defaults-fused-graphs-and-flash-attention)
lists them. GR00T N1.7's fused graph is opt-in and is not in these tables; the
[notes](#notes) give it, and how far the fused graphs' actions move from the
per-op graph's.

Rows are sorted by latency. `vision ms` is `-` for archs that do not time their
vision stage separately.

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak device buffers MiB | Peak RSS MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 2 | 256 + 128 | 15.5 | 15.8 | 15.6 | 16.2 | 1.6 | 615 | 444 |
| TurboVLA | 2 | 256 | 25.0 | 25.6 | 25.7 | 25.8 | - | 560 | 445 |
| SmolVLA | 2 | 512 | 61.7 | 62.3 | 62.3 | 62.8 | 16.4 | 1107 | 431 |
| VLA-Adapter | 1 | 224 | 62.6 | 63.1 | 63.1 | 63.5 | 22.8 | 2751 | 1065 |
| GR00T N1.5 | 1 | 224 | 101.2 | 101.8 | 101.9 | 102.1 | 12.9 | 3535 | 423 |
| VLA-JEPA | 1 | 256 | 102.1 | 102.6 | 102.5 | 103.0 | 39.4 | 3879 | 422 |
| GR00T N1.6 | 1 | 224 | 106.1 | 106.5 | 106.5 | 106.8 | 13.0 | 4566 | 410 |
| GR00T N1.7 | 1 | 256 | 178.4 | 179.4 | 179.3 | 180.3 | 39.6 | 4944 | 406 |
| π0 | 2 | 224 | 269.3 | 270.0 | 269.9 | 270.7 | 57.4 | 5433 | 476 |
| Evo-1 | 1 | 448 | 270.4 | 271.8 | 271.8 | 272.2 | 76.6 | 1456 | 516 |
| π0.5 | 2 | 224 | 275.2 | 276.4 | 276.4 | 277.2 | 56.7 | 5665 | 547 |
| OpenVLA-OFT | 1 | 224 | 434.9 | 438.5 | 438.3 | 441.3 | 79.9 | 14534 | 1035 |

### Fastest configuration: Arc B390 iGPU (SYCL)

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak device buffers MiB | Peak RSS MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | *(defaults)* | 15.5 | 15.8 | 15.6 | 16.2 | 1.6 | 615 | 444 | - |
| TurboVLA | *(defaults)* | 25.0 | 25.6 | 25.7 | 25.8 | - | 560 | 445 | - |
| SmolVLA | *(defaults)* | 61.7 | 62.3 | 62.3 | 62.8 | 16.4 | 1107 | 431 | - |
| VLA-Adapter | *(defaults)* | 62.6 | 63.1 | 63.1 | 63.5 | 22.8 | 2751 | 1065 | - |
| GR00T N1.5 | `--weight-dtype bf16 --flash-attn` | 87.8 | 88.2 | 88.3 | 88.5 | 13.0 | 3537 | 433 | -13% |
| VLA-JEPA | `--flash-attn` | 93.1 | 93.8 | 93.7 | 94.4 | 32.4 | 3884 | 431 | -9% |
| GR00T N1.6 | *(defaults)* | 106.1 | 106.5 | 106.5 | 106.8 | 13.0 | 4566 | 410 | - |
| Evo-1 | `--act-dtype bf16 --flash-attn` | 142.9 | 144.2 | 144.0 | 144.9 | 26.2 | 1371 | 508 | -47% |
| π0.5 | `--act-dtype bf16 --flash-attn` | 164.9 | 165.5 | 165.4 | 165.9 | 15.2 | 5669 | 547 | -40% |
| GR00T N1.7 | `--weight-dtype bf16 --flash-attn` | 167.9 | 168.7 | 168.7 | 169.5 | 32.2 | 4948 | 413 | -6% |
| π0 | `--act-dtype bf16 --flash-attn` | 243.6 | 244.6 | 244.5 | 245.3 | 46.3 | 5333 | 476 | -9% |
| OpenVLA-OFT | *(defaults)* | 434.9 | 438.5 | 438.3 | 441.3 | 79.9 | 14534 | 1035 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` | `--act-dtype bf16 --flash-attn` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 16.1 | 16.0 | 16.0 | 16.0 | 16.0 | 16.4 | 16.4 | 16.3 |  |
| TurboVLA | 25.5 | 25.5 | 25.5 | 25.4 | 25.7 | 25.7 | 25.3 | 25.6 |  |
| SmolVLA | 62.6 | 62.1 | 62.3 | 62.5 | 62.5 | 62.2 | 221.9 | 222.5 |  |
| VLA-Adapter | 63.0 | 63.3 | 63.3 | 63.3 | 63.3 | 63.1 | 287.0 | 362.6 |  |
| GR00T N1.5 | 101.8 | 88.6 | 101.8 | 88.4 | 101.8 | 88.2 | 441.6 | 453.2 |  |
| VLA-JEPA | 103.2 | 92.2 | 102.3 | 93.4 | 102.5 | 93.7 | 272.0 | 263.3 |  |
| GR00T N1.6 | 106.8 | 105.3 | 104.0 | 104.5 | 106.1 | 103.6 | 507.2 | 495.1 |  |
| GR00T N1.7 | 179.4 | 169.6 | 181.0 | 169.9 | 178.9 | 167.8 | 540.9 | 535.0 |  |
| π0 | 270.8 | 270.7 | 413.4 | 270.9 | 270.6 | 270.3 | 829.1 | 831.0 | 244.8 |
| Evo-1 | 267.3 | 269.9 | 270.7 | 277.0 | 271.3 | 272.4 | 721.2 | 534.6 | 144.5 |
| π0.5 | 276.4 | 277.7 | 277.2 | 276.5 | 277.1 | 277.0 | 882.9 | 880.6 | 166.0 |
| OpenVLA-OFT | 441.6 | 481.1 | 442.0 | 441.9 | 440.8 | 439.1 | 1574.7 | 1574.5 |  |

</details>

## Latency and memory: AI Boost NPU (OpenVINO NPU plugin)

Rows are sorted by latency. `vision ms` is `-` for archs that do not time their
vision stage separately.

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak device buffers MiB | Peak RSS MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| TurboVLA | 2 | 256 | 23.5 | 25.1 | 24.0 | 28.8 | - | 486 | 1790 |
| VLA-JEPA | 1 | 256 | 90.7 | 91.3 | 91.3 | 91.9 | 22.7 | 3953 | 8991 |
| VLA-Adapter | 1 | 224 | 126.7 | 128.5 | 128.0 | 130.3 | 49.8 | 3129 | 6925 |
| GR00T N1.5 | 1 | 224 | 134.0 | 135.2 | 135.0 | 136.1 | 28.5 | 3752 | 9727 |
| GR00T N1.7 | 1 | 256 | 144.9 | 145.8 | 145.8 | 146.3 | 22.8 | 5083 | 12417 |
| GR00T N1.6 | 1 | 224 | 161.1 | 161.9 | 161.8 | 162.6 | 28.4 | 4923 | 12108 |
| SmolVLA | 2 | 512 | 161.3 | 165.0 | 164.9 | 166.4 | 55.4 | 1191 | 4975 |
| Evo-1 | 1 | 448 | 256.3 | 258.2 | 258.1 | 259.2 | 78.7 | 1528 | 5579 |
| OpenVLA-OFT | 1 | 224 | 405.8 | 408.4 | 408.4 | 410.2 | 51.6 | 14975 | 35112 |
| π0.5 | 2 | 224 | 430.2 | 438.1 | 438.0 | 441.9 | 57.1 | 5784 | 14942 |
| π0 | 2 | 224 | 438.3 | 444.6 | 445.4 | 447.1 | 57.4 | 5594 | 14214 |

### Fastest configuration: AI Boost NPU (OpenVINO NPU plugin)

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak device buffers MiB | Peak RSS MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| TurboVLA | *(defaults)* | 23.5 | 25.1 | 24.0 | 28.8 | - | 486 | 1790 | - |
| VLA-JEPA | *(defaults)* | 90.7 | 91.3 | 91.3 | 91.9 | 22.7 | 3953 | 8991 | - |
| VLA-Adapter | *(defaults)* | 126.7 | 128.5 | 128.0 | 130.3 | 49.8 | 3129 | 6925 | - |
| GR00T N1.5 | *(defaults)* | 134.0 | 135.2 | 135.0 | 136.1 | 28.5 | 3752 | 9727 | - |
| GR00T N1.7 | *(defaults)* | 144.9 | 145.8 | 145.8 | 146.3 | 22.8 | 5083 | 12417 | - |
| GR00T N1.6 | *(defaults)* | 161.1 | 161.9 | 161.8 | 162.6 | 28.4 | 4923 | 12108 | - |
| SmolVLA | *(defaults)* | 161.3 | 165.0 | 164.9 | 166.4 | 55.4 | 1191 | 4975 | - |
| Evo-1 | *(defaults)* | 256.3 | 258.2 | 258.1 | 259.2 | 78.7 | 1528 | 5579 | - |
| OpenVLA-OFT | *(defaults)* | 405.8 | 408.4 | 408.4 | 410.2 | 51.6 | 14975 | 35112 | - |
| π0.5 | *(defaults)* | 430.2 | 438.1 | 438.0 | 441.9 | 57.1 | 5784 | 14942 | - |
| π0 | *(defaults)* | 438.3 | 444.6 | 445.4 | 447.1 | 57.4 | 5594 | 14214 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| TurboVLA | 32.1 | 38.4 | 32.6 | 37.8 | 32.2 | 38.4 | 32.0 | 38.7 |
| VLA-JEPA | 94.0 | 94.6 | 93.7 | 94.5 | 93.8 | 94.8 | 94.0 | 94.7 |
| VLA-Adapter | 129.8 | 128.2 | 128.9 | 129.5 | 128.0 | 130.1 | 129.3 | 129.3 |
| GR00T N1.5 | 135.6 | 137.2 | 134.9 | 137.8 | 135.8 | 137.4 | 135.3 | 137.8 |
| GR00T N1.7 | 145.8 | 144.5 | 145.6 | 144.9 | 145.7 | 144.7 | 145.8 | 144.9 |
| GR00T N1.6 | 162.4 | 163.4 | 162.1 | 163.1 | 161.4 | 163.1 | 162.0 | 162.8 |
| SmolVLA | 164.6 | 164.9 | 166.1 | 163.1 | 161.8 | 162.6 | 164.9 | 162.7 |
| Evo-1 | 257.8 | 258.0 | 257.6 | 257.7 | 257.5 | 257.3 | 258.0 | 258.0 |
| OpenVLA-OFT | 408.6 | 408.1 | 407.4 | 405.7 | 406.4 | 405.8 | 407.3 | 406.8 |
| π0.5 | 444.3 | 442.0 | 444.2 | 443.8 | 443.5 | 444.2 | 439.9 | 441.1 |
| π0 | 445.9 | 448.7 | 447.8 | 449.2 | 445.1 | 435.9 | 446.1 | 445.5 |

</details>

## Latency and memory: OpenVINO CPU plugin

Rows are sorted by latency. `vision ms` is `-` for archs that do not time their
vision stage separately.

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| TurboVLA | 2 | 256 | 238.4 | 281.2 | 260.7 | 301.1 | - | 1816 |
| VLA-JEPA | 1 | 256 | 1407.2 | 1413.3 | 1411.8 | 1415.3 | 491.4 | 11694 |
| VLA-Adapter | 1 | 224 | 1831.2 | 1838.1 | 1837.4 | 1842.8 | 1104.5 | 7890 |
| SmolVLA | 2 | 512 | 1899.5 | 1913.8 | 1914.2 | 1923.4 | 1242.1 | 4186 |
| GR00T N1.7 | 1 | 256 | 2076.5 | 2083.7 | 2081.9 | 2086.3 | 488.3 | 14927 |
| GR00T N1.6 | 1 | 224 | 2300.4 | 2311.6 | 2312.3 | 2317.3 | 558.6 | 13833 |
| GR00T N1.5 | 1 | 224 | 2456.6 | 2466.9 | 2465.6 | 2477.2 | 629.8 | 10635 |
| Evo-1 | 1 | 448 | 4626.5 | 4650.7 | 4638.1 | 4689.2 | 2031.6 | 4765 |
| π0.5 | 2 | 224 | 7522.7 | 7551.0 | 7544.4 | 7582.7 | 1265.5 | 16600 |
| π0 | 2 | 224 | 7537.1 | 7584.4 | 7585.8 | 7607.5 | 1291.6 | 15828 |
| OpenVLA-OFT | 1 | 224 | 11651.3 | 11721.3 | 11730.9 | 11750.1 | 1156.9 | 43074 |

### Fastest configuration: OpenVINO CPU plugin

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| TurboVLA | *(defaults)* | 238.4 | 281.2 | 260.7 | 301.1 | - | 1816 | - |
| VLA-JEPA | *(defaults)* | 1407.2 | 1413.3 | 1411.8 | 1415.3 | 491.4 | 11694 | - |
| VLA-Adapter | *(defaults)* | 1831.2 | 1838.1 | 1837.4 | 1842.8 | 1104.5 | 7890 | - |
| SmolVLA | *(defaults)* | 1899.5 | 1913.8 | 1914.2 | 1923.4 | 1242.1 | 4186 | - |
| GR00T N1.7 | *(defaults)* | 2076.5 | 2083.7 | 2081.9 | 2086.3 | 488.3 | 14927 | - |
| GR00T N1.6 | *(defaults)* | 2300.4 | 2311.6 | 2312.3 | 2317.3 | 558.6 | 13833 | - |
| GR00T N1.5 | `--flash-attn --mm-prec default` | 2380.8 | 2391.0 | 2388.7 | 2400.8 | 559.7 | 10656 | -3% |
| Evo-1 | *(defaults)* | 4626.5 | 4650.7 | 4638.1 | 4689.2 | 2031.6 | 4765 | - |
| π0.5 | *(defaults)* | 7522.7 | 7551.0 | 7544.4 | 7582.7 | 1265.5 | 16600 | - |
| π0 | *(defaults)* | 7537.1 | 7584.4 | 7585.8 | 7607.5 | 1291.6 | 15828 | - |
| OpenVLA-OFT | *(defaults)* | 11651.3 | 11721.3 | 11730.9 | 11750.1 | 1156.9 | 43074 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| TurboVLA | 240.7 | 237.9 | 239.7 | 240.1 | 381.4 | 246.8 | 249.2 | 372.9 |
| VLA-JEPA | 1533.2 | 1457.9 | 1533.6 | 1500.9 | 1443.4 | 1419.5 | 1541.8 | 1536.8 |
| VLA-Adapter | 1840.0 | 1848.4 | 1843.5 | 1839.2 | 1840.4 | 1841.5 | 1955.7 | 1986.7 |
| SmolVLA | 1912.5 | 1883.7 | 2013.1 | 1867.0 | 1935.9 | 1871.4 | 1920.8 | 1936.9 |
| GR00T N1.7 | 2224.8 | 2087.5 | 2212.9 | 2204.2 | 2083.8 | 2189.3 | 2198.2 | 2219.7 |
| GR00T N1.6 | 2316.6 | 2402.5 | 2426.6 | 2313.2 | 2308.9 | 2309.3 | 2315.6 | 2307.5 |
| GR00T N1.5 | 2473.4 | 2397.4 | 2477.8 | 2390.8 | 2475.4 | 2399.7 | 2572.2 | 2402.3 |
| Evo-1 | 4644.6 | 4560.2 | 4641.2 | 4584.1 | 4680.8 | 4582.7 | 4635.1 | 4583.3 |
| π0.5 | 7597.4 | 7535.8 | 7554.6 | 7544.1 | 7514.6 | 7533.8 | 7536.5 | 7515.9 |
| π0 | 7626.0 | 7430.0 | 7623.3 | 7444.1 | 7572.6 | 7375.1 | 7570.3 | 7421.3 |
| OpenVLA-OFT | 11731.3 | 11802.2 | 11768.7 | 11768.5 | 11731.1 | 11735.4 | 11782.7 | 11809.2 |

</details>

## Latency and memory: ggml CPU backend

Rows are sorted by latency. `vision ms` is `-` for archs that do not time their
vision stage separately.

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 2 | 256 + 128 | 48.0 | 50.7 | 50.8 | 51.6 | 5.8 | 622 |
| TurboVLA | 2 | 256 | 199.0 | 231.6 | 233.6 | 237.1 | - | 855 |
| VLA-JEPA | 1 | 256 | 1258.1 | 1265.0 | 1264.8 | 1269.8 | 421.0 | 3823 |
| GR00T N1.7 | 1 | 256 | 1532.5 | 1538.6 | 1537.5 | 1544.0 | 427.7 | 4884 |
| VLA-Adapter | 1 | 224 | 1645.4 | 1665.8 | 1667.4 | 1671.5 | 964.4 | 2707 |
| GR00T N1.6 | 1 | 224 | 1697.3 | 1705.1 | 1703.9 | 1711.2 | 519.0 | 4524 |
| SmolVLA | 2 | 512 | 1758.0 | 1763.2 | 1763.0 | 1767.6 | 1072.9 | 1101 |
| GR00T N1.5 | 1 | 224 | 1979.1 | 1986.3 | 1987.4 | 1991.4 | 522.3 | 3500 |
| Evo-1 | 1 | 448 | 4025.8 | 4121.8 | 4138.3 | 4145.1 | 2051.2 | 1432 |
| π0 | 2 | 224 | 6083.9 | 6160.1 | 6166.6 | 6183.4 | 1050.5 | 5356 |
| π0.5 | 2 | 224 | 6120.3 | 6134.4 | 6133.9 | 6142.2 | 1046.5 | 5657 |
| OpenVLA-OFT | 1 | 224 | 9894.3 | 9989.6 | 9995.9 | 10004.3 | 1017.9 | 14797 |

### Fastest configuration: ggml CPU backend

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | *(defaults)* | 48.0 | 50.7 | 50.8 | 51.6 | 5.8 | 622 | - |
| TurboVLA | `--flash-attn` | 184.8 | 220.4 | 221.7 | 229.2 | - | 850 | -5% |
| VLA-JEPA | `--weight-dtype f16` | 1020.9 | 1026.4 | 1026.3 | 1030.3 | 344.4 | 3823 | -19% |
| GR00T N1.7 | `--weight-dtype f16` | 1251.0 | 1255.7 | 1255.1 | 1258.8 | 348.4 | 4884 | -18% |
| VLA-Adapter | `--weight-dtype f16 --flash-attn` | 1341.4 | 1368.0 | 1370.5 | 1376.3 | 800.2 | 2707 | -18% |
| GR00T N1.6 | `--weight-dtype f16 --flash-attn` | 1363.4 | 1368.0 | 1367.0 | 1371.2 | 392.9 | 4524 | -20% |
| SmolVLA | `--weight-dtype f16 --flash-attn` | 1388.5 | 1399.4 | 1399.2 | 1406.6 | 799.3 | 1050 | -21% |
| GR00T N1.5 | `--weight-dtype f16 --flash-attn` | 1564.6 | 1568.8 | 1568.9 | 1571.5 | 390.4 | 3497 | -21% |
| Evo-1 | `--weight-dtype f16 --flash-attn` | 3256.0 | 3283.5 | 3286.3 | 3296.1 | 1503.2 | 1368 | -20% |
| π0.5 | `--weight-dtype f16` | 4783.6 | 4891.5 | 4902.1 | 4917.8 | 842.3 | 5657 | -20% |
| π0 | `--weight-dtype f16` | 4835.2 | 4919.0 | 4922.9 | 4955.6 | 850.1 | 5356 | -20% |
| OpenVLA-OFT | `--weight-dtype f16 --flash-attn` | 7879.9 | 7922.2 | 7915.3 | 7940.6 | 847.6 | 14797 | -21% |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 50.3 | 49.8 | 50.2 | 48.8 | 51.6 | 49.6 | 51.2 | 50.3 |
| TurboVLA | 208.9 | 199.8 | 246.8 | 204.0 | 299.8 | 288.7 | 247.7 | 237.0 |
| VLA-JEPA | 1441.8 | 1410.4 | 1378.6 | 1319.7 | 1276.6 | 1190.4 | 1137.3 | 1141.6 |
| GR00T N1.7 | 1669.5 | 1644.1 | 1614.5 | 1510.4 | 1540.1 | 1506.7 | 1256.1 | 1298.1 |
| VLA-Adapter | 1687.4 | 1687.1 | 1685.9 | 1678.9 | 1674.8 | 1678.3 | 1413.9 | 1386.9 |
| GR00T N1.6 | 1715.7 | 1843.2 | 1711.5 | 1662.6 | 1696.0 | 1684.6 | 1397.7 | 1376.5 |
| SmolVLA | 1766.0 | 1668.9 | 1770.0 | 1668.5 | 1771.6 | 1668.1 | 1516.7 | 1404.8 |
| GR00T N1.5 | 2027.7 | 2134.8 | 2014.4 | 2000.6 | 2010.7 | 1948.8 | 1629.6 | 1584.2 |
| Evo-1 | 4141.2 | 3926.1 | 4096.0 | 3870.1 | 4143.3 | 3871.3 | 3563.3 | 3311.2 |
| π0 | 6183.7 | 6390.6 | 6170.6 | 6381.4 | 6174.1 | 6383.2 | 4939.1 | 5089.7 |
| π0.5 | 6160.6 | 6150.6 | 6149.3 | 6099.9 | 6102.4 | 6140.8 | 4907.3 | 4933.7 |
| OpenVLA-OFT | 9978.1 | 10030.8 | 9980.8 | 9966.8 | 9960.2 | 9970.2 | 7918.2 | 7917.2 |

</details>

## Not run

### Arc B390 iGPU (OpenVINO GPU plugin)

- **BitVLA**: N/A off CUDA. The only published GGUF is int2-packed, and only CUDA builds load it:

  ```text
  vla(bitvla): int2-packed GGUF requires a CUDA build (VLA_BITVLA_CUDA_KERNELS); use the bf16 GGUF for CPU.
  ```
- **Octo-Small**: not supported on OpenVINO (the README's support matrix lists it as planned). The translator has no conversion for GroupNorm:

  ```text
  GGML OpenVINO backend ov::Exception: Check 'it != m_translator_map.end()' failed at openvino/translate_session.cpp:304:
  FrontEnd API failed with OpConversionFailure:
  Translation for operation type GGML_OP_GROUP_NORM is not implemented.
  vla(octo): obs tokenizer graph compute failed (-1)
  ```

### Arc B390 iGPU (SYCL)

- **BitVLA**: N/A off CUDA. The only published GGUF is int2-packed, and only CUDA builds load it.

### AI Boost NPU (OpenVINO NPU plugin)

- **BitVLA**: N/A off CUDA. The only published GGUF is int2-packed, and only CUDA builds load it.
- **Octo-Small**: not supported on OpenVINO. It fails on the NPU with the same GroupNorm error as on the GPU.

### OpenVINO CPU plugin

- **BitVLA**: N/A off CUDA. The only published GGUF is int2-packed, and only CUDA builds load it.
- **Octo-Small**: not supported on OpenVINO. It fails on the CPU plugin with the same GroupNorm error as on the GPU.

### ggml CPU backend

- **BitVLA**: N/A off CUDA. The only published GGUF is int2-packed, and only CUDA builds load it.

## Notes

OpenVINO compiles each graph on the first predict. The warmup calls absorb it, so none of the latencies above include it, but a process waits for it before its first action. Process wall time minus the 23 calls gives an upper bound on load plus compile, since it also counts reading the GGUF:

| | GPU plugin | NPU plugin | ggml CPU build (no compile) |
|---|--:|--:|--:|
| TurboVLA | 3-4 s | 9-10 s | 1 s |
| VLA-JEPA, VLA-Adapter, GR00T N1.5-N1.7 | 12-33 s | 76-144 s | 4-16 s |
| Evo-1 | 28-29 s | 189-193 s | 1-3 s |
| SmolVLA | 34-36 s | 404-410 s | 1-2 s |
| π0, π0.5 | 43-81 s | 532-572 s | 9-23 s |
| OpenVLA-OFT | 75-112 s | 849-873 s | 54-56 s |

Do not set `GGML_OPENVINO_CACHE_DIR` to skip the compile; [`docs/backend/ov.md`](../backend/ov.md) explains why.

π0's F32 default on the GPU costs 2.5x. With `GGML_OPENVINO_GPU_PRECISION=f16`, π0 runs at 254.6 ms mean (min 253.6) against 625.1 ms, with 15556 MiB of device buffers against 25056, but its gripper then flips early on some inputs (below). [`docs/backend/ov.md`](../backend/ov.md) explains why π0 needs F32 there.

A latency is not a correctness check. π0 is the one arch [`docs/backend/ov.md`](../backend/ov.md) lists as wrong on the NPU, so we compared its actions on 48 fixed inputs (2 views, 16-token prompts) with the ggml CPU build at F32 weights. On the GPU, at its F32 default, dims 0-5 are within 7.3e-4 and the gripper changes sign at the same step as the reference in all 48 chunks. On the NPU, dims 0-5 are within 3.7e-2 (median 8.3e-3), and in 1 of the 48 chunks the gripper flips one step early, a max|Δ| of 2.0. The GPU at F16 does the same: 2.8e-2 on dims 0-5 and one chunk in 48 with an early flip. The NPU has no F32 mode, so π0 is still not one to run there.

The chip runs under a 25 W long-term power limit, and `thermald --adaptive` was running, which can change it. PL1 read 25 W over a 28 s window at every section boundary but one: at the end of the SYCL section it read 23.75 W over a 1 s window. The first timed calls of a process can run at boost power, so on the CPU sections read `mean` or `p50` for sustained latency.

On the ggml CPU backend, Octo-Small and TurboVLA run at one of two speeds, set by the machine's state rather than by the code. In an alternating A/B against `450992c`, the commit of the September 2026 report, `450992c`'s first round gave 49.9 ms and 160.8 ms. Every later round of either commit gave 58.5-59.3 ms and 227.6-228.6 ms. Octo-Small's row above is a fast process (the other two gave 57.2 and 59.8 ms). TurboVLA's row is the slow speed; the September report's 163.0 ms was the fast one.

VLA-Adapter on the OpenVINO CPU plugin is about 5% slower than at `450992c`, which ran with its own `patch_ggml_openvino.py` on the same llama.cpp. In an alternating A/B, `450992c` gave 1755.9 and 1769.0 ms and this tree gave 1840.0 and 1850.1 ms. The vision stage accounts for it: 985-993 ms there against 1105-1112 ms here. TurboVLA did not move in the same A/B (273.0 and 289.0 ms against 275.1 and 287.5 ms).

Some rows have more than 2% spread between processes. The reported row is always the process with the lowest mean:

- On SYCL, π0's three process means were 270.0, 282.5 and 270.7 ms, π0.5's 276.4, 277.0 and 295.5 ms, and OpenVLA-OFT's 464.1, 438.5 and 446.6 ms. π0 with `--act-dtype bf16 --flash-attn` gave 261.4, 244.6 and 272.9 ms.
- On the OpenVINO GPU plugin, the first process ran slow for π0 (903.3 ms, against 633.8 and 625.1) and for OpenVLA-OFT (1749.9 ms, against 1351.5 and 1334.9). These two hold 25056 and 41761 MiB of device buffers out of 62 GiB of shared RAM. VLA-Adapter's processes gave 185.4, 216.8 and 215.6 ms. Its one-process screen gave 232.8 ms at the defaults and 183.3 ms with `--weight-dtype f16 --flash-attn`, but the re-run of that flag set (194.0, 215.7 and 221.9 ms) did not confirm the gain.

Two SYCL fused graphs are opt-in, and the tables above do not include them. Both runs used the report's protocol:

- GR00T N1.7 with `VLA_GR00T_N17_FUSE=1` runs at 114.0 ms mean (min 113.6), against 179.4 ms on the default graph. With `--weight-dtype bf16 --flash-attn` it runs at 111.6 ms.
- SmolVLA's W8A8 path, `VLA_SMOLVLA_W8A8=all` on its fused graph, runs at 52.3 ms mean (min 51.8), against 62.3 ms.

SYCL's fused graphs are not bit-identical to the per-op graph that the escape hatches restore. We measured each model's error against the ggml CPU build at F32 on 48 paired inputs. The fused graph's error was larger by a factor of 1.45 for VLA-Adapter, 1.49 for SmolVLA, 1.86 for GR00T N1.6 and 2.79 for GR00T N1.5 (geometric mean of the per-input ratios). The largest single error was 4.4e-2, on GR00T N1.5, against 8.0e-3 on the per-op graph. These were measured on the same machine at PR #73's head.

`vla_core` links with `-fsycl-device-code-split=per_kernel` and the JIT option `-cl-fp32-correctly-rounded-divide-sqrt` (PR #33). Both are link options, so they also apply to ggml-sycl's kernels in the same binary.

## Reproducing

```bash
# OpenVINO build: iGPU, NPU and OpenVINO CPU plugin
source <openvino 2026.4>/setupvars.sh
cmake -S . -B build-ov -G Ninja -DGGML_OPENVINO=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-ov -j

# one row, for example SmolVLA
GGML_OPENVINO_DEVICE=GPU ./build-ov/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
ZE_ENABLE_ALT_DRIVERS=/lib/x86_64-linux-gnu/libze_intel_npu.so.1 GGML_OPENVINO_DEVICE=NPU \
  ./build-ov/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
GGML_OPENVINO_DEVICE=CPU ./build-ov/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20

# SYCL build: Arc B390 iGPU
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl -G Ninja -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build build-sycl -j
# compute runtime 26.22 + Level Zero loader 1.16.1: Level Zero crashes, use OpenCL
export UR_ADAPTERS_FORCE_LOAD=/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_opencl.so.0
export ONEAPI_DEVICE_SELECTOR=opencl:gpu
./build-sycl/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
# GR00T and Octo also need their env (see Model configuration), for example:
VLA_GR00T_EMBODIMENT=libero_sim ./build-sycl/vla-bench --ckpt gr00tn1d7-libero-object.gguf --images 1 --size 256 --warmup 3 --reps 20

# ggml CPU backend
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpu -j
./build-cpu/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
```
