# Qualcomm Dragonwing IQ-10: Hexagon NPU

`vla-bench` times `predict()` in-process on synthetic inputs. It measures the
engine only: no transport, no simulator, and no claim about task success.

The graph runs on the Hexagon NPU, and ops the NPU rejects run on the CPU
through vla.cpp's fallback wrapper.

## Test setup

| | |
|---|---|
| Device | Qualcomm Dragonwing IQ-10 (IQX EVK): QCS8797, 18 Oryon cores, Hexagon v81 NPU (one of its four CDSP sessions in use), 60.8 GiB RAM visible |
| Threads | 16 (the default: all hardware threads, capped at 16), for the NPU's CPU fallback |
| Power | CPU governor `performance` |
| OS | Ubuntu 24.04.4 LTS, kernel 6.6.110 |
| Toolchain | `ghcr.io/snapdragon-toolchain/arm64-linux:v0.7`: clang 21.1.8, Hexagon SDK 6.6.0.0 (tools 19.0.07) |
| Commit | `bd166b8` |
| llama.cpp | `b11223` |
| Date | 2026-11-06 10:46 to 11:16 (UTC+07) |
| Build | `scripts/build_linux_hexagon.sh` (servers included); the HTP skel processed for the board and installed in `/vendor/dsp/cdsp0..3` |
| Runtime flags | defaults; the fastest flags per model are in the second table |
| Method | 3 warmups + 20 timed reps per process, 3 processes. The process with the lowest mean is reported, and memory is the peak over all three. |

**Peak RSS** is the process's peak resident set (`getrusage`, via
`/usr/bin/time`). It leaves out the weights, which live in FastRPC buffers
mapped for the NPU. **System RAM** is the largest drop in the system's
`MemAvailable` while the process ran, sampled every 0.5 s; it counts those
buffers. Nothing else ran on the board.

## Model configuration

Each model runs at its native view count and input size, with weights as
shipped. Checkpoints are the `vrfai/*` GGUFs, and ACT's is
`ravediamond/act-aloha-sim-transfer-cube-gguf`; where a repo ships several, we
use the `libero_object` variant. Settings not listed are the defaults.

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
| ACT | `act-aloha-sim-transfer-cube-gguf` | 1 | 640 x 480 | - | 100 | 14 | transformer decoder | 1 |

- **Lang tokens** is the synthetic prompt length. VLA-JEPA adds its 32
  `<embodied>` tokens with `--extra-token 151697 --extra-count 32`.
- **Action dim** is the checkpoint's padded width. LIBERO uses the first 7 dims.
- **GR00T** needs its embodiment set in the environment:
  `VLA_GR00T_EMBODIMENT=new_embodiment` for N1.5, `libero_panda` for N1.6 and
  `libero_sim` for N1.7. **Octo** needs `VLA_OCTO_UNNORM_DATASET=libero_object`.
- **Octo** takes a 256 px primary view and a 128 px wrist view.
- **ACT** is an ALOHA checkpoint (sim transfer cube): no language, a 14-dim
  bimanual state and action, run with `--images 1 --size 640 --height 480`.

## Latency and memory

Rows are sorted by latency. `vision ms` is `-` for archs that do not time their
vision stage separately.

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB | System RAM MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 2 | 256 + 128 | 81.7 | 84.6 | 82.3 | 89.6 | 21.9 | 253 | 728 |
| ACT | 1 | 640 x 480 | 147.8 | 151.2 | 150.7 | 153.6 | - | 193 | 471 |
| TurboVLA | 2 | 256 | 184.6 | 191.8 | 190.3 | 199.4 | - | 249 | 1118 |
| VLA-JEPA | 1 | 256 | 489.7 | 495.8 | 494.7 | 501.6 | 196.4 | 184 | 4023 |
| GR00T N1.5 | 1 | 224 | 530.2 | 532.0 | 531.3 | 534.3 | 138.5 | 162 | 3734 |
| SmolVLA | 2 | 512 | 583.3 | 584.0 | 583.9 | 584.7 | 229.8 | 146 | 1194 |
| VLA-Adapter | 1 | 224 | 664.0 | 668.6 | 666.6 | 673.7 | 257.8 | 872 | 3584 |
| GR00T N1.6 | 1 | 224 | 680.9 | 682.1 | 681.6 | 683.9 | 145.3 | 146 | 4889 |
| GR00T N1.7 | 1 | 256 | 739.6 | 746.4 | 745.1 | 753.8 | 196.4 | 190 | 5215 |
| π0.5 | 2 | 224 | 1114.0 | 1116.1 | 1115.5 | 1117.8 | 278.0 | 285 | 5802 |
| π0 | 2 | 224 | 1117.6 | 1120.5 | 1119.9 | 1123.8 | 277.8 | 284 | 5547 |
| OpenVLA-OFT | 1 | 224 | 1117.7 | 1120.2 | 1120.4 | 1121.8 | 262.3 | 844 | 15321 |
| Evo-1 | 1 | 448 | 1653.5 | 1661.4 | 1661.2 | 1665.6 | 815.6 | 319 | 1556 |

### Fastest configuration

We screen each model over these runtime flag sets, then re-run the fastest set
under the full protocol:

- the defaults (F16 weights, the Hexagon build's default);
- `--flash-attn` and `--mm-prec default`, alone and together;
- `--flash-attn 0`, because flash attention is on by default in Hexagon builds.

The weight dtype stays F16 in every set. `--weight-dtype f16` is the default,
and `--weight-dtype bf16` is left out: the NPU has no BF16 kernels, so every
matmul would run on the CPU (SmolVLA: 18.6 s against 0.58 s, mean, 5 reps).

The screen runs 2 warmups and 5 reps in one process per flag set. A flag set
replaces the defaults only if its mean is more than 3% lower, both in the screen
and in the full re-run.

Flash attention and the matmul precision can move the action chunk. Success
rates measured at the defaults therefore do not carry over to these rows. For
VLA-JEPA and GR00T N1.7, `--flash-attn` puts their LM back on flash attention,
which the defaults leave off because it is outside the fidelity bar against a
CPU-only build: VLA-JEPA 6.2e-3 on a served LIBERO observation (2.7e-3 on fixed
inputs), N1.7 6.2e-3 on fixed inputs (2.7e-3 served).

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB | System RAM MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | *(defaults)* | 81.7 | 84.6 | 82.3 | 89.6 | 21.9 | 253 | 728 | - |
| ACT | *(defaults)* | 147.8 | 151.2 | 150.7 | 153.6 | - | 193 | 471 | - |
| TurboVLA | *(defaults)* | 184.6 | 191.8 | 190.3 | 199.4 | - | 249 | 1118 | - |
| VLA-JEPA | `--flash-attn` | 427.8 | 434.2 | 432.6 | 441.7 | 196.1 | 184 | 3971 | -12.4% |
| GR00T N1.5 | *(defaults)* | 530.2 | 532.0 | 531.3 | 534.3 | 138.5 | 162 | 3734 | - |
| SmolVLA | *(defaults)* | 583.3 | 584.0 | 583.9 | 584.7 | 229.8 | 146 | 1194 | - |
| VLA-Adapter | *(defaults)* | 664.0 | 668.6 | 666.6 | 673.7 | 257.8 | 872 | 3584 | - |
| GR00T N1.6 | *(defaults)* | 680.9 | 682.1 | 681.6 | 683.9 | 145.3 | 146 | 4889 | - |
| GR00T N1.7 | `--flash-attn --mm-prec default` | 715.1 | 719.2 | 718.7 | 722.9 | 195.5 | 190 | 5074 | -3.6% |
| π0.5 | *(defaults)* | 1114.0 | 1116.1 | 1115.5 | 1117.8 | 278.0 | 285 | 5802 | - |
| π0 | *(defaults)* | 1117.6 | 1120.5 | 1119.9 | 1123.8 | 277.8 | 284 | 5547 | - |
| OpenVLA-OFT | *(defaults)* | 1117.7 | 1120.2 | 1120.4 | 1121.8 | 262.3 | 844 | 15321 | - |
| Evo-1 | *(defaults)* | 1653.5 | 1661.4 | 1661.2 | 1665.6 | 815.6 | 319 | 1556 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--flash-attn 0` |
|---|--:|--:|--:|--:|--:|
| Octo-Small | 90.8* | 83.0 | 83.2 | 85.3 | 88.1 |
| ACT | 154.4 | 158.4 | 154.7 | 153.2 | 179.9 |
| TurboVLA | 192.0 | 192.6 | 194.0 | 191.9 | 257.3 |
| VLA-JEPA | 498.2 | 431.3 | 494.0 | 439.8 | 602.4 |
| GR00T N1.5 | 534.0 | 526.4 | 527.0 | 526.4 | 811.6 |
| SmolVLA | 584.4 | 584.7 | 585.2 | 585.4 | 2145.2 |
| VLA-Adapter | 671.6 | 670.4 | 669.4 | 671.9 | 950.8 |
| GR00T N1.6 | 689.2 | 685.8 | 687.3 | 683.0 | 851.0 |
| GR00T N1.7 | 758.3 | 733.1 | 752.8 | 723.2 | 858.2 |
| π0.5 | 1125.2 | 1117.6 | 1134.2 | 1134.2 | 2039.2 |
| π0 | 1113.2 | 1126.1 | 1114.8 | 1126.1 | 2056.3 |
| OpenVLA-OFT | 1125.5 | 1127.8 | 1123.4 | 1123.5 | 1402.2 |
| Evo-1 | 1668.8 | 1669.2 | 1669.8 | 1657.4 | 3116.8 |

\* Octo-Small's defaults screen was re-run: the first run (672.8 ms, with a
241 ms vision stage) overlapped a transient slowdown on the board.

</details>

## Q8_0 weights

These rows run GGUFs that `scripts/quantize_gguf.py --type Q8_0` made from the
checkpoints above (ggml's block-32 Q8_0 weights). The setup is the one above,
except:

| | |
|---|---|
| Commit | `71cdb09` |
| Date | 2026-11-11 10:10 to 11:49 (UTC+07) for latency, memory and served fidelity; fixed-input fidelity 2026-11-10 |
| Runtime flags | defaults |

Each configuration packs the weights of the one before it, plus:

| Config | `quantize_gguf.py` flags | Packs |
|---|---|---|
| F16 | - | none: the shipped file, loaded as F16 (`--weight-dtype f16`, the default) |
| default | `--type Q8_0` | the 2-D weights outside the vision tower and the action head. Embeddings, the output head, norms, patch embeddings and position tables stay float in every configuration |
| vision, unpadded | `--vision` | the vision tower |
| vision | `--vision --pad-m32` | the vision tower, with FFN widths that are not a multiple of 32 zero-padded (SigLIP's 4304 to 4320 in π0, π0.5, GR00T N1.5 and N1.6, VLA-Adapter and OpenVLA-OFT) |
| vision + DiT | `--vision --pad-m32 --dit` | the DiT's weight matrices, except adaLN and the time embedding |
| ..., keep-float (*kind*, *n*) | the flags of the named configuration plus `--keep-float REGEX` | that configuration with *n* weights left float |

The `--keep-float` lists come from a greedy search: each group of blocks (a
quarter of a stack, attention or FFN) is packed alone and measured, the groups
that move the action chunk most are left float until the deviation is under the
target, then blocks are packed back one at a time while it stays under. *fixed*
uses the fixed-input deviation and *served* the served one, with a target of
2.9e-3; GR00T N1.5 and N1.6 take 1.1 times their F16 served deviation instead
(1.3e-2, 1.1e-2). *union* leaves both lists float.

`fixed max|Δ|` is measured as in [Fidelity](../backend/hexagon.md#fidelity) and
`served max|Δ|` as in [Serving](../backend/hexagon.md#serving), against the
CPU-only build at F32.

| Model | Config | Q8_0 tensors | GGUF GiB | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB | System RAM MiB | mean vs F16 | RAM vs F16 | fixed max\|Δ\| | served max\|Δ\| |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | F16 | - | 0.51 | 81.5 | 84.1 | 82.3 | 88.3 | 20.7 | 252 | 725 | - | - | 5.0e-4 | 1.4e-4 |
| Octo-Small | default | 121 / 318 | 0.22 | 79.5 | 81.6 | 80.7 | 85.5 | 20.4 | 258 | 405 | -3.0% | -44.1% | 2.5e-3 | 2.0e-3 |
| Octo-Small | vision | 123 / 318 | 0.22 | 80.2 | 83.5 | 81.1 | 90.3 | 21.6 | 258 | 402 | -0.7% | -44.6% | 1.9e-3 | 1.9e-3 |
| ACT | F16 | - | 0.13 | 147.0 | 152.5 | 152.3 | 156.0 | - | 194 | 469 | - | - | 6.1e-4 | 1.0e-3 |
| ACT | default | 35 / 147 | 0.06 | 145.9 | 150.9 | 150.9 | 152.8 | - | 192 | 405 | -1.0% | -13.6% | 2.1e-3 | 2.4e-3 |
| TurboVLA | F16 | - | 0.80 | 186.6 | 190.2 | 189.1 | 195.3 | - | 248 | 1079 | - | - | 3.7e-4 | 8.5e-4 |
| TurboVLA | default | 133 / 657 | 0.53 | 184.8 | 186.5 | 186.4 | 188.2 | - | 248 | 814 | -1.9% | -24.6% | 1.3e-2 | 4.2e-3 |
| TurboVLA | vision | 208 / 657 | 0.30 | 182.6 | 185.7 | 184.1 | 190.7 | - | 245 | 485 | -2.4% | -55.1% | 1.4e-2 | 8.3e-3 |
| TurboVLA | vision, keep-float (fixed, 32) | 176 / 657 | 0.39 | 181.2 | 185.8 | 183.6 | 191.4 | - | 245 | 611 | -2.3% | -43.4% | 2.6e-3 | 6.9e-3 |
| VLA-JEPA | F16 | - | 4.25 | 489.1 | 496.8 | 494.6 | 506.4 | 197.8 | 184 | 4018 | - | - | 2.1e-3 | 2.2e-3 |
| VLA-JEPA | default | 196 / 873 | 3.02 | 425.6 | 432.1 | 430.3 | 437.5 | 199.1 | 184 | 2718 | -13.0% | -32.4% | 1.4e-2 | 1.0e-1 |
| VLA-JEPA | vision | 300 / 873 | 2.67 | 404.8 | 411.6 | 410.8 | 416.8 | 178.4 | 96 | 2350 | -17.1% | -41.5% | 1.5e-2 | 1.6e-1 |
| VLA-JEPA | vision + DiT | 396 / 873 | 2.56 | 387.0 | 393.5 | 392.9 | 400.2 | 179.2 | 97 | 2288 | -20.8% | -43.1% | 1.3e-2 | 1.3e-1 |
| VLA-JEPA | vision + DiT, keep-float (fixed, 233) | 163 / 873 | 3.88 | 456.6 | 461.4 | 459.9 | 467.7 | 185.0 | 184 | 3614 | -7.1% | -10.1% | 2.9e-3 | 8.3e-3 |
| VLA-JEPA | vision + DiT, keep-float (served, 143) | 253 / 873 | 3.38 | 430.1 | 436.6 | 434.2 | 444.7 | 182.3 | 184 | 3117 | -12.1% | -22.4% | 8.4e-3 | 2.1e-3 |
| VLA-JEPA | vision + DiT, keep-float (union, 244) | 152 / 873 | 3.93 | 452.6 | 461.6 | 460.7 | 469.3 | 185.3 | 184 | 3679 | -7.1% | -8.4% | 3.7e-3 | 5.1e-3 |
| GR00T N1.5 | F16 | - | 6.46 | 527.3 | 529.3 | 528.9 | 531.3 | 138.7 | 161 | 3659 | - | - | 1.9e-3 | 1.2e-2 |
| GR00T N1.5 | default | 85 / 887 | 5.93 | 499.5 | 500.2 | 500.0 | 501.3 | 138.3 | 161 | 3148 | -5.5% | -14.0% | 5.1e-3 | 8.0e-2 |
| GR00T N1.5 | vision | 247 / 887 | 5.57 | 479.1 | 481.1 | 479.7 | 483.8 | 118.8 | 161 | 2754 | -9.1% | -24.7% | 7.2e-3 | 4.6e-2 |
| GR00T N1.5 | vision + DiT | 345 / 887 | 4.28 | 398.7 | 399.2 | 399.1 | 399.5 | 117.9 | 161 | 2339 | -24.6% | -36.1% | 8.4e-3 | 3.6e-2 |
| GR00T N1.5 | vision + DiT, keep-float (fixed, 191) | 154 / 887 | 5.57 | 448.3 | 449.0 | 448.7 | 450.0 | 131.8 | 162 | 3061 | -15.2% | -16.3% | 2.7e-3 | 2.8e-2 |
| GR00T N1.5 | vision + DiT, keep-float (served, 223) | 122 / 887 | 5.86 | 471.7 | 472.9 | 472.6 | 474.0 | 133.1 | 161 | 3172 | -10.7% | -13.3% | 6.8e-3 | 9.1e-3 |
| GR00T N1.5 | vision + DiT, keep-float (union, 272) | 73 / 887 | 6.27 | 489.0 | 491.2 | 489.7 | 494.8 | 135.1 | 161 | 3413 | -7.2% | -6.7% | 3.3e-3 | 1.1e-2 |
| SmolVLA | F16 | - | 1.05 | 583.4 | 584.0 | 583.8 | 584.1 | 229.9 | 146 | 1226 | - | - | 7.4e-4 | 1.9e-3 |
| SmolVLA | default | 225 / 791 | 0.76 | 568.2 | 569.0 | 568.6 | 569.8 | 228.8 | 104 | 919 | -2.6% | -25.0% | 1.2e-2 | 1.3e-2 |
| SmolVLA | vision | 297 / 791 | 0.69 | 560.4 | 561.0 | 560.9 | 561.6 | 221.3 | 104 | 848 | -3.9% | -30.8% | 3.4e-2 | 1.6e-2 |
| SmolVLA | vision, keep-float (fixed, 58) | 239 / 791 | 0.73 | 563.5 | 564.2 | 564.1 | 564.5 | 222.9 | 104 | 896 | -3.4% | -26.9% | 2.0e-3 | 1.6e-2 |
| VLA-Adapter | F16 | - | 2.59 | 664.7 | 669.2 | 666.6 | 677.5 | 257.6 | 872 | 3569 | - | - | 3.9e-4 | 3.9e-4 |
| VLA-Adapter | default | 171 / 1455 | 2.25 | 649.4 | 653.5 | 651.8 | 658.0 | 256.6 | 872 | 3200 | -2.3% | -10.3% | 4.3e-3 | 6.3e-3 |
| VLA-Adapter | vision | 367 / 1455 | 1.65 | 617.9 | 623.4 | 622.1 | 629.0 | 226.5 | 852 | 2555 | -6.8% | -28.4% | 3.1e-3 | 6.0e-3 |
| VLA-Adapter | vision, keep-float (fixed, 8) | 359 / 1455 | 1.66 | 618.2 | 622.7 | 622.6 | 626.9 | 226.5 | 852 | 2516 | -6.9% | -29.5% | 2.8e-3 | 6.3e-3 |
| VLA-Adapter | vision, keep-float (served, 40) | 327 / 1455 | 1.78 | 623.6 | 627.7 | 625.5 | 632.3 | 227.1 | 849 | 2494 | -6.2% | -30.1% | 2.9e-3 | 2.9e-3 |
| GR00T N1.6 | F16 | - | 8.53 | 681.0 | 682.6 | 682.3 | 684.1 | 145.0 | 146 | 4929 | - | - | 1.3e-3 | 1.0e-2 |
| GR00T N1.6 | default | 114 / 1094 | 7.44 | 643.4 | 644.5 | 643.9 | 646.6 | 144.2 | 134 | 3959 | -5.6% | -19.7% | 7.2e-3 | 3.4e-2 |
| GR00T N1.6 | vision, unpadded | 249 / 1094 | 7.20 | 815.6 | 817.3 | 817.3 | 818.5 | 316.7 | 134 | 3738 | +19.7% | -24.2% | 7.1e-3 | 2.5e-2 |
| GR00T N1.6 | vision | 276 / 1094 | 7.08 | 621.6 | 622.5 | 622.1 | 623.2 | 124.2 | 134 | 3588 | -8.8% | -27.2% | 8.0e-3 | 2.9e-2 |
| GR00T N1.6 | vision + DiT | 470 / 1094 | 4.52 | 461.4 | 462.2 | 461.9 | 462.9 | 124.2 | 116 | 2777 | -32.3% | -43.7% | 8.1e-3 | 2.9e-2 |
| GR00T N1.6 | vision + DiT, keep-float (fixed, 57) | 413 / 1094 | 5.26 | 481.7 | 482.3 | 482.2 | 482.7 | 125.5 | 146 | 3076 | -29.3% | -37.6% | 2.8e-3 | 3.6e-2 |
| GR00T N1.6 | vision + DiT, keep-float (served, 3) | 467 / 1094 | 4.55 | 463.2 | 464.0 | 463.6 | 465.0 | 124.3 | 146 | 2800 | -32.0% | -43.2% | 7.6e-3 | 9.2e-3 |
| GR00T N1.6 | vision + DiT, keep-float (union, 60) | 410 / 1094 | 5.29 | 480.1 | 480.7 | 480.5 | 481.2 | 125.2 | 147 | 3114 | -29.6% | -36.8% | 3.0e-3 | 8.6e-3 |
| GR00T N1.7 | F16 | - | 5.86 | 740.1 | 744.9 | 745.0 | 749.0 | 196.9 | 190 | 5108 | - | - | 2.0e-3 | 2.0e-3 |
| GR00T N1.7 | default | 112 / 1030 | 5.16 | 699.0 | 703.5 | 701.5 | 709.3 | 196.1 | 190 | 4356 | -5.6% | -14.7% | 1.1e-2 | 2.0e-2 |
| GR00T N1.7 | vision | 216 / 1030 | 4.81 | 683.3 | 688.4 | 686.0 | 694.1 | 179.5 | 190 | 3996 | -7.6% | -21.8% | 2.0e-2 | 2.2e-2 |
| GR00T N1.7 | vision + DiT | 410 / 1030 | 3.99 | 528.8 | 536.1 | 534.6 | 542.1 | 181.6 | 191 | 3151 | -28.0% | -38.3% | 2.8e-2 | 2.1e-2 |
| GR00T N1.7 | vision + DiT, keep-float (fixed, 185) | 225 / 1030 | 4.90 | 599.2 | 606.0 | 605.4 | 611.7 | 186.9 | 191 | 4051 | -18.6% | -20.7% | 2.6e-3 | 5.3e-3 |
| GR00T N1.7 | vision + DiT, keep-float (served, 157) | 253 / 1030 | 4.68 | 573.7 | 579.8 | 578.0 | 590.8 | 187.3 | 190 | 3870 | -22.2% | -24.2% | 4.2e-2 | 2.8e-3 |
| GR00T N1.7 | vision + DiT, keep-float (union, 233) | 177 / 1030 | 5.06 | 609.4 | 614.4 | 614.1 | 619.1 | 188.2 | 190 | 4249 | -17.5% | -16.8% | 3.9e-3 | 6.5e-3 |
| π0.5 | F16 | - | 11.72 | 1106.0 | 1107.8 | 1107.0 | 1111.5 | 277.0 | 284 | 5980 | - | - | 4.8e-4 | 3.7e-4 |
| π0.5 | default | 127 / 819 | 6.30 | 969.3 | 971.0 | 969.8 | 974.7 | 277.3 | 126 | 3997 | -12.3% | -33.2% | 4.9e-4 | 1.9e-3 |
| π0.5 | vision, unpadded | 262 / 819 | 6.06 | 1318.2 | 1320.2 | 1318.9 | 1324.5 | 621.0 | 126 | 3760 | +19.2% | -37.1% | 8.5e-4 | 2.7e-3 |
| π0.5 | vision | 289 / 819 | 5.94 | 939.6 | 941.3 | 940.8 | 943.1 | 236.7 | 103 | 3676 | -15.0% | -38.5% | 9.2e-4 | 2.8e-3 |
| π0 | F16 | - | 6.04 | 1118.9 | 1123.3 | 1123.7 | 1125.3 | 276.9 | 284 | 5662 | - | - | 7.7e-4 | 5.8e-4 |
| π0 | default | 127 / 780 | 4.31 | 976.4 | 978.8 | 977.8 | 981.2 | 277.0 | 126 | 3723 | -12.9% | -34.2% | 1.8e-2 | 1.7e-3 |
| π0 | vision | 289 / 780 | 3.95 | 935.7 | 937.6 | 937.3 | 938.8 | 235.9 | 103 | 3337 | -16.5% | -41.1% | 1.6e-2 | 1.7e-3 |
| π0 | default, keep-float (fixed, 8) | 119 / 780 | 4.32 | 977.0 | 978.8 | 978.3 | 980.2 | 276.9 | 126 | 3711 | -12.9% | -34.5% | 1.4e-3 | 5.2e-4 |
| OpenVLA-OFT | F16 | - | 14.03 | 1122.2 | 1125.2 | 1125.0 | 1127.6 | 262.7 | 844 | 15266 | - | - | 3.1e-4 | 7.5e-4 |
| OpenVLA-OFT | default | 227 / 958 | 8.32 | 843.5 | 845.7 | 844.8 | 848.9 | 259.5 | 844 | 9403 | -24.8% | -38.4% | 5.9e-4 | 3.5e-3 |
| OpenVLA-OFT | vision | 423 / 958 | 7.72 | 810.1 | 812.5 | 810.6 | 816.4 | 228.7 | 833 | 8821 | -27.8% | -42.2% | 7.9e-4 | 3.2e-3 |
| OpenVLA-OFT | vision, keep-float (served, 12) | 411 / 958 | 7.86 | 818.7 | 821.4 | 820.4 | 824.8 | 230.0 | 839 | 8959 | -27.0% | -41.3% | 7.9e-4 | 2.5e-3 |
| Evo-1 | F16 | - | 1.45 | 1640.1 | 1659.0 | 1653.3 | 1675.4 | 818.6 | 318 | 1582 | - | - | 6.2e-4 | 1.1e-3 |
| Evo-1 | default | 100 / 636 | 1.26 | 1643.5 | 1655.4 | 1653.6 | 1660.8 | 820.1 | 312 | 1459 | -0.2% | -7.8% | 5.4e-3 | 1.2e-2 |
| Evo-1 | vision | 196 / 636 | 1.00 | 1620.6 | 1638.8 | 1639.0 | 1645.7 | 808.1 | 312 | 1121 | -1.2% | -29.1% | 6.3e-3 | 1.4e-2 |
| Evo-1 | vision, keep-float (fixed, 25) | 171 / 636 | 1.05 | 1624.7 | 1634.5 | 1632.9 | 1647.6 | 803.5 | 319 | 1171 | -1.5% | -26.0% | 2.4e-3 | 1.1e-2 |

A per-op profile of the default, vision and vision + DiT files puts every Q8_0
matmul on the HMX. The loader zero-pads a 4304-wide FFN to 4320 at load only
when its weights are float, so without `--pad-m32` the vision tower's FFNs of
π0.5 and GR00T N1.6 keep their 4304 width (rows *vision, unpadded*).

Fixed-input deviation of the default configuration on the NPU and on the
CPU-only build, against the CPU-only build at F32:

| Model | NPU | CPU-only build |
|---|--:|--:|
| π0 | 1.8e-2 | 1.9e-2 |
| π0.5 | 4.9e-4 | 1.1e-3 |
| SmolVLA | 1.2e-2 | 1.8e-2 |
| Evo-1 | 5.4e-3 | 7.3e-3 |
| GR00T N1.5 | 5.1e-3 | 1.2e-2 |
| GR00T N1.6 | 7.2e-3 | 1.2e-2 |
| GR00T N1.7 | 1.1e-2 | 4.4e-2 |
| VLA-JEPA | 1.4e-2 | 2.7e-2 |
| VLA-Adapter | 4.4e-3 | 6.2e-3 |
| TurboVLA | 1.3e-2 | 2.1e-2 |

## Not run

- **BitVLA**: N/A off CUDA. The only published GGUF is int2-packed, and only
  CUDA builds load it.

## Notes

The HTP skel has to be processed for this board and installed in
`/vendor/dsp/cdsp0..3`: its FastRPC ignores `ADSP_LIBRARY_PATH`. See
[backend/hexagon.md](../backend/hexagon.md#linux-iq-10).

GR00T N1.6 runs one view here, as the configuration above fixes. With two views
its SigLIP2 tower's patch matmul and flash attention, batched over the views,
still run on the CPU; see
[backend/hexagon.md](../backend/hexagon.md#op-coverage-measured).

These rows time the defaults; they do not vouch for accuracy. On a served LIBERO
observation, GR00T N1.5 and N1.6 land above the 2.9e-3 fidelity bar: N1.5
closer to the reference than the CPU at F16, N1.6 slightly further; see
[backend/hexagon.md](../backend/hexagon.md#serving).

## Reproducing

```bash
scripts/build_linux_hexagon.sh
# copy build-lnx-htp/pkg to the board, install its HTP skel (see backend/hexagon.md), then on the board:
export LD_LIBRARY_PATH=$PWD/pkg/lib

# one row, for example SmolVLA
./pkg/bin/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
```
