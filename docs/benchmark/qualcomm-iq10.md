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
| OpenVLA-OFT | 1 | 224 | 1117.7 | 1120.2 | 1120.4 | 1121.8 | 262.3 | 844 | 15321 |
| π0 | 2 | 224 | 1117.6 | 1120.5 | 1119.9 | 1123.8 | 277.8 | 284 | 5547 |
| Evo-1 | 1 | 448 | 1653.5 | 1661.4 | 1661.2 | 1665.6 | 815.6 | 319 | 1556 |

### Fastest configuration

We screen each model over these runtime flag sets, then re-run the fastest set
under the full protocol:

- the defaults (F16 weights, the Hexagon build's default);
- `--flash-attn` and `--mm-prec default`, alone and together;
- `--flash-attn 0`, because flash attention is on by default in Hexagon builds.

The weight dtype stays F16 in every set. `--weight-dtype f16` is the default,
and `--weight-dtype bf16` is left out: the NPU has no BF16 kernels, so every
matmul would run on the CPU (SmolVLA: 18.7 s against 0.82 s).

The screen runs 2 warmups and 5 reps in one process per flag set. A flag set
replaces the defaults only if its mean is more than 3% lower, both in the screen
and in the full re-run.

Flash attention and the matmul precision can move the action chunk. Success
rates measured at the defaults therefore do not carry over to these rows. For
VLA-JEPA and GR00T N1.7, `--flash-attn` puts their LM back on flash attention,
which the defaults leave off because it is outside the fidelity bar (VLA-JEPA
6.7e-3 on a served LIBERO observation, N1.7 5.4e-3 on fixed inputs).

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
| OpenVLA-OFT | *(defaults)* | 1117.7 | 1120.2 | 1120.4 | 1121.8 | 262.3 | 844 | 15321 | - |
| π0 | *(defaults)* | 1117.6 | 1120.5 | 1119.9 | 1123.8 | 277.8 | 284 | 5547 | - |
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
| OpenVLA-OFT | 1125.5 | 1127.8 | 1123.4 | 1123.5 | 1402.2 |
| π0 | 1113.2 | 1126.1 | 1114.8 | 1126.1 | 2056.3 |
| Evo-1 | 1668.8 | 1669.2 | 1669.8 | 1657.4 | 3116.8 |

\* Octo-Small's defaults screen was re-run: the first run (672.8 ms, with a
241 ms vision stage) overlapped a transient slowdown on the board.

</details>

## Not run

- **BitVLA**: N/A off CUDA. The only published GGUF is int2-packed, and only
  CUDA builds load it.

## Notes

The HTP skel has to be processed for this board and installed in
`/vendor/dsp/cdsp0..3`: its FastRPC ignores `ADSP_LIBRARY_PATH`. See
[backend/hexagon.md](../backend/hexagon.md#linux-iq-10).

GR00T N1.6 runs one view here, as the configuration above fixes. With two views
one batched `MUL_MAT` of its SigLIP2 tower still runs on the CPU; see
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
