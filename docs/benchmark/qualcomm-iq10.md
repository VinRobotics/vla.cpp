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
| Commit | `f0b12ae` |
| llama.cpp | `b11223` |
| Date | 2026-11-09 17:58 to 18:28 (UTC+07) |
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
| Octo-Small | 2 | 256 + 128 | 81.2 | 85.6 | 83.6 | 91.1 | 21.9 | 252 | 743 |
| ACT | 1 | 640 x 480 | 147.7 | 153.1 | 152.0 | 157.7 | - | 194 | 470 |
| TurboVLA | 2 | 256 | 184.0 | 191.9 | 191.4 | 197.1 | - | 249 | 1047 |
| VLA-JEPA | 1 | 256 | 490.2 | 498.4 | 497.9 | 505.6 | 199.0 | 184 | 4012 |
| GR00T N1.5 | 1 | 224 | 524.7 | 525.9 | 525.7 | 527.8 | 138.6 | 162 | 3644 |
| SmolVLA | 2 | 512 | 583.5 | 584.4 | 584.2 | 585.2 | 229.9 | 145 | 1194 |
| VLA-Adapter | 1 | 224 | 662.0 | 668.3 | 666.0 | 677.9 | 257.9 | 872 | 3600 |
| GR00T N1.6 | 1 | 224 | 731.7 | 733.7 | 733.1 | 735.1 | 194.5 | 274 | 5691 |
| GR00T N1.7 | 1 | 256 | 739.2 | 742.7 | 741.9 | 746.2 | 197.7 | 191 | 5049 |
| π0.5 | 2 | 224 | 1103.9 | 1106.8 | 1106.1 | 1109.9 | 277.2 | 285 | 5790 |
| OpenVLA-OFT | 1 | 224 | 1121.4 | 1123.0 | 1121.9 | 1126.0 | 262.4 | 844 | 15323 |
| π0 | 2 | 224 | 1127.8 | 1131.2 | 1130.8 | 1134.6 | 276.9 | 285 | 5531 |
| Evo-1 | 1 | 448 | 1650.2 | 1658.4 | 1656.5 | 1666.6 | 821.1 | 319 | 1553 |

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
| Octo-Small | *(defaults)* | 81.2 | 85.6 | 83.6 | 91.1 | 21.9 | 252 | 743 | - |
| ACT | *(defaults)* | 147.7 | 153.1 | 152.0 | 157.7 | - | 194 | 470 | - |
| TurboVLA | *(defaults)* | 184.0 | 191.9 | 191.4 | 197.1 | - | 249 | 1047 | - |
| VLA-JEPA | `--flash-attn --mm-prec default` | 428.6 | 435.0 | 434.1 | 442.6 | 197.6 | 184 | 3971 | -12.7% |
| GR00T N1.5 | *(defaults)* | 524.7 | 525.9 | 525.7 | 527.8 | 138.6 | 162 | 3644 | - |
| SmolVLA | *(defaults)* | 583.5 | 584.4 | 584.2 | 585.2 | 229.9 | 145 | 1194 | - |
| VLA-Adapter | *(defaults)* | 662.0 | 668.3 | 666.0 | 677.9 | 257.9 | 872 | 3600 | - |
| GR00T N1.7 | `--flash-attn --mm-prec default` | 714.1 | 719.2 | 718.5 | 725.5 | 196.6 | 191 | 5081 | -3.2% |
| GR00T N1.6 | *(defaults)* | 731.7 | 733.7 | 733.1 | 735.1 | 194.5 | 274 | 5691 | - |
| π0.5 | *(defaults)* | 1103.9 | 1106.8 | 1106.1 | 1109.9 | 277.2 | 285 | 5790 | - |
| OpenVLA-OFT | *(defaults)* | 1121.4 | 1123.0 | 1121.9 | 1126.0 | 262.4 | 844 | 15323 | - |
| π0 | *(defaults)* | 1127.8 | 1131.2 | 1130.8 | 1134.6 | 276.9 | 285 | 5531 | - |
| Evo-1 | *(defaults)* | 1650.2 | 1658.4 | 1656.5 | 1666.6 | 821.1 | 319 | 1553 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--flash-attn 0` |
|---|--:|--:|--:|--:|--:|
| Octo-Small | 84.5 | 84.9 | 86.0 | 87.9 | 83.1 |
| ACT | 155.5 | 155.2 | 158.4 | 147.8 | 176.9 |
| TurboVLA | 191.1 | 193.3 | 188.9 | 191.5 | 252.8 |
| VLA-JEPA | 501.1 | 441.9 | 504.7 | 439.9 | 598.9 |
| GR00T N1.5 | 525.8 | 525.2 | 526.4 | 525.6 | 809.1 |
| SmolVLA | 584.3 | 584.8 | 585.6 | 584.5 | 2145.5 |
| VLA-Adapter | 677.1 | 667.2 | 673.1 | 674.4 | 950.4 |
| GR00T N1.6 | 739.0 | 736.8 | 737.4 | 741.2 | 897.7 |
| GR00T N1.7 | 749.3 | 727.4 | 744.9 | 723.3 | 851.0 |
| π0.5 | 1125.3 | 1114.0 | 1133.7 | 1115.2 | 2053.2 |
| OpenVLA-OFT | 1127.0 | 1124.8 | 1128.0 | 1121.5 | 1397.1 |
| π0 | 1113.4 | 1113.5 | 1113.9 | 1113.7 | 2070.2 |
| Evo-1 | 1657.1 | 1657.0 | 1656.2 | 1662.9 | 3106.6 |

</details>

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

GR00T N1.6's row runs its SigLIP2 tower with F32 weights and its first three
blocks' matmuls on the CPU, which keeps it within the fidelity bar when served;
the F32 tower is also why its System RAM is higher than N1.5's.

These rows time the defaults; they do not vouch for accuracy. On a served LIBERO
observation, GR00T N1.5 lands above the 2.9e-3 fidelity bar, closer to the
reference than the CPU at F16; see
[backend/hexagon.md](../backend/hexagon.md#serving).

## Reproducing

```bash
scripts/build_linux_hexagon.sh
# copy build-lnx-htp/pkg to the board, install its HTP skel (see backend/hexagon.md), then on the board:
export LD_LIBRARY_PATH=$PWD/pkg/lib

# one row, for example SmolVLA
./pkg/bin/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
```
