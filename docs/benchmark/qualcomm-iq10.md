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
| Commit | `7c76ea9`; ACT at `0f3e010` |
| llama.cpp | `b11223` |
| Date | 2026-11-05 14:26 to 14:57 (UTC+07); ACT 17:00 to 17:01 |
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
| Octo-Small | 2 | 256 + 128 | 117.6 | 119.4 | 118.4 | 120.7 | 21.2 | 253 | 723 |
| ACT | 1 | 640 x 480 | 148.5 | 152.9 | 152.5 | 157.3 | - | 194 | 467 |
| TurboVLA | 2 | 256 | 188.1 | 190.8 | 188.6 | 194.9 | - | 249 | 1075 |
| VLA-JEPA | 1 | 256 | 505.4 | 512.6 | 510.1 | 522.2 | 196.4 | 184 | 3969 |
| SmolVLA | 2 | 512 | 583.6 | 584.4 | 584.4 | 585.1 | 229.9 | 145 | 1196 |
| GR00T N1.5 | 1 | 224 | 591.6 | 594.9 | 593.4 | 597.1 | 138.9 | 162 | 3639 |
| GR00T N1.6 | 1 | 224 | 696.5 | 699.3 | 699.2 | 702.2 | 145.1 | 147 | 4692 |
| VLA-Adapter | 1 | 224 | 703.2 | 708.4 | 708.2 | 715.3 | 258.1 | 872 | 3566 |
| GR00T N1.7 | 1 | 256 | 759.5 | 766.3 | 766.0 | 773.4 | 197.6 | 191 | 5113 |
| π0 | 2 | 224 | 1114.0 | 1116.6 | 1116.4 | 1118.2 | 277.1 | 284 | 5537 |
| π0.5 | 2 | 224 | 1113.0 | 1117.3 | 1116.6 | 1120.7 | 277.9 | 285 | 5826 |
| OpenVLA-OFT | 1 | 224 | 1720.3 | 1725.5 | 1724.7 | 1731.0 | 263.4 | 844 | 15451 |
| Evo-1 | 1 | 448 | 2333.6 | 2348.3 | 2349.9 | 2360.1 | 820.6 | 319 | 1757 |

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
| Octo-Small | *(defaults)* | 117.6 | 119.4 | 118.4 | 120.7 | 21.2 | 253 | 723 | - |
| ACT | *(defaults)* | 148.5 | 152.9 | 152.5 | 157.3 | - | 194 | 467 | - |
| TurboVLA | *(defaults)* | 188.1 | 190.8 | 188.6 | 194.9 | - | 249 | 1075 | - |
| VLA-JEPA | `--flash-attn --mm-prec default` | 442.1 | 452.4 | 450.2 | 460.6 | 198.3 | 184 | 3985 | -11.7% |
| SmolVLA | *(defaults)* | 583.6 | 584.4 | 584.4 | 585.1 | 229.9 | 145 | 1196 | - |
| GR00T N1.5 | *(defaults)* | 591.6 | 594.9 | 593.4 | 597.1 | 138.9 | 162 | 3639 | - |
| GR00T N1.6 | *(defaults)* | 696.5 | 699.3 | 699.2 | 702.2 | 145.1 | 147 | 4692 | - |
| VLA-Adapter | *(defaults)* | 703.2 | 708.4 | 708.2 | 715.3 | 258.1 | 872 | 3566 | - |
| GR00T N1.7 | `--flash-attn --mm-prec default` | 727.8 | 734.1 | 733.2 | 742.4 | 196.2 | 191 | 5081 | -4.2% |
| π0 | *(defaults)* | 1114.0 | 1116.6 | 1116.4 | 1118.2 | 277.1 | 284 | 5537 | - |
| π0.5 | *(defaults)* | 1113.0 | 1117.3 | 1116.6 | 1120.7 | 277.9 | 285 | 5826 | - |
| OpenVLA-OFT | *(defaults)* | 1720.3 | 1725.5 | 1724.7 | 1731.0 | 263.4 | 844 | 15451 | - |
| Evo-1 | *(defaults)* | 2333.6 | 2348.3 | 2349.9 | 2360.1 | 820.6 | 319 | 1757 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--flash-attn 0` |
|---|--:|--:|--:|--:|--:|
| Octo-Small | 121.8 | 120.4 | 121.8 | 123.2 | 123.8 |
| ACT | 143.4 | 153.4 | 157.5 | 155.8 | 174.8 |
| TurboVLA | 196.5 | 189.3 | 192.3 | 190.9 | 257.4 |
| VLA-JEPA | 514.8 | 454.3 | 512.0 | 452.9 | 1662.4 |
| SmolVLA | 584.6 | 585.0 | 585.2 | 584.6 | 2147.2 |
| GR00T N1.5 | 594.7 | 592.6 | 600.1 | 601.0 | 877.2 |
| GR00T N1.6 | 709.6 | 714.5 | 711.1 | 713.1 | 875.3 |
| VLA-Adapter | 712.4 | 711.1 | 713.7 | 712.8 | 989.2 |
| GR00T N1.7 | 765.5 | 739.0 | 758.3 | 733.1 | 865.3 |
| π0 | 1131.6 | 1138.1 | 1118.0 | 1128.8 | 2051.6 |
| π0.5 | 1125.8 | 1109.1 | 1134.2 | 1134.6 | 2036.0 |
| OpenVLA-OFT | 1737.9 | 1722.4 | 1715.7 | 1718.5 | 2005.8 |
| Evo-1 | 2345.2 | 2352.4 | 2355.4 | 2361.4 | 3877.1 |

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
