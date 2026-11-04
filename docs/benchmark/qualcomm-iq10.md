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
| Commit | `1cab1c3` |
| llama.cpp | `b11223` |
| Date | 2026-11-04 11:52 to 12:30 (UTC+07) |
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

## Latency and memory

Rows are sorted by latency. `vision ms` is `-` for archs that do not time their
vision stage separately.

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB | System RAM MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 2 | 256 + 128 | 121.6 | 123.0 | 122.3 | 125.8 | 20.5 | 253 | 804 |
| TurboVLA | 2 | 256 | 428.1 | 433.0 | 431.6 | 436.8 | - | 438 | 1327 |
| VLA-JEPA | 1 | 256 | 440.2 | 446.3 | 444.7 | 450.2 | 195.2 | 185 | 3971 |
| GR00T N1.5 | 1 | 224 | 613.7 | 615.7 | 615.7 | 618.0 | 148.3 | 162 | 3636 |
| SmolVLA | 2 | 512 | 631.0 | 632.4 | 631.7 | 634.6 | 281.8 | 146 | 1190 |
| GR00T N1.6 | 1 | 224 | 705.8 | 708.5 | 708.0 | 710.7 | 145.0 | 146 | 4869 |
| GR00T N1.7 | 1 | 256 | 747.0 | 756.0 | 755.0 | 764.5 | 197.2 | 191 | 5090 |
| π0.5 | 2 | 224 | 1117.4 | 1121.0 | 1120.8 | 1123.3 | 292.5 | 285 | 5790 |
| π0 | 2 | 224 | 1124.4 | 1128.7 | 1128.6 | 1130.5 | 293.0 | 285 | 5534 |
| VLA-Adapter | 1 | 224 | 1733.6 | 1752.3 | 1749.5 | 1765.6 | 1187.2 | 872 | 3566 |
| Evo-1 | 1 | 448 | 2579.1 | 2595.5 | 2592.1 | 2611.9 | 854.0 | 319 | 1551 |
| OpenVLA-OFT | 1 | 224 | 4097.7 | 4107.8 | 4107.8 | 4114.2 | 1187.3 | 844 | 15327 |

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
rates measured at the defaults therefore do not carry over to these rows.

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB | System RAM MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | *(defaults)* | 121.6 | 123.0 | 122.3 | 125.8 | 20.5 | 253 | 804 | - |
| TurboVLA | *(defaults)* | 428.1 | 433.0 | 431.6 | 436.8 | - | 438 | 1327 | - |
| VLA-JEPA | *(defaults)* | 440.2 | 446.3 | 444.7 | 450.2 | 195.2 | 185 | 3971 | - |
| GR00T N1.5 | *(defaults)* | 613.7 | 615.7 | 615.7 | 618.0 | 148.3 | 162 | 3636 | - |
| SmolVLA | *(defaults)* | 631.0 | 632.4 | 631.7 | 634.6 | 281.8 | 146 | 1190 | - |
| GR00T N1.6 | *(defaults)* | 705.8 | 708.5 | 708.0 | 710.7 | 145.0 | 146 | 4869 | - |
| GR00T N1.7 | *(defaults)* | 747.0 | 756.0 | 755.0 | 764.5 | 197.2 | 191 | 5090 | - |
| π0.5 | *(defaults)* | 1117.4 | 1121.0 | 1120.8 | 1123.3 | 292.5 | 285 | 5790 | - |
| π0 | *(defaults)* | 1124.4 | 1128.7 | 1128.6 | 1130.5 | 293.0 | 285 | 5534 | - |
| VLA-Adapter | *(defaults)* | 1733.6 | 1752.3 | 1749.5 | 1765.6 | 1187.2 | 872 | 3566 | - |
| Evo-1 | *(defaults)* | 2579.1 | 2595.5 | 2592.1 | 2611.9 | 854.0 | 319 | 1551 | - |
| OpenVLA-OFT | *(defaults)* | 4097.7 | 4107.8 | 4107.8 | 4114.2 | 1187.3 | 844 | 15327 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--flash-attn 0` |
|---|--:|--:|--:|--:|--:|
| Octo-Small | 125.3 | 128.8 | 124.4 | 123.5 | 124.6 |
| TurboVLA | 438.8 | 436.5 | 439.3 | 438.2 | 535.1 |
| VLA-JEPA | 446.7 | 445.8 | 444.8 | 445.2 | 817.8 |
| GR00T N1.5 | 615.8 | 619.3 | 615.3 | 618.4 | 1096.5 |
| SmolVLA | 632.1 | 633.0 | 632.5 | 632.6 | 2834.8 |
| GR00T N1.6 | 714.2 | 712.1 | 718.7 | 714.9 | 885.7 |
| GR00T N1.7 | 765.2 | 742.0 | 761.5 | 743.6 | 1081.0 |
| π0.5 | 1121.1 | 1121.9 | 1120.5 | 1121.4 | 2062.9 |
| π0 | 1129.8 | 1128.8 | 1131.8 | 1128.5 | 2090.5 |
| VLA-Adapter | 1766.1 | 1760.1 | 1756.1 | 1757.2 | 1767.1 |
| Evo-1 | 2613.0 | 2613.7 | 2617.3 | 2617.9 | 4840.1 |
| OpenVLA-OFT | 4110.3 | 4101.9 | 4101.7 | 4108.2 | 4107.9 |

</details>

## Not run

- **BitVLA**: N/A off CUDA. The only published GGUF is int2-packed, and only
  CUDA builds load it.

## Notes

The HTP skel has to be processed for this board and installed in
`/vendor/dsp/cdsp0..3`: its FastRPC ignores `ADSP_LIBRARY_PATH`. See
[backend/hexagon.md](../backend/hexagon.md#linux-iq-10).

GR00T N1.6 runs one view here, as the configuration above fixes. With two views
its SigLIP2 tower hits the batched `MUL_MAT` the wrapper sends to the CPU; see
[backend/hexagon.md](../backend/hexagon.md#op-coverage-measured).

These rows time the defaults; they do not vouch for accuracy. On a served LIBERO
observation, GR00T N1.5, N1.6 and VLA-JEPA land above the 2.9e-3 fidelity bar:
N1.5 and VLA-JEPA closer to the reference than the CPU at F16, N1.6 slightly
further; see [backend/hexagon.md](../backend/hexagon.md#serving).

## Reproducing

```bash
scripts/build_linux_hexagon.sh
# copy build-lnx-htp/pkg to the board, install its HTP skel (see backend/hexagon.md), then on the board:
export LD_LIBRARY_PATH=$PWD/pkg/lib

# one row, for example SmolVLA
./pkg/bin/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
```
