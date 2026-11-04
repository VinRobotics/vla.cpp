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
| Commit | `f7b0e68` |
| llama.cpp | `b11223` |
| Date | 2026-11-04 15:48 to 16:23 (UTC+07) |
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
| Octo-Small | 2 | 256 + 128 | 121.3 | 124.0 | 122.8 | 128.5 | 20.4 | 253 | 731 |
| TurboVLA | 2 | 256 | 186.8 | 190.6 | 189.6 | 194.9 | - | 249 | 1009 |
| VLA-JEPA | 1 | 256 | 448.1 | 452.9 | 451.0 | 458.7 | 197.0 | 185 | 3971 |
| SmolVLA | 2 | 512 | 583.6 | 584.3 | 584.2 | 584.9 | 230.0 | 146 | 1197 |
| GR00T N1.5 | 1 | 224 | 597.1 | 599.1 | 598.6 | 600.0 | 138.9 | 162 | 3645 |
| GR00T N1.6 | 1 | 224 | 704.6 | 707.6 | 706.3 | 711.0 | 145.9 | 147 | 4765 |
| GR00T N1.7 | 1 | 256 | 756.6 | 762.6 | 761.2 | 769.1 | 195.5 | 191 | 5084 |
| VLA-Adapter | 1 | 224 | 837.8 | 853.7 | 852.5 | 863.3 | 296.0 | 872 | 3581 |
| π0.5 | 2 | 224 | 1109.9 | 1111.9 | 1112.0 | 1112.7 | 277.1 | 285 | 5839 |
| π0 | 2 | 224 | 1116.8 | 1120.5 | 1120.1 | 1123.1 | 277.6 | 285 | 5535 |
| OpenVLA-OFT | 1 | 224 | 1761.5 | 1768.2 | 1768.8 | 1772.7 | 298.5 | 844 | 15461 |
| Evo-1 | 1 | 448 | 2329.8 | 2346.0 | 2341.9 | 2360.7 | 819.2 | 319 | 1594 |

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
| Octo-Small | *(defaults)* | 121.3 | 124.0 | 122.8 | 128.5 | 20.4 | 253 | 731 | - |
| TurboVLA | *(defaults)* | 186.8 | 190.6 | 189.6 | 194.9 | - | 249 | 1009 | - |
| VLA-JEPA | *(defaults)* | 448.1 | 452.9 | 451.0 | 458.7 | 197.0 | 185 | 3971 | - |
| SmolVLA | *(defaults)* | 583.6 | 584.3 | 584.2 | 584.9 | 230.0 | 146 | 1197 | - |
| GR00T N1.5 | *(defaults)* | 597.1 | 599.1 | 598.6 | 600.0 | 138.9 | 162 | 3645 | - |
| GR00T N1.6 | *(defaults)* | 704.6 | 707.6 | 706.3 | 711.0 | 145.9 | 147 | 4765 | - |
| GR00T N1.7 | `--flash-attn` | 733.3 | 739.1 | 736.8 | 747.5 | 195.3 | 191 | 5059 | -3.1% |
| VLA-Adapter | *(defaults)* | 837.8 | 853.7 | 852.5 | 863.3 | 296.0 | 872 | 3581 | - |
| π0.5 | *(defaults)* | 1109.9 | 1111.9 | 1112.0 | 1112.7 | 277.1 | 285 | 5839 | - |
| π0 | *(defaults)* | 1116.8 | 1120.5 | 1120.1 | 1123.1 | 277.6 | 285 | 5535 | - |
| OpenVLA-OFT | *(defaults)* | 1761.5 | 1768.2 | 1768.8 | 1772.7 | 298.5 | 844 | 15461 | - |
| Evo-1 | *(defaults)* | 2329.8 | 2346.0 | 2341.9 | 2360.7 | 819.2 | 319 | 1594 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--flash-attn 0` |
|---|--:|--:|--:|--:|--:|
| Octo-Small | 124.3 | 128.9 | 129.6 | 127.7 | 124.7 |
| TurboVLA | 194.6 | 192.5 | 188.6 | 194.3 | 251.9 |
| VLA-JEPA | 453.3 | 454.5 | 453.3 | 451.9 | 605.5 |
| SmolVLA | 586.3 | 586.9 | 584.4 | 586.0 | 2147.5 |
| GR00T N1.5 | 601.3 | 600.7 | 601.7 | 600.4 | 877.5 |
| GR00T N1.6 | 708.6 | 708.4 | 706.5 | 707.5 | 876.8 |
| GR00T N1.7 | 763.3 | 737.3 | 765.2 | 741.6 | 869.0 |
| VLA-Adapter | 854.5 | 859.7 | 870.9 | 878.5 | 1151.2 |
| π0.5 | 1113.6 | 1112.9 | 1113.4 | 1113.5 | 2047.2 |
| π0 | 1141.2 | 1140.3 | 1141.9 | 1140.9 | 2044.7 |
| OpenVLA-OFT | 1776.1 | 1775.6 | 1770.0 | 1765.2 | 2045.9 |
| Evo-1 | 2339.7 | 2350.8 | 2340.9 | 2352.5 | 3868.0 |

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
