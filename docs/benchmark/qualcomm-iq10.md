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
| Commit | `e8dfa34` (PR #69, on top of #68) |
| llama.cpp | `b11223` |
| Date | 2026-11-02 11:02 to 11:48 (UTC+07) |
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
| GR00T N1.7 | 1 | 256 | 747.5 | 754.7 | 753.0 | 762.7 | 196.1 | 191 | 5083 |
| SmolVLA | 2 | 512 | 810.3 | 818.6 | 820.5 | 822.6 | 283.0 | 146 | 1187 |
| GR00T N1.5 | 1 | 224 | 1020.9 | 1024.2 | 1024.2 | 1027.0 | 563.2 | 162 | 3650 |
| GR00T N1.6 | 1 | 224 | 1110.3 | 1114.5 | 1113.6 | 1118.5 | 559.8 | 147 | 4687 |
| π0.5 | 2 | 224 | 2914.6 | 2942.6 | 2934.4 | 2967.2 | 1429.6 | 285 | 5517 |
| Evo-1 | 1 | 448 | 3519.4 | 3572.4 | 3576.7 | 3588.2 | 854.5 | 319 | 1579 |

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
| GR00T N1.7 | *(defaults)* | 747.5 | 754.7 | 753.0 | 762.7 | 196.1 | 191 | 5083 | - |
| SmolVLA | *(defaults)* | 810.3 | 818.6 | 820.5 | 822.6 | 283.0 | 146 | 1187 | - |
| GR00T N1.5 | *(defaults)* | 1020.9 | 1024.2 | 1024.2 | 1027.0 | 563.2 | 162 | 3650 | - |
| GR00T N1.6 | *(defaults)* | 1110.3 | 1114.5 | 1113.6 | 1118.5 | 559.8 | 147 | 4687 | - |
| π0.5 | *(defaults)* | 2914.6 | 2942.6 | 2934.4 | 2967.2 | 1429.6 | 285 | 5517 | - |
| Evo-1 | *(defaults)* | 3519.4 | 3572.4 | 3576.7 | 3588.2 | 854.5 | 319 | 1579 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--flash-attn 0` |
|---|--:|--:|--:|--:|--:|
| GR00T N1.7 | 768.7 | 748.5 | 761.3 | 744.5 | 1086.6 |
| SmolVLA | 823.1 | 818.9 | 819.5 | 818.5 | 3018.8 |
| GR00T N1.5 | 1026.1 | 1023.0 | 1025.3 | 1026.1 | 1501.2 |
| GR00T N1.6 | 1114.4 | 1115.1 | 1114.4 | 1113.3 | 1290.3 |
| π0.5 | 2939.0 | 2956.9 | 2959.0 | 2962.7 | 2959.6 |
| Evo-1 | 3583.7 | 3593.9 | 3591.8 | 3582.6 | 5828.5 |

</details>

## Not run

- **BitVLA**: N/A off CUDA. The only published GGUF is int2-packed, and only
  CUDA builds load it.
- **π0, VLA-JEPA, VLA-Adapter, OpenVLA-OFT, Octo-Small, TurboVLA**: not run in
  this round; their GGUFs were not on the board.

## Notes

The HTP skel has to be processed for this board and installed in
`/vendor/dsp/cdsp0..3`: its FastRPC ignores `ADSP_LIBRARY_PATH`. See
[backend/hexagon.md](../backend/hexagon.md#linux-iq-10).

GR00T N1.6 runs one view here, as the configuration above fixes. With two views
its SigLIP2 tower hits the batched `MUL_MAT` the wrapper sends to the CPU; see
[backend/hexagon.md](../backend/hexagon.md#op-coverage-measured).

These rows time the defaults; they do not vouch for accuracy. On a served LIBERO
observation, GR00T N1.5 and N1.6 land above the 2.9e-3 fidelity bar, about as
far off as the CPU at F16; see
[backend/hexagon.md](../backend/hexagon.md#serving).

## Reproducing

```bash
scripts/build_linux_hexagon.sh
# copy build-lnx-htp/pkg to the board, install its HTP skel (see backend/hexagon.md), then on the board:
export LD_LIBRARY_PATH=$PWD/pkg/lib

# one row, for example SmolVLA
./pkg/bin/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
```
