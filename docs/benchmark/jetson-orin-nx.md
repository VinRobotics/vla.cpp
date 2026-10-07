# Jetson Orin NX 16 GB (CUDA)

`vla-bench` times `predict()` in-process on synthetic inputs. It measures the
engine only: no transport, no simulator, and no claim about task success.

## Test setup

| | |
|---|---|
| Device | NVIDIA Jetson Orin NX 16 GB Developer Kit (Super), 1024-core Ampere GPU (sm_87), 16 GB LPDDR5 unified (15.3 GiB visible) |
| Host CPU | 8-core Arm Cortex-A78AE |
| Power | `MAXN_SUPER` (nvpmodel 0); `jetson_clocks` state not changed |
| OS | JetPack 6.2 (L4T R36.4.3), Ubuntu 22.04.5, kernel 5.15.148-tegra |
| Toolkit | CUDA 12.6, GCC 11.4 |
| Commit | `f7e0f7f` (main, the merge of PR #32; `src/`, `cmake/`, `tests/` and `CMakeLists.txt` identical to `c93ca0a`) |
| llama.cpp | `b11223` |
| Date | 2026-10-01 18:10 to 20:57 (UTC+02) |
| Build | `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release` (`GGML_NATIVE=ON`) |
| Runtime flags | defaults; the fastest flags per model are in the second table |
| Method | 3 warmups + 20 timed reps per process, 3 processes. The process with the lowest mean is reported. |

This report gives latency only. Memory is unified, and `nvidia-smi` reports no
per-process usage on Jetson. `getrusage` on this board did not reliably capture
the CUDA buffers, so there is no memory column. The weights and graphs are the
same as on the other Jetsons, so the footprint should match the
[Orin Nano](jetson-orin-nano.md) and [AGX Orin](jetson-agx-orin.md) reports.

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
- Every file's sha256 matches the one Hugging Face lists.

## Latency

Rows are sorted by latency. `vision ms` is `-` for archs that do not time their
vision stage separately.

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms |
|---|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 2 | 256 + 128 | 27.4 | 27.5 | 27.5 | 27.5 | 2.9 |
| TurboVLA | 2 | 256 | 79.7 | 79.8 | 79.8 | 79.8 | - |
| VLA-JEPA | 1 | 256 | 176.7 | 176.8 | 176.7 | 176.8 | 73.6 |
| GR00T N1.7 | 1 | 256 | 251.4 | 251.8 | 251.6 | 251.9 | 73.5 |
| GR00T N1.6 | 1 | 224 | 252.1 | 252.3 | 252.2 | 253.1 | 76.5 |
| GR00T N1.5 | 1 | 224 | 260.8 | 261.1 | 261.0 | 261.7 | 76.5 |
| BitVLA | 1 | 224 | 269.1 | 269.2 | 269.2 | 269.2 | 50.6 |
| VLA-Adapter | 1 | 224 | 270.7 | 270.8 | 270.8 | 270.9 | 151.8 |
| SmolVLA | 2 | 512 | 411.2 | 411.5 | 411.4 | 411.8 | 261.7 |
| π0 | 2 | 224 | 739.6 | 740.0 | 740.0 | 740.2 | 152.7 |
| π0.5 | 2 | 224 | 742.9 | 743.2 | 743.1 | 743.4 | 152.2 |
| Evo-1 | 1 | 448 | 893.2 | 894.0 | 894.1 | 894.2 | 454.0 |

### Fastest configuration

We screen each model over these runtime flag sets, then re-run the fastest set
under the full protocol:

- the defaults;
- `--flash-attn` and `--mm-prec default`, alone and together;
- `--weight-dtype bf16` and `--weight-dtype f16`, each alone and with
  `--flash-attn`;
- `--act-dtype bf16 --flash-attn`, for π0 and Evo-1 on CUDA only.

The screen runs 2 warmups and 5 reps in one process per flag set. A flag set
replaces the defaults only if its mean is more than 3% lower, both in the screen
and in the full re-run.

Flash attention, bf16 activations and lower-precision weights can move the
action chunk. Success rates measured at the defaults therefore do not carry over
to these rows.

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|
| Octo-Small | *(defaults)* | 27.4 | 27.5 | 27.5 | 27.5 | 2.9 | - |
| TurboVLA | `--weight-dtype bf16 --flash-attn` | 46.7 | 46.8 | 46.8 | 46.8 | - | -41% |
| VLA-JEPA | `--weight-dtype f16 --flash-attn` | 154.9 | 155.0 | 155.0 | 155.1 | 62.0 | -12% |
| GR00T N1.7 | `--weight-dtype f16 --flash-attn` | 233.0 | 233.1 | 233.0 | 233.1 | 62.0 | -7% |
| VLA-Adapter | `--weight-dtype f16` | 236.7 | 236.9 | 236.8 | 237.3 | 131.2 | -13% |
| GR00T N1.6 | `--weight-dtype f16` | 243.1 | 243.3 | 243.2 | 244.1 | 68.3 | -4% |
| GR00T N1.5 | `--weight-dtype f16 --flash-attn` | 249.2 | 249.5 | 249.4 | 250.1 | 70.9 | -4% |
| SmolVLA | `--flash-attn --mm-prec default` | 254.1 | 254.3 | 254.2 | 254.7 | 151.8 | -38% |
| BitVLA | *(defaults)* | 269.1 | 269.2 | 269.2 | 269.2 | 50.6 | - |
| π0 | `--weight-dtype f16 --flash-attn` | 597.6 | 597.9 | 597.8 | 598.2 | 140.0 | -19% |
| Evo-1 | `--act-dtype bf16 --flash-attn` | 626.0 | 626.3 | 626.2 | 626.4 | 196.0 | -30% |
| π0.5 | `--weight-dtype f16 --flash-attn` | 664.7 | 665.1 | 665.1 | 665.6 | 135.8 | -11% |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` | `--act-dtype bf16 --flash-attn` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 27.6 | 27.5 | 27.5 | 27.6 | 27.5 | 27.5 | 27.6 | 27.5 |  |
| TurboVLA | 79.8 | 64.9 | 79.9 | 64.5 | 62.6 | 46.9 | 64.6 | 49.1 |  |
| VLA-JEPA | 177.2 | 170.9 | 177.8 | 171.2 | 177.3 | 171.5 | 162.3 | 155.5 |  |
| GR00T N1.7 | 253.4 | 247.0 | 252.9 | 247.6 | 253.3 | 247.8 | 240.0 | 233.6 |  |
| GR00T N1.6 | 253.9 | 255.7 | 253.5 | 256.1 | 253.1 | 256.3 | 244.2 | 246.7 |  |
| GR00T N1.5 | 262.1 | 254.6 | 262.4 | 253.9 | 261.3 | 254.1 | 258.1 | 250.3 |  |
| BitVLA | 268.6 | 268.6 | 269.2 | 269.5 | 269.5 | 269.6 | 269.5 | 269.0 |  |
| VLA-Adapter | 271.2 | 270.8 | 271.6 | 270.9 | 270.8 | 270.9 | 237.4 | 237.9 |  |
| SmolVLA | 411.5 | 301.6 | 365.0 | 254.4 | 478.6 | 368.7 | 482.0 | 371.5 |  |
| π0 | 743.7 | 676.5 | 741.2 | 679.8 | 740.5 | 679.1 | 663.3 | 599.2 | 622.0 |
| π0.5 | 742.8 | 744.1 | 743.1 | 745.0 | 742.7 | 744.6 | 672.4 | 668.8 |  |
| Evo-1 | 895.2 | 676.4 | 895.1 | 676.7 | 894.7 | 677.1 | 917.0 | 698.2 | 627.9 |

</details>

## Not run

- **OpenVLA-OFT**: not attempted. Its 15.1 GB of weights do not fit in the 15.3 GiB
  of memory the board exposes.

## Notes

- No thermal throttling: across every timed window the GPU clock stayed at
  1162-1172 MHz (`tegrastats`), with a junction temperature of at most 83 °C. The
  board idled at 61-62 °C before the sweep.
- Every model keeps the GPU busy: at the defaults, `tegrastats` shows 94-98% GPU
  load during the timed calls, with the EMC (memory controller) at 23-64%, so the latencies are
  bound by GPU compute rather than by the host or memory bandwidth. Board power
  was 21-26 W under load (8.3 W idle).

## Reproducing

```bash
cmake -S . -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j

# one row, for example SmolVLA
./build-cuda/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
```
