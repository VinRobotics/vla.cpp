# Jetson AGX Orin 64GB (CUDA)

`vla-bench` times `predict()` in-process on synthetic inputs. It measures the
engine only: no transport, no simulator, and no claim about task success.

## Test setup

| | |
|---|---|
| Device | NVIDIA Jetson AGX Orin Developer Kit, 2048-core Ampere GPU (sm_87), 64 GB LPDDR5 unified |
| Host CPU | 12-core Arm Cortex-A78AE |
| Power mode | `MAXN` (nvpmodel 0); `jetson_clocks` state not changed |
| OS | JetPack 6.2 (L4T R36.4.3), Ubuntu 22.04.5 |
| Toolkit | CUDA 12.6, GCC 11.4 |
| Commit | `c93ca0a` (PR #32 head, branch `b11223-numerics-perf`) |
| llama.cpp | `b11223` |
| Date | 2026-09-30 01:09 to 04:05 (UTC+07) |
| Build | `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release` (`GGML_NATIVE=ON`) |
| Runtime flags | defaults; the fastest flags per model are in the second table |
| Method | 3 warmups + 20 timed reps per process, 3 processes. The process with the lowest mean is reported, and memory is the peak over all three. |

Memory is unified, and `nvidia-smi` reports no per-process usage on Jetson. The CUDA buffers are mapped into the process, so **Peak RSS** (`getrusage`) counts them together with host memory, and it is the process's whole footprint. VLA-JEPA, for example, peaks at 4358 MiB here. On the RTX 3090 the same model takes 4138 MiB of VRAM plus 634 MiB of host RSS.

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
| BitVLA | `bitvla-libero-gguf` (int2`) | 1 | 224 | 16 | 8 | 7 | parallel decoding | 1 |
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

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 2 | 256 + 128 | 17.5 | 23.9 | 22.5 | 30.3 | 3.1 | 947 |
| TurboVLA | 2 | 256 | 36.0 | 36.8 | 36.8 | 36.9 | - | 1238 |
| VLA-JEPA | 1 | 256 | 94.6 | 96.3 | 95.1 | 98.7 | 38.2 | 4358 |
| VLA-Adapter | 1 | 224 | 127.4 | 128.2 | 128.2 | 128.5 | 75.3 | 3698 |
| GR00T N1.5 | 1 | 224 | 131.2 | 132.3 | 131.9 | 132.7 | 38.4 | 4079 |
| GR00T N1.6 | 1 | 224 | 132.6 | 133.1 | 133.0 | 133.4 | 38.0 | 5082 |
| GR00T N1.7 | 1 | 256 | 132.7 | 134.0 | 133.7 | 135.2 | 37.6 | 5442 |
| BitVLA | 1 | 224 | 133.0 | 134.7 | 133.4 | 134.9 | 27.4 | 2219 |
| SmolVLA | 2 | 512 | 218.3 | 223.4 | 223.4 | 225.9 | 130.6 | 2527 |
| π0 | 2 | 224 | 350.3 | 350.9 | 350.6 | 351.4 | 75.5 | 5982 |
| π0.5 | 2 | 224 | 351.9 | 353.1 | 353.1 | 353.7 | 75.0 | 6008 |
| OpenVLA-OFT | 1 | 224 | 384.9 | 385.3 | 385.3 | 385.4 | 75.7 | 15391 |
| Evo-1 | 1 | 448 | 434.6 | 435.2 | 435.3 | 435.6 | 208.8 | 2075 |

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

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | *(defaults)* | 17.5 | 23.9 | 22.5 | 30.3 | 3.1 | 947 | - |
| TurboVLA | `--weight-dtype bf16 --flash-attn` | 24.1 | 26.6 | 24.5 | 32.3 | - | 932 | -28% |
| VLA-JEPA | `--weight-dtype f16 --flash-attn` | 82.4 | 85.6 | 83.6 | 93.8 | 32.9 | 4346 | -11% |
| VLA-Adapter | `--weight-dtype f16 --flash-attn` | 116.5 | 117.8 | 117.7 | 118.7 | 64.0 | 3699 | -8% |
| GR00T N1.5 | `--weight-dtype f16 --flash-attn` | 123.0 | 124.0 | 123.3 | 125.7 | 35.9 | 4007 | -6% |
| GR00T N1.7 | `--weight-dtype f16 --flash-attn` | 125.1 | 125.9 | 125.4 | 126.8 | 31.5 | 5441 | -6% |
| GR00T N1.6 | `--weight-dtype f16` | 127.6 | 128.7 | 128.6 | 129.2 | 34.7 | 5077 | -3% |
| BitVLA | *(defaults)* | 133.0 | 134.7 | 133.4 | 134.9 | 27.4 | 2219 | - |
| SmolVLA | `--flash-attn --mm-prec default` | 146.4 | 147.0 | 146.9 | 147.3 | 78.1 | 1711 | -34% |
| π0 | `--act-dtype bf16 --flash-attn` | 300.4 | 301.2 | 301.2 | 301.5 | 67.7 | 6017 | -14% |
| Evo-1 | `--act-dtype bf16 --flash-attn` | 311.1 | 311.4 | 311.5 | 311.6 | 92.0 | 2154 | -28% |
| OpenVLA-OFT | `--weight-dtype f16` | 345.2 | 345.9 | 345.8 | 346.3 | 64.8 | 15392 | -10% |
| π0.5 | *(defaults)* | 351.9 | 353.1 | 353.1 | 353.7 | 75.0 | 6008 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` | `--act-dtype bf16 --flash-attn` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 33.4 | 32.4 | 31.9 | 33.5 | 33.4 | 31.3 | 32.0 | 30.9 |  |
| TurboVLA | 40.1 | 38.3 | 39.5 | 36.7 | 35.1 | 34.8 | 51.2 | 35.4 |  |
| VLA-JEPA | 117.6 | 113.2 | 115.6 | 115.3 | 117.2 | 115.7 | 108.8 | 102.6 |  |
| VLA-Adapter | 130.7 | 133.4 | 128.7 | 128.8 | 128.6 | 129.3 | 119.5 | 118.0 |  |
| GR00T N1.5 | 144.6 | 130.8 | 143.6 | 130.2 | 134.4 | 130.2 | 129.2 | 126.9 |  |
| GR00T N1.6 | 134.8 | 136.4 | 135.9 | 135.2 | 133.4 | 135.4 | 128.6 | 130.9 |  |
| GR00T N1.7 | 146.7 | 133.4 | 135.5 | 143.8 | 144.3 | 143.9 | 136.1 | 128.2 |  |
| BitVLA | 144.4 | 144.5 | 144.6 | 144.9 | 145.5 | 144.5 | 145.4 | 144.8 |  |
| SmolVLA | 222.9 | 168.9 | 203.8 | 146.2 | 249.9 | 196.7 | 250.4 | 200.2 |  |
| π0 | 350.8 | 324.5 | 350.7 | 323.9 | 350.8 | 323.7 | 355.9 | 329.2 | 300.0 |
| π0.5 | 353.4 | 352.3 | 354.2 | 353.0 | 353.0 | 353.2 | 358.3 | 357.6 |  |
| OpenVLA-OFT | 384.4 | 383.8 | 385.3 | 383.7 | 384.8 | 385.7 | 346.2 | 346.7 |  |
| Evo-1 | 434.0 | 335.5 | 434.5 | 335.3 | 434.7 | 335.2 | 441.9 | 343.5 | 310.9 |

</details>

## Reproducing

```bash
cmake -S . -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j

# one row, for example SmolVLA
./build-cuda/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
```
