# Jetson Orin Nano Super (CUDA)

`vla-bench` times `predict()` in-process on synthetic inputs. It measures the
engine only: no transport, no simulator, and no claim about task success.

## Test setup

| | |
|---|---|
| Device | NVIDIA Jetson Orin Nano Developer Kit (Super), 1024-core Ampere GPU (sm_87), 8 GB LPDDR5 unified (7.4 GiB visible) |
| Host CPU | 6-core Arm Cortex-A78AE |
| Power mode | `MAXN_SUPER` (nvpmodel 2); `jetson_clocks` state not changed |
| OS | JetPack 6.2.2 (L4T R36.5), Ubuntu 22.04.5 |
| Toolkit | CUDA 12.6, GCC 11.4 |
| Commit | `c93ca0a` (PR #32 head, branch `b11223-numerics-perf`) |
| llama.cpp | `b11223` |
| Date | 2026-09-30 01:09 to 02:45 (UTC+07) |
| Build | `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release` (`GGML_NATIVE=ON`) |
| Runtime flags | defaults; the fastest flags per model are in the second table |
| Method | 3 warmups + 20 timed reps per process, 3 processes. The process with the lowest mean is reported, and memory is the peak over all three. |

Memory is unified, and `nvidia-smi` reports no per-process usage on Jetson. The CUDA buffers are mapped into the process, so **Peak RSS** (`getrusage`) counts them together with host memory, and it is the process's whole footprint. VLA-JEPA, for example, peaks at 4332 MiB here. On the RTX 3090 the same model takes 4138 MiB of VRAM plus 634 MiB of host RSS.

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
| Octo-Small | 2 | 256 + 128 | 31.9 | 34.4 | 33.1 | 36.3 | 4.1 | 923 |
| TurboVLA | 2 | 256 | 89.3 | 90.2 | 89.6 | 89.9 | - | 1215 |
| VLA-JEPA | 1 | 256 | 193.9 | 194.6 | 194.3 | 196.0 | 82.4 | 4332 |
| GR00T N1.7 | 1 | 256 | 271.7 | 272.1 | 272.0 | 272.2 | 82.3 | 5413 |
| GR00T N1.6 | 1 | 224 | 272.3 | 272.6 | 272.5 | 272.9 | 85.0 | 5051 |
| GR00T N1.5 | 1 | 224 | 293.4 | 295.0 | 295.2 | 296.0 | 84.8 | 4048 |
| VLA-Adapter | 1 | 224 | 297.5 | 297.9 | 297.9 | 298.1 | 167.4 | 3669 |
| BitVLA | 1 | 224 | 335.0 | 335.5 | 335.3 | 336.0 | 63.6 | 2193 |
| SmolVLA | 2 | 512 | 462.8 | 464.2 | 463.9 | 466.1 | 298.6 | 2488 |
| π0 | 2 | 224 | 845.2 | 846.1 | 846.1 | 846.7 | 169.2 | 5961 |
| π0.5 | 2 | 224 | 849.7 | 850.2 | 850.2 | 850.5 | 169.3 | 5982 |
| Evo-1 | 1 | 448 | 1011.5 | 1012.5 | 1012.6 | 1013.0 | 523.3 | 2048 |

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
| Octo-Small | *(defaults)* | 31.9 | 34.4 | 33.1 | 36.3 | 4.1 | 923 | - |
| TurboVLA | `--weight-dtype bf16 --flash-attn` | 52.4 | 53.1 | 53.1 | 53.2 | - | 895 | -41% |
| VLA-JEPA | `--weight-dtype f16 --flash-attn` | 173.7 | 174.8 | 174.6 | 176.6 | 70.6 | 4318 | -10% |
| GR00T N1.7 | `--weight-dtype f16 --flash-attn` | 254.5 | 255.0 | 254.9 | 255.6 | 70.4 | 5409 | -6% |
| VLA-Adapter | `--weight-dtype f16` | 266.2 | 266.6 | 266.5 | 266.8 | 148.1 | 3668 | -11% |
| GR00T N1.6 | *(defaults)* | 272.3 | 272.6 | 272.5 | 272.9 | 85.0 | 5051 | - |
| SmolVLA | `--flash-attn --mm-prec default` | 286.7 | 287.8 | 287.3 | 289.9 | 173.1 | 1676 | -38% |
| GR00T N1.5 | *(defaults)* | 293.4 | 295.0 | 295.2 | 296.0 | 84.8 | 4048 | - |
| BitVLA | *(defaults)* | 335.0 | 335.5 | 335.3 | 336.0 | 63.6 | 2193 | - |
| π0 | `--weight-dtype f16 --flash-attn` | 688.2 | 688.7 | 688.8 | 689.0 | 160.0 | 5901 | -19% |
| Evo-1 | `--act-dtype bf16 --flash-attn` | 720.8 | 722.7 | 722.7 | 723.6 | 233.4 | 2133 | -29% |
| π0.5 | `--weight-dtype f16 --flash-attn` | 764.8 | 765.8 | 765.7 | 766.4 | 153.2 | 5925 | -10% |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` | `--act-dtype bf16 --flash-attn` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 43.1 | 45.1 | 41.8 | 41.8 | 41.5 | 43.0 | 43.5 | 48.7 |  |
| TurboVLA | 91.2 | 76.3 | 97.3 | 72.8 | 73.5 | 60.3 | 81.0 | 65.9 |  |
| VLA-JEPA | 205.7 | 193.1 | 201.0 | 199.0 | 205.6 | 193.4 | 194.8 | 186.1 |  |
| GR00T N1.7 | 273.0 | 267.8 | 273.4 | 266.8 | 273.7 | 266.7 | 264.7 | 257.3 |  |
| GR00T N1.6 | 275.0 | 279.0 | 274.3 | 279.6 | 273.0 | 278.9 | 271.5 | 270.8 |  |
| GR00T N1.5 | 295.4 | 289.6 | 298.3 | 287.0 | 298.2 | 288.6 | 293.5 | 287.9 |  |
| VLA-Adapter | 298.7 | 299.0 | 298.9 | 298.5 | 300.0 | 298.4 | 268.1 | 269.2 |  |
| BitVLA | 336.1 | 336.3 | 335.9 | 335.9 | 335.7 | 337.0 | 336.0 | 335.6 |  |
| SmolVLA | 465.3 | 336.9 | 414.5 | 287.3 | 533.9 | 406.7 | 535.0 | 407.8 |  |
| π0 | 848.8 | 772.8 | 849.4 | 776.5 | 848.1 | 778.2 | 763.4 | 687.9 | 719.7 |
| π0.5 | 850.5 | 850.5 | 851.2 | 852.0 | 852.3 | 849.7 | 768.6 | 763.4 |  |
| Evo-1 | 1014.7 | 763.0 | 1014.6 | 762.5 | 1014.8 | 762.3 | 1029.4 | 774.8 | 721.8 |

</details>

## Not run

- **OpenVLA-OFT**: not attempted. Its 15.1 GB of weights do not fit in 8 GB of memory.

## Reproducing

```bash
cmake -S . -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j

# one row, for example SmolVLA
./build-cuda/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
```
