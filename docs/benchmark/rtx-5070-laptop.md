# RTX 5070 Laptop GPU (CUDA)

`vla-bench` times `predict()` in-process on synthetic inputs. It measures the
engine only: no transport, no simulator, and no claim about task success.

## Test setup

| | |
|---|---|
| Device | NVIDIA GeForce RTX 5070 Laptop GPU, 8 GB GDDR7, Blackwell (sm_120) |
| Host | Intel Core i9-14900HX (24 cores, 32 threads), 31 GiB RAM; laptop on AC power, platform profile `performance` |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8 |
| Driver / toolkit | 590.48.01 / CUDA 12.8, GCC 11.4 |
| Commit | `c93ca0a` (PR #32 head, branch `b11223-numerics-perf`) |
| llama.cpp | `b11223` |
| Date | 2026-09-30 00:59 to 03:56 (UTC+07) |
| Build | `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release` (`GGML_NATIVE=ON`) |
| Runtime flags | defaults; the fastest flags per model are in the second table |
| Method | 3 warmups + 20 timed reps per process, 3 processes. The process with the lowest mean is reported, and memory is the peak over all three. |

**Peak VRAM** is the process's device memory from `nvidia-smi --query-compute-apps`, sampled every 0.5 s. **Peak host RSS** is the process's peak resident set (`getrusage`).

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

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak VRAM MiB | Peak host RSS MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 2 | 256 + 128 | 4.7 | 5.0 | 5.0 | 5.0 | 0.6 | 734 | 589 |
| TurboVLA | 2 | 256 | 14.1 | 14.5 | 14.5 | 14.8 | - | 1030 | 583 |
| VLA-JEPA | 1 | 256 | 38.0 | 40.8 | 38.6 | 50.5 | 14.2 | 4006 | 728 |
| BitVLA | 1 | 224 | 54.4 | 60.0 | 55.8 | 71.8 | 11.9 | 1328 | 1178 |
| VLA-Adapter | 1 | 224 | 56.3 | 63.0 | 59.4 | 75.9 | 34.1 | 2896 | 1071 |
| GR00T N1.7 | 1 | 256 | 63.3 | 68.0 | 64.3 | 81.6 | 16.1 | 5064 | 750 |
| GR00T N1.6 | 1 | 224 | 65.5 | 72.3 | 68.6 | 87.8 | 21.5 | 4708 | 740 |
| GR00T N1.5 | 1 | 224 | 73.0 | 79.2 | 74.4 | 95.0 | 21.2 | 3668 | 765 |
| SmolVLA | 2 | 512 | 89.3 | 98.7 | 96.5 | 111.3 | 54.2 | 2056 | 805 |
| π0 | 2 | 224 | 177.1 | 195.7 | 190.7 | 212.7 | 41.3 | 5534 | 788 |
| π0.5 | 2 | 224 | 180.5 | 196.7 | 192.1 | 209.8 | 40.9 | 5528 | 809 |
| Evo-1 | 1 | 448 | 184.6 | 201.6 | 201.7 | 217.7 | 92.4 | 1604 | 805 |

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

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak VRAM MiB | Peak host RSS MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | `--flash-attn` | 4.4 | 4.7 | 4.8 | 4.8 | 0.6 | 734 | 590 | -6% |
| TurboVLA | `--weight-dtype f16 --flash-attn` | 8.6 | 9.1 | 9.1 | 9.3 | - | 586 | 642 | -37% |
| VLA-JEPA | *(defaults)* | 38.0 | 40.8 | 38.6 | 50.5 | 14.2 | 4006 | 728 | - |
| VLA-Adapter | `--weight-dtype f16 --flash-attn` | 48.1 | 53.6 | 49.6 | 63.7 | 27.4 | 2894 | 1071 | -15% |
| BitVLA | *(defaults)* | 54.4 | 60.0 | 55.8 | 71.8 | 11.9 | 1328 | 1178 | - |
| GR00T N1.5 | `--weight-dtype f16 --flash-attn` | 59.8 | 64.8 | 61.2 | 76.4 | 15.6 | 3670 | 688 | -18% |
| SmolVLA | `--flash-attn --mm-prec default` | 60.4 | 65.3 | 61.4 | 77.5 | 29.4 | 1238 | 810 | -34% |
| GR00T N1.7 | *(defaults)* | 63.3 | 68.0 | 64.3 | 81.6 | 16.1 | 5064 | 750 | - |
| GR00T N1.6 | *(defaults)* | 65.5 | 72.3 | 68.6 | 87.8 | 21.5 | 4708 | 740 | - |
| Evo-1 | `--weight-dtype f16 --flash-attn` | 129.0 | 141.8 | 138.4 | 153.9 | 36.0 | 1558 | 729 | -30% |
| π0 | `--weight-dtype f16 --flash-attn` | 135.4 | 149.0 | 145.3 | 169.3 | 31.5 | 5548 | 716 | -24% |
| π0.5 | `--weight-dtype f16 --flash-attn` | 147.1 | 161.0 | 156.6 | 174.9 | 29.0 | 5752 | 733 | -18% |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` | `--act-dtype bf16 --flash-attn` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 4.8 | 4.4 | 4.4 | 4.5 | 4.5 | 4.5 | 4.4 | 4.5 |  |
| TurboVLA | 14.7 | 12.1 | 14.3 | 12.0 | 12.4 | 10.0 | 11.2 | 9.1 |  |
| VLA-JEPA | 38.2 | 37.3 | 39.2 | 37.5 | 38.7 | 37.5 | 44.4 | 43.6 |  |
| BitVLA | 55.6 | 55.7 | 55.6 | 55.8 | 56.0 | 55.9 | 56.1 | 56.0 |  |
| VLA-Adapter | 58.6 | 58.1 | 57.9 | 57.5 | 58.5 | 58.6 | 48.7 | 48.5 |  |
| GR00T N1.7 | 64.0 | 62.9 | 63.8 | 63.3 | 64.5 | 62.9 | 73.2 | 72.2 |  |
| GR00T N1.6 | 67.9 | 68.3 | 67.6 | 68.1 | 67.8 | 68.2 | 71.1 | 71.2 |  |
| GR00T N1.5 | 73.7 | 71.3 | 73.0 | 71.2 | 73.8 | 71.9 | 62.1 | 60.5 |  |
| SmolVLA | 90.2 | 67.0 | 85.6 | 61.0 | 95.1 | 71.3 | 88.9 | 65.5 |  |
| π0 | 197.5 | 187.5 | 198.0 | 187.1 | 197.8 | 188.0 | 165.3 | 150.3 | 183.1 |
| π0.5 | 198.1 | 198.4 | 198.4 | 198.6 | 198.7 | 199.1 | 165.8 | 163.0 |  |
| Evo-1 | 203.7 | 156.0 | 204.0 | 156.5 | 204.1 | 157.4 | 191.9 | 144.0 | 151.9 |

</details>

## Not run

- **OpenVLA-OFT**: does not fit. Loading the weights runs out of memory:

  ```text
  ggml_backend_cuda_buffer_type_alloc_buffer: allocating 14377.07 MiB on device 0: cudaMalloc failed: out of memory
  alloc_tensor_range: failed to allocate CUDA0 buffer of size 15075447168
  vla(openvla_oft): alloc_weights failed (OOM?)
  vla-bench: model_load failed
  ```

## Reproducing

```bash
cmake -S . -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j

# one row, for example SmolVLA
./build-cuda/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
```
