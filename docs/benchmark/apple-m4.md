# Apple M4 (Metal)

`vla-bench` times `predict()` in-process on synthetic inputs. It measures the
engine only: no transport, no simulator, and no claim about task success.

## Test setup

| | |
|---|---|
| Device | Apple M4, 10-core GPU, 24 GB unified memory |
| Host CPU | Apple M4, 4 performance + 6 efficiency cores |
| OS | macOS 26.5.1 (25F80) |
| Toolchain | Apple clang 21.0.0, CMake 4.3.3 |
| Commit | `c93ca0a` (PR #32 head, branch `b11223-numerics-perf`) |
| llama.cpp | `b11223` |
| Date | 2026-09-30 00:59 to 01:23 (UTC+07) |
| Build | `-DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release` |
| Runtime flags | defaults; the fastest flags per model are in the second table |
| Method | 3 warmups + 20 timed reps per process, 3 processes. The process with the lowest mean is reported, and memory is the peak over all three. |

Memory is unified, so there is no separate device pool. **Peak RSS** is the process's peak resident set (`getrusage`), which includes the Metal buffers.

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
| Octo-Small | 2 | 256 + 128 | 22.6 | 22.9 | 22.9 | 23.0 | 5.7 | 838 |
| TurboVLA | 2 | 256 | 65.2 | 65.4 | 65.4 | 65.5 | - | 1127 |
| VLA-JEPA | 1 | 256 | 248.7 | 249.0 | 248.9 | 249.2 | 86.0 | 4043 |
| VLA-Adapter | 1 | 224 | 315.7 | 316.3 | 316.3 | 316.5 | 174.1 | 3712 |
| GR00T N1.7 | 1 | 256 | 350.8 | 355.0 | 355.2 | 355.6 | 85.9 | 5125 |
| SmolVLA | 2 | 512 | 357.6 | 358.1 | 358.2 | 358.4 | 215.8 | 1208 |
| GR00T N1.6 | 1 | 224 | 358.9 | 361.6 | 361.7 | 362.0 | 100.3 | 4762 |
| GR00T N1.5 | 1 | 224 | 401.5 | 401.9 | 401.9 | 402.2 | 99.4 | 3753 |
| Evo-1 | 1 | 448 | 829.3 | 830.1 | 830.1 | 830.5 | 382.1 | 1637 |
| π0 | 2 | 224 | 1141.1 | 1149.7 | 1146.3 | 1159.6 | 199.5 | 5549 |
| π0.5 | 2 | 224 | 1153.8 | 1165.1 | 1161.1 | 1180.4 | 199.3 | 5963 |
| OpenVLA-OFT | 1 | 224 | 1650.7 | 1716.1 | 1740.0 | 1751.1 | 184.2 | 15479 |

### Fastest configuration

We screen each model over these runtime flag sets, then re-run the fastest set
under the full protocol:

- the defaults;
- `--flash-attn` and `--mm-prec default`, alone and together;
- `--weight-dtype bf16` and `--weight-dtype f16`, each alone and with
  `--flash-attn`.

The screen runs 2 warmups and 5 reps in one process per flag set. A flag set
replaces the defaults only if its mean is more than 3% lower, both in the screen
and in the full re-run.

Flash attention, bf16 activations and lower-precision weights can move the
action chunk. Success rates measured at the defaults therefore do not carry over
to these rows.

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | *(defaults)* | 22.6 | 22.9 | 22.9 | 23.0 | 5.7 | 838 | - |
| TurboVLA | `--weight-dtype f16 --flash-attn` | 54.8 | 55.0 | 55.0 | 55.2 | - | 747 | -16% |
| VLA-JEPA | *(defaults)* | 248.7 | 249.0 | 248.9 | 249.2 | 86.0 | 4043 | - |
| VLA-Adapter | *(defaults)* | 315.7 | 316.3 | 316.3 | 316.5 | 174.1 | 3712 | - |
| SmolVLA | `--weight-dtype f16 --flash-attn` | 325.6 | 326.0 | 325.9 | 326.2 | 185.9 | 1208 | -9% |
| GR00T N1.7 | *(defaults)* | 350.8 | 355.0 | 355.2 | 355.6 | 85.9 | 5125 | - |
| GR00T N1.6 | *(defaults)* | 358.9 | 361.6 | 361.7 | 362.0 | 100.3 | 4762 | - |
| GR00T N1.5 | *(defaults)* | 401.5 | 401.9 | 401.9 | 402.2 | 99.4 | 3753 | - |
| Evo-1 | `--weight-dtype f16 --flash-attn` | 771.4 | 772.1 | 772.2 | 772.4 | 327.1 | 1637 | -7% |
| π0 | *(defaults)* | 1141.1 | 1149.7 | 1146.3 | 1159.6 | 199.5 | 5549 | - |
| π0.5 | *(defaults)* | 1153.8 | 1165.1 | 1161.1 | 1180.4 | 199.3 | 5963 | - |
| OpenVLA-OFT | *(defaults)* | 1650.7 | 1716.1 | 1740.0 | 1751.1 | 184.2 | 15479 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 22.9 | 23.1 | 23.2 | 22.9 | 23.1 | 23.1 | 23.1 | 23.0 |
| TurboVLA | 65.4 | 59.3 | 65.6 | 59.3 | 62.1 | 55.7 | 61.3 | 55.3 |
| VLA-JEPA | 249.2 | 251.3 | 249.0 | 250.7 | 249.2 | 250.4 | 247.0 | 248.6 |
| VLA-Adapter | 316.5 | 316.4 | 316.6 | 316.4 | 316.6 | 316.4 | 314.7 | 313.8 |
| GR00T N1.7 | 355.4 | 357.8 | 355.2 | 357.8 | 355.5 | 357.8 | 351.1 | 353.3 |
| SmolVLA | 358.8 | 329.8 | 358.6 | 330.6 | 358.4 | 329.2 | 354.2 | 326.4 |
| GR00T N1.6 | 361.9 | 363.3 | 361.6 | 363.2 | 361.5 | 363.4 | 357.6 | 357.9 |
| GR00T N1.5 | 402.0 | 401.9 | 402.2 | 402.5 | 401.9 | 402.5 | 398.3 | 399.0 |
| Evo-1 | 832.6 | 777.7 | 830.3 | 778.2 | 831.4 | 775.3 | 826.7 | 771.9 |
| π0 | 1179.1 | 1171.9 | 1178.9 | 1172.7 | 1182.7 | 1175.7 | 1172.0 | 1160.1 |
| π0.5 | 1160.7 | 1157.8 | 1156.8 | 1154.8 | 1156.8 | 1153.7 | 1132.6 | 1136.0 |
| OpenVLA-OFT | 1721.8 | 1699.7 | 1684.9 | 1681.7 | 1676.7 | 1672.2 | 1683.9 | 1684.9 |

</details>

## Not run

- **BitVLA**: N/A off CUDA. The only published GGUF is int2-packed, and only CUDA builds load it:

  ```text
  vla(bitvla): int2-packed GGUF requires a CUDA build (VLA_BITVLA_CUDA_KERNELS); use the bf16 GGUF for CPU.
  ```

## Reproducing

```bash
cmake -S . -B build-metal -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal -j

# one row, for example SmolVLA
./build-metal/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
```
