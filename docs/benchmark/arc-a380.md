# Intel Arc A380 (SYCL)

`vla-bench` times `predict()` in-process on synthetic inputs. It measures the
engine only: no transport, no simulator, and no claim about task success.

## Test setup

| | |
|---|---|
| Device | Intel Arc A380, 6 GB GDDR6 (Alchemist, DG2) |
| Host | AMD Ryzen 5 5500 (6 cores, 12 threads), 15.5 GiB RAM |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8 (i915) |
| Driver / toolkit | Intel compute runtime 24.39.31294.20, Level Zero 1.17.44 / oneAPI 2025.3 (icx/icpx) |
| Commit | `c93ca0a` (PR #32 head, branch `b11223-numerics-perf`) |
| llama.cpp | `b11223` |
| Date | 2026-09-30 01:07 to 05:33 (UTC+07) |
| Build | `-DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release -G Ninja`, after `source /opt/intel/oneapi/setvars.sh` |
| Runtime flags | defaults; the fastest flags per model are in the second table |
| Method | 3 warmups + 20 timed reps per process, 3 processes. The process with the lowest mean is reported, and memory is the peak over all three. |

**Peak VRAM** is the process's device-local memory from the i915 DRM `fdinfo` counters (`drm-total-local0`), sampled every 0.5 s. **Peak host RSS** is the process's peak resident set (`getrusage`).

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
| Octo-Small | 2 | 256 + 128 | 60.5 | 61.0 | 61.0 | 61.3 | 4.8 | 585 | 497 |
| TurboVLA | 2 | 256 | 103.5 | 103.6 | 103.6 | 103.8 | - | 885 | 561 |
| VLA-JEPA | 1 | 256 | 355.9 | 356.8 | 356.8 | 357.3 | 109.3 | 3855 | 468 |
| VLA-Adapter | 1 | 224 | 408.9 | 410.0 | 410.2 | 410.4 | 193.7 | 2750 | 1102 |
| GR00T N1.5 | 1 | 224 | 472.0 | 472.7 | 472.7 | 473.3 | 114.8 | 3531 | 496 |
| GR00T N1.7 | 1 | 256 | 615.3 | 618.4 | 618.2 | 620.9 | 109.9 | 4913 | 517 |
| GR00T N1.6 | 1 | 224 | 646.0 | 647.6 | 647.9 | 648.4 | 114.8 | 4558 | 493 |
| SmolVLA | 2 | 512 | 734.5 | 736.5 | 736.5 | 738.0 | 231.6 | 1159 | 508 |
| π0 | 2 | 224 | 921.9 | 924.2 | 924.1 | 925.4 | 228.3 | 5366 | 513 |
| π0.5 | 2 | 224 | 945.6 | 948.1 | 948.2 | 950.0 | 228.3 | 5659 | 513 |
| Evo-1 | 1 | 448 | 1047.3 | 1048.8 | 1048.8 | 1049.8 | 354.6 | 1467 | 544 |

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

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak VRAM MiB | Peak host RSS MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | *(defaults)* | 60.5 | 61.0 | 61.0 | 61.3 | 4.8 | 585 | 497 | - |
| TurboVLA | `--weight-dtype f16` | 89.7 | 89.9 | 89.9 | 90.1 | - | 541 | 561 | -13% |
| VLA-JEPA | *(defaults)* | 355.9 | 356.8 | 356.8 | 357.3 | 109.3 | 3855 | 468 | - |
| VLA-Adapter | *(defaults)* | 408.9 | 410.0 | 410.2 | 410.4 | 193.7 | 2750 | 1102 | - |
| GR00T N1.5 | *(defaults)* | 472.0 | 472.7 | 472.7 | 473.3 | 114.8 | 3531 | 496 | - |
| GR00T N1.7 | *(defaults)* | 615.3 | 618.4 | 618.2 | 620.9 | 109.9 | 4913 | 517 | - |
| GR00T N1.6 | *(defaults)* | 646.0 | 647.6 | 647.9 | 648.4 | 114.8 | 4558 | 493 | - |
| SmolVLA | *(defaults)* | 734.5 | 736.5 | 736.5 | 738.0 | 231.6 | 1159 | 508 | - |
| π0 | *(defaults)* | 921.9 | 924.2 | 924.1 | 925.4 | 228.3 | 5366 | 513 | - |
| π0.5 | *(defaults)* | 945.6 | 948.1 | 948.2 | 950.0 | 228.3 | 5659 | 513 | - |
| Evo-1 | *(defaults)* | 1047.3 | 1048.8 | 1048.8 | 1049.8 | 354.6 | 1467 | 544 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 61.4 | 65.1 | 71.3 | 65.4 | 65.3 | 61.4 | 64.6 | 62.7 |
| TurboVLA | 103.5 | 110.7 | 104.0 | 110.7 | 100.3 | 103.9 | 90.0 | 94.1 |
| VLA-JEPA | 356.6 | 346.2 | 356.9 | 346.7 | 358.2 | 346.1 | 558.6 | 552.8 |
| VLA-Adapter | 410.2 | 410.0 | 410.2 | 410.2 | 410.2 | 410.0 | 573.5 | 573.4 |
| GR00T N1.5 | 471.3 | 485.3 | 470.3 | 485.3 | 472.6 | 484.3 | 816.2 | 829.6 |
| GR00T N1.7 | 619.4 | 608.7 | 618.7 | 610.8 | 619.4 | 610.5 | 1058.2 | 1053.0 |
| GR00T N1.6 | 646.8 | 658.3 | 646.5 | 661.8 | 646.4 | 660.8 | 1043.1 | 1061.3 |
| SmolVLA | 736.8 | 953.3 | 737.1 | 950.1 | 735.3 | 951.0 | 814.8 | 1035.3 |
| π0 | 923.4 | 1389.6 | 922.7 | 1391.1 | 923.3 | 1389.7 | 1885.7 | 2345.4 |
| π0.5 | 946.7 | 948.1 | 946.9 | 947.5 | 946.7 | 947.9 | 1910.3 | 1911.4 |
| Evo-1 | 1048.1 | 1317.8 | 1048.5 | 1318.8 | 1048.7 | 1318.5 | 1393.3 | 1666.7 |

</details>

## Not run

- **BitVLA**: N/A off CUDA. The only published GGUF is int2-packed, and only CUDA builds load it:

  ```text
  vla(bitvla): int2-packed GGUF requires a CUDA build (VLA_BITVLA_CUDA_KERNELS); use the bf16 GGUF for CPU.
  ```
- **OpenVLA-OFT**: does not fit. The weights load, but the first predict runs out of memory:

  ```text
  level_zero backend failed with error: 38 (UR_RESULT_ERROR_OUT_OF_HOST_MEMORY) exception caught at ggml/src/ggml-sycl/ggml-sycl.cpp, line:5835
  Error OP CONCAT
  ```

## Reproducing

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl -G Ninja -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build build-sycl -j

# one row, for example SmolVLA
./build-sycl/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
```
