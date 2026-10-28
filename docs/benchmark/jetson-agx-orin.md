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
| Commit | `e6653e8` (branch `perf/cuda-exact-kernels`, on `1012c05`) |
| llama.cpp | `b11223` |
| Date | 2026-10-22 17:20 to 17:57, `1012c05` 2026-10-23 03:08 to 03:21 (UTC+07) |
| Build | `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release` (`GGML_NATIVE=ON`, `VLA_SPM=OFF`) |
| Runtime flags | defaults; the fastest flags per model are under [Fastest configuration](#fastest-configuration) |
| Method | 3 warmups + 20 timed reps per process, 3 processes. The process with the lowest mean is reported. |

The defaults run the bit-exact CUDA path
([backend/cuda.md](../backend/cuda.md#the-bit-exact-fast-path)): the actions are the same
bits as the plain ggml graph, so success rates measured at the defaults carry
over. The [comparison below](#against-1012c05) re-measures `1012c05`, the
commit before it, with the same protocol.

Memory is unified, and `nvidia-smi` reports no per-process usage on Jetson.
In this round `getrusage` did not count the CUDA buffers on this board (the
[Orin NX](jetson-orin-nx.md) report saw the same). **CUDA MiB** is instead the
peak of the device memory the process holds through `cudaMalloc` and
`cuMemCreate` (ggml's VMM pool), counted by an `LD_PRELOAD` shim over one run:
weights, compute buffers, and the fast path's own buffers (cuBLAS workspace,
per-shape attention and BF16 staging buffers). Host memory is not included.

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

## Latency

Rows are sorted by latency. `vision ms` is `-` for archs that do not time their
vision stage separately. GR00T N1.7 runs its vision tower inside the main graph
on this path, so it has no separate vision time either.

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | CUDA MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 2 | 256 + 128 | 14.3 | 20.2 | 17.9 | 28.1 | 2.9 | 554 |
| TurboVLA | 2 | 256 | 28.0 | 29.9 | 28.4 | 35.0 | - | 863 |
| VLA-JEPA | 1 | 256 | 60.6 | 61.4 | 61.3 | 61.7 | 19.4 | 3890 |
| BitVLA | 1 | 224 | 76.8 | 80.0 | 77.3 | 87.3 | 15.9 | 1037 |
| VLA-Adapter | 1 | 224 | 79.1 | 79.7 | 79.7 | 80.0 | 39.5 | 2792 |
| GR00T N1.7 | 1 | 256 | 93.0 | 94.0 | 94.1 | 94.2 | - | 4953 |
| GR00T N1.5 | 1 | 224 | 96.7 | 98.0 | 97.9 | 98.7 | 21.2 | 3578 |
| GR00T N1.6 | 1 | 224 | 101.2 | 102.1 | 102.2 | 102.3 | 21.4 | 4570 |
| SmolVLA | 2 | 512 | 164.3 | 165.2 | 165.0 | 166.5 | 72.6 | 1896 |
| π0.5 | 2 | 224 | 266.6 | 267.1 | 267.2 | 267.4 | 42.2 | 5630 |
| π0 | 2 | 224 | 269.0 | 269.4 | 269.5 | 269.8 | 41.6 | 5521 |
| OpenVLA-OFT | 1 | 224 | 297.4 | 297.7 | 297.7 | 298.0 | 40.9 | 14648 |
| Evo-1 | 1 | 448 | 298.5 | 299.0 | 298.8 | 299.7 | 124.0 | 1566 |

### Against `1012c05`

The same protocol on `1012c05` (upstream `main` before this path), same board
and build flags: mean latency at the defaults, and the CUDA memory as above.
The fast path adds 0 to 207 MiB; the most relative to the model is Evo-1's
167 MiB (+12%).

| Model | `1012c05` ms | this branch ms | change | `1012c05` CUDA MiB | this branch CUDA MiB |
|---|--:|--:|--:|--:|--:|
| Octo-Small | 21.2-24.9 | 18.7-21.6 | noise | 545 | 554 |
| TurboVLA | 41.0 | 29.9 | -27% | 847 | 863 |
| VLA-JEPA | 97.7 | 61.4 | -37% | 3805 | 3890 |
| BitVLA | 134.2 | 80.0 | -40% | 1035 | 1037 |
| VLA-Adapter | 126.6 | 79.7 | -37% | 2692 | 2792 |
| GR00T N1.7 | 133.8 | 94.0 | -30% | 4860 | 4953 |
| GR00T N1.5 | 131.2 | 98.0 | -25% | 3470 | 3578 |
| GR00T N1.6 | 133.1 | 102.1 | -23% | 4505 | 4570 |
| SmolVLA | 216.3 | 165.2 | -24% | 1846 | 1896 |
| π0.5 | 353.2 | 267.1 | -24% | 5630 | 5630 |
| π0 | 349.6 | 269.4 | -23% | 5338 | 5521 |
| OpenVLA-OFT | 385.0 | 297.7 | -23% | 14441 | 14648 |
| Evo-1 | 434.3 | 299.0 | -31% | 1399 | 1566 |

Octo-Small is bound by the host: its graphs are small, and the time goes to
launching them, not to the kernels. Its mean moves by 15% between runs of the
same build, so its row gives the range over four runs of upstream and five of
this branch, some taken before the rebase onto `1012c05` (no Octo code changed
in it).

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
| Octo-Small | `--flash-attn` | 13.3 | 17.8 | 16.0 | 23.2 | 2.5 | -12% |
| TurboVLA | *(defaults)* | 28.0 | 29.9 | 28.4 | 35.0 | - | - |
| VLA-JEPA | *(defaults)* | 60.6 | 61.4 | 61.3 | 61.7 | 19.4 | - |
| BitVLA | *(defaults)* | 76.8 | 80.0 | 77.3 | 87.3 | 15.9 | - |
| VLA-Adapter | *(defaults)* | 79.1 | 79.7 | 79.7 | 80.0 | 39.5 | - |
| GR00T N1.7 | *(defaults)* | 93.0 | 94.0 | 94.1 | 94.2 | - | - |
| GR00T N1.5 | *(defaults)* | 96.7 | 98.0 | 97.9 | 98.7 | 21.2 | - |
| GR00T N1.6 | *(defaults)* | 101.2 | 102.1 | 102.2 | 102.3 | 21.4 | - |
| SmolVLA | `--flash-attn --mm-prec default` | 140.5 | 141.6 | 141.1 | 143.4 | 72.4 | -14% |
| π0.5 | *(defaults)* | 266.6 | 267.1 | 267.2 | 267.4 | 42.2 | - |
| π0 | *(defaults)* | 269.0 | 269.4 | 269.5 | 269.8 | 41.6 | - |
| OpenVLA-OFT | *(defaults)* | 297.4 | 297.7 | 297.7 | 298.0 | 40.9 | - |
| Evo-1 | `--flash-attn --mm-prec default` | 282.9 | 283.3 | 283.1 | 284.1 | 107.6 | -5% |

The flags that change numerics leave the bit-exact path, so on most models they
are now slower than the defaults. Octo-Small's `--flash-attn` lead is within its
run-to-run noise (see above).

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` | `--act-dtype bf16 --flash-attn` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 23.1 | 20.9 | 25.5 | 26.1 | 25.3 | 25.8 | 25.7 | 26.6 |  |
| TurboVLA | 32.9 | 33.9 | 34.4 | 35.9 | 42.3 | 32.8 | 33.8 | 35.2 |  |
| VLA-JEPA | 61.2 | 61.6 | 62.5 | 60.3 | 62.4 | 60.4 | 87.1 | 86.7 |  |
| BitVLA | 86.2 | 88.6 | 86.4 | 88.6 | 87.6 | 87.8 | 139.7 | 88.2 |  |
| VLA-Adapter | 80.9 | 80.8 | 80.3 | 80.1 | 80.1 | 80.1 | 119.9 | 118.5 |  |
| GR00T N1.7 | 93.7 | 92.9 | 93.9 | 92.1 | 94.0 | 92.7 | 102.8 | 101.8 |  |
| GR00T N1.5 | 98.9 | 97.1 | 98.2 | 99.3 | 97.8 | 99.4 | 107.5 | 112.2 |  |
| GR00T N1.6 | 102.2 | 105.6 | 102.5 | 105.3 | 102.4 | 105.2 | 112.8 | 116.6 |  |
| SmolVLA | 164.2 | 164.9 | 142.5 | 142.3 | 193.8 | 192.9 | 203.1 | 195.2 |  |
| π0.5 | 268.7 | 269.0 | 268.0 | 269.6 | 269.0 | 268.0 | 340.4 | 340.8 |  |
| π0 | 269.6 | 315.6 | 270.0 | 316.2 | 271.1 | 315.4 | 338.7 | 330.3 | 301.8 |
| OpenVLA-OFT | 298.5 | 300.7 | 306.0 | 299.1 | 299.0 | 299.4 | 348.7 | 346.5 |  |
| Evo-1 | 300.9 | 286.3 | 299.4 | 285.1 | 301.2 | 285.7 | 322.1 | 299.7 | 311.0 |

</details>

## FoldQuant checkpoints

The calibrated FoldQuant LIBERO models (`vrfai/pi05-libero-w4a4`,
`vrfai/GR00T-N1.5-LIBERO-4suite-combined-w4a4`, `vrfai/GR00T-N1.6-LIBERO-w4a4`,
`vrfai/GR00T-N1.7-LIBERO-w4a4` (`libero_object`), and their `-res8` variants),
converted with `scripts/convert_quantized_model_to_gguf.py`
([QUANTIZATION.md](../QUANTIZATION.md)). `w4a4-res8` keeps the LLM's o and down
sites at INT8 (`site_bits = "o:8,down:8"`). Integer sums are exact in any
order, and the fused nodes keep ggml's rounding, so the actions are the same
bits as on `1012c05`; the action md5 of every file matched at one and two views.

Measured on 2026-10-26 with `jetson_clocks` on (GPU at 1.3 GHz), at `1d0b925`
(this branch before the rebase onto `5499b77`, which brought only SYCL and
OpenVINO changes) and `1012c05`, same build flags. Method as above, with the two
builds alternating inside each round. GR00T N1.6's quantized files take
252 px views (81 image tokens), from their config; the BF16 GGUF takes 224.

| Model | Config | Views | Input | `1012c05` ms | this branch ms | change |
|---|---|--:|--:|--:|--:|--:|
| GR00T N1.7 | w4a4 | 1 | 256 | 84.3 | 55.5 | -34% |
| GR00T N1.7 | w4a4-res8 | 1 | 256 | 85.3 | 56.4 | -34% |
| GR00T N1.5 | w4a4 | 1 | 224 | 96.9 | 67.4 | -30% |
| GR00T N1.5 | w4a4-res8 | 1 | 224 | 98.7 | 69.2 | -30% |
| GR00T N1.6 | w4a4 | 1 | 252 | 106.7 | 71.9 | -33% |
| GR00T N1.6 | w4a4-res8 | 1 | 252 | 108.1 | 72.8 | -33% |
| π0.5 | w4a4 | 2 | 224 | 216.0 | 142.9 | -34% |
| π0.5 | w4a4-res8 | 2 | 224 | 234.7 | 152.0 | -35% |

## Reproducing

```bash
cmake -S . -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j

# one row, for example SmolVLA
./build-cuda/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20

# a FoldQuant row, for example GR00T N1.7 w4a4
python scripts/convert_quantized_model_to_gguf.py --quantized-model GR00T-N1.7-LIBERO-w4a4/libero_object \
    --out gr00tn1d7-libero-object-w4a4.gguf
VLA_GR00T_EMBODIMENT=libero_sim ./build-cuda/vla-bench --ckpt gr00tn1d7-libero-object-w4a4.gguf \
    --images 1 --size 256 --warmup 3 --reps 20
```
