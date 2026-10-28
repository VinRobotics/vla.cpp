# Intel Core i9-14900HX (CPU)

`vla-bench` times `predict()` in-process on synthetic inputs. It measures the
engine only: no transport, no simulator, and no claim about task success.

## Test setup

| | |
|---|---|
| Device | Intel Core i9-14900HX (laptop), 8 performance + 16 efficiency cores, 32 threads; on AC power, platform profile `performance` |
| Memory | 31 GiB |
| Power | PL1 = PL2 = 250 W; `intel_pstate` `powersave` governor |
| Threads | 16 (the default: all hardware threads, capped at 16) |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8 |
| Toolchain | GCC 11.4, CMake 3.22 |
| Commit | `c93ca0a` (PR #32 head, branch `b11223-numerics-perf`) |
| llama.cpp | `b11223` |
| Date | 2026-09-30 00:59 to 03:56 (UTC+07) |
| Build | `-DCMAKE_BUILD_TYPE=Release` (`GGML_NATIVE=ON`), CPU backend only |
| Runtime flags | defaults; the fastest flags per model are in the second table |
| Method | 3 warmups + 20 timed reps per process, 3 processes. The process with the lowest mean is reported, and memory is the peak over all three. |

**Peak RSS** is the process's peak resident set (`getrusage`).

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
| Octo-Small | 2 | 256 + 128 | 67.0 | 71.8 | 72.2 | 75.0 | 8.4 | 622 |
| TurboVLA | 2 | 256 | 314.7 | 325.9 | 325.5 | 333.7 | - | 854 |
| VLA-JEPA | 1 | 256 | 1227.2 | 1273.3 | 1280.7 | 1294.7 | 400.9 | 3822 |
| GR00T N1.7 | 1 | 256 | 1505.5 | 1530.8 | 1533.0 | 1545.2 | 400.5 | 4883 |
| VLA-Adapter | 1 | 224 | 1659.3 | 1689.6 | 1684.4 | 1729.9 | 980.8 | 2706 |
| GR00T N1.6 | 1 | 224 | 1688.9 | 1721.9 | 1717.9 | 1745.8 | 496.1 | 4524 |
| SmolVLA | 2 | 512 | 1822.6 | 1855.6 | 1854.4 | 1874.7 | 1179.1 | 1099 |
| GR00T N1.5 | 1 | 224 | 1935.4 | 1966.1 | 1968.8 | 1983.0 | 494.0 | 3499 |
| Evo-1 | 1 | 448 | 4118.2 | 4155.7 | 4152.5 | 4184.5 | 1875.2 | 1431 |
| π0.5 | 2 | 224 | 5735.0 | 5913.9 | 5799.2 | 6092.9 | 1001.6 | 5657 |
| π0 | 2 | 224 | 6182.1 | 6342.8 | 6250.5 | 6507.3 | 1035.7 | 5353 |
| OpenVLA-OFT | 1 | 224 | 9126.4 | 9636.7 | 9836.8 | 9854.8 | 1008.6 | 14796 |

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
| Octo-Small | *(defaults)* | 67.0 | 71.8 | 72.2 | 75.0 | 8.4 | 622 | - |
| TurboVLA | *(defaults)* | 314.7 | 325.9 | 325.5 | 333.7 | - | 854 | - |
| VLA-JEPA | *(defaults)* | 1227.2 | 1273.3 | 1280.7 | 1294.7 | 400.9 | 3822 | - |
| GR00T N1.7 | *(defaults)* | 1505.5 | 1530.8 | 1533.0 | 1545.2 | 400.5 | 4883 | - |
| SmolVLA | `--flash-attn` | 1630.3 | 1661.5 | 1658.8 | 1681.5 | 982.4 | 1048 | -10% |
| VLA-Adapter | *(defaults)* | 1659.3 | 1689.6 | 1684.4 | 1729.9 | 980.8 | 2706 | - |
| GR00T N1.6 | *(defaults)* | 1688.9 | 1721.9 | 1717.9 | 1745.8 | 496.1 | 4524 | - |
| GR00T N1.5 | *(defaults)* | 1935.4 | 1966.1 | 1968.8 | 1983.0 | 494.0 | 3499 | - |
| Evo-1 | `--flash-attn --mm-prec default` | 3906.7 | 3952.3 | 3951.2 | 3981.4 | 1711.6 | 1366 | -5% |
| π0.5 | *(defaults)* | 5735.0 | 5913.9 | 5799.2 | 6092.9 | 1001.6 | 5657 | - |
| π0 | *(defaults)* | 6182.1 | 6342.8 | 6250.5 | 6507.3 | 1035.7 | 5353 | - |
| OpenVLA-OFT | *(defaults)* | 9126.4 | 9636.7 | 9836.8 | 9854.8 | 1008.6 | 14796 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 72.5 | 74.2 | 72.3 | 73.4 | 75.7 | 72.3 | 73.5 | 72.0 |
| TurboVLA | 326.7 | 321.6 | 328.5 | 316.7 | 327.4 | 321.4 | 334.9 | 336.0 |
| VLA-JEPA | 1280.7 | 1289.1 | 1268.9 | 1284.5 | 1297.0 | 1275.7 | 1337.9 | 1349.0 |
| GR00T N1.7 | 1530.0 | 1523.1 | 1528.9 | 1522.2 | 1537.9 | 1530.8 | 1562.2 | 1575.5 |
| VLA-Adapter | 1710.7 | 1697.9 | 1714.1 | 1710.8 | 1705.3 | 1693.9 | 1794.8 | 1796.8 |
| GR00T N1.6 | 1736.3 | 1752.0 | 1726.9 | 1764.6 | 1723.6 | 1777.8 | 1725.8 | 1808.8 |
| SmolVLA | 1843.8 | 1696.3 | 1866.4 | 1700.3 | 1872.5 | 1699.9 | 1927.8 | 1742.6 |
| GR00T N1.5 | 1977.4 | 2009.7 | 1993.0 | 2003.7 | 1972.1 | 1983.2 | 2028.2 | 2083.7 |
| Evo-1 | 4195.4 | 3981.5 | 4182.1 | 3964.9 | 4183.0 | 3966.7 | 4254.8 | 4055.9 |
| π0.5 | 6015.6 | 6034.9 | 6020.5 | 6012.2 | 6022.9 | 6038.4 | 6055.9 | 6032.6 |
| π0 | 6393.7 | 6344.9 | 6397.6 | 6335.9 | 6407.8 | 6321.3 | 6460.6 | 6513.6 |
| OpenVLA-OFT | 9592.9 | 9575.3 | 9568.9 | 9583.8 | 9589.3 | 9713.6 | 10282.6 | 10295.8 |

</details>

## Not run

- **BitVLA**: N/A off CUDA. The only published GGUF is int2-packed, and only CUDA builds load it:

  ```text
  vla(bitvla): int2-packed GGUF requires a CUDA build (VLA_BITVLA_CUDA_KERNELS); use the bf16 GGUF for CPU.
  ```

## Reproducing

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j

# one row, for example SmolVLA
./build-cpu/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
```
