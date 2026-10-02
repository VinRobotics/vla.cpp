# Snapdragon X: Hexagon NPU and Oryon CPU

`vla-bench` times `predict()` in-process on synthetic inputs. It measures the
engine only: no transport, no simulator, and no claim about task success.

One `build-wos-htp` binary covers both sections. By default the graph runs on the Hexagon NPU, and
ops the NPU rejects run on the CPU through vla.cpp's fallback wrapper. `VLA_DEVICE=cpu` skips the
NPU, so the same binary gives the CPU numbers.

## Test setup

| | |
|---|---|
| Device | ASUS Vivobook 14 X1407QA laptop: Snapdragon X X1-26-100 (8 Oryon cores), Hexagon v73 NPU (driver 30.0.220.3000), 16 GB RAM (15.6 GiB visible) |
| Threads | 8 (the default: all hardware threads), for the CPU runs and the NPU's CPU fallback |
| Power | on AC power, Windows power plan `Balanced` |
| OS | Windows 11 Home, build 26200 |
| Toolchain | Visual Studio clang 22.1.3, Hexagon SDK 6.6.0.0 (tools 19.0.07) |
| Commit | `c93ca0a` (PR #32 head, branch `b11223-numerics-perf`) |
| llama.cpp | `b11223` |
| Date | 2026-09-30 01:40 to 05:58 (UTC+07) |
| Build | `scripts/build_windows_snapdragon.ps1 -Backend htp -HtpCert <cert.pfx> -LlamaDir <llama.cpp b11223>` (servers included) |
| Runtime flags | defaults; the fastest flags per model are in the second table |
| Method | 3 warmups + 20 timed reps per process, 3 processes. The process with the lowest mean is reported, and memory is the peak over all three. |

**Peak working set** is the process's peak working set from `GetProcessMemoryInfo`. On the NPU the weights live in FastRPC shared memory, which counts toward the working set. **Peak private** is the process's peak private commit (`PeakPagefileUsage`).

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

### How the fastest configuration is picked

We screen each model over these runtime flag sets, then re-run the fastest set
under the full protocol:

- the defaults;
- `--flash-attn` and `--mm-prec default`, alone and together;
- `--weight-dtype bf16` and `--weight-dtype f16`, each alone and with
  `--flash-attn`;
- `--flash-attn 0`, because flash attention is on by default in Hexagon builds.

The screen runs 2 warmups and 5 reps in one process per flag set. A flag set
replaces the defaults only if its mean is more than 3% lower, both in the screen
and in the full re-run.

Flash attention, bf16 activations and lower-precision weights can move the
action chunk. Success rates measured at the defaults therefore do not carry over
to these rows.

## Latency and memory: Hexagon NPU (with CPU fallback)

Rows are sorted by latency. `vision ms` is `-` for archs that do not time their
vision stage separately.

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak working set MiB | Peak private MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 2 | 256 + 128 | 239.7 | 266.3 | 265.4 | 288.4 | 51.2 | 679 | 218 |
| VLA-JEPA | 1 | 256 | 821.0 | 846.2 | 846.0 | 859.0 | 369.2 | 3900 | 1105 |
| SmolVLA | 2 | 512 | 1239.3 | 1246.9 | 1247.7 | 1249.5 | 430.6 | 1129 | 334 |
| GR00T N1.7 | 1 | 256 | 1400.2 | 1421.3 | 1417.2 | 1438.2 | 367.9 | 4964 | 625 |
| TurboVLA | 2 | 256 | 2280.8 | 2292.7 | 2289.2 | 2305.5 | - | 948 | 208 |
| GR00T N1.5 | 1 | 224 | 2755.7 | 2773.9 | 2773.0 | 2784.5 | 1802.1 | 3560 | 151 |
| GR00T N1.6 | 1 | 224 | 2940.0 | 2957.1 | 2956.9 | 2970.7 | 1861.1 | 4601 | 164 |
| VLA-Adapter | 1 | 224 | 4072.1 | 4099.2 | 4101.9 | 4114.7 | 2888.8 | 2796 | 861 |
| π0 | 2 | 224 | 5730.0 | 5739.3 | 5740.4 | 5743.9 | 3614.4 | 5384 | 230 |
| Evo-1 | 1 | 448 | 7208.6 | 7292.2 | 7302.9 | 7344.8 | 1054.1 | 1499 | 276 |
| π0.5 | 2 | 224 | 9321.7 | 9363.2 | 9367.4 | 9395.1 | 4531.4 | 5701 | 230 |

### Fastest configuration: Hexagon NPU (with CPU fallback)

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak working set MiB | Peak private MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | *(defaults)* | 239.7 | 266.3 | 265.4 | 288.4 | 51.2 | 679 | 218 | - |
| TurboVLA | `--weight-dtype f16 --flash-attn` | 490.1 | 554.7 | 541.9 | 600.9 | - | 586 | 208 | -76% |
| VLA-JEPA | *(defaults)* | 821.0 | 846.2 | 846.0 | 859.0 | 369.2 | 3900 | 1105 | - |
| SmolVLA | *(defaults)* | 1239.3 | 1246.9 | 1247.7 | 1249.5 | 430.6 | 1129 | 334 | - |
| GR00T N1.7 | *(defaults)* | 1400.2 | 1421.3 | 1417.2 | 1438.2 | 367.9 | 4964 | 625 | - |
| GR00T N1.5 | *(defaults)* | 2755.7 | 2773.9 | 2773.0 | 2784.5 | 1802.1 | 3560 | 151 | - |
| GR00T N1.6 | *(defaults)* | 2940.0 | 2957.1 | 2956.9 | 2970.7 | 1861.1 | 4601 | 164 | - |
| VLA-Adapter | *(defaults)* | 4072.1 | 4099.2 | 4101.9 | 4114.7 | 2888.8 | 2796 | 861 | - |
| π0 | *(defaults)* | 5730.0 | 5739.3 | 5740.4 | 5743.9 | 3614.4 | 5384 | 230 | - |
| Evo-1 | *(defaults)* | 7208.6 | 7292.2 | 7302.9 | 7344.8 | 1054.1 | 1499 | 276 | - |
| π0.5 | *(defaults)* | 9321.7 | 9363.2 | 9367.4 | 9395.1 | 4531.4 | 5701 | 230 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` | `--flash-attn 0` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 275.6 | 270.9 | 262.0 | 272.9 | 278.9 | 270.7 | 262.8 | 280.2 | 271.8 |
| VLA-JEPA | 846.2 | 849.0 | 848.2 | 853.7 | 6512.3 | 6373.1 | 856.8 | 846.5 | 1424.1 |
| SmolVLA | 1251.1 | 1251.4 | 1245.3 | 1250.3 | 7625.0 | 7714.3 | 1248.8 | 1254.5 | 4753.7 |
| GR00T N1.7 | 1418.6 | 1415.1 | 1426.4 | 1415.0 | 7752.4 | 8351.3 | 1425.8 | 1416.1 | 1910.3 |
| TurboVLA | 2284.2 | 2285.7 | 2282.8 | 2294.4 | 1437.3 | 1426.4 | 552.1 | 516.9 | 2591.1 |
| GR00T N1.5 | 2786.6 | 2787.3 | 2775.8 | 2778.1 | 10053.6 | 10119.7 | 2777.8 | 2787.3 | 3772.6 |
| GR00T N1.6 | 2962.3 | 2960.8 | 2969.8 | 2962.5 | 8713.5 | 8558.1 | 2962.7 | 2959.9 | 3510.4 |
| VLA-Adapter | 4160.1 | 4123.3 | 4095.0 | 4095.5 | 9240.4 | 9441.1 | 4135.3 | 4121.3 | 4140.0 |
| π0 | 5736.0 | 5734.2 | 5731.2 | 5735.4 | 59145.8 | 57928.4 | 5724.3 | 5716.5 | 9426.5 |
| Evo-1 | 7331.0 | 7301.4 | 7314.1 | 7321.5 | 20437.4 | 20437.9 | 7325.5 | 7316.4 | 12709.7 |
| π0.5 | 9377.9 | 9397.3 | 9412.7 | 9348.2 | 58598.5 | 60551.1 | 9368.5 | 9394.8 | 9335.6 |

</details>

## Latency and memory: Oryon CPU (`VLA_DEVICE=cpu`)

Rows are sorted by latency. `vision ms` is `-` for archs that do not time their
vision stage separately.

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak working set MiB | Peak private MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 2 | 256 + 128 | 109.5 | 111.6 | 111.1 | 114.3 | 13.1 | 623 | 729 |
| TurboVLA | 2 | 256 | 548.9 | 591.2 | 617.6 | 625.4 | - | 847 | 1020 |
| VLA-JEPA | 1 | 256 | 1656.5 | 1668.1 | 1664.1 | 1674.2 | 575.5 | 3826 | 4886 |
| GR00T N1.7 | 1 | 256 | 1988.1 | 2018.0 | 2002.6 | 2043.9 | 570.8 | 4886 | 5460 |
| VLA-Adapter | 1 | 224 | 2013.1 | 2181.2 | 2180.4 | 2209.5 | 1279.4 | 2708 | 3508 |
| SmolVLA | 2 | 512 | 2080.9 | 2312.7 | 2321.2 | 2337.6 | 1540.8 | 1063 | 1352 |
| GR00T N1.6 | 1 | 224 | 2226.0 | 2352.5 | 2272.7 | 2524.1 | 717.9 | 4524 | 4648 |
| GR00T N1.5 | 1 | 224 | 2736.6 | 2863.4 | 2761.5 | 3080.6 | 734.4 | 3488 | 3599 |
| Evo-1 | 1 | 448 | 5518.9 | 5885.5 | 6116.9 | 6142.7 | 2785.1 | 1363 | 1493 |
| π0.5 | 2 | 224 | 9140.6 | 10325.4 | 10198.6 | 11022.6 | 1614.3 | 5643 | 5674 |
| π0 | 2 | 224 | 9224.5 | 10401.7 | 10298.1 | 11023.1 | 1556.6 | 5321 | 5472 |

### Fastest configuration: Oryon CPU (`VLA_DEVICE=cpu`)

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak working set MiB | Peak private MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | *(defaults)* | 109.5 | 111.6 | 111.1 | 114.3 | 13.1 | 623 | 729 | - |
| TurboVLA | `--weight-dtype f16` | 358.8 | 363.3 | 363.1 | 367.3 | - | 485 | 655 | -39% |
| VLA-JEPA | *(defaults)* | 1656.5 | 1668.1 | 1664.1 | 1674.2 | 575.5 | 3826 | 4886 | - |
| GR00T N1.7 | *(defaults)* | 1988.1 | 2018.0 | 2002.6 | 2043.9 | 570.8 | 4886 | 5460 | - |
| VLA-Adapter | *(defaults)* | 2013.1 | 2181.2 | 2180.4 | 2209.5 | 1279.4 | 2708 | 3508 | - |
| SmolVLA | *(defaults)* | 2080.9 | 2312.7 | 2321.2 | 2337.6 | 1540.8 | 1063 | 1352 | - |
| GR00T N1.6 | *(defaults)* | 2226.0 | 2352.5 | 2272.7 | 2524.1 | 717.9 | 4524 | 4648 | - |
| GR00T N1.5 | *(defaults)* | 2736.6 | 2863.4 | 2761.5 | 3080.6 | 734.4 | 3488 | 3599 | - |
| Evo-1 | *(defaults)* | 5518.9 | 5885.5 | 6116.9 | 6142.7 | 2785.1 | 1363 | 1493 | - |
| π0.5 | *(defaults)* | 9140.6 | 10325.4 | 10198.6 | 11022.6 | 1614.3 | 5643 | 5674 | - |
| π0 | *(defaults)* | 9224.5 | 10401.7 | 10298.1 | 11023.1 | 1556.6 | 5321 | 5472 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` | `--flash-attn 0` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 125.1 | 133.6 | 126.2 | 126.0 | 124.5 | 125.9 | 128.9 | 126.1 | 126.2 |
| TurboVLA | 619.8 | 622.1 | 620.0 | 621.3 | 1138.8 | 1141.9 | 361.8 | 366.7 | 629.7 |
| VLA-JEPA | 1912.1 | 1763.2 | 1673.6 | 1753.8 | 6252.8 | 5802.0 | 2477.1 | 1910.1 | 1723.4 |
| GR00T N1.7 | 2244.6 | 2067.0 | 2253.5 | 2088.2 | 7437.8 | 7080.7 | 2077.1 | 2123.7 | 2272.3 |
| VLA-Adapter | 2465.6 | 2454.5 | 2684.9 | 2445.1 | 7794.7 | 7421.5 | 2381.1 | 2458.7 | 2463.8 |
| SmolVLA | 2883.6 | 2629.0 | 2642.4 | 2642.4 | 7254.5 | 7268.4 | 2698.5 | 2638.0 | 3001.2 |
| GR00T N1.6 | 2583.0 | 2395.4 | 2336.1 | 2386.6 | 8396.8 | 8137.0 | 2298.5 | 2452.0 | 2419.4 |
| GR00T N1.5 | 3192.7 | 3127.0 | 3110.0 | 3114.5 | 10416.3 | 10197.1 | 3089.9 | 3109.3 | 3353.4 |
| Evo-1 | 6223.3 | 6346.8 | 6301.9 | 6390.8 | 18169.2 | 17479.1 | 6138.2 | 6340.4 | 6515.3 |
| π0.5 | 11258.2 | 11366.8 | 11239.9 | 11168.1 | 33696.1 | 34020.8 | 11163.7 | 11248.8 | 11292.2 |
| π0 | 11321.0 | 11457.8 | 11441.3 | 11290.5 | 33961.6 | 34012.7 | 10983.4 | 11208.4 | 11245.1 |

</details>

## Not run

### Hexagon NPU (with CPU fallback)

- **BitVLA**: N/A off CUDA. The only published GGUF is int2-packed, and only CUDA builds load it:

  ```text
  vla(bitvla): int2-packed GGUF requires a CUDA build (VLA_BITVLA_CUDA_KERNELS); use the bf16 GGUF for CPU.
  ```
- **OpenVLA-OFT**: not attempted. Its 14 GiB of weights do not fit next to Windows in 16 GB of RAM.

### Oryon CPU (`VLA_DEVICE=cpu`)

- **BitVLA**: N/A off CUDA. The only published GGUF is int2-packed, and only CUDA builds load it:

  ```text
  vla(bitvla): int2-packed GGUF requires a CUDA build (VLA_BITVLA_CUDA_KERNELS); use the bf16 GGUF for CPU.
  ```
- **OpenVLA-OFT**: not attempted. Its 14 GiB of weights do not fit next to Windows in 16 GB of RAM.

## Notes

`ADSP_LIBRARY_PATH` must be unset, or name the skels built with the binary. On this machine it pointed at the skels of an older llama.cpp build (b11201). With those loaded, every model aborted at its first graph with `ggml-hex: dspqueue_read failed: 0x00000072`. With it unset, the backend loads the skels next to the executable, and every model runs.

In this build the default weight dtype is F16, for the CPU runs as well as the NPU. The screen shows it: `--weight-dtype f16` tracks the defaults, while `--weight-dtype bf16` is about 3x slower. TurboVLA's checkpoint is F32 and stays F32 at the defaults, which is why `--weight-dtype f16` helps it so much.

The CPU rows vary more than on the desktop hosts. For example, π0.5's `min` is 9.1 s against a 10.3 s `mean`. The laptop ran on the `Balanced` power plan, and neither clocks nor temperature were controlled.

## Reproducing

```powershell
.\scripts\build_windows_snapdragon.ps1 -Backend htp -LlamaDir <llama.cpp b11223> -HtpCert <cert.pfx>
Remove-Item Env:ADSP_LIBRARY_PATH -ErrorAction SilentlyContinue

# one row on the NPU, for example SmolVLA
.\build-wos-htp\bin\vla-bench.exe --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
# the same row on the CPU
$env:VLA_DEVICE = "cpu"; .\build-wos-htp\bin\vla-bench.exe --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
```
