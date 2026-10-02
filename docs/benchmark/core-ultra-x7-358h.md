# Intel Core Ultra X7 358H: Arc B390 iGPU, AI Boost NPU and CPU

`vla-bench` times `predict()` in-process on synthetic inputs. It measures the
engine only: no transport, no simulator, and no claim about task success.

Four ways to run on this chip, from two builds. The OpenVINO build selects its device with
`GGML_OPENVINO_DEVICE`: `GPU` for the Arc B390 iGPU, `NPU` for the AI Boost NPU and `CPU` for
OpenVINO's CPU plugin. Every OpenVINO run was checked against ggml's `OpenVINO: using device` line,
so a silent fallback to another device would have been counted as a failure. A build without
OpenVINO gives ggml's own CPU backend.

## Test setup

| | |
|---|---|
| Device | Intel Core Ultra X7 358H (Panther Lake): 16 cores, 16 threads; Arc B390 iGPU; AI Boost NPU; 62 GiB of shared RAM |
| Threads | 16 (the default: all hardware threads, capped at 16), for both CPU sections |
| Power | platform profile `balanced`; PL1 25 W over a 27 s window, PL2 65 W; `intel_pstate` `powersave` governor |
| OS | Ubuntu 24.04.4 LTS, kernel 7.0 |
| Toolkit | OpenVINO 2026.4.0, the version `scripts/install_ov.sh` pins, unpacked into a private directory; the system's 2026.2.1 was not used. GPU compute runtime 26.22.38646.4, NPU driver 1.33.0, Level Zero loader 1.16.1, GCC 13.3 |
| Commit | `450992c` (PR #32 head, branch `b11223-numerics-perf`) |
| llama.cpp | `b11223` |
| Date | 2026-09-30 09:49 to 16:20 (UTC+07) |
| Build | OpenVINO: `-DGGML_OPENVINO=ON -DCMAKE_BUILD_TYPE=Release` (`GGML_NATIVE=ON`), after sourcing OpenVINO's `setupvars.sh`. ggml CPU backend: `-DCMAKE_BUILD_TYPE=Release` (`GGML_NATIVE=ON`) |
| Runtime flags | defaults; the fastest flags per model are in the second table |
| Method | 3 warmups + 20 timed reps per process, 3 processes. The process with the lowest mean is reported, and memory is the peak over all three. |

Memory is shared with the CPU. **Peak device buffers** is the process's GPU or NPU buffer memory from the DRM `fdinfo` counters, sampled every 0.5 s: `drm-total-gtt`, `-system` and `-stolen` for the xe GPU driver, and `drm-total-memory` for the intel_vpu NPU driver. **Peak RSS** is the process's peak resident set (`getrusage`). The two are separate counters, so adding them can double-count a buffer the CPU also maps. The two CPU sections report peak RSS only.

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
  `--flash-attn`.

The screen runs 2 warmups and 5 reps in one process per flag set. A flag set
replaces the defaults only if its mean is more than 3% lower, both in the screen
and in the full re-run.

Flash attention, bf16 activations and lower-precision weights can move the
action chunk. Success rates measured at the defaults therefore do not carry over
to these rows.

## Latency and memory: Arc B390 iGPU (OpenVINO GPU plugin)

Rows are sorted by latency. `vision ms` is `-` for archs that do not time their
vision stage separately.

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak device buffers MiB | Peak RSS MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| TurboVLA | 2 | 256 | 35.6 | 36.3 | 36.3 | 36.9 | - | 1080 | 1537 |
| GR00T N1.5 | 1 | 224 | 145.5 | 146.6 | 146.6 | 147.5 | 43.0 | 10556 | 6726 |
| VLA-JEPA | 1 | 256 | 152.4 | 154.5 | 154.6 | 155.5 | 28.9 | 9387 | 5321 |
| GR00T N1.6 | 1 | 224 | 225.1 | 231.1 | 230.9 | 238.3 | 20.4 | 14630 | 10159 |
| GR00T N1.7 | 1 | 256 | 230.8 | 232.2 | 231.9 | 233.8 | 29.9 | 14931 | 11043 |
| VLA-Adapter | 1 | 224 | 232.6 | 258.5 | 237.3 | 303.2 | 122.3 | 7424 | 3747 |
| SmolVLA | 2 | 512 | 499.8 | 505.4 | 505.4 | 511.7 | 115.7 | 4769 | 3614 |
| π0.5 | 2 | 224 | 581.5 | 586.4 | 586.0 | 589.3 | 86.8 | 15754 | 11428 |
| π0 | 2 | 224 | 795.8 | 802.8 | 803.0 | 805.1 | 87.2 | 25041 | 16994 |
| Evo-1 | 1 | 448 | 841.5 | 849.9 | 850.4 | 853.6 | 232.6 | 4965 | 3075 |
| OpenVLA-OFT | 1 | 224 | 1494.4 | 1588.3 | 1554.1 | 1689.6 | 169.2 | 41486 | 15266 |

### Fastest configuration: Arc B390 iGPU (OpenVINO GPU plugin)

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak device buffers MiB | Peak RSS MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| TurboVLA | `--weight-dtype bf16 --flash-attn` | 31.9 | 33.0 | 32.6 | 33.4 | - | 1330 | 1216 | -9% |
| GR00T N1.5 | *(defaults)* | 145.5 | 146.6 | 146.6 | 147.5 | 43.0 | 10556 | 6726 | - |
| VLA-JEPA | *(defaults)* | 152.4 | 154.5 | 154.6 | 155.5 | 28.9 | 9387 | 5321 | - |
| GR00T N1.6 | *(defaults)* | 225.1 | 231.1 | 230.9 | 238.3 | 20.4 | 14630 | 10159 | - |
| GR00T N1.7 | *(defaults)* | 230.8 | 232.2 | 231.9 | 233.8 | 29.9 | 14931 | 11043 | - |
| VLA-Adapter | *(defaults)* | 232.6 | 258.5 | 237.3 | 303.2 | 122.3 | 7424 | 3747 | - |
| SmolVLA | `--weight-dtype f16 --flash-attn` | 455.2 | 459.9 | 460.4 | 461.7 | 68.8 | 2393 | 1851 | -9% |
| π0.5 | *(defaults)* | 581.5 | 586.4 | 586.0 | 589.3 | 86.8 | 15754 | 11428 | - |
| Evo-1 | `--flash-attn --mm-prec default` | 752.1 | 757.7 | 757.3 | 760.3 | 139.8 | 4854 | 3039 | -11% |
| π0 | `--flash-attn --mm-prec default` | 759.8 | 766.3 | 767.8 | 769.7 | 85.0 | 25043 | 16969 | -5% |
| OpenVLA-OFT | *(defaults)* | 1494.4 | 1588.3 | 1554.1 | 1689.6 | 169.2 | 41486 | 15266 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| TurboVLA | 38.2 | 33.2 | 37.1 | 33.1 | 36.1 | 32.5 | 36.8 | 33.2 |
| GR00T N1.5 | 146.8 | 152.0 | 147.1 | 151.4 | 146.4 | 150.2 | 146.4 | 150.9 |
| VLA-JEPA | 155.3 | 152.1 | 154.5 | 152.0 | 154.7 | 151.2 | 154.4 | 151.1 |
| GR00T N1.6 | 228.6 | 225.0 | 230.4 | 224.6 | 237.6 | 224.9 | 230.1 | 224.8 |
| GR00T N1.7 | 232.5 | 230.1 | 232.2 | 229.8 | 232.5 | 228.9 | 232.8 | 229.4 |
| VLA-Adapter | 267.2 | 290.4 | 271.2 | 271.9 | 253.1 | 251.8 | 265.9 | 287.8 |
| SmolVLA | 506.3 | 462.0 | 502.3 | 459.1 | 507.7 | 457.1 | 504.9 | 456.9 |
| π0.5 | 586.2 | 584.8 | 587.7 | 586.1 | 589.0 | 586.0 | 585.4 | 585.3 |
| π0 | 804.2 | 769.5 | 801.1 | 765.6 | 803.0 | 768.6 | 800.1 | 769.6 |
| Evo-1 | 851.5 | 762.2 | 850.2 | 758.7 | 848.0 | 760.9 | 846.7 | 758.7 |
| OpenVLA-OFT | 1476.0 | 1480.6 | 1570.6 | 1471.4 | 1550.8 | 2222.0 | 1423.7 | 2479.9 |

</details>

## Latency and memory: AI Boost NPU (OpenVINO NPU plugin)

Rows are sorted by latency. `vision ms` is `-` for archs that do not time their
vision stage separately.

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak device buffers MiB | Peak RSS MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| TurboVLA | 2 | 256 | 63.8 | 67.3 | 65.9 | 72.7 | - | 895 | 1314 |
| VLA-Adapter | 1 | 224 | 316.0 | 426.4 | 433.3 | 437.1 | 151.8 | 2762 | 5038 |
| OpenVLA-OFT | 1 | 224 | 468.7 | 472.3 | 472.4 | 474.1 | 119.6 | 14521 | 29642 |
| π0 | 2 | 224 | 836.6 | 851.3 | 849.0 | 855.9 | 131.9 | 5756 | 11602 |
| SmolVLA | 2 | 512 | 1656.5 | 1678.9 | 1679.6 | 1702.1 | 1220.5 | 1486 | 2894 |

### Fastest configuration: AI Boost NPU (OpenVINO NPU plugin)

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak device buffers MiB | Peak RSS MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| TurboVLA | `--weight-dtype f16` | 59.9 | 63.6 | 62.4 | 69.0 | - | 523 | 877 | -5% |
| VLA-Adapter | *(defaults)* | 316.0 | 426.4 | 433.3 | 437.1 | 151.8 | 2762 | 5038 | - |
| OpenVLA-OFT | *(defaults)* | 468.7 | 472.3 | 472.4 | 474.1 | 119.6 | 14521 | 29642 | - |
| π0 | *(defaults)* | 836.6 | 851.3 | 849.0 | 855.9 | 131.9 | 5756 | 11602 | - |
| SmolVLA | *(defaults)* | 1656.5 | 1678.9 | 1679.6 | 1702.1 | 1220.5 | 1486 | 2894 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| TurboVLA | 72.5 | 75.1 | 73.2 | 72.6 | 70.4 | 71.0 | 70.2 | 70.2 |
| VLA-Adapter | 425.2 | 424.1 | 432.3 | 430.7 | 431.8 | 429.1 | 436.7 | 421.5 |
| OpenVLA-OFT | 483.6 | 480.6 | 477.0 | 483.9 | 469.8 | 473.1 | 475.7 | 475.2 |
| π0 | 883.7 | 823.1 | 881.1 | 832.0 | 857.8 | 824.9 | 872.1 | 841.9 |
| SmolVLA | 1697.6 | 1703.9 | 1665.3 | 1659.4 | 1696.4 | 1679.3 | 1701.0 | 1688.8 |

</details>

## Latency and memory: OpenVINO CPU plugin

Rows are sorted by latency. `vision ms` is `-` for archs that do not time their
vision stage separately.

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| TurboVLA | 2 | 256 | 240.7 | 242.3 | 241.9 | 243.6 | - | 1825 |
| VLA-JEPA | 1 | 256 | 1314.3 | 1420.7 | 1431.6 | 1445.5 | 475.5 | 11702 |
| VLA-Adapter | 1 | 224 | 1612.5 | 1623.4 | 1620.8 | 1636.0 | 905.3 | 7868 |
| SmolVLA | 2 | 512 | 1942.8 | 1950.2 | 1950.4 | 1955.3 | 1053.3 | 4545 |
| GR00T N1.7 | 1 | 256 | 2034.7 | 2048.9 | 2038.3 | 2053.4 | 460.9 | 14936 |
| GR00T N1.5 | 1 | 224 | 2221.0 | 2405.0 | 2406.8 | 2432.8 | 560.2 | 10627 |
| GR00T N1.6 | 1 | 224 | 2243.0 | 2269.8 | 2250.7 | 2342.5 | 523.1 | 13842 |
| Evo-1 | 1 | 448 | 4496.6 | 4681.7 | 4756.0 | 4776.2 | 1910.9 | 4748 |
| π0 | 2 | 224 | 6924.5 | 7436.4 | 7492.7 | 7516.8 | 1087.9 | 15819 |
| π0.5 | 2 | 224 | 7025.7 | 7528.9 | 7584.9 | 7616.4 | 1094.3 | 16522 |
| OpenVLA-OFT | 1 | 224 | 11604.9 | 11655.3 | 11647.8 | 11700.6 | 1040.6 | 43047 |

### Fastest configuration: OpenVINO CPU plugin

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| TurboVLA | *(defaults)* | 240.7 | 242.3 | 241.9 | 243.6 | - | 1825 | - |
| VLA-JEPA | *(defaults)* | 1314.3 | 1420.7 | 1431.6 | 1445.5 | 475.5 | 11702 | - |
| VLA-Adapter | *(defaults)* | 1612.5 | 1623.4 | 1620.8 | 1636.0 | 905.3 | 7868 | - |
| SmolVLA | *(defaults)* | 1942.8 | 1950.2 | 1950.4 | 1955.3 | 1053.3 | 4545 | - |
| GR00T N1.7 | *(defaults)* | 2034.7 | 2048.9 | 2038.3 | 2053.4 | 460.9 | 14936 | - |
| GR00T N1.5 | *(defaults)* | 2221.0 | 2405.0 | 2406.8 | 2432.8 | 560.2 | 10627 | - |
| GR00T N1.6 | *(defaults)* | 2243.0 | 2269.8 | 2250.7 | 2342.5 | 523.1 | 13842 | - |
| Evo-1 | *(defaults)* | 4496.6 | 4681.7 | 4756.0 | 4776.2 | 1910.9 | 4748 | - |
| π0 | *(defaults)* | 6924.5 | 7436.4 | 7492.7 | 7516.8 | 1087.9 | 15819 | - |
| π0.5 | *(defaults)* | 7025.7 | 7528.9 | 7584.9 | 7616.4 | 1094.3 | 16522 | - |
| OpenVLA-OFT | *(defaults)* | 11604.9 | 11655.3 | 11647.8 | 11700.6 | 1040.6 | 43047 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| TurboVLA | 243.0 | 243.6 | 243.4 | 242.8 | 248.0 | 248.6 | 247.2 | 249.0 |
| VLA-JEPA | 1564.5 | 1515.8 | 1576.7 | 1565.6 | 1532.9 | 1547.8 | 1538.7 | 1556.6 |
| VLA-Adapter | 1851.5 | 1834.8 | 1825.1 | 1811.8 | 1852.4 | 1844.8 | 1793.9 | 1793.2 |
| SmolVLA | 1949.0 | 1996.8 | 1948.2 | 1990.5 | 1950.6 | 1994.9 | 1953.6 | 1997.5 |
| GR00T N1.7 | 2271.9 | 2040.0 | 2266.9 | 2261.1 | 2039.4 | 2222.5 | 2039.1 | 2034.5 |
| GR00T N1.5 | 2421.0 | 2425.6 | 2548.2 | 2521.8 | 2433.5 | 2531.9 | 2422.0 | 2427.9 |
| GR00T N1.6 | 2430.1 | 2250.6 | 2254.7 | 2398.4 | 2249.9 | 2461.8 | 2253.9 | 2253.4 |
| Evo-1 | 4906.3 | 4805.6 | 4889.3 | 4901.2 | 4842.8 | 4849.3 | 4766.9 | 4795.5 |
| π0 | 7567.3 | 7550.0 | 7528.8 | 7536.5 | 7541.7 | 7550.6 | 7569.9 | 7571.0 |
| π0.5 | 7666.3 | 7647.3 | 7613.1 | 7603.7 | 7620.1 | 7627.3 | 7599.9 | 7595.8 |
| OpenVLA-OFT | 11732.4 | 11706.9 | 11726.0 | 11705.5 | 11676.9 | 11744.3 | 11681.2 | 11690.1 |

</details>

## Latency and memory: ggml CPU backend

Rows are sorted by latency. `vision ms` is `-` for archs that do not time their
vision stage separately.

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 2 | 256 + 128 | 79.9 | 130.0 | 138.4 | 142.2 | 16.6 | 623 |
| TurboVLA | 2 | 256 | 160.0 | 163.0 | 162.5 | 164.4 | - | 855 |
| VLA-JEPA | 1 | 256 | 1071.8 | 1306.5 | 1321.2 | 1332.1 | 399.2 | 3824 |
| GR00T N1.7 | 1 | 256 | 1267.9 | 1529.9 | 1540.3 | 1552.9 | 420.0 | 4884 |
| VLA-Adapter | 1 | 224 | 1666.3 | 1672.9 | 1672.2 | 1677.6 | 974.7 | 2708 |
| GR00T N1.6 | 1 | 224 | 1671.7 | 1713.2 | 1712.3 | 1727.0 | 519.7 | 4524 |
| SmolVLA | 2 | 512 | 1788.5 | 1920.7 | 1959.3 | 1975.0 | 901.7 | 1101 |
| GR00T N1.5 | 1 | 224 | 1981.2 | 1991.4 | 1990.3 | 1999.1 | 523.7 | 3500 |
| Evo-1 | 1 | 448 | 4104.1 | 4122.2 | 4120.4 | 4133.9 | 2053.6 | 1432 |
| π0.5 | 2 | 224 | 6110.4 | 6130.1 | 6130.5 | 6141.5 | 1048.2 | 5657 |
| π0 | 2 | 224 | 6153.5 | 6170.8 | 6169.4 | 6182.0 | 1053.9 | 5356 |
| OpenVLA-OFT | 1 | 224 | 9973.4 | 10011.6 | 10014.2 | 10044.8 | 1025.6 | 14797 |

### Fastest configuration: ggml CPU backend

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | `--weight-dtype bf16 --flash-attn` | 108.0 | 111.3 | 111.5 | 114.3 | 16.1 | 623 | -14% |
| TurboVLA | *(defaults)* | 160.0 | 163.0 | 162.5 | 164.4 | - | 855 | - |
| VLA-JEPA | `--weight-dtype f16` | 881.3 | 1041.8 | 1074.4 | 1082.8 | 331.3 | 3824 | -20% |
| GR00T N1.7 | `--weight-dtype f16 --flash-attn` | 981.2 | 1238.8 | 1253.5 | 1304.5 | 273.3 | 4884 | -19% |
| GR00T N1.6 | `--weight-dtype f16 --flash-attn` | 1051.0 | 1384.6 | 1410.5 | 1438.1 | 357.2 | 4524 | -19% |
| VLA-Adapter | `--weight-dtype f16` | 1374.1 | 1380.0 | 1379.1 | 1387.2 | 808.9 | 2707 | -18% |
| SmolVLA | `--weight-dtype f16 --flash-attn` | 1501.1 | 1549.7 | 1550.8 | 1570.7 | 693.3 | 1051 | -19% |
| GR00T N1.5 | `--weight-dtype f16 --flash-attn` | 1555.9 | 1576.6 | 1578.0 | 1588.8 | 384.0 | 3498 | -21% |
| Evo-1 | `--weight-dtype f16 --flash-attn` | 3263.7 | 3277.0 | 3277.9 | 3285.5 | 1499.0 | 1368 | -21% |
| π0.5 | `--weight-dtype f16 --flash-attn` | 4903.4 | 4917.5 | 4917.2 | 4927.7 | 850.3 | 5657 | -20% |
| π0 | `--weight-dtype f16` | 4933.4 | 4951.2 | 4949.7 | 4966.4 | 860.6 | 5355 | -20% |
| OpenVLA-OFT | `--weight-dtype f16 --flash-attn` | 7925.4 | 7947.1 | 7947.8 | 7957.9 | 852.2 | 14797 | -21% |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--weight-dtype bf16` | `--weight-dtype bf16 --flash-attn` | `--weight-dtype f16` | `--weight-dtype f16 --flash-attn` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 120.5 | 119.5 | 114.5 | 118.5 | 111.9 | 110.8 | 116.2 | 119.3 |
| TurboVLA | 201.4 | 190.7 | 201.9 | 189.3 | 300.4 | 287.6 | 245.8 | 231.4 |
| VLA-JEPA | 1206.8 | 1184.3 | 1574.5 | 1501.2 | 1494.5 | 1104.9 | 919.0 | 1139.7 |
| GR00T N1.7 | 1603.7 | 1641.2 | 1432.1 | 1568.9 | 1681.9 | 1389.4 | 1122.9 | 1089.8 |
| VLA-Adapter | 1720.5 | 1693.8 | 1695.9 | 1682.8 | 1697.5 | 1752.6 | 1533.7 | 1542.7 |
| GR00T N1.6 | 1652.7 | 1830.0 | 1736.6 | 1620.1 | 1871.4 | 1619.2 | 1287.4 | 1281.7 |
| SmolVLA | 1980.5 | 1861.1 | 1991.3 | 1873.8 | 1986.9 | 1872.2 | 1673.5 | 1566.2 |
| GR00T N1.5 | 1996.0 | 2135.1 | 1994.7 | 2136.8 | 1997.5 | 2137.7 | 1722.8 | 1702.3 |
| Evo-1 | 4136.1 | 3867.8 | 4140.3 | 3866.8 | 4139.2 | 3864.0 | 3569.0 | 3283.7 |
| π0.5 | 6164.1 | 6162.5 | 6164.4 | 6171.1 | 6160.7 | 6159.4 | 4920.4 | 4918.6 |
| π0 | 6193.3 | 6401.2 | 6179.6 | 6409.5 | 6174.2 | 6395.9 | 4954.6 | 5175.8 |
| OpenVLA-OFT | 10040.8 | 10041.5 | 10033.6 | 10025.2 | 10001.9 | 9994.9 | 7960.8 | 7958.7 |

</details>

## Not run

### Arc B390 iGPU (OpenVINO GPU plugin)

- **BitVLA**: N/A off CUDA. The only published GGUF is int2-packed, and only CUDA builds load it:

  ```text
  vla(bitvla): int2-packed GGUF requires a CUDA build (VLA_BITVLA_CUDA_KERNELS); use the bf16 GGUF for CPU.
  ```
- **Octo-Small**: not supported on OpenVINO (the README's support matrix lists it as planned). The translator has no conversion for one of its ops:

  ```text
  GGML OpenVINO backend ov::Exception: Check 'it != m_translator_map.end()' failed at openvino/translate_session.cpp:304
  ```

### AI Boost NPU (OpenVINO NPU plugin)

- **π0.5**: the NPU compiler rejects the graph's dynamic shape:

  ```text
  Compilation failed. vclAllocatedExecutableCreate4 result: 0x78000004 - [NPU_VCL] Compiler returned msg:
  IE.Reshape doesn't support dynamic shapes
  vla(pi05): ggml_backend_graph_compute failed (-1)
  ```
- **GR00T N1.5**: the NPU compiler fails without giving a reason:

  ```text
  Compilation failed. vclAllocatedExecutableCreate4 result: 0x78000004 - [NPU_VCL] Compiler returned msg:
  Compilation failed
  vla(gr00tn1d5): graph compute failed (-1)
  ```
- **GR00T N1.6**: the NPU compiler aborts on a dynamic dimension (exit code 250):

  ```text
  LLVM ERROR: Failed to infer result type(s):
  "IE.Add"(...) {} : (tensor<1x1x51x1536xf32>, tensor<1x1x?x1536xf32, {bounds = ... [1, 1, 132, 1536] ...}>) -> ( ??? )
  ```
- **GR00T N1.7**: the NPU compiler fails without giving a reason:

  ```text
  Compilation failed. vclAllocatedExecutableCreate4 result: 0x78000004 - [NPU_VCL] Compiler returned msg:
  Compilation failed
  vla(gr00tn1d7): graph compute failed (-1)
  ```
- **VLA-JEPA**: the NPU compiler aborts on a dynamic dimension (exit code 250):

  ```text
  LLVM ERROR: Failed to infer result type(s):
  "IE.Add"(...) {} : (tensor<1x1x40x768xf32>, tensor<1x1x?x768xf32, {bounds = ... [1, 1, 68, 768] ...}>) -> ( ??? )
  ```
- **Evo-1**: the NPU compiler rejects the graph's dynamic shape:

  ```text
  Compilation failed. vclAllocatedExecutableCreate4 result: 0x78000004 - [NPU_VCL] Compiler returned msg:
  IE.Reshape doesn't support dynamic shapes
  vla(evo1): ggml_backend_graph_compute failed (-1)
  ```
- **BitVLA**: N/A off CUDA. The only published GGUF is int2-packed, and only CUDA builds load it:

  ```text
  vla(bitvla): int2-packed GGUF requires a CUDA build (VLA_BITVLA_CUDA_KERNELS); use the bf16 GGUF for CPU.
  ```
- **Octo-Small**: not supported on OpenVINO (the README's support matrix lists it as planned). The translator has no conversion for one of its ops:

  ```text
  GGML OpenVINO backend ov::Exception: Check 'it != m_translator_map.end()' failed at openvino/translate_session.cpp:304
  ```

### OpenVINO CPU plugin

- **BitVLA**: N/A off CUDA. The only published GGUF is int2-packed, and only CUDA builds load it:

  ```text
  vla(bitvla): int2-packed GGUF requires a CUDA build (VLA_BITVLA_CUDA_KERNELS); use the bf16 GGUF for CPU.
  ```
- **Octo-Small**: not supported on OpenVINO (the README's support matrix lists it as planned). The translator has no conversion for one of its ops:

  ```text
  GGML OpenVINO backend ov::Exception: Check 'it != m_translator_map.end()' failed at openvino/translate_session.cpp:304
  ```

### ggml CPU backend

- **BitVLA**: N/A off CUDA. The only published GGUF is int2-packed, and only CUDA builds load it:

  ```text
  vla(bitvla): int2-packed GGUF requires a CUDA build (VLA_BITVLA_CUDA_KERNELS); use the bf16 GGUF for CPU.
  ```

## Notes

OpenVINO compiles each graph on the first predict, which can take up to a minute on the GPU. The warmup calls absorb it, so none of the latencies above include it. Do not set `GGML_OPENVINO_CACHE_DIR`; `docs/backend/ov.md` explains why.

π0 runs at F32 inference precision on the GPU by default (`GGML_OPENVINO_GPU_PRECISION=f32`, set by vla.cpp for π0 alone), which costs about 3x.

A latency is not a correctness check. `docs/backend/ov.md` lists π0 as wrong on the NPU (max|Δ| 1.7), measured at an earlier llama.cpp pin. Its NPU output was not re-verified here.

The chip runs under a 25 W long-term power limit. The first timed calls of a process can run at boost power, so on the CPU sections read `mean` or `p50` for sustained latency.

## Reproducing

```bash
# OpenVINO build: iGPU, NPU and OpenVINO CPU plugin
source <openvino 2026.4>/setupvars.sh
cmake -S . -B build-ov -G Ninja -DGGML_OPENVINO=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-ov -j

# one row, for example SmolVLA
GGML_OPENVINO_DEVICE=GPU ./build-ov/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
ZE_ENABLE_ALT_DRIVERS=/lib/x86_64-linux-gnu/libze_intel_npu.so.1 GGML_OPENVINO_DEVICE=NPU \
  ./build-ov/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
GGML_OPENVINO_DEVICE=CPU ./build-ov/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20

# ggml CPU backend
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpu -j
./build-cpu/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
```
