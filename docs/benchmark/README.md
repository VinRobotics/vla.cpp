# Benchmarks

Per-device latency and memory for every architecture, measured with `vla-bench` on
synthetic inputs at `c93ca0a` (PR #32), llama.cpp `b11223`. The Core Ultra X7
358H ran at `450992c`, which differs from `c93ca0a` only by an OpenVINO fix
(`3ff447c`) and docs. Each report lists the device, the build, each model's
configuration, latency and peak memory at the defaults, and the fastest runtime
flags for that device.

The table below gives the minimum latency in ms at the defaults, the one number
every report has. `N/A` means the arch does not run on that backend, and `-`
means the model does not fit or fails there; each report says which and why.

| Device | Octo-Small | TurboVLA | VLA-JEPA | VLA-Adapter | GR00T N1.5 | GR00T N1.6 | GR00T N1.7 | BitVLA | SmolVLA | π0 | π0.5 | OpenVLA-OFT | Evo-1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| [RTX 5090 (CUDA), author's numbers](rtx-5090.md) | 2.69 | 4.88 | 14.3 | 16.0 | 16.1 | 19.4 | 19.5 | 20.8 | 38.2 | 28.8 | 29.2 | 34.5 | 49.1 |
| [RTX 3090 (CUDA)](rtx-3090.md) | 4.3 | 9.0 | 27.3 | 34.2 | 36.6 | 36.3 | 35.3 | 26.9 | 61.0 | 97.5 | 97.8 | 134.2 | 108.6 |
| [RTX 3060 (CUDA)](rtx-3060.md) | 8.3 | 21.6 | 57.1 | 74.7 | 86.8 | 78.1 | 82.8 | 59.1 | 125.1 | 228.2 | 229.9 | - | 230.8 |
| [RTX 5070 Laptop (CUDA)](rtx-5070-laptop.md) | 4.7 | 14.1 | 38.0 | 56.3 | 73.0 | 65.5 | 63.3 | 54.4 | 89.3 | 177.1 | 180.5 | - | 184.6 |
| [Jetson AGX Orin (CUDA)](jetson-agx-orin.md) | 17.5 | 36.0 | 94.6 | 127.4 | 131.2 | 132.6 | 132.7 | 133.0 | 218.3 | 350.3 | 351.9 | 384.9 | 434.6 |
| [Jetson Orin Nano Super (CUDA)](jetson-orin-nano.md) | 31.9 | 89.3 | 193.9 | 297.5 | 293.4 | 272.3 | 271.7 | 335.0 | 462.8 | 845.2 | 849.7 | - | 1011.5 |
| [Apple M4 (Metal)](apple-m4.md) | 22.6 | 65.2 | 248.7 | 315.7 | 401.5 | 358.9 | 350.8 | N/A | 357.6 | 1141.1 | 1153.8 | 1650.7 | 829.3 |
| [Intel Arc B390 iGPU (OpenVINO)](core-ultra-x7-358h.md) | N/A | 35.6 | 152.4 | 232.6 | 145.5 | 225.1 | 230.8 | N/A | 499.8 | 795.8 | 581.5 | 1494.4 | 841.5 |
| [Intel Arc A380 (SYCL)](arc-a380.md) | 60.5 | 103.5 | 355.9 | 408.9 | 472.0 | 646.0 | 615.3 | N/A | 734.5 | 921.9 | 945.6 | - | 1047.3 |
| [Snapdragon X Hexagon NPU](snapdragon-x-hexagon.md) | 239.7 | 2280.8 | 821.0 | 4072.1 | 2755.7 | 2940.0 | 1400.2 | N/A | 1239.3 | 5730.0 | 9321.7 | - | 7208.6 |
| [Intel AI Boost NPU (OpenVINO)](core-ultra-x7-358h.md) | N/A | 63.8 | - | 316.0 | - | - | - | N/A | 1656.5 | 836.6 | - | 468.7 | - |
| [Intel Core i7-14700F (CPU)](core-i7-14700f.md) | 39.8 | 195.1 | 825.5 | 1140.6 | 1913.8 | 1627.9 | 1453.4 | N/A | 1689.2 | 6455.7 | 6068.1 | 9869.0 | 4159.6 |
| [Intel Core i9-14900HX (CPU)](core-i9-14900hx.md) | 67.0 | 314.7 | 1227.2 | 1659.3 | 1935.4 | 1688.9 | 1505.5 | N/A | 1822.6 | 6182.1 | 5735.0 | 9126.4 | 4118.2 |
| [Intel Core Ultra X7 358H (CPU)](core-ultra-x7-358h.md) | 79.9 | 160.0 | 1071.8 | 1666.3 | 1981.2 | 1671.7 | 1267.9 | N/A | 1788.5 | 6153.5 | 6110.4 | 9973.4 | 4104.1 |
| [Intel Core Ultra X7 358H (OpenVINO CPU)](core-ultra-x7-358h.md) | N/A | 240.7 | 1314.3 | 1612.5 | 2221.0 | 2243.0 | 2034.7 | N/A | 1942.8 | 6924.5 | 7025.7 | 11604.9 | 4496.6 |
| [Intel Core i5-12400F (CPU)](core-i5-12400f.md) | 78.1 | 392.7 | 1668.5 | 2165.7 | 2681.4 | 2216.8 | 1973.6 | N/A | 2288.4 | 8953.5 | 8637.9 | - | 5435.0 |
| [AMD Ryzen 5 5500 (CPU)](ryzen-5-5500.md) | 96.6 | 475.7 | 2133.0 | 2781.7 | 3447.0 | 2826.0 | 2518.8 | N/A | 3124.3 | 11321.3 | 11225.6 | - | 7347.2 |
| [Snapdragon X Oryon (CPU)](snapdragon-x-hexagon.md) | 109.5 | 548.9 | 1656.5 | 2013.1 | 2736.6 | 2226.0 | 1988.1 | N/A | 2080.9 | 9224.5 | 9140.6 | - | 5518.9 |

Notes on the table:

- The RTX 5090 row comes from the PR #32 author and was not re-measured. Its
  numbers are a minimum over 5 alternating rounds, not the protocol the other
  reports use (3 warmups + 20 timed reps, best of 3 processes).
- On the i7-14700F, `min` for TurboVLA, VLA-JEPA and VLA-Adapter catches boost
  power at the start of a process and sits about 30% below `p50`. Its report
  explains this.
- The Core Ultra X7 358H runs under a 25 W long-term power limit, and its ggml
  CPU row shows the same effect: `min` sits 42% below `p50` for Octo-Small and
  about 18% below for VLA-JEPA and GR00T N1.7. On its AI Boost NPU, VLA-Adapter's
  `min` is 27% below `p50`.
- The OpenVINO rows have no Octo-Small, because OpenVINO cannot translate one of
  its ops yet. The NPU compiler rejects the graphs of six archs; the report
  quotes each error.
- [`docs/backend/ov.md`](../backend/ov.md) lists π0's output on the AI Boost NPU
  as wrong at an earlier llama.cpp pin. It was not re-checked in this round, so
  that cell is a latency only.
