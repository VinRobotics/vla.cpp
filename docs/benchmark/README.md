# Benchmarks

Per-device latency and memory for every architecture, measured with `vla-bench` on
synthetic inputs at `c93ca0a` (PR #32), llama.cpp `b11223`. The Jetson Orin NX
ran at `f7e0f7f`, which differs from `c93ca0a` only by docs, release tooling and
OpenVINO-only scripts. The five Core Ultra X7 358H rows (Arc B390 SYCL and
OpenVINO, AI Boost NPU, OpenVINO CPU and ggml CPU) ran later, in one round, at
main `a83bcad`, after PR #73 (same llama.cpp pin). The Jetson
AGX Orin row ran at `e6653e8`, on the bit-exact CUDA path
([backend/cuda.md](../backend/cuda.md#the-bit-exact-fast-path)); its report also re-measures
`1012c05`, before that path, for comparison. The Radeon RX 9060 XT ROCm row
ran at `d0e5d00`, PR #35 merged with main `f464653` (same llama.cpp pin), with
fixed-input checks against the CPU build in
[backend/rocm.md](../backend/rocm.md). The Qualcomm Dragonwing IQ-10 row ran at
`1cab1c3` (same llama.cpp pin). Each report
lists the device, the build, each model's configuration, latency and peak memory
at the defaults, and the fastest runtime flags for that device.

The table below gives the minimum latency in ms at the defaults, the one number
every report has. `N/A` means the arch does not run on that backend, and `-`
means the model does not fit or fails there; each report says which and why.
`n/r` means not run.

| Device | Octo-Small | TurboVLA | VLA-JEPA | VLA-Adapter | GR00T N1.5 | GR00T N1.6 | GR00T N1.7 | BitVLA | SmolVLA | π0 | π0.5 | OpenVLA-OFT | Evo-1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| [RTX 5090 (CUDA), author's numbers](rtx-5090.md) | 2.69 | 4.88 | 14.3 | 16.0 | 16.1 | 19.4 | 19.5 | 20.8 | 38.2 | 28.8 | 29.2 | 34.5 | 49.1 |
| [RTX 3090 (CUDA)](rtx-3090.md) | 4.3 | 9.0 | 27.3 | 34.2 | 36.6 | 36.3 | 35.3 | 26.9 | 61.0 | 97.5 | 97.8 | 134.2 | 108.6 |
| [RTX 3060 (CUDA)](rtx-3060.md) | 8.3 | 21.6 | 57.1 | 74.7 | 86.8 | 78.1 | 82.8 | 59.1 | 125.1 | 228.2 | 229.9 | - | 230.8 |
| [RTX 5070 Laptop (CUDA)](rtx-5070-laptop.md) | 4.7 | 14.1 | 38.0 | 56.3 | 73.0 | 65.5 | 63.3 | 54.4 | 89.3 | 177.1 | 180.5 | - | 184.6 |
| [Radeon RX 9060 XT (ROCm)](rx-9060-xt.md) | 14.0 | 33.8 | 65.9 | 70.1 | 79.5 | 94.6 | 97.1 | N/A | 207.7 | 217.9 | 220.6 | 182.2 | 233.6 |
| [Jetson AGX Orin (CUDA)](jetson-agx-orin.md) | 14.3 | 28.0 | 60.6 | 79.1 | 96.7 | 101.2 | 93.0 | 76.8 | 164.3 | 269.0 | 266.6 | 297.4 | 298.5 |
| [Jetson Orin NX 16 GB (CUDA)](jetson-orin-nx.md) | 27.4 | 79.7 | 176.7 | 270.7 | 260.8 | 252.1 | 251.4 | 269.1 | 411.2 | 739.6 | 742.9 | - | 893.2 |
| [Jetson Orin Nano Super (CUDA)](jetson-orin-nano.md) | 31.9 | 89.3 | 193.9 | 297.5 | 293.4 | 272.3 | 271.7 | 335.0 | 462.8 | 845.2 | 849.7 | - | 1011.5 |
| [Apple M4 (Metal)](apple-m4.md) | 22.6 | 65.2 | 248.7 | 315.7 | 401.5 | 358.9 | 350.8 | N/A | 357.6 | 1141.1 | 1153.8 | 1650.7 | 829.3 |
| [Intel Arc B390 iGPU (SYCL)](core-ultra-x7-358h.md) | 15.5 | 25.0 | 102.1 | 62.6 | 101.2 | 106.1 | 178.4 | N/A | 61.7 | 269.3 | 275.2 | 434.9 | 270.4 |
| [Intel Arc B390 iGPU (OpenVINO)](core-ultra-x7-358h.md) | N/A | 28.5 | 116.3 | 170.2 | 106.0 | 122.8 | 131.4 | N/A | 213.1 | 620.5 | 232.5 | 1262.5 | 566.0 |
| [Intel Arc A380 (SYCL)](arc-a380.md) | 60.5 | 103.5 | 355.9 | 408.9 | 472.0 | 646.0 | 615.3 | N/A | 734.5 | 921.9 | 945.6 | - | 1047.3 |
| [Snapdragon X Hexagon NPU](snapdragon-x-hexagon.md) | 239.7 | 2280.8 | 821.0 | 4072.1 | 2755.7 | 2940.0 | 1400.2 | N/A | 1239.3 | 5730.0 | 9321.7 | - | 7208.6 |
| [Qualcomm Dragonwing IQ-10 Hexagon NPU](qualcomm-iq10.md) | 121.6 | 428.1 | 440.2 | 1733.6 | 613.7 | 705.8 | 747.0 | N/A | 631.0 | 1124.4 | 1117.4 | 4097.7 | 2579.1 |
| [Intel AI Boost NPU (OpenVINO)](core-ultra-x7-358h.md) | N/A | 23.5 | 90.7 | 126.7 | 134.0 | 161.1 | 144.9 | N/A | 161.3 | 438.3 | 430.2 | 405.8 | 256.3 |
| [Intel Core i7-14700F (CPU)](core-i7-14700f.md) | 39.8 | 195.1 | 825.5 | 1140.6 | 1913.8 | 1627.9 | 1453.4 | N/A | 1689.2 | 6455.7 | 6068.1 | 9869.0 | 4159.6 |
| [Intel Core i9-14900HX (CPU)](core-i9-14900hx.md) | 67.0 | 314.7 | 1227.2 | 1659.3 | 1935.4 | 1688.9 | 1505.5 | N/A | 1822.6 | 6182.1 | 5735.0 | 9126.4 | 4118.2 |
| [Intel Core Ultra X7 358H (CPU)](core-ultra-x7-358h.md) | 48.0 | 199.0 | 1258.1 | 1645.4 | 1979.1 | 1697.3 | 1532.5 | N/A | 1758.0 | 6083.9 | 6120.3 | 9894.3 | 4025.8 |
| [Intel Core Ultra X7 358H (OpenVINO CPU)](core-ultra-x7-358h.md) | N/A | 238.4 | 1407.2 | 1831.2 | 2456.6 | 2300.4 | 2076.5 | N/A | 1899.5 | 7537.1 | 7522.7 | 11651.3 | 4626.5 |
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
  CPU row shows the same effect: `min` sits 15% below `p50` for TurboVLA. On
  that row, Octo-Small and TurboVLA also run at one of two speeds, set by the
  machine's state rather than the code; the report has the A/B.
- On the Arc B390 SYCL row, the SYCL defaults include PR #73's fused graphs for
  SmolVLA, VLA-Adapter, GR00T N1.5 and GR00T N1.6. SYCL there goes through the
  OpenCL adapter, because the Level Zero adapter crashes with that machine's
  compute runtime (26.22) and loader (1.16.1).
- The OpenVINO rows have no Octo-Small, because OpenVINO cannot translate its
  GroupNorm yet.
- π0's AI Boost NPU cell is a latency only. On 48 inputs against the ggml CPU
  build at F32, its gripper flips one step early in 1 chunk in 48, as on the GPU
  at F16; the NPU has no F32 mode. The report and
  [`docs/backend/ov.md`](../backend/ov.md) have the numbers.
