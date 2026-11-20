# Arduino VENTUNO Q: Hexagon NPU

`vla-bench` times `predict()` in-process on synthetic inputs. It measures the
engine only: no transport, no simulator, and no claim about task success.

The graph runs on the Hexagon NPU, and ops the NPU rejects run on the CPU
through vla.cpp's fallback wrapper.

## Test setup

| | |
|---|---|
| Device | Arduino VENTUNO Q: Qualcomm QCS8275, 4 Cortex-A78C and 4 Cortex-A55 cores, Hexagon v75 NPU (one CDSP session in use; 4 HVX threads, HMX, 8 MB VTCM), 14 GiB RAM visible |
| Threads | 8 (the default: all hardware threads), for the NPU's CPU fallback; a second table pins 4 threads to the A78C cores |
| Power | CPU governor `performance` (the board boots with `schedutil` and `ondemand`) |
| OS | Ubuntu 24.04.4 LTS, kernel 6.8.0-1078-qcom, `qcom-fastrpc1` 1.0.15 |
| Toolchain | `ghcr.io/snapdragon-toolchain/arm64-linux:v0.7`: clang 21.1.8, Hexagon SDK 6.6.0.0 (tools 19.0.07) |
| Commit | `d1fe64b` (main) |
| llama.cpp | `b11223` |
| Date | 2026-11-19 17:24 to 17:51 (UTC+07) |
| Build | `scripts/build_linux_hexagon.sh` (servers included); HTP skels as built, loaded from the package through `ADSP_LIBRARY_PATH` |
| Runtime flags | defaults; the fastest flags per model are in the third table |
| Method | 3 warmups + 20 timed reps per process, 3 processes. The process with the lowest mean is reported, and memory is the peak over all three. |

**Peak RSS** is the process's peak resident set (`getrusage`, via
`/usr/bin/time`). It leaves out the weights, which live in FastRPC buffers
mapped for the NPU. **System RAM** is the largest drop in the system's
`MemAvailable` while the process ran, sampled every 0.5 s; it counts those
buffers. Nothing else ran on the board.

## Model configuration

Each model runs at its native view count and input size, with weights as
shipped. Checkpoints are the `vrfai/*` GGUFs, and ACT's is
`ravediamond/act-aloha-sim-transfer-cube-gguf`. Settings not listed are the
defaults. Only the models that ran are listed; the others are under
[Not run](#not-run).

| Model | Checkpoint | Views | Input | Lang tokens | Action chunk | Action dim | Action head | Steps |
|---|---|--:|--:|--:|--:|--:|---|--:|
| SmolVLA | `smolvla-libero-gguf` | 2 | 512 | 16 | 50 | 7 | flow matching | 10 |
| Evo-1 | `evo1-libero-gguf` | 1 | 448 | 16 | 50 | 7 | flow matching | 32 |
| Octo-Small | `octo-small-libero-gguf` | 2 | 256 + 128 | 16 | 4 | 7 | DDPM diffusion | 20 |
| TurboVLA | `turbovla-libero-gguf` | 2 | 256 | 16 | 12 | 7 | single-pass decoder | 1 |
| ACT | `act-aloha-sim-transfer-cube-gguf` | 1 | 640 x 480 | - | 100 | 14 | transformer decoder | 1 |

- **Lang tokens** is the synthetic prompt length.
- **Action dim** is the checkpoint's padded width. LIBERO uses the first 7 dims.
- **Octo** needs `VLA_OCTO_UNNORM_DATASET=libero_object`, and takes a 256 px
  primary view and a 128 px wrist view.
- **ACT** is an ALOHA checkpoint (sim transfer cube): no language, a 14-dim
  bimanual state and action, run with `--images 1 --size 640 --height 480`.
- **Evo-1** needs `GGML_HEXAGON_MBUF=256` on this board
  (see [Notes](#notes)); every Evo-1 row below has it set.

## Latency and memory

Rows are sorted by latency. `vision ms` is `-` for archs that do not time their
vision stage separately.

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB | System RAM MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 2 | 256 + 128 | 170.6 | 203.3 | 199.6 | 243.7 | 64.4 | 196 | 862 |
| ACT | 1 | 640 x 480 | 325.4 | 352.4 | 351.8 | 368.1 | - | 137 | 395 |
| TurboVLA | 2 | 256 | 687.7 | 720.4 | 717.6 | 748.8 | - | 192 | 1263 |
| SmolVLA | 2 | 512 | 1063.4 | 1074.2 | 1073.6 | 1078.7 | 524.6 | 89 | 1180 |
| Evo-1 | 1 | 448 | 5663.2 | 5708.3 | 5703.2 | 5742.4 | 3776.9 | 262 | 1663 |

### Four threads on the A78C cores

The same protocol with `VLA_N_THREADS=4 taskset -c 0-3`, which keeps the CPU
fallback off the A55 cores. It is faster on every model.

| Model | Views | Input | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB | System RAM MiB | vs 8 threads |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 2 | 256 + 128 | 157.6 | 169.9 | 169.8 | 179.5 | 47.4 | 196 | 802 | -16.4% |
| ACT | 1 | 640 x 480 | 234.9 | 246.3 | 244.9 | 257.2 | - | 137 | 421 | -30.1% |
| TurboVLA | 2 | 256 | 522.3 | 530.9 | 529.1 | 540.5 | - | 192 | 1082 | -26.3% |
| SmolVLA | 2 | 512 | 1036.4 | 1039.9 | 1039.7 | 1042.5 | 495.6 | 89 | 1171 | -3.2% |
| Evo-1 | 1 | 448 | 4481.4 | 4547.3 | 4547.6 | 4593.1 | 2899.0 | 262 | 1639 | -20.3% |

### Fastest configuration

We screen each model over these runtime flag sets, then re-run the fastest set
under the full protocol:

- the defaults (F16 weights, the Hexagon build's default);
- `--flash-attn` and `--mm-prec default`, alone and together;
- `--flash-attn 0`, because flash attention is on by default in Hexagon builds.

The weight dtype stays F16 in every set, as on the
[IQ-10](qualcomm-iq10.md#fastest-configuration). The screen and the re-run use
four threads on the A78C cores, the faster thread setting above.

The screen runs 2 warmups and 5 reps in one process per flag set. A flag set
replaces the defaults only if its mean is more than 3% lower, both in the screen
and in the full re-run.

No flag set beat the defaults. Octo-Small's `--flash-attn --mm-prec default`
was 7% lower in the screen and 5% higher in the full re-run (178.8 ms against
169.9 ms).

| Model | Fastest flags | min ms | mean ms | p50 ms | p90 ms | vision ms | Peak RSS MiB | System RAM MiB | vs defaults |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | *(defaults)* | 157.6 | 169.9 | 169.8 | 179.5 | 47.4 | 196 | 802 | - |
| ACT | *(defaults)* | 234.9 | 246.3 | 244.9 | 257.2 | - | 137 | 421 | - |
| TurboVLA | *(defaults)* | 522.3 | 530.9 | 529.1 | 540.5 | - | 192 | 1082 | - |
| SmolVLA | *(defaults)* | 1036.4 | 1039.9 | 1039.7 | 1042.5 | 495.6 | 89 | 1171 | - |
| Evo-1 | *(defaults)* | 4481.4 | 4547.3 | 4547.6 | 4593.1 | 2899.0 | 262 | 1639 | - |

<details><summary>Screen means (ms) per flag set</summary>

| Model | defaults | `--flash-attn` | `--mm-prec default` | `--flash-attn --mm-prec default` | `--flash-attn 0` |
|---|--:|--:|--:|--:|--:|
| Octo-Small | 178.4 | 171.6 | 263.6 | 165.7 | 178.8 |
| ACT | 247.4 | 246.0 | 246.7 | 246.3 | 298.6 |
| TurboVLA | 535.5 | 535.9 | 530.7 | 535.8 | 655.4 |
| SmolVLA | 1036.2 | 1040.4 | 1037.6 | 1039.0 | 5221.9 |
| Evo-1 | 4602.9 | 4560.4 | 4572.1 | 4560.5 | 7183.3 |

</details>

## Not run

- **VLA-JEPA, π0**: fail. One NPU session can map about 3.2 GB on this board
  (see [Notes](#notes)), and these models' buffers exceed it, with the default
  buffer size and with `GGML_HEXAGON_MBUF=256`:

  ```text
  ggml-hex: HTP0 buffer mapping failed : domain_id 3 size 1066545152 fd 22 error 0x00000001
  terminate called after throwing an instance of 'std::runtime_error'
    what():  ggml-hex: fastrpc_mmap failed (see log for details)
  ```

- **π0.5, GR00T N1.5, GR00T N1.6, GR00T N1.7, VLA-Adapter, OpenVLA-OFT**: not
  attempted. Each takes 3.6 GB to 15.3 GB of system RAM on the
  [IQ-10](qualcomm-iq10.md#latency-and-memory), above the same limit.
- **BitVLA**: N/A. Its only published GGUF is int2-packed, which only CUDA
  builds load.

## Notes

**Setting up the board.** The package from `scripts/build_linux_hexagon.sh`
runs unchanged, with no skel processing and nothing installed as root, unlike
the [IQ-10](../backend/hexagon.md#installing-the-skel). The board ships
`libcdsprpc.so.1` without the unversioned name ggml-hexagon opens, and vla.cpp
sets `ADSP_LIBRARY_PATH` by default only on Windows, so:

```bash
ln -s /usr/lib/aarch64-linux-gnu/libcdsprpc.so.1 pkg/lib/libcdsprpc.so
export LD_LIBRARY_PATH=$PWD/pkg/lib ADSP_LIBRARY_PATH=$PWD/pkg/lib
```

Without `ADSP_LIBRARY_PATH` the session fails with
`failed to open session ... error 0x80000406`.

**The mapping limit.** ggml-hexagon maps every buffer of a model into the
session when the model loads, in chunks of up to 1 GiB (`GGML_HEXAGON_MBUF`, in
MiB). On this board the kernel refuses a mapping
(`qcom,fastrpc-cb ...: failed to map buffer`) once the total is large enough,
and smaller chunks reach a higher total. Measured with π0:

| Chunk size | Mapped before the first failure |
|---|--:|
| 1 GiB (default) | 1369 MiB |
| 256 MiB (`GGML_HEXAGON_MBUF=256`) | 3181 MiB |

`GGML_HEXAGON_VMEM` does not move the limit: π0 failed at the same buffer with
it set to 2800, 2000 and 1500. SmolVLA, Octo-Small, TurboVLA and ACT fit at the
defaults, and Evo-1 only with 256 MiB chunks. Upstream reworked buffer
allocation and mapping after this pin
([llama.cpp #32144](https://github.com/ggml-org/llama.cpp/pull/32144)); that
change is later than the `v1.0.0` tag and was not tested here.

**Fidelity.** `vla_predict_check` on fixed inputs at the Hexagon defaults,
against a CPU-only build (`-DGGML_HEXAGON=OFF`) of the same commit with
`--weight-dtype f32 --flash-attn 0`, as in
[backend/hexagon.md](../backend/hexagon.md#fidelity). The bar is 2.9e-3.

| Model | Views | max\|Δ\| | IQ-10 |
|---|--:|--:|--:|
| SmolVLA | 2 | 5.9e-4 | 7.4e-4 |
| Evo-1 | 1 | 8.1e-4 | 6.2e-4 |
| Octo-Small | 1 | 1.0e-3 | 5.0e-4 |
| TurboVLA | 2 | 4.4e-4 | 3.7e-4 |
| ACT | 1 | 6.1e-4 | 6.1e-4 |

**Against the board's CPU.** The CPU-only build, one process per cell, 2
warmups and 5 reps at its defaults, or 1 warmup and 3 reps where marked F32.
These are not under the protocol above.

| Model | NPU, 4 threads, mean ms | CPU, 4 threads, mean ms | CPU, 8 threads, mean ms |
|---|--:|--:|--:|
| Octo-Small | 169.9 | 266.4 | 336.5 |
| ACT | 246.3 | 498.2 | 616.7 |
| TurboVLA | 530.9 | 1292.6 | 1585.6 |
| SmolVLA | 1039.9 | 7416.4 (F32) | 51363.3 |
| Evo-1 | 4547.3 | 19837.4 (F32) | - |

SmolVLA's 51 s is the CPU-only build's default BF16 weights, for which these
cores have no fast matmul; its and Evo-1's four-thread cells use
`--weight-dtype f32`, and Evo-1 was not timed at the default.

**Idle gaps.** `vla-bench --gap-ms` leaves the NPU idle between calls, as a
policy server does. Octo-Small ran 2000 reps with 100 ms gaps (mean 184.5 ms)
and SmolVLA 400 reps with 500 ms gaps (mean 1041.0 ms), both on four threads,
without an error or a stall. The host hang reported on this board upstream
([llama.cpp #30309](https://github.com/ggml-org/llama.cpp/issues/30309)) was
seen with two sessions under `llama-server`; one session in one process is the
only shape tested here, and `vla-server` was not soaked.

## Reproducing

```bash
# on an x86-64 host with Docker
scripts/build_linux_hexagon.sh
# copy build-lnx-htp/pkg to the board, then on the board:
ln -s /usr/lib/aarch64-linux-gnu/libcdsprpc.so.1 pkg/lib/libcdsprpc.so
export LD_LIBRARY_PATH=$PWD/pkg/lib ADSP_LIBRARY_PATH=$PWD/pkg/lib

# one row, for example SmolVLA
./pkg/bin/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20

# the same row on four threads
VLA_N_THREADS=4 taskset -c 0-3 ./pkg/bin/vla-bench --ckpt smolvla-libero.gguf --images 2 --size 512 --warmup 3 --reps 20
```
