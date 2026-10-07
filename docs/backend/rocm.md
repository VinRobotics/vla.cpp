# ROCm/HIP backend

`vla.cpp` can run a VLA model through the HIP backend in the pinned
`llama.cpp`/ggml dependency. This page records the validated configuration and
the current model coverage. ROCm is selected at configure time, in a separate
build directory; it is not selected automatically on an AMD machine.

The results below are from upstream `main@1adf078` merged with this HIP
backend on Linux 6.17.0-40, Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`),
ROCm 7.14.60850, and the repository's llama.cpp tag `b11223` (ggml commit
`4da6337`). Other GPUs and ROCm versions have not been measured here.

## Build and select the device

Install HIP, hipBLAS, rocBLAS, Ninja, and the host dependencies in the main
README. Confirm the GPU target with `rocminfo`; substitute that target for
`gfx1151` if using another GPU.

```bash
hipconfig --version
rocminfo | grep -m1 gfx
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -B build-rocm -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 \
    -DVLA_BUILD_TESTS=ON -DVLA_BUILD_SERVER=OFF -DVLA_SPM=OFF
cmake --build build-rocm -j$(nproc)
```

The flags above match this validation. `VLA_BUILD_SERVER=OFF` avoids the
protobuf and ZeroMQ build dependencies; it does not disable `vla-cli`,
`vla-bench`, or the test harness. The server and simulator path was not
retested on this revision.

If ROCm is installed outside the system library search path, export its
library directory before running CTest or any binary. On the test system,
CTest initially failed to load `libhipblas.so.3` until this path was set.

```bash
export VLA_ROCM_ROOT="$(hipconfig -R)"
export LD_LIBRARY_PATH="$VLA_ROCM_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ctest --test-dir build-rocm --output-on-failure
```

`VLA_DEVICE=<n>` chooses the HIP device ordinal (default 0). A successful
model load prints `backend = ROCm/HIP (device 0: AMD Radeon 8060S Graphics)`.
If initialization fails, the core logs the reason and uses CPU. An invalid
ordinal (`VLA_DEVICE=999`) was checked: it fell back to CPU and returned the
same fixed-input action bytes as the CPU build.

ggml-hip still exposes CUDA-named entry points and may print `CUDA graph`
messages. The backend banner identifies the actual device. vla.cpp's own
CUDA-only BF16 and BitVLA kernels are excluded from the HIP build; the tested
`libvla_core.so` linked `libggml-hip`, hipBLAS, rocBLAS and the HIP runtime,
with no CUDA runtime or cuBLAS dependency. CMake rejects HIP combined with
CUDA or Vulkan, and rejects `GGML_BACKEND_DL=ON` with HIP.

## Fixed-input validation

`vla_predict_check` fixes images, language tokens, state and diffusion noise.
Both the CPU reference and HIP run used BF16 resident weights.
The CPU build of this candidate produced byte-identical actions to upstream
`main@1adf078` for both rows below. Each HIP row returned the complete
50 × 32 action array, with finite values and identical action bytes across
five independent processes. CPU and HIP Release builds each passed all ten
first-party CTest cases.

| Published GGUF | Views × image size | CPU reference | HIP vs CPU max abs | RMS | Cosine |
|---|---:|---|---:|---:|---:|
| SmolVLA LIBERO | 2 × 512 | BF16 | 1.992e-3 | 2.421e-4 | 0.9999991 |
| π0.5 LIBERO | 2 × 224 | BF16 | 8.792e-4 | 1.169e-4 | 0.9999998 |

The GGUFs came from `vrfai/smolvla-libero-gguf` (local file
`smolvla-libero.gguf`, SHA-256
`6fb2d475c98b4c2cef3e27c4eff4e67b483740cbf983fff320a3b8a5e5f74fe8`)
and `vrfai/pi05-libero-gguf` (local file `pi05-libero.gguf`, SHA-256
`f9b585b12cbe56bc6b531d40cc45dd1f9036478d5b9f200c1b32aaf323ef6471`).
The comparisons are against the same GGUF and weight dtype on CPU; they are
engine checks, not LIBERO success-rate measurements.

To repeat the fixed-input checks, build CPU with the same `b11223` pin and
`-DVLA_BUILD_TESTS=ON`, then run the corresponding commands on both builds:

```bash
export SMOLVLA_GGUF=/path/to/smolvla-libero.gguf
export PI05_GGUF=/path/to/pi05-libero.gguf
VLA_IMG_SIZE=512 ./build-rocm/tests/vla_predict_check "$SMOLVLA_GGUF" '' 2 --weight-dtype bf16 > smolvla-hip.txt
VLA_IMG_SIZE=224 ./build-rocm/tests/vla_predict_check "$PI05_GGUF" '' 2 --weight-dtype bf16 > pi05-hip.txt
VLA_IMG_SIZE=512 ./build-cpu/tests/vla_predict_check "$SMOLVLA_GGUF" '' 2 --weight-dtype bf16 > smolvla-cpu.txt
VLA_IMG_SIZE=224 ./build-cpu/tests/vla_predict_check "$PI05_GGUF" '' 2 --weight-dtype bf16 > pi05-cpu.txt
```

The tools print the `action_len=` line followed by action values; compare that
portion of each output. Use five fresh HIP processes to check repeatability.

## Synthetic-input engine latency

These are in-process `predict()` timings with fixed synthetic inputs. They
exclude transport and simulator work. Each row used three fresh processes,
each with three warmups and 20 timed calls, BF16 resident weights, GPU
performance level `high` (2.9 GHz observed before and after the rounds), and
CPU governor `performance`. The stock `vla-bench` reports P50, P90 and mean
Vision time for each round. The table uses the median of the three round P50
values, the worst round P90, and the median of the three round Vision means.
All values are milliseconds.

| Model | Views × size | Tokens | P50 | Worst P90 | Vision mean |
|---|---:|---:|---:|---:|---:|
| SmolVLA LIBERO | 2 × 512 | 48 | 395.0 | 396.9 | 210.3 |
| π0.5 LIBERO | 2 × 224 | 128 | 389.1 | 403.1 | 47.1 |

The π0.5 rounds had P50 values 388.1, 389.1 and 389.3 ms; the worst P90 was
403.1 ms. These results use upstream `main@1adf078` and llama.cpp `b11223`.
Do not compare them as a speedup against an older revision without a paired
benchmark.

Run each command three times in fresh processes, changing the log path for
each round:

```bash
./build-rocm/vla-bench --ckpt "$SMOLVLA_GGUF" --images 2 --size 512 \
    --tokens 48 --weight-dtype bf16 --warmup 3 --reps 20 \
    > smolvla-round1.log 2>&1
./build-rocm/vla-bench --ckpt "$PI05_GGUF" --images 2 --size 224 \
    --tokens 128 --weight-dtype bf16 --warmup 3 --reps 20 \
    > pi05-round1.log 2>&1
```

Within a round, `vla-bench` calculates P50 and P90 from sorted samples with
linear interpolation at `p × (n - 1)`. Keep the logs and an environment
manifest when presenting new results.

## Current limits

- The README's ROCm `Y` entries cover only the two configurations measured
  above on Linux `gfx1151`. Other models tested on the earlier `b10729` pin
  need fresh numerical and benchmark checks on `b11223` before being marked
  supported. π0's HIP numerical path also remains under investigation.
- BitVLA's published int2 GGUF requires vla.cpp's CUDA-only kernels. That
  path is not available in a HIP build.
- FoldQuant quantized GGUFs added on the newer mainline were not measured on
  HIP. This PR does not add native HIP FoldQuant integer kernels.
- The server/client route, task success rate, and long-duration soak have not
  been rerun on this upstream revision. Synthetic tokens and images do not
  establish policy equivalence in a robot task.
- The VLA core uses one HIP backend for a full graph and has no per-op CPU
  scheduler fallback. An unsupported HIP op fails prediction instead of
  silently running part of the graph on CPU.
