# ROCm/HIP backend

`vla.cpp` can run a VLA model through the HIP backend in the pinned
`llama.cpp`/ggml dependency. ROCm is selected at configure time, in a separate
build directory; it is not selected automatically on an AMD machine. This page
records the validated configurations, what runs on the GPU, and the current
limits.

| GPU | Target | ROCm | llama.cpp | Models checked | Latency |
|---|---|---|---|---|---|
| Radeon RX 9060 XT, 16 GB | `gfx1200` | 7.14.1 | `b11223` | all 12 archs that load off CUDA, plus ACT | [rx-9060-xt.md](../benchmark/rx-9060-xt.md) |
| Radeon 8060S (Ryzen AI Max+ 395) | `gfx1151` | 7.14.60850 | `b11223` | SmolVLA, π0.5 | [below](#radeon-8060s-gfx1151-1) |

Other GPUs and ROCm versions have not been measured.

## Build and select the device

Install HIP, hipBLAS, rocBLAS and hipBLASLt with their development files, and
the host dependencies in the main README. Confirm the GPU target with
`rocminfo` and pass it as `GPU_TARGETS`:

```bash
hipconfig --version
rocminfo | grep -m1 gfx
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -B build-rocm -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON -DGPU_TARGETS=gfx1200 -DVLA_BUILD_TESTS=ON
cmake --build build-rocm -j$(nproc)
```

Ninja is optional; the RX 9060 XT build used the default Makefiles generator,
with the server and SentencePiece on (the defaults). The gfx1151 validation
below used `-DVLA_BUILD_SERVER=OFF -DVLA_SPM=OFF`, which avoids the protobuf and
ZeroMQ build dependencies; it does not disable `vla-cli`, `vla-bench`, or the
test harness.

With AMD's per-target ROCm 7.14 packages, `amdrocm7.14-gfx1200` installs the
hipBLAS and rocBLAS libraries but not their headers or CMake files, and ggml's
configure step then fails to find `hipblas`. They come from
`amdrocm-blas-dev7.14` and `amdrocm-hipblas-common-dev7.14`.

If ROCm is installed outside the system library search path, export its
library directory before running CTest or any binary. On the gfx1151 system,
CTest initially failed to load `libhipblas.so.3` until this path was set.

```bash
export VLA_ROCM_ROOT="$(hipconfig -R)"
export LD_LIBRARY_PATH="$VLA_ROCM_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ctest --test-dir build-rocm --output-on-failure
```

`VLA_DEVICE=<n>` chooses the HIP device ordinal (default 0). A successful
model load prints `backend = ROCm/HIP (device 0: AMD Radeon RX 9060 XT)`.
If initialization fails, the core logs the reason and uses CPU. An invalid
ordinal (`VLA_DEVICE=999`) was checked on both GPUs: it fell back to CPU and
returned the same fixed-input action bytes as the CPU build.

ggml-hip still exposes CUDA-named entry points and may print `CUDA graph`
messages. The backend banner identifies the actual device. CMake rejects HIP
combined with CUDA or Vulkan, and rejects `GGML_BACKEND_DL=ON` with HIP.

## What runs on the GPU

ggml-hip compiles ggml's CUDA kernels for AMD, and the core reaches it through
`ggml_backend_cuda_init`. The whole graph runs on the GPU; there is no per-op
CPU fallback, so an op HIP does not support fails prediction instead of
silently running on the CPU.

vla.cpp's own CUDA kernels are not compiled into a HIP build, and the paths
that need them stay off:

- the bit-exact fused nodes ([cuda.md](cuda.md#the-bit-exact-fast-path)), so
  `VLA_FUSED_OPS` has no effect;
- BF16 activations: `--act-dtype bf16` is ignored with a message;
- FoldQuant's integer GEMMs: a FoldQuant GGUF is read back as float weights;
- BitVLA's int2 kernels: the published int2 GGUF fails to load.

Graph choices made for ggml-cuda's kernels rather than for vla.cpp's do carry
over. SmolVLA keeps its LM GEMM weights in F32 under the default
`--mm-prec f32`, as on CUDA, instead of converting them from BF16 on every
call, and caches its cross-attention K/V as F16. On the RX 9060 XT this took
SmolVLA's mean from 228.1 to 208.5 ms with byte-identical actions; the F32
copies raise its peak VRAM from 1541 to 2303 MiB.

GEMMs go through hipBLAS and rocBLAS, which hands them to hipBLASLt on
gfx1200. Leave that on: forcing `ROCBLAS_USE_HIPBLASLT=0` made π0 3.5x slower
(774 vs 222 ms).

## Fixed-input validation

`vla_predict_check` fixes images, language tokens, state and diffusion noise,
so each run compares the same action chunk. These are engine checks against
the same GGUF on CPU, not LIBERO success-rate measurements.

### Radeon RX 9060 XT (gfx1200)

Every published GGUF that loads off CUDA was run at its native view count and
input size (the configuration in the
[benchmark report](../benchmark/rx-9060-xt.md#model-configuration)), with
weights as shipped, on the CPU and HIP builds of `6d54233`. SmolVLA's actions
were re-checked at `d0e5d00`, the only later change to a model, and are
byte-identical on both builds. Octo-Small and TurboVLA ship F32 weights; the
others keep BF16 weights resident by default. CTest passed 10/10 on both
builds.

| Model | Actions | HIP vs CPU max abs | RMS | Cosine | F32 weights: HIP vs CPU max abs | CPU vs CPU F32, max abs | HIP vs CPU F32, max abs |
|---|--:|--:|--:|--:|--:|--:|--:|
| Octo-Small | 28 | 2.98e-07 | 1.42e-07 | 1.0000000 | (F32 GGUF) | - | - |
| TurboVLA | 84 | 7.23e-07 | 2.66e-07 | 1.0000000 | (F32 GGUF) | - | - |
| VLA-JEPA | 49 | 9.71e-03 | 3.92e-03 | 0.9999639 | 9.14e-05 | 1.32e-02 | 9.59e-03 |
| VLA-Adapter | 56 | 3.74e-03 | 9.99e-04 | 0.9999933 | 1.55e-06 | 2.10e-03 | 2.83e-03 |
| GR00T N1.5 | 512 | 2.86e-03 | 7.71e-04 | 0.9999962 | 3.35e-04 | 4.49e-03 | 5.84e-03 |
| GR00T N1.6 | 6400 | 1.90e-03 | 3.04e-04 | 0.9999995 | 3.39e-04 | 5.32e-03 | 6.49e-03 |
| GR00T N1.7 | 5280 | 5.76e-03 | 3.58e-04 | 0.9999993 | 3.86e-04 | 1.83e-02 | 1.43e-02 |
| SmolVLA | 1600 | 3.93e-03 | 4.57e-04 | 0.9999969 | 1.74e-04 | 5.57e-03 | 1.78e-03 |
| π0 | 1600 | 4.90e-03 | 4.42e-04 | 0.9999978 | 4.79e-04 | 7.53e-03 | 2.63e-03 |
| π0.5 | 1600 | 7.60e-04 | 1.28e-04 | 0.9999997 | 4.62e-05 | 5.59e-04 | 9.72e-04 |
| Evo-1 | 1200 | 2.31e-03 | 3.06e-04 | 0.9999989 | 2.11e-06 | 2.11e-03 | 1.76e-03 |

- **HIP vs CPU** compares the two builds at their defaults.
- **F32 weights** runs both with `--weight-dtype f32`. HIP and CPU then agree to
  within 4.8e-4, so the kernels match; what is left is summation order.
- The last two columns measure each backend's default against the CPU F32 run.
  HIP's BF16 path lands in the same range as the CPU's own BF16 path, between
  0.3x (SmolVLA) and 1.7x (π0.5) of the CPU's max abs. The gap in the first
  column is BF16 rounding on both sides, not a HIP kernel error.
- Every model returned finite actions, and two fresh HIP processes per model
  (OpenVLA-OFT included) returned identical action bytes.
- **OpenVLA-OFT** runs on HIP (its 15 GB GGUF fits in 16 GB of VRAM) but has
  no CPU reference here: its BF16 weights alone exceed the host's 15.4 GiB of
  RAM.
- **BitVLA** is not run: the published GGUF is int2-packed and needs the CUDA
  kernels.
- **ACT** (`khanhnd61/act_so101-multi-task-clean`, converted with
  `scripts/convert_act_to_gguf.py`, two 640x480 views) matches CPU to 9.9e-5
  max abs, on actions spanning ±150. It runs its convolutions as im2col, as on
  CPU.

Each local GGUF's SHA-256 matched the one Hugging Face lists. Each run was:

```bash
# CPU build and HIP build, same GGUF and environment
VLA_IMG_SIZE=<input> ./build-*/tests/vla_predict_check <gguf> '' <views> [--weight-dtype f32]
```

with the per-model environment from the benchmark report. VLA-JEPA adds
`VLA_EXTRA_TOKEN=151697 VLA_EXTRA_COUNT=32`, and Octo-Small, which takes
exactly 16 tokens, `VLA_EXTRA_TOKEN=0 VLA_EXTRA_COUNT=10`.

### Radeon 8060S (gfx1151)

Both the CPU reference and HIP run used BF16 resident weights. The CPU build of
this candidate produced byte-identical actions to upstream `main@1adf078` for
both rows below. Each HIP row returned the complete 50 × 32 action array, with
finite values and identical action bytes across five independent processes.
CPU and HIP Release builds each passed all ten first-party CTest cases.

| Published GGUF | Views × image size | CPU reference | HIP vs CPU max abs | RMS | Cosine |
|---|---:|---|---:|---:|---:|
| SmolVLA LIBERO | 2 × 512 | BF16 | 1.992e-3 | 2.421e-4 | 0.9999991 |
| π0.5 LIBERO | 2 × 224 | BF16 | 8.792e-4 | 1.169e-4 | 0.9999998 |

The GGUFs came from `vrfai/smolvla-libero-gguf` (local file
`smolvla-libero.gguf`, SHA-256
`6fb2d475c98b4c2cef3e27c4eff4e67b483740cbf983fff320a3b8a5e5f74fe8`)
and `vrfai/pi05-libero-gguf` (local file `pi05-libero.gguf`, SHA-256
`f9b585b12cbe56bc6b531d40cc45dd1f9036478d5b9f200c1b32aaf323ef6471`).
These results are from upstream `main@1adf078` merged with this HIP backend,
on Linux 6.17.0-40 and ggml commit `4da6337`.

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
exclude transport and simulator work.

### Radeon RX 9060 XT (gfx1200)

The full report, with memory, the fastest runtime flags per model and the
method (3 warmups + 20 reps, best of 3 processes), is
[rx-9060-xt.md](../benchmark/rx-9060-xt.md). At the defaults, on `d0e5d00`:

| Model | Views × input | p50 ms | vision ms | Peak VRAM MiB |
|---|--:|--:|--:|--:|
| Octo-Small | 2 × 256 + 128 | 14.0 | 1.9 | 817 |
| ACT | 2 × 640x480 | 24.9 | - | 539 |
| TurboVLA | 2 × 256 | 34.0 | - | 1112 |
| VLA-JEPA | 1 × 256 | 66.4 | 22.2 | 4167 |
| VLA-Adapter | 1 × 224 | 70.9 | 34.9 | 3144 |
| GR00T N1.5 | 1 × 224 | 80.1 | 21.1 | 3925 |
| GR00T N1.6 | 1 × 224 | 95.5 | 20.6 | 4871 |
| GR00T N1.7 | 1 × 256 | 98.2 | 22.2 | 5225 |
| OpenVLA-OFT | 1 × 224 | 183.2 | 36.1 | 14899 |
| SmolVLA | 2 × 512 | 208.7 | 51.9 | 2303 |
| π0 | 2 × 224 | 219.0 | 40.7 | 5791 |
| π0.5 | 2 × 224 | 221.8 | 40.5 | 5793 |
| Evo-1 | 1 × 448 | 234.0 | 79.5 | 1871 |

ACT is not part of the benchmark set; its row is the converted
`khanhnd61/act_so101-multi-task-clean` checkpoint from the validation above,
run with `--images 2 --size 640 --height 480`. Its convolutions run as im2col.
ggml's direct convolution has a WMMA path for RDNA 3 and 4: forcing it for ACT
measured 15.3 ms, with max abs 4.9e-3 against CPU. Turning it on needs a
check for WMMA- or MFMA-capable targets, as `cuda_conv_has_mma()` does for
NVIDIA, and is left for a follow-up.

### Radeon 8060S (gfx1151)

Each row used three fresh processes, each with three warmups and 20 timed
calls, BF16 resident weights, GPU performance level `high` (2.9 GHz observed
before and after the rounds), and CPU governor `performance`. The stock
`vla-bench` reports P50, P90 and mean Vision time for each round. The table
uses the median of the three round P50 values, the worst round P90, and the
median of the three round Vision means. All values are milliseconds.

| Model | Views × size | Tokens | P50 | Worst P90 | Vision mean |
|---|---:|---:|---:|---:|---:|
| SmolVLA LIBERO | 2 × 512 | 48 | 395.0 | 396.9 | 210.3 |
| π0.5 LIBERO | 2 × 224 | 128 | 389.1 | 403.1 | 47.1 |

The π0.5 rounds had P50 values 388.1, 389.1 and 389.3 ms; the worst P90 was
403.1 ms. These results use upstream `main@1adf078` and llama.cpp `b11223`,
before the SmolVLA change above. Do not compare them as a speedup against an
older revision without a paired benchmark.

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

- Only Linux `gfx1200` and `gfx1151` have been measured. Other AMD targets need
  their own numerical and benchmark checks.
- BitVLA's published int2 GGUF requires vla.cpp's CUDA-only kernels and does
  not load in a HIP build.
- FoldQuant GGUFs run as float weights on HIP (there are no HIP integer
  kernels) and were not measured.
- vla.cpp's fused CUDA nodes and BF16 activation path are CUDA-only, so HIP
  runs the plain ggml graph.
- The server/client route, task success rate, and long-duration soak have not
  been rerun on HIP. Synthetic tokens and images do not establish policy
  equivalence in a robot task.
- The VLA core uses one HIP backend for a full graph and has no per-op CPU
  scheduler fallback. An unsupported HIP op fails prediction instead of
  silently running part of the graph on CPU.
