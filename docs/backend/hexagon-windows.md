# vla.cpp on Snapdragon X (Windows on Arm): Hexagon NPU, Adreno GPU and CPU

Measured 2026-09 against llama.cpp build 11201 (`2145525a4`), passed in with
`-LlamaDir`. The `b11223` tag that `CMakeLists.txt` pins was not tested.

## Summary

vla.cpp now builds natively on a Snapdragon X laptop in three flavours:

- the Oryon CPU;
- the Hexagon NPU (HTP), with ops the NPU rejects running on the CPU;
- the Adreno GPU (OpenCL), with the same CPU fallback.

`vla-server`, `vla-cli`, `vla-bench` and the tests all build.

Eleven of the twelve published checkpoints run on all three. BitVLA does not: its only published GGUF is int2-packed, which only CUDA builds load. OpenVLA-OFT was not attempted, because at F16 it needs about 14 GB (more than the 8 GB budget).

Every accelerator result below was checked against a CPU-backend reference on identical inputs before its latency was recorded.

- **SmolVLA runs fastest on the NPU:** 1.23 s per action chunk against 2.57 s on the CPU and 3.05 s on the GPU. It is within 1.5e-3 of the CPU reference.
- **The NPU also wins on π0 (5.6 s vs 10.4 s CPU) and VLA-JEPA (0.85 s vs 1.6 s).** Both models are sensitive to precision, and both land 4.8e-3 to 5.6e-3 from an F32 reference, above the 2.9e-3 bar the other backends are held to. Running the same F16 weights on the CPU moves them further (8.5e-3 and 1.2e-2).
- **On the other eight models the CPU is fastest**, once it runs F16 weights instead of the BF16 it defaults to. Oryon has no BF16 matmul, so the shipped default is 3-4x slower than it needs to be. Both accelerators still agree with the reference on these models, most to within 1e-3.
- **Five ggml-hexagon kernels give wrong answers for shapes vla.cpp uses.** The first three can flip the gripper channel (max |Δ| 0.7 to 1.9):
  - GELU runs as GELU_QUICK.
  - A broadcast with a gap is mis-indexed.
  - IM2COL is wrong unless the conv is a patch embedding.
  - F16 copies read back wrong on the host.
  - Op fusion costs precision.

  The CPU fallback now routes around each of them, and each is described [below](#hexagon-issues-found-and-worked-around).

The NPU path needs the same one-off setup as llama.cpp's Hexagon backend: a newer NPU driver, test-signing, and a self-signed certificate. See [NPU prerequisites](#npu-prerequisites).

## Device

| Item | Value |
|---|---|
| Laptop | ASUS Vivobook 14 X1407QA |
| SoC | Snapdragon X X1-26-100, 8 Oryon cores |
| GPU | Adreno X1-45, driver 31.0.112.0 |
| NPU | Hexagon v73, driver 30.0.220.3000; 4 HVX threads, 1 HMX unit, 8 MB VTCM |
| RAM | 16 GB |
| OS | Windows 11 Home, build 26200 |

## Toolchain

| Component | Version / location | Used for |
|---|---|---|
| Visual Studio | 2026 (18), MSVC 14.51, ARM64 libraries | Its bundled Clang 22.1.3, CMake and Ninja build everything |
| Windows SDK / WDK | 10.0.26100 | `inf2cat` and `signtool`, to sign the HTP skels |
| Hexagon SDK | 6.6.0.0, tools 19.0.07 (`HEXAGON_SDK_ROOT`, `HEXAGON_TOOLS_ROOT`) | Building the HTP skels |
| OpenCL SDK | 2.3.2 (`OPENCL_SDK_ROOT`) | The Adreno backend |
| vcpkg | `C:\vcpkg` | protobuf 6.33.4 (triplet `arm64-windows-clangcl`, in this repo), zeromq 4.3.5 and cppzmq (stock `arm64-windows`) for `vla-server` |
| Python | 3.14.7 ARM64, venv in `.venv` | `transformers` 5.17, `tokenizers`, `gguf`, `numpy`: tokenizing prompts and `scripts/quantize_gguf.py` |

vcpkg needs no admin rights. protobuf has to be compiled by clang, like
vla.cpp. Otherwise clang's and MSVC's different mangling of `__restrict` leaves
the servers with unresolved protobuf symbols. The overlay triplet in
`cmake/vcpkg-triplets` builds it with Visual Studio's clang-cl. ZeroMQ is a C
API, so the stock MSVC build serves. Run these from a Visual Studio developer
shell:

```
git clone --depth 1 https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat -disableMetrics
C:\vcpkg\vcpkg install zeromq cppzmq --triplet arm64-windows
C:\vcpkg\vcpkg install protobuf --triplet arm64-windows-clangcl --overlay-triplets=<vla.cpp>\cmake\vcpkg-triplets
```

## Building

The build script sets up the Visual Studio shell, the compiler flags llama.cpp's Snapdragon preset uses, vcpkg, and (for the NPU) skel signing:

```
.\scripts\build_windows_snapdragon.ps1 -Backend htp    -LlamaDir <llama.cpp> -HtpCert <cert.pfx>
.\scripts\build_windows_snapdragon.ps1 -Backend opencl -LlamaDir <llama.cpp>
.\scripts\build_windows_snapdragon.ps1 -Backend cpu    -LlamaDir <llama.cpp>
```

Each build goes into `build-wos-<backend>`, with every binary and DLL in `build-wos-<backend>\bin`. `-NoServer` skips `vla-server`, SentencePiece and their protobuf and ZeroMQ dependencies, so Octo `--text` needs `--tokens` there.

`-LlamaDir` points the build at an existing llama.cpp checkout through `FETCHCONTENT_SOURCE_DIR_LLAMA`. Without it, the `b11223` pin in `CMakeLists.txt` applies, which was not tested here.

The HTP build also signs `libggml-htp-v*.so` with the certificate and copies the skels and their catalog next to the binaries. At startup the Hexagon backend points `ADSP_LIBRARY_PATH` at the executable's own folder, but only if the variable is unset. If it is already set, for example by a llama.cpp install, the skels it names must come from the same llama.cpp commit.

### NPU prerequisites

These follow llama.cpp's [Windows guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/windows.md). They are one-off per machine, and the steps that need admin rights are marked.

1. **Update the NPU driver.** Install Qualcomm's HND package from the Qualcomm Software Center (driver 30.0.220 or later; admin and a reboot). The driver shipped with the laptop (30.0.143) lacks the `dspqueue_*` functions, and the backend fails with `failed to dlsym dspqueue_create`.
2. **Enable test-signing** (admin and a reboot). Secure Boot must be off first, in the firmware setup. Then run `bcdedit /set TESTSIGNING ON`.
3. **Create and trust a code-signing certificate** (admin, for the trust step):

   ```
   New-SelfSignedCertificate -Subject "CN=GGML.HTP.v1" -Type CodeSigningCert -CertStoreLocation Cert:\CurrentUser\My
   # export it to <cert.pfx> and <cert.cer>, then:
   certutil -addstore Root <cert.cer>
   certutil -addstore TrustedPublisher <cert.cer>
   ```

4. **Pass the `.pfx` to the build** with `-HtpCert`. Without it the skels are unsigned, and opening a session fails with error `0x80000406`.

### What had to change for Windows

| Problem | Fix |
|---|---|
| `fseeko` and `popen` are POSIX; `clock_gettime` in the test harness | `_fseeki64`, `_popen` with cmd.exe quoting, `std::chrono` |
| `vla_core` became a DLL (llama.cpp turns on `BUILD_SHARED_LIBS`) that exports nothing, so every consumer failed to link | The two internal libraries are static on Windows |
| DLLs land in `<build>\bin` and Windows has no rpath | Executables go to `bin` too; the servers' vcpkg DLLs are copied next to them |
| protobuf and ZeroMQ were required even for `vla-cli` | New option `VLA_BUILD_SERVER` (default ON); ZeroMQ is also found through its CMake package, since vcpkg ships no pkg-config |
| SentencePiece adds `-fPIC`, which clang rejects when targeting Windows | Patched out at fetch time, on Windows only |
| vcpkg's protobuf shim hands SentencePiece the DLL instead of its import library | SentencePiece's protobuf dependency is linked as the `protobuf::libprotobuf` target, from the top-level directory |
| clang and MSVC mangle `__restrict` parameters differently, so clang code cannot link against an MSVC-built protobuf | protobuf built with clang-cl (the overlay triplet above) |
| `setvbuf(stdout, NULL, _IOLBF, 0)` in both servers: the Windows CRT has no line buffering and treats size 0 as invalid, so it fails fast (`0xC0000409`) before printing anything | `_IONBF` on Windows |
| vcpkg's abseil passes the MSVC-only flag `-ignore:4221` to its consumers | Stripped from the imported targets |
| ggml-hexagon's catalog-signing step can run before the skels exist in a parallel build | The script builds the four skels first |

## Running

```
$env:VLA_PYTHON = "$PWD\.venv\Scripts\python.exe"   # tokenizer for --text
.\build-wos-htp\bin\vla-cli.exe --ckpt models\smolvla\smolvla-libero.gguf --image assets\front.jpg --text "pick up the black bowl" --pretty
.\build-wos-htp\bin\vla-server.exe models\smolvla\smolvla-libero.gguf
```

`vla-server` on the NPU answered a two-view SmolVLA request over ZeroMQ in
1.25 s server-side (1.67 s for the first, cold request), matching
`vla_predict_check`. The LIBERO client could not be run, because it needs
PyTorch and LIBERO; the smoke test used a 20-line pyzmq client instead.

| Variable / flag | Effect in the Hexagon and OpenCL builds |
|---|---|
| `VLA_DEVICE=cpu` | Skip the accelerator. The same binary then produces the CPU reference. |
| `VLA_DEVICE=<n>` | Registry device index: an HTP session, or an OpenCL GPU |
| `--weight-dtype f16\|f32\|bf16` | Resident dtype for GEMM weights. New: `f16`, which is also the default on both accelerators. |
| `--flash-attn on\|off` | On by default in Hexagon builds, off elsewhere |
| `VLA_FALLBACK_STATS=1` | Print how each graph was split between the accelerator and the CPU |
| `GGML_HEXAGON_OPFUSION` | Defaults to `0` here (see below); set to `1` to trade precision for speed |
| `GGML_OPENCL_ADRENO_XMEM_GEMM` | Defaults to `0` here (see below); set to `1` to trade precision for speed |

Each process prints which ops the accelerator gave to the CPU, once per op type, and a summary when it exits:

```
vla: HTP0+CPU rejects GET_ROWS (f32 <- bf16, [960,48,1,1], e.g. 'node_15'); running it on CPU
vla: HTP0+CPU: 7 of 35 graphs split; 7 nodes in 7 runs went to CPU, 90.2 MiB copied in, 1.2 MiB out, 90.2 MiB of weights mirrored on host
```

## How it works

### The CPU fallback

The engine drives one backend per model through `gallocr`, with no scheduler. That works for CUDA, Metal, SYCL and OpenVINO, which run every op the archs build. Hexagon and OpenCL do not. An NPU rejects an op when it has no kernel for it (GELU_ERF), or when the tile does not fit VTCM (softmax rows that are not a multiple of 32). Neither can be known until the graph exists.

`src/backend_fallback.cpp` wraps the accelerator in a backend that looks like it to the arch: same device, same buffer type, so weights and activations still live on the accelerator. Its `graph_compute` splits each graph:

- Runs of ops the accelerator accepts go to it unchanged.
- Runs it rejects are copied to host memory, computed on a CPU backend, and copied back.
- Weights that a CPU op reads are copied once and kept.

No arch's call sites changed, and CUDA, Metal, SYCL and OpenVINO builds do not compile the wrapper at all. The Hexagon- and OpenCL-specific graph changes (`fa_kv`, `gelu`, the default weight dtype and flash attention) sit behind the same build flags and reduce to the old calls elsewhere. SmolVLA's CPU BF16 output was checked bit-identical before and after them. The other archs were not re-checked against a pre-change build, because the unmodified tree does not build on Windows.

The wrapper is also where Hexagon's wrong kernels are refused (next section). Forcing every op through the CPU path (`GGML_HEXAGON_OPFILTER=.*`) reproduces a pure-CPU run bit for bit.

### Hexagon issues found and worked around

Each issue below was found by moving op families to the CPU with `GGML_HEXAGON_OPFILTER` until the action chunk matched the reference. None are vla.cpp bugs; all should go upstream.

| Issue | Seen on | Error on the NPU | After the fix | Where fixed |
|---|---|---:|---:|---|
| Unary `GELU` runs `x*sigmoid(1.702x)`, which is GELU_QUICK. Upstream maps both ops to one kernel. | SmolVLA (SigLIP) | 1.9 | 1.5e-3 | `vla::gelu()` builds the tanh form exactly as `x*sigmoid(2c(x+0.044715x³))` from ops the NPU gets right. The wrapper refuses any remaining `GGML_UNARY_OP_GELU`. |
| Binary op with `src1` broadcast over one dimension but not a higher one, e.g. `[64,12,261,2] * [64,1,261,1]` | TurboVLA (RoPE), Octo | 0.72 | 6.0e-4 | Wrapper sends these to the CPU |
| `IM2COL` accepts any geometry, but is right only for patch embeddings (kernel = stride > 1, no padding) | Octo (stride-2 pad-1 stem, 1x1 projection) | 1.05 | 5.2e-4 | Wrapper sends other geometries to the CPU; SigLIP's 16x16/16 stays on the NPU |
| An F32→F16 `CPY` computed on the NPU reads back wrong on the host, though the NPU's own consumers see it correctly | SmolVLA with FA forced to CPU | 1.2 | 1.7e-3 | A copy that feeds a CPU op runs on the CPU too |
| Op fusion (MUL_MAT+ADD, RMS_NORM+MUL) | SmolVLA | 1.2e-2 | 1.6e-3 | `GGML_HEXAGON_OPFUSION` defaults to 0 |

Two further changes keep work on the NPU:

- **F16 K/V for flash attention.** The HTP kernel takes F16 or Q8_0 K/V only. The archs pass F32, which sent every SmolVLA attention through the CPU: 352 runs and 348 MiB of copies per chunk. `vla::fa_kv()` casts K/V before the permute in Hexagon builds, and is a no-op elsewhere. Evo-1 keeps F32 K/V (its attention runs on the CPU), because the cast measured 3.9e-2.
- **Flash attention on by default.** In the towers, the explicit path's 1024x1024 F32 score matmuls took SmolVLA's vision stage to 3.8 s. Flash attention does it in 0.44 s.

Not fixed, left on the CPU:

- `GELU_ERF` has no HTP kernel (DINOv2 and SigLIP-so400m towers).
- `SOFT_MAX` needs rows that are a multiple of 32.
- `GROUP_NORM`, `SIN` and `COS` are also unsupported.

Evo-1 is where this costs the most: 3,900 CPU runs and 11 GB of copies per chunk, which is why the NPU is slower than the CPU there.

### Adreno (OpenCL) issues

| Issue | Error | After the fix | Fix |
|---|---:|---:|---|
| The "xmem" F16xF32 GEMM (weights prepacked into images) rounds activations | SmolVLA 5.0e-3 | 2.9e-4 | `GGML_OPENCL_ADRENO_XMEM_GEMM` defaults to 0. It is about 1.4x faster on SmolVLA where it works. |
| The same GEMM aborts with `CL_OUT_OF_RESOURCES` on GR00T N1.7's larger weights | crash | 2.9e-4 | same |
| MUL_MAT has no BF16 kernel | every GEMM on CPU | – | F16 resident weights by default |

### Weights: F16, not BF16

The shipped GGUFs are BF16, and most archs keep them BF16 in memory. HTP has no BF16 kernel for any op, and Adreno none for MUL_MAT, so both accelerator builds default to F16 (`vla::default_weight_dtype`). F16 holds every normal-range BF16 value exactly, and `--weight-dtype` still overrides.

SmolVLA used to convert every weight to its resident dtype, so a `quantize_gguf.py` file would not load for it on any platform. It now keeps a GEMM weight the file stores packed, as the other archs already do, and it dequantizes any packed tensor it keeps as float.

## Benchmark

### Setup

| Item | Value |
|---|---|
| Tool | `vla_predict_check`: fixed images, tokens, state and noise; `VLA_BENCH_ITERS=3` after 3 warm-ups; minimum reported |
| Inputs | Each model at its native size and view count (the README's table); VLA-JEPA with its 32 `<embodied>` tokens; Octo's prompt padded to 16 tokens, un-normalized against `libero_object` |
| Checkpoints | The published `vrfai/*` GGUFs (GR00T N1.7 and VLA-Adapter: `libero_object`) |
| Fidelity | max \|Δ\| of the action chunk against the CPU backend with `--weight-dtype f32`. A second number against the CPU BF16 default is given where it is tighter, following [ov.md](ov.md#picking-the-right-baseline). The bar the other backends are held to is 2.9e-3. |
| Memory | Peak working set of the process, from `GetProcessMemoryInfo` |
| Threads | 8 (all cores); power plan and temperature not controlled |

### Latency (ms per action chunk)

CPU BF16 is what a stock build does today. CPU F16 is the fair CPU baseline. HTP and OpenCL use their defaults (F16 weights).

| Model | Views × size | CPU BF16 | CPU F16 | NPU (HTP) | GPU (OpenCL) | Fastest |
|---|---|--:|--:|--:|--:|---|
| SmolVLA | 2 × 512 | 7,609 | 2,570 | **1,227** | 3,048 | NPU |
| VLA-JEPA | 1 × 256 | 5,557 | 1,617 | **845** | 2,402 | NPU |
| π0 | 2 × 224 | 30,937 | 10,387 | **5,563** | 13,644 | NPU |
| π0.5 | 2 × 224 | 29,957 | 9,987 | **9,280** | 13,358 | NPU (7% faster) |
| GR00T N1.7 | 1 × 256 | 6,700 | **1,983** | 3,068 | 3,535 | CPU |
| GR00T N1.6 | 1 × 224 | 7,627 | **2,232** | 5,019 | 4,032 | CPU |
| GR00T N1.5 | 1 × 224 | 9,299 | **3,186** | 5,215 | 4,572 | CPU |
| VLA-Adapter | 1 × 224 | 7,281 | **2,140** | 4,127 | 3,655 | CPU |
| Evo-1 | 1 × 448 | 16,451 | **6,451** | 7,297 | 9,321 | CPU |
| TurboVLA | 2 × 256 | 598 | **388** | 2,428 | 717 | CPU |
| Octo-Small | 1 × 256 | 104 | **88** | 196 | 148 | CPU |

TurboVLA and Octo ship F32 GGUFs, so their "CPU BF16" column is the F32 default.

### Fidelity (max |Δ| against CPU F32)

| Model | CPU F16 | NPU (HTP) | GPU (OpenCL) |
|---|--:|--:|--:|
| SmolVLA | 6.7e-4 | 2.0e-3 (1.5e-3 vs BF16) | 2.9e-4 |
| VLA-JEPA | 8.5e-3 | **4.8e-3** | 1.7e-4 |
| π0 | 1.2e-2 | **5.6e-3** | 1.4e-3 |
| π0.5 † | 6.5e-4 | 7.1e-4 | 6.3e-4 |
| GR00T N1.7 | 2.0e-2 | 1.6e-3 | 2.9e-4 |
| GR00T N1.6 | 1.0e-3 | 4.1e-4 | 3.1e-4 |
| GR00T N1.5 | 9.4e-4 | 1.5e-3 | 8.9e-4 |
| VLA-Adapter | 1.5e-3 | 7.0e-4 | 3.0e-5 |
| Evo-1 | 1.6e-3 | 4.1e-4 | 2.1e-4 |
| TurboVLA | 7.1e-3 | 6.0e-4 | 1.1e-6 |
| Octo-Small | 0 | 5.2e-4 | 8.2e-4 |

The GR00T N1.5 and N1.6 rows predate flash attention reaching their towers.
Hexagon turns it on by default, so both now run it there; they were not
re-measured.

† Against the CPU BF16 run. An F32 reference for π0.5 needs more memory than the 8 GB budget.

The accelerators are often closer to F32 than the CPU running the same F16 weights. ggml's CPU matmul converts activations to the weight's type (F16 here), while HTP and Adreno keep them in F32.

Bold marks the two results above the 2.9e-3 bar. Both models are precision-sensitive: their own CPU F16 run is worse. The fix, F32 weights, would need 12 GB (π0) and 8.6 GB (VLA-JEPA). Turning flash attention off only brings them to 4.9e-3 and 3.4e-3, at 1.6-2x the latency.

### Peak memory (GB)

Peak working set of the whole process:

| Model | CPU BF16 | CPU F16 | NPU | GPU |
|---|--:|--:|--:|--:|
| SmolVLA | 1.15 | 1.15 | 1.27 | 1.85 |
| VLA-JEPA | 3.75 | 3.75 | 3.82 | 4.20 |
| π0 | 5.20 | 5.20 | 5.25 | 5.67 |
| π0.5 | 5.63 | 5.63 | 5.71 | 6.10 |
| GR00T N1.7 | 6.00 | 6.00 | 6.07 | 7.44 |
| GR00T N1.6 | 5.62 | 5.62 | 5.70 | 7.09 |
| GR00T N1.5 | 4.57 | 4.57 | 4.66 | 6.02 |
| VLA-Adapter | 2.64 | 2.64 | 2.73 | 3.77 |
| Evo-1 | 1.40 | 1.40 | 1.47 | 1.84 |
| TurboVLA | 0.83 | 0.48 | 0.93 | 1.40 |
| Octo-Small | 0.61 | 0.61 | 0.66 | 1.11 |

Every configuration stays under 8 GB. The NPU's weights live in FastRPC shared
memory: they count toward the working set, but the process's private commit stays
under 1.4 GB. The GPU build is the largest, because the OpenCL driver's buffers
count on top of the host copy made during loading.

### Q8_0

`scripts/quantize_gguf.py --type Q8_0` packs the LM backbones. It saves memory; it does not preserve fidelity. The CPU itself moves π0 by 4.9e-2 and the GR00Ts by 7e-3 to 4e-2 on the Q8_0 files. Latency and fidelity on the Q8_0 files (fidelity against the CPU F32 run of the original):

| Model | Weights BF16 → Q8_0 | NPU | GPU | CPU (Q8_0) |
|---|---|--:|--:|--:|
| SmolVLA | 1.13 → 0.72 GB | 1,140 ms, 1.2e-2 | 3,015 ms, 9.7e-3 | 2,512 ms, 1.0e-2 |
| π0 | 6.49 → 4.33 GB | 5,010 ms, 2.3e-2 | 7,336 ms, 2.7e-1 | 8,893 ms, 4.9e-2 |
| π0.5 † | 12.59 → 5.85 GB | 8,666 ms, 1.5e-3 | 7,009 ms, 4.4e-3 | 8,947 ms, 2.3e-3 |
| GR00T N1.6 | 9.16 → 7.99 GB | 4,930 ms, 7.9e-3 | 3,473 ms, 8.5e-3 | 6,509 ms, 6.6e-3 |
| VLA-JEPA | 4.57 → 3.24 GB | 745 ms, 2.5e-2 | 1,477 ms, 4.9e-2 | 2,979 ms, 5.2e-2 |

SmolVLA's CPU Q8_0 run keeps its float weights at F16; the other rows keep the CPU default, BF16. The π0 Q8_0 result on the GPU (0.27) is far worse than the same file on the CPU, and was not investigated. Evo-1's Q8_0 file failed to load on every backend with a `ggml_view` assertion; it now loads on the CPU and the NPU. The Adreno GPU refuses one made by the old quantizer, because ggml-opencl ignores view offsets on quantized weights; requantize with the current `scripts/quantize_gguf.py`, which keeps the action expert float.

## Observations

- **The NPU wins where the graph stays on it.**
  - SmolVLA's whole graph runs on the NPU except 7 embedding row fetches, and its vision tower is 4x faster there than on the CPU.
  - π0 and π0.5 lose only their odd-width softmaxes to the CPU.
  - Where GELU_ERF or a 1025-wide softmax sits in every tower layer (Evo-1, TurboVLA, VLA-Adapter), the copies cost more than the NPU saves.
- **The CPU's shipped default is its slow path.** BF16 → F16 alone makes every BF16 model 2.5-3.5x faster on Oryon, with fidelity inside or near the bar. That is a one-flag win (`--weight-dtype f16`) for any ARM CPU without BF16 matmul.
- **The Adreno GPU is never the fastest here**, but it is the most faithful accelerator: every model is within 1.4e-3 of F32.
- **The NPU's vision is slow on SigLIP-so400m.** π0's tower takes 3.5 s on the NPU against 1.9 s on the CPU, with nothing falling back. So400m's 4304-wide MLP is not a multiple of 32, which likely keeps it off the matrix unit; this was not profiled.

## Limitations

- **No task-success numbers.** LIBERO does not run on Windows on Arm here, so fidelity is the only check that actions are right.
- **One device, one driver, one llama.cpp commit.** ggml-hexagon changes fast, and several of the bugs above may already be fixed upstream.
- **Few repetitions, no power or thermal control.** The minimum of 3 runs is reported, and the laptop was on its default power plan.
- **BitVLA and OpenVLA-OFT were not run.** BitVLA's published GGUF is int2 for CUDA; a BF16 GGUF would have to be converted from the PyTorch checkpoint. OpenVLA-OFT exceeds the 8 GB budget at F16 and was not downloaded.
- **Q8_0 on the NPU uses dynamically quantized activations**, which is part of its error; not separated here.

## Known issues

- The NPU prints `ggml-hex: FASTRPC_GET_DOMAINS query failed (0x6c), using static CDSP domains` on every run. It works regardless.
- The Adreno driver refuses `GGML_OPENCL_ADRENO_USE_LARGE_BUFFER` ("not supported by driver"), so no single allocation can exceed about 1 GB. F16 weights keep every model here under that.
- `Launch-VsDevShell.ps1` prints `'vswhere.exe' is not recognized`; the build is not affected.

## Security state

Using the NPU leaves the machine with Secure Boot off, test-signing on, and a self-signed certificate trusted as a root authority. Anyone who can read the `.pfx` can sign code that machine will trust, so keep it private. To undo, run the following; the NPU backend then stops loading, while the CPU and GPU are unaffected:

```
certutil -delstore Root <thumbprint>
certutil -delstore TrustedPublisher <thumbprint>
bcdedit /set TESTSIGNING OFF
```

Then turn Secure Boot back on in the firmware setup.
