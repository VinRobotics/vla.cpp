# Prebuilt binaries

Each [release](https://github.com/VinRobotics/vla.cpp/releases) has a
`vla.cpp-<tag>-<platform>.tar.gz` that extracts to a `vla.cpp-<tag>-<platform>/`
directory holding `libvla`, `vla.h` and the binaries side by side. The Linux ones
ship `vla-cli`, `vla-bench`, `vla-server` and `vlm-server`; macOS ships `vla-cli`
and `vla-bench`.

| Platform | Needs |
|---|---|
| `linux-x86_64-cpu` | AVX2 (Haswell or newer) |
| `linux-x86_64-cuda-12.8` | AVX2; sm_75/80/86/89/90/120 |
| `linux-x86_64-cuda-13.4` | AVX2; sm_75/80/86/89/90/120, driver 580 or newer |
| `linux-aarch64-cpu` | ARMv8.2-A with dotprod and fp16 (Cortex-A76, Neoverse N1 or newer) |
| `linux-aarch64-cuda-13.4` | sm_87 (Orin), sm_110 (Thor), sm_121 (DGX Spark); a CUDA 13 driver |
| `macos-arm64-metal` | Apple Silicon |

```bash
TAG=<release tag>
PLATFORM=linux-x86_64-cuda-13.4
curl -LO https://github.com/VinRobotics/vla.cpp/releases/download/$TAG/vla.cpp-$TAG-$PLATFORM.tar.gz
tar -xzf vla.cpp-$TAG-$PLATFORM.tar.gz
./vla.cpp-$TAG-$PLATFORM/vla-cli --help
```

## Linux

The Linux tarballs are built on Ubuntu 24.04 and do not load on an older glibc
such as Ubuntu 22.04 or JetPack 6; build from source there. Apart from the CUDA
runtime and ZeroMQ, they carry every library they use.

- `vla-server` and `vlm-server` need `sudo apt install libzmq5`.
- For a CUDA tarball, also extract the matching
  `cudart-vla.cpp-<tag>-<platform>.tar.gz` in the same place unless the CUDA
  runtime is already installed; it drops `libcudart`, `libcublas` and
  `libcublasLt` next to the binaries.

```bash
curl -LO https://github.com/VinRobotics/vla.cpp/releases/download/$TAG/cudart-vla.cpp-$TAG-$PLATFORM.tar.gz
tar -xzf cudart-vla.cpp-$TAG-$PLATFORM.tar.gz
```

## macOS

The macOS build has no SentencePiece, so `vla-cli --text` tokenizes through
`scripts/tokenize_prompt.py`, which ships in the tarball and needs `transformers`
in the active Python. Keep `default.metallib` next to the binaries.

## Windows

There are no Windows builds yet; see [backend/hexagon-windows.md](backend/hexagon-windows.md)
to build from source. The Docker image is covered in [DOCKER.md](DOCKER.md).
