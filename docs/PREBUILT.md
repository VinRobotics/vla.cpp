# Prebuilt binaries

Each [release](https://github.com/VinRobotics/vla.cpp/releases) has a
`vla.cpp-<tag>-<platform>.tar.gz` that extracts to a `vla.cpp-<tag>-<platform>/`
directory holding `libvla`, `vla.h` and the binaries side by side. The Linux ones
ship `vla-cli`, `vla-bench`, `vla-server` and `vlm-server`; macOS ships `vla-cli`
and `vla-bench`. The platforms and what each needs are in the README's
[Install](../README.md#prebuilt-binaries) section.

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
