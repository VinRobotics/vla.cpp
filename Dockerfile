# vla-server, CPU or CUDA. cmake fetches llama.cpp at build time.
#   GPU sm_120 (default):  docker build -t vla-cpp .
#   GPU other arch:        --build-arg CUDA_ARCH=89   (89=RTX40 90=H100 87=Orin; sm_120 needs CUDA>=12.8)
#   older card:            --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu24.04 --build-arg CUDA_ARCH=86
#   CPU:                   --build-arg BACKEND=cpu --build-arg BASE_IMAGE=ubuntu:24.04 -t vla-cpp-cpu
#   run:  docker run --gpus all -p5555:5555 -v $PWD/models:/models vla-cpp --bind tcp://*:5555 /models/M.gguf
#         (CDI hosts use --device nvidia.com/gpu=all)

ARG BASE_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04
FROM ${BASE_IMAGE}

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates pkg-config \
        libzmq3-dev cppzmq-dev libprotobuf-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

ARG BACKEND=cuda
ARG CUDA_ARCH=120
# nvcc can segfault on the flash-attn kernels under high -j; lower JOBS if so.
ARG JOBS=
# CUDA: -devel ships only a libcuda stub (real driver injected at runtime), so
# link ggml-cuda's driver calls against it. CPU build skips this.
RUN set -eux; \
    if [ "$BACKEND" = "cuda" ]; then \
        export LIBRARY_PATH=/usr/local/cuda/lib64/stubs; \
        cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON \
              -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
              -DCMAKE_SHARED_LINKER_FLAGS="-lcuda" -DCMAKE_EXE_LINKER_FLAGS="-lcuda"; \
    else \
        cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF; \
    fi; \
    cmake --build build -j"${JOBS:-$(nproc)}" --target vla-server; \
    cp build/vla-server /usr/local/bin/vla-server

EXPOSE 5555
ENTRYPOINT ["vla-server"]
