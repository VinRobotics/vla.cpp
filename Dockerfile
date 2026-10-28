# vla-server, CPU or CUDA. cmake fetches llama.cpp at build time; the image keeps
# only the binaries and their libs on a -runtime (or plain ubuntu) base.
#   GPU, CUDA 12.9, sm_75..sm_121:  docker build -t vla-cpp .
#   one arch, faster build:   --build-arg CUDA_ARCH=120  (86=RTX30 89=RTX40 90=H100 87=Orin 120=RTX50)
#   CUDA 13 (driver 580+):    --build-arg CUDA_VERSION=13.4.1
#   CPU:                      --build-arg BACKEND=cpu -t vla-cpp-cpu
#   run:  docker run --gpus all -p5555:5555 -v $PWD/models:/models vla-cpp --bind tcp://*:5555 /models/M.gguf
#         (CDI hosts use --device nvidia.com/gpu=all)

ARG CUDA_VERSION=12.9.1
ARG BACKEND=cuda

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS cuda-build
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS cuda-run
FROM ubuntu:24.04 AS cpu-build
FROM ubuntu:24.04 AS cpu-run

FROM ${BACKEND}-build AS build

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates pkg-config python3 \
        libzmq3-dev cppzmq-dev libprotobuf-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

ARG BACKEND
ARG CUDA_ARCH="75-real;80-real;86-real;89-real;90;120-real;121-real"
ARG GGML_NATIVE=ON
# nvcc can segfault on the flash-attn kernels under high -j; lower JOBS if so.
ARG JOBS=
# CUDA: -devel ships only a libcuda stub (real driver injected at runtime), so
# link ggml-cuda's driver calls against it. CPU build skips this.
RUN set -eux; \
    if [ "$BACKEND" = "cuda" ]; then \
        export LIBRARY_PATH=/usr/local/cuda/lib64/stubs; \
        set -- -DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}" \
               -DCMAKE_SHARED_LINKER_FLAGS="-lcuda" -DCMAKE_EXE_LINKER_FLAGS="-lcuda"; \
    else \
        set -- -DGGML_CUDA=OFF; \
    fi; \
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=${GGML_NATIVE} \
          -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON "$@"; \
    cmake --build build -j"${JOBS:-$(nproc)}" --target vla-server vla-cli; \
    mkdir /app; \
    cp build/vla-server build/vla-cli /app/; \
    cp -P build/*.so* build/bin/*.so* /app/; \
    cp LICENSE.md /app/; \
    cp build/_deps/llama-src/LICENSE /app/LICENSE.llama.cpp; \
    cp build/_deps/llama-src/licenses/LICENSE-jsonhpp /app/LICENSE.jsonhpp; \
    cp build/_deps/sentencepiece-src/LICENSE /app/LICENSE.sentencepiece; \
    cp build/_deps/sentencepiece-src/third_party/darts_clone/LICENSE /app/LICENSE.darts_clone

FROM ${BACKEND}-run

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgomp1 libprotobuf-lite32t64 libprotobuf32t64 libzmq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/ /app/
ENV PATH=/app:$PATH

EXPOSE 5555
ENTRYPOINT ["vla-server"]
