cmake_minimum_required(VERSION 3.22)
project(vla_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(_vla_accel "")
foreach(_flag GGML_CUDA GGML_SYCL GGML_METAL GGML_OPENVINO)
    if(${_flag})
        list(APPEND _vla_accel ${_flag})
    endif()
endforeach()
list(LENGTH _vla_accel _vla_accel_n)
if(_vla_accel_n GREATER 1)
    string(REPLACE ";" ", " _vla_accel_str "${_vla_accel}")
    message(FATAL_ERROR
        "Enable one accelerator backend at a time; got ${_vla_accel_str}. "
        "Configure a separate build directory per backend.")
endif()

set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(_vla_llama_patch "")
if(GGML_CUDA)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(_vla_llama_patch PATCH_COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_ggml_cuda_ext_hook.py <SOURCE_DIR>)
elseif(GGML_OPENVINO)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(_vla_llama_patch PATCH_COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_ggml_openvino.py <SOURCE_DIR>)
endif()
# Overridable so a regression can be bisected against another tag in a separate
# build dir (-DVLA_LLAMA_TAG=b10326) without editing this file. The patch
# anchors in scripts/patch_ggml_cuda_ext_hook.py and scripts/patch_ggml_openvino.py
# are checked against the default.
set(VLA_LLAMA_TAG "b10331" CACHE STRING "llama.cpp tag to fetch")

include(FetchContent)
FetchContent_Declare(llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
    GIT_TAG        ${VLA_LLAMA_TAG}
    GIT_SHALLOW    TRUE
    ${_vla_llama_patch}
)
FetchContent_MakeAvailable(llama)

# Drop the whole fetched tree from the default build: llama.cpp's ~20 tool
# binaries and their impl libraries are dead weight here. What we link (llama,
# ggml, mtmd, llama-common) is still built, pulled in as a dependency. Walking
# the tree rather than naming targets keeps a bump from adding them back. Any of
# them builds on request: cmake --build build --target llama-mtmd-cli
function(vla_exclude_fetched_targets dir)
    get_property(subdirs DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
    foreach(sub IN LISTS subdirs)
        vla_exclude_fetched_targets(${sub})
    endforeach()
    get_property(targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
    foreach(tgt IN LISTS targets)
        set_target_properties(${tgt} PROPERTIES EXCLUDE_FROM_ALL TRUE)
    endforeach()
endfunction()
vla_exclude_fetched_targets(${llama_SOURCE_DIR})

add_library(vla_core
    src/model.cpp
    src/loader.cpp
    src/options.cpp
    src/modules/action_expert.cpp
    src/modules/dit_head.cpp
    src/modules/encoder.cpp
    src/modules/prompt.cpp
    src/modules/qwen3_lm.cpp
    src/modules/siglip_vit.cpp
    src/models/smolvla.cpp
    src/models/pi0.cpp
    src/models/pi05.cpp
    src/models/evo1.cpp
    src/models/gr00tn1d5.cpp
    src/models/gr00tn1d6.cpp
    src/models/gr00tn1d7.cpp
    src/models/bitvla.cpp
    src/models/vla_adapter.cpp
    src/models/openvla_oft.cpp
    src/models/vla_jepa.cpp
)
target_include_directories(vla_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${llama_SOURCE_DIR}/vendor
)
# The VLA archs call no llama_* API; only vlm_core needs llama.
target_link_libraries(vla_core PUBLIC ggml)

if(GGML_CUDA)
    target_compile_definitions(vla_core PUBLIC GGML_USE_CUDA)

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # SASS per supported GPU, PTX only for the newest arch so future cards JIT.
        set(_vla_cuda_archs 80-real 86-real 87-real 89-real 90-real)
        set(_vla_cuda_ptx 90-virtual)
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
            list(APPEND _vla_cuda_archs 100-real 120-real)
            set(_vla_cuda_ptx 120-virtual)
        else()
            message(WARNING
                "CUDA ${CUDAToolkit_VERSION} < 12.8: omitting Blackwell sm_100/sm_120. "
                "Pass -DCMAKE_CUDA_ARCHITECTURES=<arch> explicitly to override.")
        endif()
        set(CMAKE_CUDA_ARCHITECTURES ${_vla_cuda_archs} ${_vla_cuda_ptx}
            CACHE STRING "" FORCE)
    endif()
    add_library(bitvla_cuda_kernels STATIC
        src/kernels/bitvla/bitnet_kernels.cu
        src/kernels/bitvla/bitvla_lm_cuda.cu
        src/kernels/bitvla/bitvla_vit_cuda.cu
        src/kernels/bitvla/bitvla_fp32head_cuda.cu
    )
    set_target_properties(bitvla_cuda_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(bitvla_cuda_kernels PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_features(bitvla_cuda_kernels PRIVATE cxx_std_17)
    target_compile_options(bitvla_cuda_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xptxas=-O3>
    )
    target_link_libraries(bitvla_cuda_kernels PUBLIC CUDA::cublas CUDA::cudart)
    target_link_libraries(vla_core PRIVATE bitvla_cuda_kernels)
    target_compile_definitions(vla_core PUBLIC VLA_BITVLA_CUDA_KERNELS)
    target_include_directories(vla_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

    # ggml-cuda link: the hook resolves ggml_cuda_ext_forward out of it.
    add_library(vla_cuda_ops STATIC src/cuda/vla_cuda_bf16.cu)
    target_include_directories(vla_cuda_ops PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${llama_SOURCE_DIR}/ggml/include
    )
    set_target_properties(vla_cuda_ops PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_features(vla_cuda_ops PRIVATE cxx_std_17)
    target_compile_options(vla_cuda_ops PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xptxas=-O3>
    )
    target_link_libraries(vla_cuda_ops PUBLIC CUDA::cublas CUDA::cudart ggml-cuda)
    target_link_libraries(vla_core PRIVATE vla_cuda_ops)
endif()

if(GGML_SYCL AND NOT GGML_CUDA)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        message(FATAL_ERROR
            "GGML_SYCL=ON needs the oneAPI DPC++ compiler, but CMAKE_CXX_COMPILER is "
            "'${CMAKE_CXX_COMPILER_ID}'. Source /opt/intel/oneapi/setvars.sh and configure a "
            "fresh build dir with -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx. "
            "See docs/backend/sycl.md.")
    endif()
    target_compile_definitions(vla_core PUBLIC GGML_USE_SYCL)
endif()

# Precedence matches the ladder in src/backend.h.
if(GGML_METAL AND NOT GGML_CUDA AND NOT GGML_SYCL)
    target_compile_definitions(vla_core PUBLIC GGML_USE_METAL)
endif()

if(GGML_OPENVINO AND NOT GGML_CUDA AND NOT GGML_SYCL AND NOT GGML_METAL)
    # ggml's own OpenVINO backend target finds the toolkit; this only tells the
    # archs which branch of the backend.h ladder to compile. Fail early with a
    # pointer to setupvars.sh rather than deep inside the fetched tree.
    find_package(OpenVINO QUIET COMPONENTS Runtime)
    if(NOT OpenVINO_FOUND)
        message(FATAL_ERROR
            "GGML_OPENVINO=ON but the OpenVINO runtime was not found. Install it "
            "(scripts/install_ov.sh) and 'source /opt/intel/openvino/setupvars.sh' "
            "in the shell that configures. See docs/backend/ov.md.")
    endif()
    target_compile_definitions(vla_core PUBLIC GGML_USE_OPENVINO)
endif()

add_library(vlm_core
    src/vlm/engine.cpp
)
target_include_directories(vlm_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE
        ${llama_SOURCE_DIR}/common
        ${llama_SOURCE_DIR}/tools/mtmd
        ${llama_SOURCE_DIR}/vendor
)
target_link_libraries(vlm_core PUBLIC llama ggml mtmd llama-common)

find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZeroMQ REQUIRED IMPORTED_TARGET libzmq)

find_path(CPPZMQ_INCLUDE_DIR NAMES zmq.hpp)
if(NOT CPPZMQ_INCLUDE_DIR)
    message(FATAL_ERROR
        "cppzmq (zmq.hpp) not found. Install 'cppzmq-dev' (Debian/Ubuntu) or set "
        "CMAKE_PREFIX_PATH to your cppzmq install prefix.")
endif()

set(VLA_PROTO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/serving/vla.proto)
set(VLA_PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto-gen/serving)
file(MAKE_DIRECTORY ${VLA_PROTO_GEN_DIR})

add_custom_command(
    OUTPUT  ${VLA_PROTO_GEN_DIR}/vla.pb.h
            ${VLA_PROTO_GEN_DIR}/vla.pb.cc
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/src/serving
            --cpp_out=${VLA_PROTO_GEN_DIR}
            ${VLA_PROTO_SRC}
    DEPENDS ${VLA_PROTO_SRC}
    VERBATIM
)

add_executable(vla-server
    src/serving/server.cpp
    ${VLA_PROTO_GEN_DIR}/vla.pb.cc
)
target_include_directories(vla-server PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/proto-gen
    ${llama_SOURCE_DIR}/vendor/stb
    ${CPPZMQ_INCLUDE_DIR}
)
target_link_libraries(vla-server PRIVATE
    vla_core
    protobuf::libprotobuf
    PkgConfig::ZeroMQ
)

set(VLM_PROTO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/serving/vlm.proto)
set(VLM_PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto-gen/serving)
file(MAKE_DIRECTORY ${VLM_PROTO_GEN_DIR})

add_custom_command(
    OUTPUT  ${VLM_PROTO_GEN_DIR}/vlm.pb.h
            ${VLM_PROTO_GEN_DIR}/vlm.pb.cc
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/src/serving
            --cpp_out=${VLM_PROTO_GEN_DIR}
            ${VLM_PROTO_SRC}
    DEPENDS ${VLM_PROTO_SRC}
    VERBATIM
)

add_executable(vlm-server
    src/serving/vlm-server.cpp
    ${VLM_PROTO_GEN_DIR}/vlm.pb.cc
)
target_include_directories(vlm-server PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/proto-gen
    ${llama_SOURCE_DIR}/vendor/stb
    ${CPPZMQ_INCLUDE_DIR}
)
target_link_libraries(vlm-server PRIVATE
    vlm_core
    protobuf::libprotobuf
    PkgConfig::ZeroMQ
)

add_library(vla SHARED src/vla_c_api.cpp)
target_include_directories(vla
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(vla PRIVATE vla_core)
set_target_properties(vla PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/vla.h
)

add_executable(vla-cli
    src/serving/vla-cli.cpp
)
target_include_directories(vla-cli PRIVATE
    ${llama_SOURCE_DIR}/vendor/stb
)
target_link_libraries(vla-cli PRIVATE vla_core)
# --text shells out to the tokenizer script; VLA_TOKENIZE_SCRIPT overrides it.
target_compile_definitions(vla-cli PRIVATE VLA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(vla-bench
    src/serving/vla-bench.cpp
)
target_link_libraries(vla-bench PRIVATE vla_core)

# Warnings and LTO for our own targets only, never the llama.cpp subtree.
set(VLA_FIRST_PARTY_TARGETS vla_core vlm_core vla vla-server vlm-server vla-cli vla-bench)

foreach(tgt IN LISTS VLA_FIRST_PARTY_TARGETS)
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endforeach()

include(CheckIPOSupported)
check_ipo_supported(RESULT VLA_IPO_OK OUTPUT VLA_IPO_MSG)
if(VLA_IPO_OK AND CMAKE_BUILD_TYPE STREQUAL "Release")
    foreach(tgt IN LISTS VLA_FIRST_PARTY_TARGETS)
        set_property(TARGET ${tgt} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
    endforeach()
else()
    message(STATUS "vla: IPO/LTO disabled (${VLA_IPO_MSG})")
endif()

option(VLA_BUILD_TESTS "Build vla.cpp test harnesses and unit tests" OFF)
if(VLA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
