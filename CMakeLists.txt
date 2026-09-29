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

include(GNUInstallDirs)
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}" CACHE STRING "")
elseif(UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}" CACHE STRING "")
endif()

set(_vla_accel "")
foreach(_flag GGML_CUDA GGML_SYCL GGML_METAL GGML_OPENVINO GGML_HEXAGON GGML_OPENCL)
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
# backend.h calls ggml_backend_<accel>_init directly, and GGML_BACKEND_DL builds
# the accelerators as loadable modules that are not linked into ggml. Say so here
# rather than at the undefined symbol.
if(_vla_accel_n GREATER 0 AND GGML_BACKEND_DL)
    message(FATAL_ERROR
        "GGML_BACKEND_DL=ON is not supported with ${_vla_accel}: the backend is "
        "built as a module and src/backend.h links its init directly.")
endif()
foreach(_flag GGML_VULKAN GGML_HIP GGML_MUSA GGML_CANN GGML_WEBGPU)
    if(${_flag})
        message(WARNING "${_flag}=ON reaches vlm-server through llama only; src/backend.h has no ${_flag} path, so the VLA archs run on CPU.")
    endif()
endforeach()

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

    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
        # SASS per supported GPU, sm_90 PTX so future cards JIT.
        set(_vla_cuda_archs 80-real 86-real 87-real 89-real 90-real)
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
            list(APPEND _vla_cuda_archs 100-real 120-real)
        else()
            message(WARNING
                "CUDA ${CUDAToolkit_VERSION} < 12.8: omitting Blackwell sm_100/sm_120. "
                "Pass -DCMAKE_CUDA_ARCHITECTURES=<arch> explicitly to override.")
        endif()
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.9)
            list(APPEND _vla_cuda_archs 121-real)
        endif()
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
            list(APPEND _vla_cuda_archs 110-real)
        endif()
        set(CMAKE_CUDA_ARCHITECTURES ${_vla_cuda_archs} 90-virtual
            CACHE STRING "" FORCE)
    endif()
    set(GGML_CUDA_FA_QUANTS "f16-f16" CACHE STRING "")
elseif(GGML_OPENVINO)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(_vla_llama_patch PATCH_COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_ggml_openvino.py <SOURCE_DIR>)

    # Before the fetch: ggml's own ggml-openvino/CMakeLists.txt does
    # find_package(OpenVINO REQUIRED ...) and would get there first, so this
    # message only reaches anyone if it runs earlier. Same components it asks for.
    find_package(OpenVINO QUIET COMPONENTS Runtime Threading)
    if(NOT OpenVINO_FOUND)
        message(FATAL_ERROR
            "GGML_OPENVINO=ON but the OpenVINO runtime was not found. Install it "
            "(scripts/install_ov.sh) and 'source /opt/intel/openvino/setupvars.sh' "
            "in the shell that configures. See docs/backend/ov.md.")
    endif()
endif()
# Overridable so a regression can be bisected against another tag in a separate
# build dir (-DVLA_LLAMA_TAG=b10326) without editing this file. The patch
# anchors in scripts/patch_ggml_cuda_ext_hook.py and scripts/patch_ggml_openvino.py
# are checked against the default.
set(_vla_llama_tag_default "b11223")
set(VLA_LLAMA_TAG "${_vla_llama_tag_default}" CACHE STRING "llama.cpp tag to fetch")
# A cache entry survives an edit to the line above, so an existing build dir keeps
# the tag it was first configured with and quietly builds the wrong llama.cpp.
if(NOT VLA_LLAMA_TAG STREQUAL _vla_llama_tag_default)
    message(WARNING
        "VLA_LLAMA_TAG is ${VLA_LLAMA_TAG}, not the default ${_vla_llama_tag_default}. "
        "If that was not deliberate, this build dir predates the bump: reconfigure with "
        "-DVLA_LLAMA_TAG=${_vla_llama_tag_default} or use a fresh one.")
endif()

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

# SentencePiece, and through it a protobuf, is only needed to tokenize --text with
# a tokenizer stored in the GGUF. Optional so a build without it has no protobuf.
option(VLA_SPM "Link SentencePiece for in-GGUF tokenizers (needs system protobuf)" ON)
if(DEFINED VLA_OCTO)
    message(DEPRECATION "VLA_OCTO is deprecated, use VLA_SPM")
    set_property(CACHE VLA_SPM PROPERTY VALUE ${VLA_OCTO})
    unset(VLA_OCTO CACHE)
endif()
if(VLA_SPM)
    # The system protobuf, not sentencepiece's vendored protobuf-lite: vla-server
    # aborts at static-init if two protobuf runtimes reach the same binary.
    set(SPM_ENABLE_SHARED    OFF        CACHE BOOL   "" FORCE)
    set(SPM_BUILD_TEST       OFF        CACHE BOOL   "" FORCE)
    set(SPM_ENABLE_TCMALLOC  OFF        CACHE BOOL   "" FORCE)
    set(SPM_PROTOBUF_PROVIDER "package" CACHE STRING "" FORCE)
    if(NOT WIN32)
        find_package(Protobuf QUIET)
        if(Protobuf_VERSION VERSION_GREATER_EQUAL 4.22)
            set(SPM_ABSL_PROVIDER "package" CACHE STRING "" FORCE)
        endif()
    endif()
    set(_vla_spm_patch "")
    if(WIN32 AND NOT MSVC)
        set(_vla_spm_patch PATCH_COMMAND ${CMAKE_COMMAND} -P
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch_sentencepiece_fpic.cmake)
    endif()
    FetchContent_Declare(sentencepiece
        GIT_REPOSITORY https://github.com/google/sentencepiece
        GIT_TAG        v0.2.1
        GIT_SHALLOW    TRUE
        ${_vla_spm_patch}
    )
    FetchContent_MakeAvailable(sentencepiece)
    # Only the static library is linked; its CLI tools are dead weight.
    set_property(DIRECTORY ${sentencepiece_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL TRUE)
    # vcpkg's FindProtobuf shim reports the DLL, not its import library, as
    # PROTOBUF_LITE_LIBRARY, and sentencepiece links that path verbatim. Link the
    # imported target instead, which carries the import library and protobuf's
    # abseil deps. The full runtime, not lite: vla-server links it too, and one
    # protobuf runtime per binary is the rule above.
    # Imported targets are directory-scoped, so find it again up here.
    if(WIN32)
        find_package(Protobuf CONFIG QUIET)
    endif()
    # The link goes on vla_core below rather than on sentencepiece-static:
    # names in a target's link interface resolve in the directory that made the
    # target, and sentencepiece's directory has its own copies of abseil's
    # imported targets, which still carry the MSVC flag stripped further down.
    if(WIN32 AND TARGET protobuf::libprotobuf)
        set_property(TARGET sentencepiece-static PROPERTY
            INTERFACE_LINK_LIBRARIES Threads::Threads)
        set(VLA_SPM_PROTOBUF protobuf::libprotobuf)
    endif()
endif()

# llama.cpp turns BUILD_SHARED_LIBS on, so these follow it. That is harmless on
# ELF, which exports everything, but a Windows DLL exports nothing unannotated:
# the import library comes out empty and every consumer fails to link. Nothing
# outside this tree links the cores, so on Windows they are static.
if(WIN32)
    set(VLA_CORE_LIB_TYPE STATIC)
else()
    set(VLA_CORE_LIB_TYPE "")
endif()

add_library(vla_core ${VLA_CORE_LIB_TYPE}
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
    src/models/turbovla.cpp
    src/models/octo.cpp
    src/tokenizer.cpp
)
target_include_directories(vla_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${llama_SOURCE_DIR}/vendor
)
# The VLA archs call no llama_* API; only vlm_core needs llama.
target_link_libraries(vla_core PUBLIC ggml)
if(VLA_SPM)
    target_include_directories(vla_core PRIVATE ${sentencepiece_SOURCE_DIR}/src)
    target_link_libraries(vla_core PRIVATE sentencepiece-static ${VLA_SPM_PROTOBUF})
    target_compile_definitions(vla_core PRIVATE VLA_USE_SPM)
endif()

if(GGML_CUDA)
    target_compile_definitions(vla_core PUBLIC GGML_USE_CUDA)

    enable_language(CUDA)
    add_library(bitvla_cuda_kernels STATIC
        src/kernels/bitvla/bitnet_kernels.cu
        src/kernels/bitvla/bitvla_lm_cuda.cu
        src/kernels/bitvla/bitvla_vit_cuda.cu
        src/kernels/bitvla/bitvla_fp32head_cuda.cu
    )
    # Device-link each archive on its own. Without this, vla_core nvlinks the
    # relocatable device code of both CUDA libraries together, and CUDA 12.x
    # rejects the mix because only bitvla compiles with --use_fast_math:
    # "nvlink fatal: Cicc option values for '-ftz' do not match".
    set_target_properties(bitvla_cuda_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
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
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
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
    # Only tells the archs which branch of the backend.h ladder to compile; the
    # toolkit was located before the fetch above.
    target_compile_definitions(vla_core PUBLIC GGML_USE_OPENVINO)
endif()

# Hexagon and OpenCL reject some ops the archs build; the wrapper in
# backend_fallback.cpp runs those on the CPU. It reaches into ggml's internal
# headers for the backend vtable, hence the private include.
if(GGML_HEXAGON OR GGML_OPENCL)
    target_sources(vla_core PRIVATE src/backend_fallback.cpp)
    target_include_directories(vla_core SYSTEM PRIVATE ${llama_SOURCE_DIR}/ggml/src)
    if(GGML_HEXAGON)
        target_compile_definitions(vla_core PUBLIC GGML_USE_HEXAGON)
    else()
        target_compile_definitions(vla_core PUBLIC GGML_USE_OPENCL)
    endif()
endif()

add_library(vlm_core ${VLA_CORE_LIB_TYPE}
    src/vlm/engine.cpp
)
target_include_directories(vlm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(vlm_core SYSTEM PRIVATE
    ${llama_SOURCE_DIR}/common
    ${llama_SOURCE_DIR}/tools/mtmd
    ${llama_SOURCE_DIR}/vendor
)
target_link_libraries(vlm_core PUBLIC llama ggml mtmd llama-common)

# vla-cli and vla-bench link neither protobuf nor ZeroMQ; turning this off lets
# them configure on a machine that has not installed the serving stack.
option(VLA_BUILD_SERVER "Build vla-server and vlm-server (needs protobuf, libzmq, cppzmq)" ON)
if(VLA_BUILD_SERVER)
    find_package(Protobuf REQUIRED)
    if(WIN32)
        # vcpkg ships ZeroMQ's CMake package and no pkg-config.
        find_package(ZeroMQ CONFIG REQUIRED)
        set(VLA_ZMQ_TARGET libzmq)
    else()
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(ZeroMQ REQUIRED IMPORTED_TARGET libzmq)
        set(VLA_ZMQ_TARGET PkgConfig::ZeroMQ)
    endif()

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
        ${VLA_ZMQ_TARGET}
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
        ${VLA_ZMQ_TARGET}
    )
endif()

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
# --text shells out to the tokenizer script when the GGUF has no tokenizer.
target_compile_definitions(vla-cli PRIVATE VLA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(vla-bench
    src/serving/vla-bench.cpp
)
target_link_libraries(vla-bench PRIVATE vla_core)

install(TARGETS vla COMPONENT libvla)
install(FILES LICENSE.md ${llama_SOURCE_DIR}/licenses/LICENSE-jsonhpp
    DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT libvla)
install(FILES ${llama_SOURCE_DIR}/LICENSE DESTINATION ${CMAKE_INSTALL_DOCDIR}
    RENAME LICENSE.llama.cpp COMPONENT libvla)
install(TARGETS vla_core vla-cli vla-bench)
install(FILES scripts/tokenize_prompt.py DESTINATION ${CMAKE_INSTALL_DATADIR}/vla)
if(VLA_BUILD_SERVER)
    install(TARGETS vlm_core vla-server vlm-server)
endif()

# Warnings and LTO for our own targets only, never the llama.cpp subtree.
set(VLA_FIRST_PARTY_TARGETS vla_core vlm_core vla vla-cli vla-bench)
if(VLA_BUILD_SERVER)
    list(APPEND VLA_FIRST_PARTY_TARGETS vla-server vlm-server)
endif()
# The CUDA targets were the only first-party code compiled without warnings.
if(GGML_CUDA)
    list(APPEND VLA_FIRST_PARTY_TARGETS bitvla_cuda_kernels vla_cuda_ops)
endif()

option(VLA_WERROR "Treat warnings in first-party code as errors" OFF)
foreach(tgt IN LISTS VLA_FIRST_PARTY_TARGETS)
    target_compile_options(${tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra $<$<BOOL:${VLA_WERROR}>:-Werror>>)
endforeach()

# llama.cpp sends its DLLs to <build>/bin, and Windows has no rpath: a binary
# finds a DLL beside itself or on PATH. Put ours next to them.
if(WIN32)
    foreach(tgt IN LISTS VLA_FIRST_PARTY_TARGETS)
        set_target_properties(${tgt} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    endforeach()
    # Same problem for the servers' protobuf / abseil / libzmq DLLs, which live
    # in the package manager's tree.
    if(VLA_BUILD_SERVER)
        foreach(tgt vla-server vlm-server)
            add_custom_command(TARGET ${tgt} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        $<TARGET_RUNTIME_DLLS:${tgt}> $<TARGET_FILE_DIR:${tgt}>
                COMMAND_EXPAND_LISTS)
        endforeach()
    endif()
endif()

# The NPU-side HTP skels are built (and, with HEXAGON_HTP_CERT, signed) inside
# the fetched tree. FastRPC finds them through ADSP_LIBRARY_PATH, which the
# Hexagon rung defaults to the binary's own directory, so copy them there.
if(GGML_HEXAGON AND TARGET ggml-hexagon)
    set(_vla_htp_dir ${CMAKE_BINARY_DIR}/_deps/llama-build/ggml/src/ggml-hexagon)
    add_custom_target(vla-htp-skels ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
        COMMAND ${CMAKE_COMMAND} -DSRC=${_vla_htp_dir} -DDST=${CMAKE_BINARY_DIR}/bin
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_htp_skels.cmake
        DEPENDS ggml-hexagon
        COMMENT "copying HTP skels next to the binaries"
        VERBATIM)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT VLA_IPO_OK OUTPUT VLA_IPO_MSG)
if(VLA_IPO_OK AND CMAKE_BUILD_TYPE STREQUAL "Release")
    foreach(tgt IN LISTS VLA_FIRST_PARTY_TARGETS)
        set_property(TARGET ${tgt} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
    endforeach()
else()
    message(STATUS "vla: IPO/LTO disabled (${VLA_IPO_MSG})")
endif()

# vcpkg's abseil hands every consumer "-ignore:4221", an MSVC link.exe flag,
# through INTERFACE_LINK_LIBRARIES. clang++ driving lld-link rejects it as an
# unknown argument. The warning it silences is harmless, so drop the flag.
if(WIN32 AND NOT MSVC)
    get_directory_property(_vla_imported IMPORTED_TARGETS)
    foreach(tgt IN LISTS _vla_imported)
        get_target_property(_libs ${tgt} INTERFACE_LINK_LIBRARIES)
        if(_libs AND "-ignore:4221" IN_LIST _libs)
            list(REMOVE_ITEM _libs "-ignore:4221")
            set_property(TARGET ${tgt} PROPERTY INTERFACE_LINK_LIBRARIES "${_libs}")
        endif()
    endforeach()
endif()

option(VLA_BUILD_TESTS "Build vla.cpp test harnesses and unit tests" OFF)
if(VLA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
