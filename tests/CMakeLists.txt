# vla.cpp tests, enabled with -DVLA_BUILD_TESTS=ON. predict_check needs a real
# GGUF so it is built but not run by ctest; the unit tests below are pure.

add_executable(vla_predict_check predict_check.cpp)
target_link_libraries(vla_predict_check PRIVATE vla_core)
target_compile_options(vla_predict_check PRIVATE -Wall -Wextra)

add_executable(test_vision_common test_vision_common.cpp)
target_include_directories(test_vision_common PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(test_vision_common PRIVATE -Wall -Wextra)
add_test(NAME vision_common COMMAND test_vision_common)

add_executable(test_rope_conventions test_rope_conventions.cpp)
target_include_directories(test_rope_conventions PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(test_rope_conventions PRIVATE -Wall -Wextra)
add_test(NAME rope_conventions COMMAND test_rope_conventions)

# Built as C so the public header stays C-clean.
add_executable(test_c_api test_c_api.c)
target_link_libraries(test_c_api PRIVATE vla)
target_compile_options(test_c_api PRIVATE -Wall -Wextra)
add_test(NAME c_api COMMAND test_c_api)

add_executable(test_dit_common test_dit_common.cpp)
target_include_directories(test_dit_common PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_dit_common PRIVATE ggml)
target_compile_options(test_dit_common PRIVATE -Wall -Wextra)
add_test(NAME dit_common COMMAND test_dit_common)

# Defines GGML_USE_OPENVINO itself so the pass is exercised on any build.
add_executable(test_graph_names test_graph_names.cpp)
target_include_directories(test_graph_names PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_graph_names PRIVATE ggml)
target_compile_options(test_graph_names PRIVATE -Wall -Wextra)
add_test(NAME graph_names COMMAND test_graph_names)

add_executable(test_qwen3vl_vit test_qwen3vl_vit.cpp)
target_include_directories(test_qwen3vl_vit PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_qwen3vl_vit PRIVATE ggml)
target_compile_options(test_qwen3vl_vit PRIVATE -Wall -Wextra)
add_test(NAME qwen3vl_vit COMMAND test_qwen3vl_vit)

# Links vla_core: it calls the real config_is_sane rather than a copy.
add_executable(test_config_guard test_config_guard.cpp)
target_include_directories(test_config_guard PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_config_guard PRIVATE vla_core)
target_compile_options(test_config_guard PRIVATE -Wall -Wextra)
add_test(NAME config_guard COMMAND test_config_guard)

# FoldQuant CPU reference ops as GGML_OP_CUSTOM nodes on the CPU backend. The
# golden checksum is shared with tests/py/test_foldquant_ref.py.
add_executable(test_foldquant_cpu_op test_foldquant_cpu_op.cpp)
target_include_directories(test_foldquant_cpu_op PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_foldquant_cpu_op PRIVATE vla_core)
target_compile_options(test_foldquant_cpu_op PRIVATE -Wall -Wextra -ffp-contract=off)
add_test(NAME foldquant_cpu_op COMMAND test_foldquant_cpu_op)

# A/B harness for the two BitVLA ternary-GEMM tilings. Built so it cannot rot,
# not registered with ctest: it needs a GPU and is read by hand.
# VLA_BITVLA_NARROW_GEMM=1 selects the old one-tile-per-CTA kernel at runtime.
if(GGML_CUDA)
    add_executable(bitvla_gemm_check bitvla_gemm_check.cu)
    target_include_directories(bitvla_gemm_check PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(bitvla_gemm_check PRIVATE bitvla_cuda_kernels)
endif()

# FoldQuant kernel micro-benchmark at GR00T shapes against cuBLAS BF16. Built
# so it cannot rot, not registered with ctest: it needs a GPU and is read by hand.
if(GGML_CUDA)
    add_executable(foldquant_gemm_check foldquant_gemm_check.cu)
    target_include_directories(foldquant_gemm_check PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(foldquant_gemm_check PRIVATE vla_fq_kernels CUDA::cublas)
endif()

# FoldQuant CUDA kernels versus the CPU reference, byte-identical; run twice,
# the second time through the VLA_FQ_CPU_REF=1 staging shim. Skips (exit 0)
# without a CUDA device.
if(GGML_CUDA)
    add_executable(test_foldquant_cuda_op test_foldquant_cuda_op.cpp)
    target_include_directories(test_foldquant_cuda_op PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_foldquant_cuda_op PRIVATE vla_core vla_cuda_ops ggml)
    target_compile_options(test_foldquant_cuda_op PRIVATE -Wall -Wextra -ffp-contract=off)
    add_test(NAME foldquant_cuda_op COMMAND test_foldquant_cuda_op)
    add_test(NAME foldquant_cuda_op_ref COMMAND test_foldquant_cuda_op)
    set_tests_properties(foldquant_cuda_op_ref PROPERTIES ENVIRONMENT "VLA_FQ_CPU_REF=1;GGML_CUDA_DISABLE_GRAPHS=1")
endif()

# Regression test for the in-tree BF16 CUDA kernels and the ggml hook they ride
# on. Skips itself (exit 0) when no CUDA device is present.
if(GGML_CUDA)
    add_executable(test_bf16_cuda_ops test_bf16_cuda_ops.cpp)
    target_include_directories(test_bf16_cuda_ops PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_bf16_cuda_ops PRIVATE vla_core vla_cuda_ops ggml)
    target_compile_options(test_bf16_cuda_ops PRIVATE -Wall -Wextra)
    add_test(NAME bf16_cuda_ops COMMAND test_bf16_cuda_ops)
endif()
