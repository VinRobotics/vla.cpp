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

add_executable(test_config_guard test_config_guard.cpp)
target_include_directories(test_config_guard PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(test_config_guard PRIVATE -Wall -Wextra)
add_test(NAME config_guard COMMAND test_config_guard)
