# Copyright 2026 VinRobotics - Apache-2.0
#
# Standalone build for the vla-ci LAN control-plane tools. Deliberately decoupled
# from the main project (no llama.cpp / CUDA / vla_core) so it builds in seconds
# on every server and on the orchestrator. Only deps: protobuf + ZeroMQ (cppzmq
# header), the same stack vla-server already requires.
#
#   cmake -S ci/agent -B ci/agent/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build ci/agent/build -j
# Produces: ci/agent/build/vla-ci-agent   (run on each platform server)
#           ci/agent/build/vla-ci-ctl     (run on the orchestrator)
cmake_minimum_required(VERSION 3.22)
project(vla_ci_agent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZeroMQ REQUIRED IMPORTED_TARGET libzmq)

set(PROTO ${CMAKE_CURRENT_SOURCE_DIR}/vla_ctl.proto)
set(GEN ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${GEN})
add_custom_command(
    OUTPUT  ${GEN}/vla_ctl.pb.cc ${GEN}/vla_ctl.pb.h
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR} --cpp_out=${GEN} ${PROTO}
    DEPENDS ${PROTO}
    VERBATIM
)

foreach(tool vla-ci-agent vla-ci-ctl)
    string(REPLACE "vla-ci-" "" src ${tool})
    add_executable(${tool} ${src}.cpp ${GEN}/vla_ctl.pb.cc)
    target_include_directories(${tool} PRIVATE ${GEN} ${Protobuf_INCLUDE_DIRS})
    target_link_libraries(${tool} PRIVATE protobuf::libprotobuf PkgConfig::ZeroMQ)
endforeach()
