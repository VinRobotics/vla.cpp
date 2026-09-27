# vcpkg triplet: arm64 Windows, DLLs, built with Visual Studio's clang-cl.
#
# vla.cpp builds with clang, and protobuf's headers switch on __clang__. The one
# seen to matter: PROTOBUF_RESTRICT is __restrict under clang and empty under
# MSVC, the two mangle it differently, and clang-compiled generated code then
# fails to link against an MSVC-built protobuf (RepeatedPtrFieldBase::
# InternalSwap). Building protobuf with the same compiler removes the mismatch
# without redefining keywords.
#
# Only protobuf (and abseil, which it pulls in) need it. ZeroMQ is a C API, so
# the stock MSVC triplet serves.
#
#   vcpkg install protobuf --triplet arm64-windows-clangcl ^
#       --overlay-triplets=<vla.cpp>\cmake\vcpkg-triplets
#   vcpkg install zeromq cppzmq --triplet arm64-windows
set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/clangcl-toolchain.cmake")
# A chainloaded toolchain makes vcpkg skip vcvars by default, and the link step
# still needs rc, mt and the SDK library paths from it.
set(VCPKG_LOAD_VCVARS_ENV ON)
