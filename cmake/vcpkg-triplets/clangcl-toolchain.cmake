# Chainloaded by arm64-windows-clangcl.cmake: vcpkg's own Windows toolchain, with
# Visual Studio's bundled clang-cl as the compiler. clang-cl takes the MSVC flags
# that toolchain sets.
#
# vcpkg scrubs PATH and VCPKG_ROOT, so clang-cl is looked up in the standard
# Visual Studio install locations. vcpkg's toolchains directory is found from
# CMAKE_TOOLCHAIN_FILE (scripts/buildsystems/vcpkg.cmake) and handed on to
# try_compile, which sees neither of those.
if(NOT _VLA_VCPKG_TOOLCHAINS)
    get_filename_component(_VLA_VCPKG_TOOLCHAINS "${CMAKE_TOOLCHAIN_FILE}" DIRECTORY)
    set(_VLA_VCPKG_TOOLCHAINS "${_VLA_VCPKG_TOOLCHAINS}/../toolchains")
endif()
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES _VLA_VCPKG_TOOLCHAINS)

file(GLOB _vla_llvm_dirs
    "$ENV{ProgramFiles}/Microsoft Visual Studio/*/*/VC/Tools/Llvm/ARM64/bin"
    "C:/Program Files/Microsoft Visual Studio/*/*/VC/Tools/Llvm/ARM64/bin")
find_program(_vla_clang_cl clang-cl REQUIRED
    HINTS "$ENV{VCINSTALLDIR}/Tools/Llvm/ARM64/bin" ${_vla_llvm_dirs})
set(CMAKE_C_COMPILER   "${_vla_clang_cl}")
set(CMAKE_CXX_COMPILER "${_vla_clang_cl}")
include("${_VLA_VCPKG_TOOLCHAINS}/windows.cmake")
# vcpkg sets CMAKE_RC_FLAGS to "/c65001 /DWIN32" unconditionally, and CMake's
# Ninja generator runs the C compiler over .rc files for dependencies: clang-cl
# takes /c65001 for a file name. rc.exe defaults to the right code page anyway.
set(CMAKE_RC_FLAGS "/DWIN32" CACHE STRING "" FORCE)
