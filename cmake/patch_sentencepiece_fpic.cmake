# sentencepiece v0.2.1 adds -fPIC for every compiler that is not MSVC, and
# clang targeting arm64-pc-windows-msvc rejects the flag as a hard error.
# Windows has no PIC to ask for, so drop it. Run as the FetchContent patch step
# with the working directory at the sentencepiece source root.
set(_f "src/CMakeLists.txt")
file(READ "${_f}" _s)
string(REPLACE " -fPIC" "" _s "${_s}")
file(WRITE "${_f}" "${_s}")
