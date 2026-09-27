# Build vla.cpp natively on a Snapdragon X laptop (Windows on Arm).
#
#   .\scripts\build_windows_snapdragon.ps1 -Backend htp     # Hexagon NPU (+ CPU fallback)
#   .\scripts\build_windows_snapdragon.ps1 -Backend opencl  # Adreno GPU (+ CPU fallback)
#   .\scripts\build_windows_snapdragon.ps1 -Backend cpu     # Oryon CPU only
#
# Binaries land in build-wos-<backend>\bin. See docs/backend/hexagon-windows.md
# for the one-off setup this assumes: Visual Studio with its bundled Clang,
# CMake and Ninja; the Hexagon and OpenCL SDKs (HEXAGON_SDK_ROOT,
# HEXAGON_TOOLS_ROOT, OPENCL_SDK_ROOT); vcpkg for the server's dependencies; and,
# for the NPU, a code-signing certificate for the HTP skels.
param(
    [ValidateSet("htp", "opencl", "cpu")][string]$Backend = "htp",
    # A llama.cpp checkout to build against. Empty fetches the tag CMakeLists.txt pins.
    [string]$LlamaDir = "",
    [string]$VsRoot = "C:\Program Files\Microsoft Visual Studio\18\Insiders",
    [string]$VcpkgRoot = $(if ($env:VCPKG_ROOT) { $env:VCPKG_ROOT } else { "C:\vcpkg" }),
    # Signs the HTP skels. Without it the NPU refuses to load them.
    [string]$HtpCert = $env:HEXAGON_HTP_CERT,
    [string]$WindowsSdkBin = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0",
    [switch]$NoServer,
    [switch]$Reconfigure
)
# Not "Stop": Windows PowerShell turns any stderr line from a native tool (a CMake
# warning) into a terminating error. Exit codes are checked after each step.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

& "$VsRoot\Common7\Tools\Launch-VsDevShell.ps1" -Arch arm64 -HostArch arm64 -SkipAutomaticLocation | Out-Null
$env:PATH = "$VsRoot\VC\Tools\Llvm\ARM64\bin;$env:PATH"
Set-Location $root

$dir = "build-wos-$Backend"
$flags = "-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -D_GNU_SOURCE"
# protobuf and abseil must be built by clang too, with the triplet in
# cmake/vcpkg-triplets: clang code does not link against an MSVC-built protobuf.
# ZeroMQ is a C API and comes from the stock triplet, searched second.
$triplet  = "$VcpkgRoot\installed\arm64-windows-clangcl"
$tripletC = "$VcpkgRoot\installed\arm64-windows"

# llama.cpp's toolchain file (vendored: a fetch has it only after configure)
# drives clang against the MSVC ABI.
$toolchain = "$root\cmake\arm64-windows-llvm.cmake"

$args_ = @(
    "-S", ".", "-B", $dir, "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_TOOLCHAIN_FILE=$($toolchain -replace '\\','/')",
    "-DCMAKE_C_FLAGS=$flags", "-DCMAKE_CXX_FLAGS=$flags",
    "-DGGML_OPENMP=OFF", "-DGGML_LLAMAFILE=OFF", "-DLLAMA_OPENSSL=OFF",
    "-DVLA_BUILD_TESTS=ON"
)
if ($LlamaDir) { $args_ += "-DFETCHCONTENT_SOURCE_DIR_LLAMA=$($LlamaDir -replace '\\','/')" }
if ($NoServer) {
    # Octo tokenizes in-process through SentencePiece, which needs protobuf too.
    $args_ += @("-DVLA_BUILD_SERVER=OFF", "-DVLA_OCTO=OFF")
} else {
    $args_ += @(
        "-DCMAKE_PREFIX_PATH=$($triplet -replace '\\','/');$($tripletC -replace '\\','/');$($env:OPENCL_SDK_ROOT -replace '\\','/')",
        # vcpkg's protobuf is found through its CMake package, which carries the
        # abseil dependencies the old FindProtobuf module does not.
        "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON", "-Dprotobuf_MODULE_COMPATIBLE=ON",
        "-DProtobuf_PROTOC_EXECUTABLE=$($triplet -replace '\\','/')/tools/protobuf/protoc.exe"
    )
}
switch ($Backend) {
    "htp" {
        if (-not $HtpCert) { Write-Warning "no -HtpCert / HEXAGON_HTP_CERT: the HTP skels will be unsigned and the NPU will not load them" }
        $env:HEXAGON_HTP_CERT = $HtpCert
        $env:WINDOWS_SDK_BIN  = $WindowsSdkBin
        $args_ += @("-DGGML_HEXAGON=ON", "-DGGML_OPENCL=OFF", "-DPREBUILT_LIB_DIR=windows_aarch64",
                    "-DHEXAGON_SDK_ROOT=$env:HEXAGON_SDK_ROOT", "-DHEXAGON_TOOLS_ROOT=$env:HEXAGON_TOOLS_ROOT")
    }
    "opencl" { $args_ += @("-DGGML_OPENCL=ON", "-DGGML_HEXAGON=OFF") }
    "cpu"    { $args_ += @("-DGGML_OPENCL=OFF", "-DGGML_HEXAGON=OFF") }
}

if ($Reconfigure -or -not (Test-Path "$dir\CMakeCache.txt")) {
    cmake @args_
    if ($LASTEXITCODE -ne 0) { throw "configure failed" }
}
if ($Backend -eq "htp") {
    # ggml-hexagon signs the skel catalog in a step that depends on the .so
    # files but not on the targets that build them, so a parallel build can
    # reach inf2cat first. Build the skels on their own before everything else.
    cmake --build $dir -j 8 --target htp-v73 htp-v75 htp-v79 htp-v81
    if ($LASTEXITCODE -ne 0) { throw "HTP skel build failed" }
}
cmake --build $dir -j 8
if ($LASTEXITCODE -ne 0) { throw "build failed" }
Write-Host "built $dir\bin"
