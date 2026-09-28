# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    # Empty picks the newest Visual Studio (prereleases included) through vswhere.
    [string]$VsRoot = "",
    # Empty picks VCPKG_ROOT or C:\vcpkg, whichever holds the clang-cl triplet.
    # Visual Studio's dev shell points VCPKG_ROOT at its own bundled vcpkg.
    [string]$VcpkgRoot = "",
    # Signs the HTP skels. Without it the NPU refuses to load them.
    [string]$HtpCert = $env:HEXAGON_HTP_CERT,
    # Empty picks the newest SDK under Windows Kits\10\bin (inf2cat, signtool).
    [string]$WindowsSdkBin = "",
    [switch]$NoServer,
    [switch]$Reconfigure
)
# Not "Stop": Windows PowerShell turns any stderr line from a native tool (a CMake
# warning) into a terminating error. Exit codes are checked after each step.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot

if (-not $VsRoot) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $VsRoot = & $vswhere -latest -prerelease -products * -property installationPath
    }
    if (-not $VsRoot) { throw "Visual Studio not found; pass -VsRoot" }
}
if (-not $VcpkgRoot) {
    $VcpkgRoot = @($env:VCPKG_ROOT, "C:\vcpkg") | Where-Object {
        $_ -and (Test-Path "$_\installed\arm64-windows-clangcl")
    } | Select-Object -First 1
    if (-not $VcpkgRoot -and -not $NoServer) {
        throw "vcpkg with the arm64-windows-clangcl triplet not found; pass -VcpkgRoot"
    }
}
if (-not $WindowsSdkBin) {
    $WindowsSdkBin = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin" -Directory -Filter "10.*" |
        Sort-Object { [version] $_.Name } | Select-Object -Last 1 -ExpandProperty FullName
}

& "$VsRoot\Common7\Tools\Launch-VsDevShell.ps1" -Arch arm64 -HostArch arm64 -SkipAutomaticLocation | Out-Null
$env:PATH = "$VsRoot\VC\Tools\Llvm\ARM64\bin;$env:PATH"
Set-Location $root

$dir = "build-wos-$Backend"
$jobs = $env:NUMBER_OF_PROCESSORS
$flags = "-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -D_GNU_SOURCE"
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
    cmake --build $dir -j $jobs --target htp-v73 htp-v75 htp-v79 htp-v81
    if ($LASTEXITCODE -ne 0) { throw "HTP skel build failed" }
}
cmake --build $dir -j $jobs
if ($LASTEXITCODE -ne 0) { throw "build failed" }
Write-Host "built $dir\bin"
