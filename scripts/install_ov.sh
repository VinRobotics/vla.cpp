#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/.openvino_install_work"

OS_ID=""
OS_VERSION=""

log() {
  printf '[install_openvino_runtime] %s\n' "$*"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Error: '$1' is required but not installed." >&2
    exit 1
  }
}

# Digests of the two archives the defaults below pin. These land in /opt under
# sudo, so a bad download is a root-level problem.
OPENVINO_SHA256_2204="d327ede0a5dd29ad6e73d156aa7fe43b7d8fc0eac5e789f17bf9d381ab333e7d"
OPENVINO_SHA256_2404="0bd86d578beb1e8805655f593315c69bf909008e212f801ed76e3a350927736f"

# verify_sha256 <file> <expected-or-empty> <url>. An overridden version has no
# digest here, so fall back to the one the mirror publishes: that catches a
# truncated or corrupted download, not a compromised mirror.
verify_sha256() {
  local path="$1" want="$2" url="$3"
  if [[ -z "${want}" ]]; then
    want="$(curl -fsSL "${url}.sha256" | awk 'NR==1 {print $1}')"
    if [[ ! "${want}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "Error: no usable checksum published for ${url}" >&2
      exit 1
    fi
    log "Version overridden, using the mirror's own checksum."
  fi
  if ! printf '%s  %s\n' "${want}" "${path}" | sha256sum -c - >/dev/null; then
    echo "Error: checksum mismatch for ${path}" >&2
    exit 1
  fi
  log "Checksum OK: $(basename "${path}")"
}

detect_os() {
  if [[ ! -f /etc/os-release ]]; then
    echo "Error: /etc/os-release not found; cannot detect Ubuntu version." >&2
    exit 1
  fi

  # shellcheck disable=SC1091
  source /etc/os-release

  OS_ID="${ID:-}"
  OS_VERSION="${VERSION_ID:-}"

  if [[ "${OS_ID}" != "ubuntu" ]]; then
    echo "Error: this installer supports Ubuntu only. Detected ID='${OS_ID:-unknown}'." >&2
    exit 1
  fi

  # Every package name below is amd64. Say so now, not after the first 404.
  local arch
  arch="$(uname -m)"
  if [[ "${arch}" != "x86_64" ]]; then
    echo "Error: Intel publishes these packages for x86_64 only. Detected '${arch}'." >&2
    exit 1
  fi
}

prepare_common_tools() {
  need_cmd bash
  need_cmd sudo
  need_cmd apt-get
  need_cmd wget
  need_cmd curl
  need_cmd tar
  need_cmd dpkg
  need_cmd sha256sum
  need_cmd find
  need_cmd sort

  mkdir -p "${WORK_DIR}"
}

prepare_common_dependencies() {
  log "Installing common dependencies..."
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    libcurl4-openssl-dev \
    libtbb12 \
    cmake \
    ninja-build \
    python3-pip \
    curl \
    wget \
    tar \
    libopencl1 \
    ocl-icd-opencl-dev \
    opencl-headers \
    opencl-clhpp-headers \
    clinfo
}

add_render_group() {
  # USER is unbound, not empty, in a container or a cron shell, and set -u would
  # abort here with the drivers already installed.
  local target_user="${SUDO_USER:-${USER:-}}"
  if [[ -z "${target_user}" ]]; then
    return
  fi

  if id -nG "${target_user}" | grep -qw render; then
    log "User ${target_user} is already in the render group."
  else
    log "Adding ${target_user} to render group..."
    sudo gpasswd -a "${target_user}" render || true
    log "Re-login (or restart shell session) to apply render group membership."
  fi
}

install_gpu_2204() {
  local download_dir="${WORK_DIR}/intel_gpu_2204"
  local igc_base_url="https://github.com/intel/intel-graphics-compiler/releases/download/v2.10.8"
  local crt_base_url="https://github.com/intel/compute-runtime/releases/download/25.13.33276.16"
  local checksum_file="ww13.sum"
  local igc_sums=(
    "85bb185f5c9f0700321c6ce9362a4aa85b6cdf02f74a7b8685424d0be024fb0e  intel-igc-core-2_2.10.8+18926_amd64.deb"
    "21e5ee9f0b5798335d82a88f98e80bccd9539380782e60893d6d050eb801d322  intel-igc-opencl-2_2.10.8+18926_amd64.deb"
  )
  local packages=(
    "${igc_base_url}/intel-igc-core-2_2.10.8+18926_amd64.deb"
    "${igc_base_url}/intel-igc-opencl-2_2.10.8+18926_amd64.deb"
    "${crt_base_url}/intel-level-zero-gpu-dbgsym_1.6.33276.16_amd64.ddeb"
    "${crt_base_url}/intel-level-zero-gpu_1.6.33276.16_amd64.deb"
    "${crt_base_url}/intel-opencl-icd-dbgsym_25.13.33276.16_amd64.ddeb"
    "${crt_base_url}/intel-opencl-icd_25.13.33276.16_amd64.deb"
    "${crt_base_url}/libigdgmm12_22.7.0_amd64.deb"
  )

  log "Installing Intel GPU drivers for Ubuntu 22.04..."
  mkdir -p "${download_dir}"
  cd "${download_dir}"

  for url in "${packages[@]}"; do
    wget --no-continue "${url}"
  done
  wget --no-continue "${crt_base_url}/${checksum_file}"

  sha256sum -c "${checksum_file}"
  printf '%s\n' "${igc_sums[@]}" | sha256sum -c -

  shopt -s nullglob
  local artifacts=( *.deb *.ddeb )
  shopt -u nullglob
  if [[ ${#artifacts[@]} -eq 0 ]]; then
    echo "Error: no GPU package files found for Ubuntu 22.04." >&2
    exit 1
  fi

  sudo dpkg -i "${artifacts[@]}" || sudo apt-get install -f -y
  cd "${SCRIPT_DIR}"
  rm -rf "${download_dir}"
}

install_npu_2204() {
  local download_dir="${WORK_DIR}/intel_npu_2204"
  local npu_tarball="linux-npu-driver-v1.26.0.20251125-19665715237-ubuntu2204.tar.gz"
  local npu_url="https://github.com/intel/linux-npu-driver/releases/download/v1.26.0/${npu_tarball}"
  local level_zero_deb="level-zero_1.24.2+u22.04_amd64.deb"
  local level_zero_url="https://github.com/oneapi-src/level-zero/releases/download/v1.24.2/${level_zero_deb}"
  local npu_sha256="cfdbcc9adc1ea20d498ebd9cbdb5c212f6fc940e1034ef7a72e239a8636f653a"
  local level_zero_sha256="7c304e93835d96025c90f6a3d8f2ce5edf142c24da9f4871113a4f0225fef22e"

  log "Installing Intel NPU drivers for Ubuntu 22.04..."
  mkdir -p "${download_dir}"
  cd "${download_dir}"

  # Download before purging. The other order leaves a machine with no NPU driver
  # at all if the fetch fails.
  wget --no-continue "${npu_url}"
  wget --no-continue "${level_zero_url}"
  verify_sha256 "${npu_tarball}" "${npu_sha256}" "${npu_url}"
  verify_sha256 "${level_zero_deb}" "${level_zero_sha256}" "${level_zero_url}"
  tar -xf "${npu_tarball}"

  mapfile -t npu_debs < <(find . -type f -name '*.deb' ! -name 'level-zero*.deb' | sort)
  if [[ ${#npu_debs[@]} -eq 0 ]]; then
    echo "Error: no Intel NPU .deb packages found for Ubuntu 22.04." >&2
    exit 1
  fi

  sudo dpkg --purge --force-remove-reinstreq \
    intel-driver-compiler-npu \
    intel-fw-npu \
    intel-level-zero-npu \
    intel-level-zero-npu-dbgsym || true

  sudo dpkg -i "${npu_debs[@]}" || sudo apt-get install -f -y
  sudo dpkg -i "${level_zero_deb}" || sudo apt-get install -f -y

  add_render_group

  cd "${SCRIPT_DIR}"
  rm -rf "${download_dir}"
}

install_runtime_2204() {
  local download_dir="${WORK_DIR}/openvino_runtime_2204"
  local openvino_version="${OPENVINO_VERSION:-2026.4}"
  local openvino_build="${OPENVINO_BUILD:-22959.99c81491cc3}"
  local openvino_archive="openvino_toolkit_ubuntu22_${openvino_version}.0.${openvino_build}_x86_64.tgz"
  local openvino_dirname="openvino_toolkit_ubuntu22_${openvino_version}.0.${openvino_build}_x86_64"
  local openvino_url="https://storage.openvinotoolkit.org/repositories/openvino/packages/${openvino_version}/linux/${openvino_archive}"
  local install_root="${INSTALL_ROOT:-/opt/intel}"
  local install_dir="${install_root}/openvino_${openvino_version}"
  local symlink_path="${install_root}/openvino"
  local archive_path="${download_dir}/openvino_${openvino_version}.tgz"
  local expected_sha=""
  if [[ "${openvino_version}" == "2026.4" && "${openvino_build}" == "22959.99c81491cc3" ]]; then
    expected_sha="${OPENVINO_SHA256_2204}"
  fi

  log "Installing OpenVINO runtime for Ubuntu 22.04..."
  sudo mkdir -p "${install_root}"
  mkdir -p "${download_dir}"

  curl -fL "${openvino_url}" --output "${archive_path}"
  verify_sha256 "${archive_path}" "${expected_sha}" "${openvino_url}"
  rm -rf "${download_dir:?}/${openvino_dirname}"
  tar -xf "${archive_path}" -C "${download_dir}"

  sudo rm -rf "${install_dir}"
  sudo mv "${download_dir}/${openvino_dirname}" "${install_dir}"
  sudo ln -sfn "openvino_${openvino_version}" "${symlink_path}"

  if [[ ! -f "${symlink_path}/setupvars.sh" ]]; then
    echo "Error: ${symlink_path}/setupvars.sh was not found after installation." >&2
    exit 1
  fi

  rm -rf "${download_dir}"
}

install_gpu_2404() {
  local download_dir="${WORK_DIR}/intel_gpu_2404"
  local igc_base_url="https://github.com/intel/intel-graphics-compiler/releases/download/v2.41.5"
  local crt_base_url="https://github.com/intel/compute-runtime/releases/download/26.35.39758.10"
  local checksum_file="ww35.sum"
  local igc_sums=(
    "0a6e64a663ae65a0fa02d6912ae3b6b37cf85b90c21cc423fd9fef70aaf4f628  intel-igc-core-2_2.41.5+22716_amd64.deb"
    "779e1b9e88098eb25711e9a8f67c2752665bad22f134aa40ed5649f6e1b87058  intel-igc-opencl-2_2.41.5+22716_amd64.deb"
  )
  local packages=(
    "${igc_base_url}/intel-igc-core-2_2.41.5+22716_amd64.deb"
    "${igc_base_url}/intel-igc-opencl-2_2.41.5+22716_amd64.deb"
    "${crt_base_url}/intel-ocloc-dbgsym_26.35.39758.10-0_amd64.ddeb"
    "${crt_base_url}/intel-ocloc_26.35.39758.10-0_amd64.deb"
    "${crt_base_url}/intel-opencl-icd-dbgsym_26.35.39758.10-0_amd64.ddeb"
    "${crt_base_url}/intel-opencl-icd_26.35.39758.10-0_amd64.deb"
    "${crt_base_url}/libigdgmm12_22.10.0_amd64.deb"
    "${crt_base_url}/libze-intel-gpu1-dbgsym_26.35.39758.10-0_amd64.ddeb"
    "${crt_base_url}/libze-intel-gpu1_26.35.39758.10-0_amd64.deb"
  )

  log "Installing Intel GPU drivers for Ubuntu 24.04..."
  mkdir -p "${download_dir}"
  cd "${download_dir}"

  for url in "${packages[@]}"; do
    wget --no-continue "${url}"
  done
  wget --no-continue "${crt_base_url}/${checksum_file}"

  sha256sum -c "${checksum_file}"
  printf '%s\n' "${igc_sums[@]}" | sha256sum -c -

  shopt -s nullglob
  local artifacts=( *.deb *.ddeb )
  shopt -u nullglob
  if [[ ${#artifacts[@]} -eq 0 ]]; then
    echo "Error: no GPU package files found for Ubuntu 24.04." >&2
    exit 1
  fi

  sudo dpkg -i "${artifacts[@]}" || sudo apt-get install -f -y
  cd "${SCRIPT_DIR}"
  rm -rf "${download_dir}"
}

install_npu_2404() {
  local download_dir="${WORK_DIR}/intel_npu_2404"
  local npu_release="v1.38.0"
  local npu_archive="linux-npu-driver-v1.38.0.20260910-34487311128-ubuntu2404.tar.gz"
  local npu_url="https://github.com/intel/linux-npu-driver/releases/download/${npu_release}/${npu_archive}"
  local npu_sha256="1efcd4b60c22abee751d8f2705962cbcc2a569de45c7e0e670cf08afbfcdc1d2"
  local npu_packages=(
    intel-driver-compiler-npu
    intel-fw-npu
    intel-level-zero-npu
    intel-level-zero-npu-dbgsym
  )

  log "Installing Intel NPU drivers for Ubuntu 24.04..."
  mkdir -p "${download_dir}"
  cd "${download_dir}"

  # Download before purging, so a failed fetch does not leave the machine with
  # no NPU driver at all.
  wget --no-continue "${npu_url}"
  verify_sha256 "${npu_archive}" "${npu_sha256}" "${npu_url}"
  tar -xf "${npu_archive}"

  shopt -s nullglob
  local debs=( *.deb )
  shopt -u nullglob
  if [[ ${#debs[@]} -eq 0 ]]; then
    echo "Error: no Intel NPU .deb packages found for Ubuntu 24.04." >&2
    exit 1
  fi

  sudo dpkg --purge --force-remove-reinstreq "${npu_packages[@]}" || true
  sudo dpkg -i "${debs[@]}" || sudo apt-get install -f -y

  # The Level Zero loader the NPU plugin dlopens. 22.04 needs it from GitHub;
  # 24.04 has it in the archive. See docs/backend/ov.md for ZE_ENABLE_ALT_DRIVERS.
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y libze1

  add_render_group

  cd "${SCRIPT_DIR}"
  rm -rf "${download_dir}"
}

install_runtime_2404() {
  local download_dir="${WORK_DIR}/openvino_runtime_2404"
  local openvino_version="${OPENVINO_VERSION:-2026.4}"
  local openvino_build="${OPENVINO_BUILD:-0.22959.99c81491cc3}"
  local openvino_archive="openvino_toolkit_ubuntu24_${openvino_version}.${openvino_build}_x86_64.tgz"
  local openvino_dirname="openvino_toolkit_ubuntu24_${openvino_version}.${openvino_build}_x86_64"
  local openvino_url="https://storage.openvinotoolkit.org/repositories/openvino/packages/${openvino_version}/linux/${openvino_archive}"
  local install_root="${INSTALL_ROOT:-/opt/intel}"
  local install_dir="${install_root}/openvino_${openvino_version}"
  local symlink_path="${install_root}/openvino"
  local archive_path="${download_dir}/openvino_${openvino_version}.tgz"
  local expected_sha=""
  if [[ "${openvino_version}" == "2026.4" && "${openvino_build}" == "0.22959.99c81491cc3" ]]; then
    expected_sha="${OPENVINO_SHA256_2404}"
  fi

  log "Installing OpenVINO runtime for Ubuntu 24.04..."
  sudo mkdir -p "${install_root}"
  mkdir -p "${download_dir}"

  curl -fL "${openvino_url}" --output "${archive_path}"
  verify_sha256 "${archive_path}" "${expected_sha}" "${openvino_url}"
  rm -rf "${download_dir:?}/${openvino_dirname}"
  tar -xf "${archive_path}" -C "${download_dir}"

  sudo rm -rf "${install_dir}"
  sudo mv "${download_dir}/${openvino_dirname}" "${install_dir}"
  sudo ln -sfn "openvino_${openvino_version}" "${symlink_path}"

  if [[ ! -f "${symlink_path}/setupvars.sh" ]]; then
    echo "Error: ${symlink_path}/setupvars.sh was not found after installation." >&2
    exit 1
  fi

  rm -rf "${download_dir}"
}

run_installation() {
  case "${OS_VERSION}" in
    22.04)
      install_gpu_2204
      install_npu_2204
      install_runtime_2204
      ;;
    24.04)
      install_gpu_2404
      install_npu_2404
      install_runtime_2404
      ;;
    *)
      echo "Error: unsupported Ubuntu version '${OS_VERSION:-unknown}'. Supported versions: 22.04, 24.04." >&2
      exit 1
      ;;
  esac
}

main() {
  detect_os
  prepare_common_tools
  prepare_common_dependencies

  log "Detected Ubuntu ${OS_VERSION}."
  run_installation
  rm -rf "${WORK_DIR}"
  
  log "All OpenVINO installation steps completed successfully."
  log "To load OpenVINO in current shell: source /opt/intel/openvino/setupvars.sh"
}

main "$@"
