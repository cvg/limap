#!/bin/bash
# cibuildwheel before-all for the manylinux container.
#
# Unlike pycolmap, limap does not pre-build COLMAP here: it is pulled in with
# FetchContent during the wheel build itself (src/thirdparty/CMakeLists.txt).
# This script only has to provide a toolchain, ccache and a bootstrapped vcpkg.
set -e -x
uname -a

export PATH="/usr/bin"

yum install -y dnf-plugins-core epel-release
yum config-manager --set-enabled powertools

yum install -y \
    gcc-toolset-12-gcc \
    gcc-toolset-12-gcc-c++ \
    gcc-toolset-12-gcc-gfortran \
    kernel-headers \
    perl-IPC-Cmd \
    scl-utils \
    git \
    cmake3 \
    ninja-build \
    curl \
    zip \
    unzip \
    tar \
    perl

source scl_source enable gcc-toolset-12

# The ccache shipped by the container is too old, so download and cache it.
COMPILER_TOOLS_DIR="${CONTAINER_COMPILER_CACHE_DIR}/bin"
mkdir -p "${COMPILER_TOOLS_DIR}"
if [ ! -f "${COMPILER_TOOLS_DIR}/ccache" ]; then
    FILE="ccache-4.13.6-linux-x86_64-glibc"
    curl -sSLO "https://github.com/ccache/ccache/releases/download/v4.13.6/${FILE}.tar.xz"
    echo "508b2a1217dc6e04a23e967c7b95a0fb45d8a7e16fde9e180919698f2e2be060  ${FILE}.tar.xz" | sha256sum --check
    tar -xf "${FILE}.tar.xz"
    cp "${FILE}/ccache" "${COMPILER_TOOLS_DIR}"
fi
export PATH="${COMPILER_TOOLS_DIR}:${PATH}"
ln -sf "${COMPILER_TOOLS_DIR}/ccache" /usr/local/bin/ccache
ccache --zero-stats

git clone https://github.com/microsoft/vcpkg "${VCPKG_INSTALLATION_ROOT}"
cd "${VCPKG_INSTALLATION_ROOT}"
./bootstrap-vcpkg.sh
./vcpkg integrate install
