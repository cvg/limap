#!/bin/bash
set -e -x
uname -a
CURRDIR=$(pwd)

export PATH="/usr/bin"

# Install toolchain under AlmaLinux 8,
# see https://almalinux.pkgs.org/8/almalinux-appstream-x86_64/
yum install -y \
    gcc-toolset-12-gcc \
    gcc-toolset-12-gcc-c++ \
    gcc-toolset-12-gcc-gfortran \
    scl-utils \
    git \
    cmake3 \
    ninja-build \
    curl \
    zip \
    unzip \
    tar

source scl_source enable gcc-toolset-12

git clone https://github.com/microsoft/vcpkg ${VCPKG_INSTALLATION_ROOT}
cd ${VCPKG_INSTALLATION_ROOT}
git checkout ${VCPKG_COMMIT_ID}
./bootstrap-vcpkg.sh
./vcpkg integrate install
