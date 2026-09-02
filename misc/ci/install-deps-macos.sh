#!/bin/bash
# cibuildwheel before-all for macOS. See install-deps-almalinux.sh: COLMAP is
# built in-tree by FetchContent, so this only provides the toolchain and vcpkg.
set -x -e

# Fix `brew link` error.
find /usr/local/bin -lname '*/Library/Frameworks/Python.framework/*' -delete

brew uninstall cmake || true   # Workaround for CI failures.
# libomp is required: limap links OpenMP::OpenMP_CXX, and AppleClang does not
# ship an OpenMP runtime of its own.
brew install git cmake ninja gfortran ccache libomp
brew link --force libomp
ccache --zero-stats

sudo xcode-select --reset

# When building lapack-reference, vcpkg/cmake looks for gfortran.
ln -sf "$(which gfortran-14)" "$(dirname "$(which gfortran-14)")/gfortran"

git clone https://github.com/microsoft/vcpkg "${VCPKG_INSTALLATION_ROOT}"
cd "${VCPKG_INSTALLATION_ROOT}"
./bootstrap-vcpkg.sh
./vcpkg integrate install
