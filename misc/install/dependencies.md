# System dependencies

LIMAP needs a C++17 toolchain, CMake >= 3.17, and the libraries below. Most of
them are COLMAP's, so these instructions follow its
[official guide](https://colmap.github.io/install.html) per platform. COLMAP
itself, with PoseLib, JLinkage and libigl, is built in-tree by `FetchContent`.

## Linux (Ubuntu / Debian)

```bash
sudo apt-get install \
    ninja-build \
    build-essential \
    libeigen3-dev \
    libflann-dev \
    libopenimageio-dev \
    openimageio-tools \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
```

**Boost >= 1.84** is required, to match the backend the `pycolmap` wheels are
built with (see [colmap#4672](https://github.com/colmap/colmap/issues/4672)). Ubuntu 24.04 ships 1.83; in a conda environment:

```bash
conda install -c conda-forge libboost-devel
```

Otherwise build it (only `graph` and `program_options` are compiled):

```bash
wget https://archives.boost.io/release/1.90.0/source/boost_1_90_0.tar.bz2
tar xf boost_1_90_0.tar.bz2 && cd boost_1_90_0
./bootstrap.sh --prefix="$HOME/.local/boost" --with-libraries=graph,program_options
./b2 -j"$(nproc)" --with-graph --with-program_options install
export BOOST_ROOT="$HOME/.local/boost"
```

## macOS

Dependencies from [Homebrew](https://brew.sh/), following COLMAP:

```bash
brew install \
    cmake \
    ninja \
    boost \
    eigen \
    openimageio \
    curl \
    libomp \
    metis \
    glog \
    googletest \
    ceres-solver \
    suitesparse \
    glew \
    cgal \
    sqlite3
brew link --force libomp
```

`brew link --force libomp` is required: LIMAP links `OpenMP::OpenMP_CXX` and
AppleClang ships no OpenMP runtime of its own. The package then installs as on
Linux, with no extra CMake arguments.

## Windows

*Recommended dependencies:* Visual Studio 2019 or later.

As for COLMAP, the recommended way on Windows is [vcpkg](https://vcpkg.io/).
`vcpkg.json` at the repository root declares every C++ dependency, so vcpkg
builds them in manifest mode during the CMake configure:

```powershell
git clone https://github.com/microsoft/vcpkg
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
```

Then, from a Visual Studio developer shell (`misc/ci/enter_vs_dev_shell.ps1`
enters the latest installed instance), pass the toolchain file and the triplet
through to CMake:

```powershell
# CMake needs forward slashes in the toolchain path.
$TOOLCHAIN = "$PWD/vcpkg/scripts/buildsystems/vcpkg.cmake".Replace('\', '/')
python -m pip install -Ive ".[all]" `
    -Cbuild-dir=./pylimap_build `
    -Ccmake.define.CMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" `
    -Ccmake.define.VCPKG_TARGET_TRIPLET="x64-windows-release"
```

The first configure is slow -- vcpkg builds Ceres, Boost and OpenImageIO from
source, into `<build-dir>/vcpkg_installed`. Pass `-Cbuild-dir` so reinstalls
reuse them instead of starting over in a fresh temporary directory. The same
route works elsewhere with `x64-linux-release` or `arm64-osx-release`.

> [!WARNING]
> vcpkg builds `lapack-reference` for Ceres, and the LLVM `flang` bundled with
> Visual Studio miscompiles it
> ([llvm#201254](https://github.com/llvm/llvm-project/issues/201254)). CI
> removes that `flang`; local installations are left alone. If the build fails
> with `'ssum' is not an object that can appear in an expression`, hide
> `VC/Tools/Llvm/**/flang*.exe` from your PATH and configure again.
