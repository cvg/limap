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

Many of the above dependencies are for the third-party COLMAP following its [official guide](https://colmap.github.io/install.html).

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
