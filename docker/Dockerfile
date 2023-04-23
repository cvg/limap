# this is our first build stage, it will not persist in the final image
FROM ubuntu as intermediate

# install git
RUN apt-get update && \
    apt-get install -y git

# add credentials on build
ARG SSH_PRIVATE_KEY
RUN mkdir /root/.ssh/
RUN echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_ed25519
RUN chmod 600 /root/.ssh/id_ed25519

# make sure your domain is accepted
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN git clone --recursive git@github.com:cvg/limap.git

# From here, final image
FROM nvidia/cuda:11.5.2-devel-ubuntu20.04

ARG COLMAP_VERSION=3.8
# ARG CUDA_ARCHITECTURES=86
ARG CUDA_ARCHITECTURES=all

# Prevent stop building ubuntu at time zone selection.  
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building.
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    wget \
    software-properties-common \
    lsb-core

# CMake Install (Default version does not meet the LIMAP requirement)
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
    gpg --dearmor - | \
    tee /etc/apt/trusted.gpg.d/kitware.gpg > \
    /dev/null

RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt update && \
    apt install kitware-archive-keyring && \
    rm /etc/apt/trusted.gpg.d/kitware.gpg && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 42D5A192B819C5DA

RUN apt update && \
    apt install -y cmake

# Build and install COLMAP.
RUN git clone https://github.com/colmap/colmap.git
RUN cd colmap && \
    git reset --hard ${COLMAP_VERSION} && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    ninja && \
    ninja install && \
    cd .. && rm -rf colmap

# Enable GUI
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# PoseLib Dependency for LIMAP
RUN git clone --recursive https://github.com/vlarsson/PoseLib.git && \
    cd PoseLib && \
    mkdir build && cd build && \
    cmake .. && \
    make install -j8

RUN apt-get update && \
    apt-get install -y \
    libhdf5-dev

# Only Python 3.9 seems to satisfy all dependencies
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && \
    apt-get install -y --fix-missing\
    python3.9-dev \
    python3.9-venv

ENV VIRTUAL_ENV=/opt/venv
RUN /usr/bin/python3.9 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the repository from the first image
COPY --from=intermediate /limap /limap
RUN python -m pip install torch==1.12.0 torchvision==0.13.0 --index-url https://download.pytorch.org/whl/cu115
RUN python -m pip install --upgrade pip setuptools && \
    cd limap && \
    python --version && \
    pip --version && \
    python -m pip install -r requirements.txt && \
    python -m pip install -Ive .

