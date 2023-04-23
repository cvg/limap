# LIMAP Docker Image Build from Dockerfile
This document explains how to build docker image for LIMAP. This document assumes that readers' systems meet following requirements:
- x86-64 (amd64) architecture
- GPU that supports CUDA 11.5
- Ubuntu

### Dependencies
- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)

### Github ssh setting
You need to set up a ssh key for Github account to clone the LIMAP when building the image. Follow [this instruction](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) to generate a key, and then [this instruction](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) to register the key to your Github account.

### Building Docker Image
Download the attached Dockerfile and run the below command at where the Dockerfile is.
```bash
docker build --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_ed25519)" -t="3dv:latest" .
```

Run the built Docker image with the following command.
```bash
docker run \
  --rm \
  -it \
  --gpus all \
  --shm-size 50G \
  --device=/dev/dri:/dev/dri \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  3dv:latest \
  bash
```

In case you want to run a GUI application on the container, you should allow X server connection from the host side:
```bash
xhost +local:*
```
