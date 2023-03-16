## Install PoseLib as Dependency
Clone the repository, use the main repo if you have Eigen >= 3.4:
```bash
git clone --recursive https://github.com/vlarsson/PoseLib.git
```
Or clone this fork that fixes the alignment and compatible with COLMAP if you have Eigen 3.3:
```bash
git clone --recursive -b alignment_colmap https://github.com/MarkYu98/PoseLib.git
```

Build and install:
```bash
cd PoseLib
mkdir build && cd build
cmake ..
sudo make install -j8
```