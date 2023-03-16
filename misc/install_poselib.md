## Install PoseLib as Dependency
Clone the repository
```bash
git clone --recursive https://github.com/vlarsson/PoseLib.git
cd PoseLib
```
Switch to "alignment" branch that fixes the alignment and compatible with COLMAP if you have Eigen 3.3 (you can stay on master if you have Eigen >= 3.4):
```bash
git checkout alignment
```

Build and install:
```bash
mkdir build && cd build
cmake ..
sudo make install -j8
```