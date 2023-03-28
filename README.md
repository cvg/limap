# LIMAP 

## Dependencies
* CMake >= 3.17
* COLMAP [[Guide](https://colmap.github.io/install.html)]
* PoseLib [[Guide](misc/install/poselib.md)]
* HDF5
```bash
sudo apt-get install libhdf5-dev
```
* Python 3.9 + required packages
```bash
git submodule update --init --recursive

# Refer to https://pytorch.org/get-started/previous-versions/ to install pytorch compatible with your CUDA
python -m pip install torch==1.11.0 torchvision==0.12.0 
python -m pip install -r requirements.txt
```

## Installation

```
python -m pip install -Ive . 
```
To double check if the package is successfully installed:
```
python -c "import limap"
```

## Quickstart

Download the test scene **(100 images)** with the following command.
```bash
bash scripts/quickstart.sh
```

To run **Fitnmerge** on Hypersim:
```bash
python runners/hypersim/fitnmerge.py --output_dir outputs/quickstart_fitnmerge
```

To run **Line Reconstruction** on Hypersim:
```bash
python runners/hypersim/triangulation.py --output_dir outputs/quickstart_triangulation
```

[Tips] Options are stored in the config folder: ``cfgs``. You can easily change the options with the Python argument parser. In the following shows an example:
```bash
python runners/hypersim/triangulation.py --sfm.hloc.descriptor sift --line2d.detector.method lsd \
                                         --line2d.visualize --triangulation.IoU_threshold 0.2 \
                                         --skip_exists --n_visible_views 5
```
In particular, ``skip_exists`` is a very useful option to avoid running point-based SfM and line detection/description repeatedly in each pass.

## Supported line detectors, matchers and VP estimators

If you wish to use the methods with **separate installation needed** you need to install it yourself with the corresponding guides. This is to avoid potential issues at the LIMAP installation to ensure a quicker start.

**Note**: PR on integration of new features are very welcomed.

**The following line detectors are currently supported:**
- [LSD](https://github.com/iago-suarez/pytlsd)
- [SOLD2](https://github.com/cvg/SOLD2)
- [HAWPv3](https://github.com/cherubicXN/hawp)
- [TP-LSD](https://github.com/Siyuada7/TP-LSD) (separate installation needed [[Guide](misc/install/tp_lsd.md)]) 
- [DeepLSD](https://github.com/cvg/DeepLSD) (separate installation needed [[Guide](https://github.com/cvg/DeepLSD)])

**The following line descriptors/matchers are currently supported:**
- [LBD](https://github.com/iago-suarez/pytlbd) (separate installation needed [[Guide](misc/install/lbd.md)])
- [SOLD2](https://github.com/cvg/SOLD2)
- [LineTR](https://github.com/yosungho/LineTR)
- [L2D2](https://github.com/hichem-abdellali/L2D2)
- Endpoint matching with [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) + Nearest Neighbors
- Endpoint matching with [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) + [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)

**The following vanishing point estimators are currently supported:**
- [JLinkage](https://github.com/B1ueber2y/JLinkage)
- [Progressive-X](https://github.com/danini/progressive-x) (separation installation needed [[Guide](https://github.com/danini/progressive-x/blob/master/README.md)])

