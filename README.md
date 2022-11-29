# LIMAP 

## Dependencies
* CMake >= 3.17
* Python 3.9
* COLMAP

```bash
git submodule update --init --recursive
sudo apt-get install libhdf5-dev
python -m pip install torch==1.11.0 torchvision==0.12.0
python -m pip install -r requirements.txt
```

## Installation

You can install ``limap`` package with:
```
python setup.py develop
```

or you can also use ``pip`` alternatively:
```
python -m pip install -ve .
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
python runners/hypersim/triangulation.py --sfm.fbase sift --line2d.detector.method lsd \
                                         --line2d.visualize --triangulation.IoU_threshold 0.2 \
                                         --skip_exists --n_visible_views 5
```
In particular, ``skip_exists`` is a very useful option to avoid running point-based SfM and line detection/description repeatedly in each pass.

## Supported line detectors and descriptors

The following line detectors are currently supported:
- [LSD](https://github.com/iago-suarez/pytlsd)
- [SOLD2](https://github.com/cvg/SOLD2)
- [HAWPv3](https://github.com/cherubicXN/hawp)

The line detector [TP-LSD](https://github.com/Siyuada7/TP-LSD) can be additionally used, but needs a separate installation. You will need to switch to GCC 7 for the following compilation (but can use again GCC 9 at test time):
```bash
python -m pip install -e ./third-party/TP-LSD/tp_lsd/modeling/DCNv2
python -m pip install -e ./third-party/TP-LSD
```

The following line descriptors/matchers are currently supported:
- [LBD](https://github.com/iago-suarez/pytlbd)
- [SOLD2](https://github.com/cvg/SOLD2)
- [LineTR](https://github.com/yosungho/LineTR)
- [L2D2](https://github.com/hichem-abdellali/L2D2)
- Nearest neighbor matching of the endpoints with [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)
- Matching of the endpoints with [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) + [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
