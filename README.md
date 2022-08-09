# LIMAP 

## TODO

* [ ] CameraRig optimization.

## Dependencies
* CMake >= 3.17
* Python 3.9
* COLMAP

```bash
git submodule update --init --recursive
python -m pip install -r requirements.txt
cd third-party/Hierarchical-Localization && python -m pip install -e . && cd ../..
```

Pretrained models for [SOLD2](https://github.com/cvg/SOLD2), [S2DNet](https://github.com/germain-hug/S2DNet-Minimal), [SuperPoint](https://github.com/magicleap/SuperGluePretrainedNetwork) and [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) need to be downloaded:
```
bash download_pretrained_models.sh
```

To compile LIMAP:
```bash
sudo apt-get install libhdf5-dev
mkdir -p build && cd build
cmake -DPYTHON_EXECUTABLE=${path-to-your-python-executable} ..
make -j8
cd ..
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

