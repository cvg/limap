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

Pretrained models for [SOLD2](https://github.com/cvg/SOLD2) and [S2DNet](https://github.com/germain-hug/S2DNet-Minimal) (only for featuremetric optimization) need to be downloaded:
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
python runners/hypersim/fitnmerge.py
```

To run **Line Reconstruction** on Hypersim:
```bash
python runners/hypersim/triangulation.py
```

