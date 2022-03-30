# LIMAP 

## TODO

* [ ] Abstract Detection, Matching, Fitting and Merging 

## Dependencies
* CMake >= 3.17
* Python 3.9
* COLMAP

```bash
git submodule update --init --recursive
python -m pip install -r requirements.txt
cd third-party/Hierarchical-Localization && python -m pip install -e . && cd ../..
```

To compile LIMAP:
```bash
sudo apt-get install libhdf5-dev
mkdir -p build && cd build
cmake -DPYTHON_EXECUTABLE=${path-to-your-python-executable} ..
make -j8
cd ..
```

For SOLD2 line detector, the pretrained models need to be downloaded from the links provided in the [SOLD2 repo](https://github.com/cvg/SOLD2) and put into `core/detector/SOLD2/pretrained_models`. A one-button download script is provided here to make things easier:
```bash
bash scripts/download_sold2_model.sh
```

## Quickstart

Download the test scene **(100 images)** with the following command.
```bash
bash scripts/quickstart.sh
```

To run **Fitnmerge** on Hypersim:
```bash
python runners/hypersim_fitnmerge.py
```

To run **Line Reconstruction** on Hypersim:
```bash
python runners/hypersim_triangulation.py
```

