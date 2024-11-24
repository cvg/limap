## LBD installation

Install OpenCV dependency
```bash
sudo apt-get install libopencv-dev libopencv-contrib-dev libarpack++2-dev libarpack2-dev libsuperlu-dev
```
Then install the pytlbd Python package from [Iago Suárez](https://github.com/iago-suarez)
```bash
git clone --recursive https://github.com/iago-suarez/pytlbd.git ./third-party/pytlbd
python -m pip install -e ./third-party/pytlbd
```
The LBD is listed as a dependency with separate installation to avoid the OpenCV dependency. When using the LBD matcher inside LIMAP, please use the parallelization with ``--line2d.matcher.n_jobs 8`` (8 cores, you can use even more CPU cores if applicable). 

