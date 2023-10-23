Installation
=================================

------------------
Dependencies
------------------

* CMake >= 3.17
* COLMAP 3.8
  
  Follow `official document <https://colmap.github.io/install.html>`_ to install COLMAP. Make sure to use the tag 3.8. (COLMAP has been under active development since summer 2023, so we currently only support COLMAP 3.8 or before)

* PoseLib
  
  .. code-block:: bash

    git clone --recursive https://github.com/vlarsson/PoseLib.git
    cd PoseLib
    mkdir build && cd build
    cmake ..
    sudo make install -j8

* HDF5

  .. code-block:: bash

    sudo apt-get install libhdf5-dev

* OpenCV (only for installing `pytlbd <https://github.com/B1ueber2y/limap-internal/blob/main/requirements.txt#L33>`_, it's fine to remove it from `requirements.txt` and use LIMAP without OpenCV)

  .. code-block:: bash

    sudo apt-get install libopencv-dev libopencv-contrib-dev libarpack++2-dev libarpack2-dev libsuperlu-dev

* Python 3.9 + required packages

  Install PyTorch >= 1.12.0 (Note: LIMAP has not been tested with PyTorch 2)

  * CPU version
  
  .. code-block:: bash

    pip install torch==1.12.0 torchvision==0.13.0


  * Or refer to https://pytorch.org/get-started/previous-versions/ to install PyTorch compatible with your CUDA version
  
  Install other Python packages

  .. code-block:: bash

      git submodule update --init --recursive
      pip install -r requirements.txt

------------------
Install LIMAP
------------------

.. code-block:: bash

    pip install -Ive .

Alternatively:

.. code-block:: bash

    mkdir build && cd build
    cmake -DPYTHON_EXECUTABLE=`which python` ..
    make -j8

To test if LIMAP is successfully installed, try ``import limap`` and it should reports no error.

.. code-block:: bash

    python -c "import limap"
