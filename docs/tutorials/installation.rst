Installation
=================================

------------------
Dependencies
------------------

* Python 3.9/10/11
* CMake >= 3.17
* CUDA (for deep learning based detectors/matchers)
* System dependencies [`Command line <https://github.com/cvg/limap/blob/main/misc/install/dependencies.md>`_]
  
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
