Installation
=================================

------------------
Dependencies
------------------

* Python 3.10/11/12
* CMake >= 3.17
* CUDA (for deep learning based detectors/matchers)
* System dependencies [`Command line <https://github.com/cvg/limap/blob/main/misc/install/dependencies.md>`_]

------------------
Install LIMAP
------------------

.. code-block:: bash

    git submodule update --init --recursive
    python -m pip install -r requirements.txt
    python -m pip install -Ive .

To double check if the package is successfully installed:

.. code-block:: bash

    python -c "import limap; print(limap.__version__)"

For faster incremental rebuilds during development (reuses the CMake build directory instead of rebuilding from scratch):

.. code-block:: bash

    python -m pip install -Cbuild-dir=./pylimap_build --no-build-isolation -Ive .
