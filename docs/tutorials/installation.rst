Installation
=================================

------------------
Dependencies
------------------

* Python 3.10/11/12/13
* CMake >= 3.17
* CUDA (for deep learning based detectors/matchers)
* System dependencies [`Command line <https://github.com/cvg/limap/blob/main/misc/install/dependencies.md>`_]

*Note that one cannot visualize reconstructions on Python 3.13, as there are
no published wheels available for open3d and our 3D viewer depends on it.*

------------------
Install modes
------------------

**Full** -- everything the reconstruction pipelines need. Use this unless you
know you want one of the others:

.. code-block:: bash

    python -m pip install -Ive ".[all]"
    python -m pip install -r requirements.txt   # git-sourced detectors and matchers

**Core only** -- the compiled library and its Python API, without visualisation
or the 2D detectors. Enough to use ``limap.geometry``, ``limap.scene``,
``limap.estimators`` and ``limap.sfm`` as a library:

.. code-block:: bash

    python -m pip install -Ive .

**Developer** -- adds pytest and the pinned formatters on top of the full install:

.. code-block:: bash

    python -m pip install -Ive ".[all,dev]"
    python -m pip install -r requirements.txt

To double check if the package is successfully installed:

.. code-block:: bash

    python -c "import limap; print(limap.__version__)"

------------------
Extras
------------------

.. list-table::
   :header-rows: 1

   * - Extra
     - Contents
   * - ``viz``
     - matplotlib, seaborn, open3d -- needed by ``limap.visualize``
   * - ``line2d``
     - einops, scikit-image, pillow -- support code for the 2D line detectors
   * - ``dev``
     - pytest, ruff, clang-format
   * - ``all``
     - ``viz`` + ``line2d``

A few things worth knowing:

* **Running a reconstruction needs** ``hloc``, which comes from
  ``requirements.txt`` rather than from the package metadata: it is not
  published on PyPI, so it cannot be declared as a dependency. Without it the
  package imports fine, but the point frontend will fail when it is first used.
* **open3d publishes no wheels for Python 3.13+**, and all 3D visualization
  depends on it, so ``visualize_holistic_recon.py``,
  ``visualize_colmap_model.py`` and ``limap.visualize``'s 3D helpers do not run
  there. Reconstruction itself is unaffected -- no pipeline touches open3d.
* Several further methods (HAWP, TP-LSD, LBD, RoMa, Progressive-X) are not
  installed by any of the above. Each is cloned and pip-installed separately.
  See the per-method guides under `misc/install/
  <https://github.com/cvg/limap/tree/main/misc/install>`_.

For faster incremental rebuilds during development (reuses the CMake build
directory instead of rebuilding from scratch):

.. code-block:: bash

    python -m pip install -Cbuild-dir=./pylimap_build --no-build-isolation -Ive .
