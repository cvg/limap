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

    python -m pip install -r requirements.txt   # git-sourced detectors and matchers
    python -m pip install -Ive ".[all]"

**Core only** -- the compiled library and its Python API, and nothing else:

.. code-block:: bash

    python -m pip install -Ive .

This gives you the geometry types, reading and writing of reconstructions, and
the estimators and bundle adjustment, all operating on data you already have.
It does **not** let you reconstruct from images, which needs ``hloc``, nor
detect 2D lines, nor visualize anything. This is also the mode a published
wheel provides -- the extras and the git-sourced detectors are always opt-in on
top.

**Developer** -- adds pytest and the pinned formatters on top of the full install:

.. code-block:: bash

    python -m pip install -r requirements.txt
    python -m pip install -Ive ".[all,dev]"

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
  ``visualize_colmap_model.py`` and the 3D helpers in ``limap.visualize`` do
  not run there. The ``pxwplanar`` plane detector also depends on open3d, so it is
  skipped on 3.13 as well. Line and point reconstruction are unaffected -- no
  pipeline touches open3d.
* Several further methods (HAWP, TP-LSD, LBD, RoMa, Progressive-X) are not
  installed by any of the above. Each is cloned and pip-installed separately.
  See the per-method guides under `misc/install/
  <https://github.com/cvg/limap/tree/main/misc/install>`_.

For faster incremental rebuilds during development (reuses the CMake build
directory instead of rebuilding from scratch):

.. code-block:: bash

    python -m pip install -Cbuild-dir=./pylimap_build --no-build-isolation -Ive .
