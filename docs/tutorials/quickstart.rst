Quickstart
=================================

Some examples are prepared for users to quickly try out LIMAP for mapping, localization and SfM with lines and the other structured primitives.

------------------
Line Mapping
------------------

For this example we are using the first scene ``ai_001_001`` from `Hypersim <https://github.com/apple/ml-hypersim>`_ dataset. Download the test scene **(100 images)** with the following command:

.. code-block:: bash

    bash scripts/quickstart.sh

First, prepare the Hypersim scene by undistorting the images and creating a COLMAP model:

.. code-block:: bash

    python runners/hypersim/undistort_images.py \
        --data_dir data \
        --scene_id ai_001_001 \
        --output_dir outputs/quickstart \
        --max_image_dim 800

Then, run point-line triangulation on the undistorted images:

.. code-block:: bash

    python -m limap.cli.automatic_point_line_triangulation \
        -m outputs/quickstart/undistorted/sparse \
        -i outputs/quickstart/undistorted/images \
        -o outputs/quickstart/triangulation

To visualize the full reconstruction (points + lines):

.. code-block:: bash

    python visualize_holistic_recon.py --input_dir outputs/quickstart/triangulation/final_model --cam_scale 0.1

To visualize points only (using pycolmap):

.. code-block:: bash

    python visualize_colmap_model.py --input_dir outputs/quickstart/triangulation/final_model --cam_scale 0.1

To additionally reconstruct the vanishing points, planes and the wireframe on the same scene, swap the CLI for ``python -m limap.cli.automatic_structure_triangulation`` with the same arguments (a GPU is needed for plane detection). See :doc:`triangulation`.

[**Tips**] Options are stored in the config folder ``cfgs`` (default: ``cfgs/structure_triangulation/default.yaml``). You can override the config file with ``-c``, or override individual options directly on the command line. The ``--skip_exists`` option is useful to avoid re-running point-based SfM and line detection/description in each pass.

------------------------------------------
Holistic Incremental SfM
------------------------------------------

The same scene can be reconstructed from scratch, with no input poses, using
the holistic incremental mapper, which jointly optimizes points, lines,
vanishing points, planes and the wireframe:

.. code-block:: bash

    python experiments/benchmark_sfm.py \
        --dataset hypersim \
        --scenes ai_001_001 \
        --data_dir data \
        --output_dir outputs/quickstart_sfm

This writes ``outputs/quickstart_sfm/hypersim/ai_001_001/holistic/models/`` and
prints the relative pose AUC against the ground-truth poses. Add
``--methods holistic points_only`` to run the point-only baseline alongside it;
it reuses the same frontend, so the comparison isolates the mapper.

See :doc:`sfm` for the mapper itself and for running the frontend separately.

-------------------------------------------------
Hybrid Localization with Points and Lines
-------------------------------------------------

We provide an example of hybrid point-line localization on the *Stairs* scene of the `7Scenes <https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/>`_ dataset. Prepare the dataset following hloc's `7Scenes pipeline <https://github.com/cvg/Hierarchical-Localization/tree/master/hloc/pipelines/7Scenes>`_ (scene images together with the SIFT SfM models, DenseVLAD retrieval pairs, and rendered depth maps), laid out under a single ``datasets/7scenes`` root. Then run:

.. code-block:: bash

    python runners/7scenes/localization.py --dataset datasets/7scenes -s stairs --skip_exists

Add ``--use_dense_depth`` to build the line map from rendered depth maps instead of triangulation, or ``--use_points_only`` for the point-only baseline. The runner prints the pose errors for point-only (hloc) versus hybrid point-line localization; an improved accuracy from adding lines is expected. See :doc:`localization` for the full tutorial.
