Holistic incremental SfM
=================================

When no camera poses are available, LIMAP recovers them together with the 3D model. The holistic incremental mapper registers the images and bundle-adjusts points, lines, vanishing points, planes and the wireframe jointly, rather than reconstructing points first and fitting the structures onto frozen poses afterwards.

This is the pipeline of our `ECCV 2024 <https://arxiv.org/abs/2409.19811>`_ and `ECCV 2026 <https://arxiv.org/abs/2609.04026>`_ papers; see the *Citations* section of the README for details.

A reconstruction runs in two stages: a **frontend** that detects and matches features over the images, and a **mapper** that incrementally registers the images and optimizes the reconstruction.

-----------------------------------------
Frontend: detection and matching
-----------------------------------------

The frontend writes two databases:

* ``database.db`` - the COLMAP database, holding keypoints and their matches
* ``structure_database.db`` - the structure database, holding 2D lines, groups and their associations

For SfM the entry point is :py:func:`limap.runners.structure_frontend_from_images`, which takes the image directory and an in-memory ``pycolmap.Reconstruction``. It has to be in memory: COLMAP's on-disk model format drops unposed frames, which is precisely what we have at this stage.

.. note::

   ``python -m limap.cli.structure_frontend`` is the *posed* variant (:py:func:`limap.runners.structure_frontend_from_model`), intended for the pipelines in :doc:`triangulation`. It requires a COLMAP model and cannot be used for SfM from scratch.

The detectors, matchers and their options are shared with the mapping pipelines; see :doc:`line2d` and :doc:`groups`.

-----------------------------------------
Incremental mapper
-----------------------------------------

Given the two databases, the mapper runs the incremental reconstruction:

.. code-block:: bash

    python -m limap.cli.structure_incremental_sfm \
        --db_path ${COLMAP_DATABASE} \
        --structure_db_path ${STRUCTURE_DATABASE} \
        --image_path ${IMAGE_PATH} \
        --output_dir ${OUTPUT_DIR}

Each reconstructed model is written to ``${OUTPUT_DIR}`` as a COLMAP model, with the 3D structures alongside it under ``structures/`` (see :doc:`output`).

-----------------------------------------
End-to-end incremental SfM
-----------------------------------------

In practice the two stages are run together. :py:func:`limap.runners.automatic_structure_incremental_reconstruction` calls the frontend and then the mapper, starting from the images alone. The dataset runner ``runners/hypersim/automatic_structure_incremental_reconstruction.py`` wraps it and is configured through ``cfgs/structure_incremental_reconstruction/``.

For a worked example on the quickstart scene, together with the relative pose AUC against the ground truth, see :doc:`quickstart`.

-----------------------------------------
Relation to triangulation
-----------------------------------------

If the camera poses are already known, for instance from an existing COLMAP reconstruction, there is no need to run SfM: use the pipelines in :doc:`triangulation` instead, which keep the given poses and reconstruct the structures on top of them.
