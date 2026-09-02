Holistic 3D mapping
=================================

-----------------------------------------
Line mapping on a set of posed images
-----------------------------------------

As one of the main features, LIMAP supports line reconstruction on a set of posed images provided as a COLMAP model, optionally jointly with points. The main entry point is the CLI:

.. code-block:: bash

    python -m limap.cli.automatic_point_line_triangulation \
        -m ${COLMAP_MODEL_PATH} \
        -i ${IMAGE_PATH} \
        -o ${OUTPUT_DIR}

where ``-m`` is the path to the COLMAP sparse model (cameras/images/points), ``-i`` the image folder, and ``-o`` the output directory. The reconstruction (points + lines) is written to ``${OUTPUT_DIR}/final_model``. The configuration defaults to ``cfgs/structure_triangulation/default.yaml`` and can be overridden with ``-c`` or per-key command-line flags.

For a complete worked example on Hypersim, see :doc:`quickstart`, or the dataset runner ``runners/hypersim/structure_triangulation.py``.

------------------------------------------------
Holistic mapping with groups and the wireframe
------------------------------------------------

``automatic_point_line_triangulation`` reconstructs points and lines only. To additionally reconstruct the groups (vanishing points and planes) and the wireframe, and to optimize all of them jointly with the camera poses, use:

.. code-block:: bash

    python -m limap.cli.automatic_structure_triangulation \
        -m ${COLMAP_MODEL_PATH} \
        -i ${IMAGE_PATH} \
        -o ${OUTPUT_DIR}

The arguments are the same as above. The bundle adjustment then additionally enforces the vanishing point and plane constraints (orthogonality and parallelism) on the associated lines and points. The 3D structures are written to ``${OUTPUT_DIR}/final_model/structures/``, alongside the COLMAP model itself; see :doc:`output`.

Plane detection runs a monocular network and needs a GPU; see :doc:`groups` for the detectors involved and for how to switch groups off.

---------------------------------------------------------------------
Line mapping on a set of unposed images by running COLMAP first
---------------------------------------------------------------------

To run line mapping on a set of unposed images, first pose the images with `COLMAP <https://colmap.github.io>`_ following the guide `here <https://colmap.github.io/cli.html>`_. Then pass the resulting sparse model to the triangulation CLI above via the ``-m`` argument, along with the corresponding image folder via ``-i``.

This poses the cameras from points alone and reconstructs the structures afterwards, on top of frozen poses. LIMAP can instead recover the poses and the structures together, so that lines and groups constrain the bundle adjustment throughout the reconstruction; see :doc:`sfm`.

-----------------------------------------
Using auxiliary depth maps
-----------------------------------------

When depth maps are available, the 3D line map can be built with geometry-guided line reconstruction instead of triangulation. See the runner ``runners/hypersim/geometry_guided_line_reconstruction.py`` and the :py:meth:`limap.runners.line_reconstruction_with_depth_maps` API, configured via ``cfgs/geometry_guided_line_reconstruction/``.
