Visualization
=================================

We provide command-line scripts to visualize the reconstruction produced by LIMAP. To visualize the full reconstruction (points + lines):

.. code-block:: bash

    python visualize_holistic_recon.py --input_dir ${OUTPUT_DIR}/final_model --cam_scale 0.1

To visualize the points only (using pycolmap):

.. code-block:: bash

    python visualize_colmap_model.py --input_dir ${OUTPUT_DIR}/final_model --cam_scale 0.1

Here ``${OUTPUT_DIR}`` is the output directory passed to the triangulation CLI (see :doc:`triangulation`). We use `Open3D <http://www.open3d.org/>`_ as the backend for the visualization.

``visualize_holistic_recon.py`` also renders the structured primitives of a holistic reconstruction. Useful options include ``--color_lines_by_vps`` to color the 3D lines by their associated vanishing point, ``--manhattan_planes`` / ``--atlanta_planes`` to restrict the displayed planes to a Manhattan or Atlanta world, ``--plane_alpha`` and ``--cylinder_alpha`` to control primitive transparency, and ``--min_track_length`` / ``--min_associated_features`` to filter out weakly supported structures. Pass ``--help`` for the full list.
