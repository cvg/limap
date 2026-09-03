limap.visualize package
=======================

.. note::

   Everything here needs ``matplotlib``, ``seaborn`` or ``open3d``, which are
   installed with the ``viz`` extra: ``python -m pip install "limap[viz]"``.

2D visualization
----------------

.. currentmodule:: limap.visualize

.. autofunction:: draw_2d_points

.. autofunction:: draw_2d_lines

.. autofunction:: draw_2d_vpresult

.. autofunction:: make_big_image

3D visualization
----------------

.. autofunction:: open3d_get_3d_points

.. autofunction:: open3d_get_3d_lines

.. autofunction:: open3d_get_camera_frustums

.. autofunction:: open3d_visualize_3d_lines

.. autofunction:: open3d_get_plane_mesh

.. autofunction:: open3d_get_sphere_mesh

.. autofunction:: open3d_get_cylinder_mesh

.. autoclass:: PlaneMesh
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: SphereMesh
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: CylinderMesh
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

Textured meshes
---------------

Meshes carrying the image texture of the group they were fitted to.

.. autofunction:: create_textured_plane_mesh

.. autofunction:: create_textured_sphere_mesh

.. autofunction:: create_textured_cylinder_mesh

.. autofunction:: get_group_binary_mask

.. autofunction:: project_points_to_image

Ranges
------

The visualizers clip to a range, which is best computed robustly from the
reconstruction rather than from its extremes.

.. autofunction:: compute_robust_range_points

.. autofunction:: compute_robust_range_lines
