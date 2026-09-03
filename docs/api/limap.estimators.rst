limap.estimators package
========================

Estimation of 3D structures and camera poses: triangulation of lines,
robust fitting of 3D lines and group primitives, absolute pose estimation
and the bundle adjusters.

.. toctree::
   :maxdepth: 3
   :caption: Module Contents:

   limap.estimators.triangulation
   limap.estimators.line3d
   limap.estimators.group3d
   limap.estimators.absolute_pose
   limap.estimators.bundle_adjustment

RANSAC
------

.. autoclass:: limap.estimators.PoseLibRansacOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: limap.estimators.PoseLibRansacStats
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise
