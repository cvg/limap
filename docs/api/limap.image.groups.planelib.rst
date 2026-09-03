Plane detection
===============

Instantiate a plane detector
----------------------------

.. currentmodule:: limap.image.groups.planelib

.. autofunction:: get_plane_detector

.. autoclass:: DetectorOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: PxwPlanarOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

Base interface
--------------

.. currentmodule:: limap.image.groups.planelib.base_plane_detector

.. autoclass:: BasePlaneDetector
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: BasePlaneDetectorOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

Conversion utilities
--------------------

.. currentmodule:: limap.image.groups.planelib

.. autofunction:: convert_plane_mask_to_groups2d

Visualization
-------------

.. autofunction:: visualize_top_components

.. autofunction:: visualize_single_component

.. autofunction:: visualize_plane_tracks
