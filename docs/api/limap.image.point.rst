Point detection and matching
============================

Keypoint detection and matching are delegated to `hloc
<https://github.com/cvg/Hierarchical-Localization>`_; only the options
selecting the method are defined here. The results are written into COLMAP's
``database.db``.

.. currentmodule:: limap.image.point

.. autoclass:: PointDetectionOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: PointMatcherOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise
