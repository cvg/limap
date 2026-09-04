Joint point-line detection
==========================

Some networks predict keypoints and line segments from a single shared
backbone. Running them through the separate point and line paths would encode
each image twice, so the joint path drives one pass per image and writes the
point features, the segments and the line descriptors together.

Detection
---------

.. currentmodule:: limap.image.joint_point_line

.. autofunction:: joint_point_line_detection

Options
-------

.. autoclass:: JointPointLineDetectionOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise
