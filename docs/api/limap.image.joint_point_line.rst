Joint point-line matching
=========================

GlueStick and its successors match points and lines together, in one pass over
an image pair. Running a point matcher and a line matcher separately over the
same pairs does the shared work twice, so a joint matcher replaces both: it
writes its point half in hloc's match file format and its line half in the
per-image files :func:`limap.scene.import_line_matches` reads, which leaves the
import and the geometric verification downstream unchanged.

The junctions a joint matcher matches are the endpoints of the lines plus the
keypoints of the same network pass that described them. The description step
exports those keypoints into the COLMAP database
(:func:`joint_point_line_description`), so the network runs once per image
rather than once for the line descriptors and again for the point features, and
the junctions the matcher sees are exactly the ones the descriptors were built
from.

Instantiate a joint matcher
---------------------------

.. module:: limap.image.joint_point_line

.. autofunction:: get_joint_matcher

.. autofunction:: get_joint_matcher_class

Base interface and results
--------------------------

.. currentmodule:: limap.image.joint_point_line.base_joint_matcher

.. autoclass:: BaseJointMatcher
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. currentmodule:: limap.image.joint_point_line

.. autoclass:: JointMatchResult
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

Options
-------

.. autoclass:: JointPointLineMatcherOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: JointMatcherOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: BaseJointMatcherOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

Association
-----------

.. autofunction:: joint_point_line_description

.. autofunction:: joint_point_line_matching

.. autofunction:: remap_point_matches

.. autofunction:: write_hloc_features
