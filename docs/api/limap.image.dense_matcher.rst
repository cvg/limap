Dense matching
==============

Dense matchers produce a pixel-wise warp between two images. It is used to
associate points, lines and groups without relying on sparse descriptors.

Instantiate a dense matcher
---------------------------

.. currentmodule:: limap.image.dense_matcher

.. autofunction:: get_dense_matcher

Base interface and results
--------------------------

.. currentmodule:: limap.image.dense_matcher.base_dense_matcher

.. autoclass:: BaseDenseMatcher
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. currentmodule:: limap.image.dense_matcher

.. autoclass:: DenseMatchingResult
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: BiDenseMatchingResult
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

Options
-------

.. autoclass:: DenseMatchingOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: PointDenseMatchingOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: LineDenseMatchingOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: GroupDenseMatchingOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

Association
-----------

.. autofunction:: associate_via_dense_matching

.. autofunction:: match_points_via_dense_matching

.. autofunction:: match_lines_via_dense_matching

.. autofunction:: match_groups_via_dense_matching

Metrics
-------

.. autofunction:: compute_point_distance_matrix

.. autofunction:: compute_line_distance_matrix

.. autofunction:: compute_mask_overlap_matrix
