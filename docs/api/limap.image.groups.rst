Group detection, description and matching
=========================================

Vanishing points, planes and parametric primitives are detected per image and
associated across images. Vanishing point and plane detectors have registries
of their own, on the pages beside this one.

Detection and description
-------------------------

.. module:: limap.image.groups

.. autoclass:: VPDetectionOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: vp_detection

.. autoclass:: PlaneDetectionOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: plane_detection

.. autoclass:: GroupDescriptionOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: group_description

.. autofunction:: sam3_group_description

Matching by voting
------------------

Groups that carry no descriptor can still be matched, by voting with the
point and line correspondences that fall inside them.

.. autoclass:: GroupVotingOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: match_groups_by_voting

.. currentmodule:: limap.image.group_voting

.. autofunction:: vote_unmatched_groups

Group masks I/O
---------------

.. currentmodule:: limap.image.groups

.. autofunction:: get_group_mask_filename

.. autofunction:: read_group_mask

.. autofunction:: read_group_masks

.. autofunction:: write_group_mask

.. autofunction:: write_group_masks

SAM 3 utilities
---------------

.. autofunction:: load_sam3_data_for_image

.. autofunction:: convert_sam3_masks_to_groups2d
