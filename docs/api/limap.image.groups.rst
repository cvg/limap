Group detection, description and matching
=========================================

Detection and description
-------------------------

.. automodule:: limap.image.groups.group_ops
   :members:
   :undoc-members:
   :show-inheritance:

Options
-------

.. automodule:: limap.image.groups.specs
   :members:
   :undoc-members:
   :show-inheritance:

Matching by voting
------------------

Groups that carry no descriptor can still be matched, by voting with the
point and line correspondences that fall inside them.

.. autoclass:: limap.image.groups.GroupVotingOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: limap.image.groups.match_groups_by_voting

.. automodule:: limap.image.group_voting
   :members:
   :undoc-members:
   :show-inheritance:

Group masks I/O
---------------

.. automodule:: limap.image.groups.group_io
   :members:
   :undoc-members:
   :show-inheritance:

SAM 3 utilities
---------------

.. automodule:: limap.image.groups.sam3_utils
   :members:
   :undoc-members:
   :show-inheritance:
