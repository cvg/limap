limap.scene package
===================

A reconstruction is a plain COLMAP model with the LIMAP structures beside it:
:class:`HolisticReconstruction` pairs a ``pycolmap.Reconstruction`` with a
:class:`StructureReconstruction` holding the 3D lines, groups, wireframe and
the per-image 2D structures. This module defines those types, the structure
database that carries the frontend output, and the helpers that fill it.

.. currentmodule:: limap.scene

Reconstruction
--------------

The containers a pipeline reads and writes, serialized as a COLMAP model
plus a ``structures/`` folder; see :doc:`the output format
</tutorials/output>`. The manager holds several of them, as COLMAP's
``ReconstructionManager`` does for the point models.

.. autoclass:: HolisticReconstruction
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: StructureReconstruction
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: HolisticReconstructionManager
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

2D structures
-------------

What a single image observes: the line segments, groups and wireframe
detected in it, together with the features associated to them.

.. autoclass:: Structure2d
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: AssociatedFeature2d
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: Group2d
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: Wireframe2d
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: WireframeConnection2d
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: create_wireframe2d

3D structures
-------------

Their counterparts in the reconstruction. The ``WithActiveLabels`` variants
additionally track which of their observations are currently active, a
transient state that is not serialized.

.. autoclass:: AssociatedFeature3d
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: Group3d
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: Group3dWithActiveLabels
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: Line3dWithActiveLabels
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: Wireframe3d
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: WireframeConnection3d
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: WireframeVotingOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

Matches and correspondences
---------------------------

Line and group correspondences ride in the same ``TwoViewGeometry``
structures as the point matches, and are collected into a pair of
``colmap::CorrespondenceGraph`` instances, one for lines and one for groups.

.. autoclass:: LineMatch
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: LineMatches
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: GroupMatch
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: GroupMatches
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: StructureCorrespondenceGraph
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

Structure database
------------------

``structure_database.db`` is LIMAP's counterpart of COLMAP's ``database.db``:
it stores the 2D lines and groups, their associations and their matches.

.. autoclass:: StructureDatabase
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: StructureDatabaseCache
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: StructureDatabaseTransaction
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: create_structure_db

.. autofunction:: initialize_structures

.. autofunction:: initialize_structures_from_reconstruction

.. autofunction:: import_line_detections

.. autofunction:: import_line_matches

.. autofunction:: import_group_descriptions

Depth and point cloud readers
-----------------------------

Interfaces for the geometry-guided pipelines, which reconstruct lines from
depth maps or a point cloud instead of triangulating them.

.. autoclass:: BaseDepthReader
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: BasePointCloudReader
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

COLMAP MVS models
-----------------

.. autoclass:: COLMAPMVSModel
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: COLMAPMVSImage
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: compute_neighbors

