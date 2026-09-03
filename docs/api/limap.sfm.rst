limap.sfm package
=================

Triangulation of lines and groups -- globally over all images at once, or
incrementally as images are registered -- together with the structure-aware
incremental mapper and the pipeline driving it.

.. currentmodule:: limap.sfm

Global triangulation
--------------------

Triangulates every structure from all its observations at once, for a
reconstruction whose poses are already known.

.. autoclass:: GlobalLineTriangulationOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: GlobalLineTriangulationController
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: global_line_triangulation

.. autofunction:: global_line_triangulation_pipeline

.. autoclass:: GlobalGroupTriangulationOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: GlobalGroupTriangulationController
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: global_group_triangulation

.. autofunction:: global_triangulate_structure

.. autoclass:: GlobalStructureTriangulationOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: global_structure_triangulation

Incremental triangulation
-------------------------

The triangulators used inside incremental SfM: structures are extended and
re-triangulated as each new image is registered.

.. autoclass:: IncrementalLineTriangulatorOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: IncrementalLineTriangulator
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: incremental_line_triangulation_pipeline

.. autoclass:: IncrementalGroupTriangulatorOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: IncrementalGroupTriangulator
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: IncrementalStructureTriangulatorOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: IncrementalStructureTriangulator
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: incremental_triangulate_structure

Incremental mapper and pipeline
-------------------------------

The mapper registers images with hybrid point and line correspondences and
keeps the structures up to date; the pipeline drives it from initialization
to the final bundle adjustment.

.. autoclass:: StructureIncrementalMapperOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: StructureIncrementalMapper
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: StructureIncrementalPipelineOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: StructureIncrementalPipeline
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: StructureIncrementalPipelineStatus
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: StructureLocalBundleAdjustmentReport
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: StructureObservationManager
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

Filtering, verification and merging
-----------------------------------

Cleanup applied during and after mapping: dropping structures that lost their
support, verifying group associations, and merging duplicated 3D lines.

.. autoclass:: GroupVerificationOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: FilterGroupsStats
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: filter_group_associations

.. autofunction:: verify_vp_matches

.. autofunction:: verify_plane_matches

.. autofunction:: delete_supportless_groups

.. autoclass:: LineMergingOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: merge_fitted_lines_3d

