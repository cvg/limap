limap.evaluation package
========================

Evaluation of a reconstructed line map against a ground truth: a mesh, a
point cloud, or a set of reference 3D lines. The mesh and point cloud
evaluators share an interface -- point and line distances to the ground-truth
surface, and the inlier and outlier parts of a segment under a threshold.

.. currentmodule:: limap.evaluation

Evaluate w.r.t. a mesh
----------------------

.. autoclass:: MeshEvaluator
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

Evaluate w.r.t. a point cloud
-----------------------------

Distances are queried through a K-D tree, which can be built once and then
saved and reloaded.

.. autoclass:: PointCloudEvaluator
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

Evaluate w.r.t. reference lines
-------------------------------

Recall of the reference lines covered by the reconstruction, and of the
reconstructed lines supported by a reference.

.. autoclass:: RefLineEvaluator
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise
