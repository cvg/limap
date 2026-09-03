limap.image package
===================

The 2D frontend: detection, description, matching and association of points,
lines and groups, together with the operations that write them into the
COLMAP database and the LIMAP structure database.

.. toctree::
   :maxdepth: 3
   :caption: Module Contents:

   limap.image.point
   limap.image.line
   limap.image.groups
   limap.image.vplib
   limap.image.planelib
   limap.image.dense_matcher

Image description and association
---------------------------------

End-to-end frontend steps running all enabled feature types over a set of
images.

.. automodule:: limap.image.specs
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: limap.image.process
   :members:
   :undoc-members:
   :show-inheritance:
