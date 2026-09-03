Image description and association
=================================

The frontend steps that run over a whole set of images: creating the two
databases, describing every image with all enabled feature types, and
associating the images pairwise. They dispatch to the per-feature operations
documented on the pages beside this one.

.. currentmodule:: limap.image

.. autofunction:: create_empty_databases

.. autoclass:: ImageDescriptionOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: image_description

.. autoclass:: ImageAssociationOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: image_association
