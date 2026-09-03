limap.util package
==================

Shared utilities: the basic value types, the KD-tree used across the
pipelines, configuration loading and file I/O.

Types
-----

.. py:data:: limap.util.Color
   :type: tuple[float, float, float]

   An RGB color triplet. The 2D helpers use the ``[0, 255]`` range
   expected by OpenCV, the 3D ones the ``[0, 1]`` range expected by Open3D.

.. py:data:: limap.util.Ranges
   :type: tuple[numpy.ndarray, numpy.ndarray]

   An axis-aligned 3D bounding box, as its minimum and maximum corner.

.. autoclass:: limap.util.KDTree
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

Configuration
-------------

YAML configuration files under ``cfgs/`` are resolved with
``base_config_file`` inheritance and then overridden by unknown
``--key.subkey value`` command line arguments.

.. automodule:: limap.util.config
   :members:
   :undoc-members:
   :show-inheritance:

File I/O
--------

.. automodule:: limap.util.io
   :members:
   :undoc-members:
   :show-inheritance:

Model weights
-------------

.. automodule:: limap.util.model_weights
   :members:
   :undoc-members:
   :show-inheritance:
