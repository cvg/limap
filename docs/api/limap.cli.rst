limap.cli package
=================

Command line entry points, each run with ``python -m limap.cli.<name>``.
A CLI module parses its arguments together with the YAML configuration under
``cfgs/`` (see :func:`limap.util.config.load_config`), converts them into the
``Options`` dataclass of the corresponding runner and calls it. They are thin
by design: the pipelines themselves live in :doc:`limap.runners`.

Triangulation from a COLMAP model
---------------------------------

Complete pipelines that run the frontend and the triangulation in one go, on
a scene whose camera poses are already known.

Points and lines
^^^^^^^^^^^^^^^^

.. automodule:: limap.cli.automatic_point_line_triangulation
   :members:
   :undoc-members:
   :exclude-members: parse_config, parse_args
   :show-inheritance:

Points, lines, groups and wireframe
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: limap.cli.automatic_structure_triangulation
   :members:
   :undoc-members:
   :exclude-members: parse_config, parse_args
   :show-inheritance:

Frontend
--------

Detection, description and matching alone, writing the two databases that the
triangulation modules below consume.

.. automodule:: limap.cli.structure_frontend
   :members:
   :undoc-members:
   :exclude-members: parse_config, parse_args
   :show-inheritance:

Triangulation from a structure database
---------------------------------------

The second half of the automatic pipelines, taking the frontend output
instead of computing it.

Global line triangulation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: limap.cli.global_line_triangulation
   :members:
   :undoc-members:
   :exclude-members: parse_config, parse_args
   :show-inheritance:

Global structure triangulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: limap.cli.global_structure_triangulation
   :members:
   :undoc-members:
   :exclude-members: parse_config, parse_args
   :show-inheritance:

Incremental line triangulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: limap.cli.incremental_line_triangulation
   :members:
   :undoc-members:
   :exclude-members: parse_config, parse_args
   :show-inheritance:

Incremental structure triangulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: limap.cli.incremental_structure_triangulation
   :members:
   :undoc-members:
   :exclude-members: parse_config, parse_args
   :show-inheritance:

Incremental SfM
---------------

.. automodule:: limap.cli.structure_incremental_sfm
   :members:
   :undoc-members:
   :exclude-members: parse_config, parse_args
   :show-inheritance:
