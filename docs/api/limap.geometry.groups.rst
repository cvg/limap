Group primitives
================

"Groups" is the umbrella term for vanishing points, planes and parametric
primitives (sphere, cylinder, ellipsoid, cuboid, cone). Each type is
identified by a :class:`~limap.geometry.GroupType` and parameterized by a
flat vector of parameters, whose layout and length depend on the type. The
helpers below query, initialize and normalize those parameters.

.. autoclass:: limap.geometry.GroupType
   :members:
   :undoc-members:
   :member-order: groupwise

.. autofunction:: limap.geometry.get_num_params_in_2d_by_group_type

.. autofunction:: limap.geometry.get_num_params_in_3d_by_group_type

.. autofunction:: limap.geometry.get_default_group_params_2d

.. autofunction:: limap.geometry.get_default_group_params_3d

.. autofunction:: limap.geometry.initialize_group_params

.. autofunction:: limap.geometry.normalize_group_params_3d

.. autofunction:: limap.geometry.check_group_params_3d
