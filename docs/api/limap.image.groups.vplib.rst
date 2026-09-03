Vanishing point estimation
==========================

Instantiate a vanishing point detector
--------------------------------------

.. currentmodule:: limap.image.groups.vplib

.. autofunction:: get_vp_detector

.. autoclass:: DetectorOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. currentmodule:: limap.image.groups.vplib.register_vp_detector

.. autoclass:: JLinkageOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

Base interface
--------------

.. currentmodule:: limap.image.groups.vplib.base_vp_detector

.. autoclass:: BaseVPDetector
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. currentmodule:: limap.image.groups.vplib

.. autoclass:: BaseVPDetectorOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

Results
-------

A detector returns a :class:`VPResult` per image, which the frontend converts
into the 2D groups stored in the structure database.

.. autoclass:: VPResult
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: convert_vpresult_to_groups2d

.. autofunction:: convert_vpresults_to_groups2d

JLinkage
--------

The bundled J-Linkage implementation. Its parameters
(:class:`JLinkageOptions` here) are distinct from the registry-level
:class:`~limap.image.groups.vplib.register_vp_detector.JLinkageOptions` above.

.. autoclass:: JLinkageOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: JLinkage
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise
