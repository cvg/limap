Line detection, description and matching
========================================

Line segments are detected, optionally described, and matched across image
pairs. Methods are selected by name through the registries below; each
returns an implementation of the base interfaces.

Instantiate a line detector / descriptor
----------------------------------------

.. module:: limap.image.line

.. autofunction:: get_detector

.. autofunction:: get_extractor

.. autofunction:: get_uncertainty2d

.. autoclass:: DetectorOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: ExtractorOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

Instantiate a line matcher
--------------------------

.. autofunction:: get_matcher

.. autoclass:: MatcherOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. currentmodule:: limap.image.line.register_matcher

.. autoclass:: SuperGlueMatcherOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: DenseMatcherOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

Base interfaces
---------------

Implemented by every method under ``limap/image/line/``; a new method needs a
subclass of these plus one branch in the registry above.

.. currentmodule:: limap.image.line.base_detector

.. autoclass:: BaseDetector
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: BaseDetectorOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. currentmodule:: limap.image.line.base_matcher

.. autoclass:: BaseMatcher
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autoclass:: BaseMatcherOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

Detection and matching over a set of images
-------------------------------------------

The batch helpers the pipelines call, writing their output into the structure
database.

.. currentmodule:: limap.image.line

.. autoclass:: LineDetectionOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: line_detection

.. autoclass:: LineMatcherOptions
   :members:
   :undoc-members:
   :special-members: __init__
   :member-order: groupwise

.. autofunction:: line_matching

.. autofunction:: exhaustive_line_matching
