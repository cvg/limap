Vanishing points, planes and groups
====================================

Beyond points and lines, LIMAP reconstructs **groups**: structured primitives
that many features share. A group is a vanishing point, a plane, or one of the
parametric primitives (sphere, cylinder, ellipsoid, cuboid, cone). The points
and lines associated with a group are constrained to it during bundle
adjustment, which is what makes the reconstruction *holistic*.

The detectors live under :py:mod:`limap.image.groups` and follow the same
registry pattern as the line detectors in :doc:`line2d`.

-----------------------------------------------------
Vanishing point detection
-----------------------------------------------------

Vanishing points are estimated from the 2D lines detected on an image:

.. code-block:: python

    import limap.image.groups.vplib as vplib

    # lines: list[limap.geometry.Line2d] for one image (see line2d)
    detector = vplib.get_vp_detector("jlinkage", vplib.DetectorOptions())
    vpresult = detector.detect_vp(lines)

`JLinkage <https://github.com/B1ueber2y/JLinkage>`_ and
`Progressive-X <https://github.com/danini/progressive-x>`_ (separate
installation needed) are supported.

-----------------------------------------------------
Plane detection
-----------------------------------------------------

Plane segmentation is monocular: it runs on a single image, with neither depth
nor pose required.

.. code-block:: python

    from pathlib import Path
    import limap.image.groups.planelib as planelib

    detector = planelib.get_plane_detector(
        "pxwplanar", planelib.DetectorOptions()
    )
    plane_mask = detector.detect_plane_mask(Path("example.png"))

The detector is `pxwplanar
<https://github.com/alpayozkan/PixelwisePlanarity>`_, pulled in by
``requirements.txt``; its weights are downloaded from Hugging Face on first
use. Segmentation parameters are exposed through
``planelib.PxwPlanarOptions``.

-----------------------------------------------------
Over many images
-----------------------------------------------------

The batch helpers mirror the line ones and write their results into the
structure database:

* :py:func:`limap.image.groups.vp_detection`
* :py:func:`limap.image.groups.plane_detection`
* :py:func:`limap.image.groups.group_description`

In the mapping pipelines, groups are switched on with the ``use_groups``
option (enabled by default; see ``cfgs/structure_triangulation/default.yaml``,
and ``default_cpu.yaml`` for a configuration that disables them).
