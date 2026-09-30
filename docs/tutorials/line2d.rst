Line detection, description and matching
============================================

LIMAP provides modular interfaces for line detection, description and matching under :py:mod:`limap.image.line`.

The detectors, descriptors and matchers below are external methods; their BibTeX
entries are collected `here <https://github.com/cvg/limap/blob/main/misc/citations.md>`_. Please cite the ones you use.

-----------------------------------------------------
Minimal example on line detection and description
-----------------------------------------------------

Detection and description operate directly on an image file path. Here is a minimal example running `DeepLSD <https://github.com/cvg/DeepLSD>`_ detection and `SOLD2 <https://github.com/cvg/SOLD2>`_ description on an image ``example.png``:

.. code-block:: python

    from pathlib import Path
    import limap.image.line as line2d

    image_path = Path("example.png")
    detector = line2d.get_detector("deeplsd", line2d.DetectorOptions())
    segs = detector.detect(image_path)          # (N, 5): x1, y1, x2, y2, score
    extractor = line2d.get_extractor("sold2", line2d.ExtractorOptions())
    desc = extractor.extract(image_path, segs)  # descriptors for the detected segments

A detector that also describes can do both in one network pass. `UPAL <https://github.com/francois141/upal>`_ additionally predicts keypoints, so the whole frontend can run off a single pass per image with ``use_joint_point_line_detection`` (see ``cfgs/structure_triangulation/upal.yaml``):

.. code-block:: python

    detector = line2d.get_detector("upal", line2d.DetectorOptions())
    segs, desc = detector.detect_and_extract(image_path)

-----------------------------------------------------
Minimal example on line matching
-----------------------------------------------------

The matcher type must be compatible with the extractor. Here is a minimal example running the `SOLD2 <https://github.com/cvg/SOLD2>`_ matcher on two sets of descriptors:

.. code-block:: python

    import limap.image.line as line2d

    # desc1, desc2: descriptors extracted from two images (see above)
    extractor = line2d.get_extractor("sold2", line2d.ExtractorOptions())
    matcher = line2d.get_matcher("sold2", line2d.MatcherOptions(), extractor)
    matches = matcher.match_pair(desc1, desc2)

-----------------------------------------------------
Joint point-line matching
-----------------------------------------------------

`GlueStick <https://github.com/cvg/GlueStick>`_ and its lighter successor
`LightGlueStick <https://github.com/aubingazhib/LightGlueStick>`_ match points
and lines in a single pass, so the matcher above throws away half of what the
network already computed. :py:mod:`limap.image.joint_point_line` keeps both
halves: it describes an image from the line descriptors *and* the point
features that fed the COLMAP database, and returns the point matches alongside
the line matches.

.. code-block:: python

    from limap.image.joint_point_line import (
        JointPointLineMatcherOptions,
        get_joint_matcher,
    )

    options = JointPointLineMatcherOptions()
    matcher = get_joint_matcher(options.method, options.matching_options)
    desc1 = matcher.describe(descinfo_folder, feature_path, img_id1, name1)
    desc2 = matcher.describe(descinfo_folder, feature_path, img_id2, name2)
    result = matcher.match_pair(desc1, desc2)
    result.point_matches0  # per keypoint of image 1, its match or -1
    result.line_matches    # (N, 2) pairs of line indices

In a pipeline this replaces both ``point_matcher`` and ``line_matcher``; turn
it on with ``use_joint_point_line_matcher``, as in
``cfgs/structure_triangulation/gluestick_joint.yaml`` and
``cfgs/structure_triangulation/lightgluestick_joint.yaml``. The point matches
index the keypoints of the COLMAP database, so the point detector has to be the
one the joint matcher was trained with -- SuperPoint, for both of these. They
also read the same ``wireframe`` line descriptors, so switching between them
costs nothing but the matching step.

-----------------------------------------------------
Visualization
-----------------------------------------------------

Here is an example on visualizing the detected segments:

.. code-block:: python

    import cv2
    import limap.visualize

    image = cv2.imread("example.png")
    # segs is (N, 5); reshape the endpoints to (2, 2) per line
    lines = [seg[:4].reshape(2, 2) for seg in segs]
    image = limap.visualize.draw_2d_lines(image, lines, (0, 255, 0))
    cv2.imshow("detections", image)
    cv2.waitKey(0)

----------------------------------------------------
Multiple images
----------------------------------------------------

To run line detection, description and matching over many images at once, use the batch helpers in :py:mod:`limap.image.line`:

* :py:func:`limap.image.line.line_detection`
* :py:func:`limap.image.line.line_matching`
* :py:func:`limap.image.line.exhaustive_line_matching`

The output detections, descriptions and matches are saved into the corresponding output folders.
