Line detection, description and matching 
============================================

We support modular interfaces for running line detection, description and matching in LIMAP. In addition, we also provide minimal examples here and a mimimal test scripts at ``runners/tests/line2d.py``.

-----------------------------------------------------
Minimal example on line detection and description
-----------------------------------------------------
To use the interface you need to construct a :class:`limap.base.CameraView` instance for each image. Since the intrinsic and extrinsic parameters are not needed, you can leave it uninitialized. Here shows an minimal example on running line detection and description with `DeepLSD <https://github.com/cvg/DeepLSD>`_ and `SOLD2 <https://github.com/cvg/SOLD2>`_ on an image `example.png`:

.. code-block:: python

    import limap.util.config
    import limap.base
    import limap.line2d
    view = limap.base.CameraView("example.png") # initiate an limap.base.CameraView instance for detection. 
    # You can specify the height and width to resize into in the limap.base.Camera instance at initialization (as the example below).
    # view = limap.base.CameraView(limap.base.Camera("SIMPLE_PINHOLE", hw=(400, 400)), "example.png")
    detector = limap.line2d.get_detector({"method": "deeplsd", "skip_exists": False}) # get a line detector
    segs = detector.detect(view) # detection
    extractor = limap.line2d.get_extractor({"method": "sold2", "skip_exists": False}) # get a line descriptor extractor
    desc = extractor.extract(view, segs) # description

-----------------------------------------------------
Minimal example on line matching 
-----------------------------------------------------
And here shows a minimal example on running line matcher with `SOLD2 <https://github.com/cvg/SOLD2>`_. Note that the type of matcher should align with the type of extractors in terms of compatibility.

.. code-block:: python

    global desc1, desc2 # read in some extracted descriptors
    import limap.util.config
    import limap.base
    import limap.line2d
    extractor = limap.line2d.get_extractor({"method": "sold2", "skip_exists": False}) # get a line extractor
    matcher = limap.line2d.get_matcher({"method": "sold2", "skip_exists": False, "n_jobs": 1, "topk": 0}, extractor) # initiate a line matcher
    matches = matcher.match_pair(desc1, desc2) # matching

-----------------------------------------------------
Visualization
-----------------------------------------------------
Here shows an example on visualizing the detections:

.. code-block:: python

    global view # the limap.base.CameraView instance used for detection
    import cv2
    import limap.visualize
    img = view.read_image(set_gray = False)
    img = limap.visualize.draw_segments(img, segs, (0, 255, 0))
    cv2.imshow("detetions", img)
    cv2.waitKey(0)

----------------------------------------------------
Multiple images
----------------------------------------------------
To run line detection, description and matching on multiple images, one can resort to the following API:

* :py:meth:`limap.runners.functions.compute_2d_segs`
* :py:meth:`limap.runners.functions.compute_exhaustive_matches`
* :py:meth:`limap.runners.functions.compute_matches`

The outputs detections, descriptions and matches will be saved into the corresponding output folders.


