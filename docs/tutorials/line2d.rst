Line detection, description and matching 
============================================

We provide interfaces for running line detection, description and matching with supported modules in LIMAP. 

-----------------------------------------------------
Minimal example on line detection and description
-----------------------------------------------------
To use the interface you need to construct a :class:`limap.base.CameraView` instance for each image. Since the intrinsic and extrinsic parameters are not needed, you can leave it uninitialized. Here shows an minimal example on running line detection and description with `DeepLSD <https://github.com/cvg/DeepLSD>`_ and `SOLD2 <https://github.com/cvg/SOLD2>`_ on an image `example.png`:

.. code-block:: python

    import limap.util.config
    import limap.base
    import limap.line2d
    cfg_detector = limap.util.config.load_config("cfgs/examples/line2d_detector.yaml") # example config file
    cfg_detector["line2d"]["detector"]["method"] = "deeplsd"
    cfg_detector["line2d"]["extractor"]["method"] = "sold2"
    view = limap.base.CameraView(limap.base.Camera(0), "example.png") # initiate an limap.base.CameraView instance for detection. You can specify the height and width to resize into in the limap.base.Camera instance at initialization.
    detector = limap.line2d.get_detector(cfg_detector) # get a line detector
    segs = detector.detect(view) # detection
    desc = detector.extract(view, segs) # description

-----------------------------------------------------
Minimal example on line matching 
-----------------------------------------------------
And here shows an minimal example on running line matcher with `SOLD2 <https://github.com/cvg/SOLD2>`_. Note that the type of matcher should align with the type of extractors in terms of compatibility.

.. code-block:: python

    global desc1, desc2 # read in some extracted descriptors
    import limap.util.config
    import limap.base
    import limap.line2d
    cfg_detector = limap.util.config.load_config("cfgs/examples/line2d_detector.yaml") # example config file
    cfg_detector["line2d"]["extractor"]["method"] = "sold2"
    extractor = limap.line2d.get_detector(cfg_detector) # get a line extractor
    cfg_matcher = limap.util.config.load_config("cfgs/examples/line2d_match.yaml") # example config file
    cfg_detector["line2d"]["matcher"]["method"] = "sold2"
    matcher = limap.line2d.get_matcher(cfg_detector, extractor) # initiate a line matcher
    matches = matcher.match_pair(desc1, desc2) # matching

----------------------------------------------------
Multiple images
----------------------------------------------------
To run line detection, description and matching on multiple images, one can resort to the following API:

* :py:meth:`limap.runners.functions.compute_2d_segs`
* :py:meth:`limap.runners.functions.compute_exhaustive_matches`
* :py:meth:`limap.runners.functions.compute_matches`

The outputs detections, descriptions and matches will be saved into the corresponding output folders.



