Line mapping 
=================================

-----------------------------------------
Line mapping on a set of posed images
-----------------------------------------

As one of the main features, LIMAP supports line reconstruction on a set of posed images, optionally with assistance of the point-based SfM model or the depth map. We support to use the poses directly from `COLMAP <https://colmap.github.io/>`_, `Bundler <https://www.cs.cornell.edu/~snavely/bundler/>`_ or `VisualSfM <http://ccwu.me/vsfm/index.html>`_. One can also use customized poses and intrinsics with the main interface :py:meth:`limap.runners.line_triangulation` API by constructing a :class:`limap.base.ImageCollection` instance as the input.

To give some references, one can construct the :class:`limap.base.ImageCollection` instance as in `Example for Hypersim <https://github.com/cvg/limap/blob/main/runners/hypersim/loader.py#L34-L41>`_ or in `Example for COLMAP <https://github.com/cvg/limap/blob/main/limap/pointsfm/colmap_reader.py#L31-L47>`_.

Here shows a minimal example on running line mapping on the constructed input:

.. code-block:: python

    global imagecols # construct your own imagecols by setting up the intrinsics and extrinsics
    import limap.util.config
    import limap.runners
    import limap.visualize
    cfg = limap.util.config.load_config("cfgs/triangulation/default.yaml") # load the example config
    cfg["output_dir"] = "outputs/TBA" # specify an output directory
    linetracks = limap.runners.line_triangulation(cfg, imagecols, neighbors=None, ranges=None) # run mapping, you can also specify visual neighboring information if applicable (for example, in a video stream you can use the sequential timestamps to construct visual neighbors)
    # visualize
    visualizer = limap.visualize.Open3DTrackVisualizer(linetracks)
    visualizer.report()
    visualizer.vis_reconstruction(imagecols, n_visibile_views=4)

All the output and intermediate data will be stored at the specified output directory. 

---------------------------------------------------------------------
Line mapping on a set of unposed images by running COLMAP first
---------------------------------------------------------------------

To run the line mapping on a set of unposed images, we now suggest to run COLMAP first to get the poses. Specifically, first pose the images with `COLMAP <https://colmap.github.io>`_ following the guide `here <https://colmap.github.io/cli.html>`_. Then, use the `COLMAP interface <https://github.com/cvg/limap/blob/main/runners/colmap_triangulation.py>`_ in LIMAP to build 3D line maps by:

.. code-block:: bash

    python runners/colmap_triangulation.py -c ${CONFIG_FILE} -a ${COLMAP_FOLDER} --output_dir ${OUTPUT_DIR}

And the line maps will be stored in the specified output folder. To use point SfM to improve robustness, add additional option ``--triangulation.use_pointsfm.enable --triangulation.use_pointsfm.colmap_folder ${COLMAP_FOLDER}`` in the end of the command.

The interface for Bundler, VisualSfM and other datasets are all stored in the ``runners`` folder, building on top of the main interface :py:meth:`limap.runners.line_triangulation` API.

-----------------------------------------
Using auxiliary depth maps
-----------------------------------------

To use auxiliary depth information, we can run line reconstruction with the :py:meth:`limap.runners.line_fitnmerge` API. One needs to write a customized depth loader inheriting :class:`limap.base.depth_reader_base.BaseDepthReader`. An example on Hypersim dataset is provided `here <https://github.com/cvg/limap/blob/main/runners/hypersim/loader.py#L10-L19>`_. Each depth loader consists of the file name of the depth image and its width, height and other information, along with the method for loading.


