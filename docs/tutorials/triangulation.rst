Line mapping 
=================================

As one of the main features, LIMAP supports line reconstruction on a set of posed images, optionally with assistance of the point-based SfM model or the depth map. We currently support to use the poses from `COLMAP <https://colmap.github.io/>`_, `Bundler <https://www.cs.cornell.edu/~snavely/bundler/>`_ and `VisualSfM <http://ccwu.me/vsfm/index.html>`_. One can also use customized poses and intrinsics with the main interface :py:meth:`limap.runners.line_triangulation` API by constructing a :class:`limap.base.ImageCollection` instance as the input.

-----------------------------------------
Line mapping on a set of images
-----------------------------------------

Specifically, to run the line mapping on a set of images, first pose the images with `COLMAP <https://colmap.github.io>`_ following the guide `here <https://colmap.github.io/cli.html>`_. Then, use the `COLMAP interface <https://github.com/cvg/limap/blob/main/runners/colmap_triangulation.py>`_ in LIMAP to build 3D line maps by:

.. code-block:: bash

    python runners/colmap_triangulation.py -c ${CONFIG_FILE} -a ${COLMAP_FOLDER} --output_path ${OUTPUT_PATH}

And the line maps will be stored in the specified output folder. To use point SfM to improve robustness, add additional option ``--triangulation.usepointsfm.enable --triangulation.use_pointsfm.colmap_folder ${COLMAP_FOLDER}`` in the end of the command.

The interface for Bundler, VisualSfM and other datasets are all stored in the ``runners`` folder, building on top of the main interface :py:meth:`limap.runners.line_triangulation` API.

-----------------------------------------
Using auxiliary depth maps
-----------------------------------------

To use auxiliary depth information, we can run line reconstruction with the :py:meth:`limap.runners.line_fitnmerge` API. One needs to write a customized depth loader inheriting :class:`limap.base.depth_reader_base.BaseDepthReader`. An example on Hypersim dataset is provided `here <https://github.com/cvg/limap/blob/main/runners/hypersim/loader.py#L10-L19>`_. Each depth loader consists of the file name of the depth image and its width, height and other information, along with the method for loading.


