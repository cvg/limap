Visualization
=================================

We provide the visualization interface `here <https://github.com/cvg/limap/blob/main/visualize_3d_lines.py>`_ to visualize the reconstructed line maps. Optionally, the camera frustums can also be visualized with an additional :class:`limap.base.ImageCollection` instance as the input. An example visualization command could be:

.. code-block:: bash

    python visualize_3d_lines.py --input_dir outputs/quickstart_triangulation/finaltracks \
                                 # add the camera frustums with "--imagecols outputs/quickstart_triangulation/imagecols.npy"

We use `Open3D <http://www.open3d.org/>`_ as the backend for the visualization.

