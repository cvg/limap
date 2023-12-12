Quickstart
=================================

Some examples are prepared for users to quickly try out LIMAP for mapping and localization with lines.

------------------
Line Mapping
------------------

For this example we are using the first scene ``ai_001_001`` from `Hypersim <https://github.com/apple/ml-hypersim>`_ dataset. Download the test scene **(100 images)** with the following command:

.. code-block:: bash

    bash scripts/quickstart.sh

To run line mapping using **Line Mapping** (RGB-only) on Hypersim:

.. code-block:: bash

    python runners/hypersim/triangulation.py --output_dir outputs/quickstart_triangulation

To run line mapping using **Fitnmerge** (line mapping with available depth maps) on Hypersim:

.. code-block:: bash

    python runners/hypersim/fitnmerge.py --output_dir outputs/quickstart_fitnmerge

To run **Visualization** of the 3D line maps after the reconstruction:

.. code-block:: bash

    python visualize_3d_lines.py --input_dir outputs/quickstart_triangulation/finaltracks \
                                 # add the camera frustums with "--imagecols outputs/quickstart_triangulation/imagecols.npy"

[**Tips**] Options are stored in the config folder: ``cfgs``. You can easily change the options with the Python argument parser. Here's an example:

.. code-block:: bash

    python runners/hypersim/triangulation.py --line2d.detector.method lsd \
                                             --line2d.visualize --triangulation.IoU_threshold 0.2 \
                                             --skip_exists --n_visible_views 5

In particular, ``skip_exists`` is a very useful option to avoid running point-based SfM and line detection/description repeatedly in each pass.

Also, the combination  ``LSD detector + Endpoints NN matcher`` can be enabled with ``--default_config_file cfgs/triangulation/default_fast.yaml`` for high efficiency (while with non-negligible performance degradation).

-------------------------------------------------
Hybrid Localization with Points and Lines
-------------------------------------------------

We provide two query examples for localization from the Stairs scene in the `7Scenes <https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/>`_ Dataset, where traditional point-based methods normally struggle due to the repeated steps and lack of texture. The examples are provided in ``.npy`` files: ``runners/tests/localization/data/localization/localization_test_data_stairs_[1|2].npy``, which contains the necessary 2D-3D point and line correspondences along with the necessary configurations.

To run the examples, for instance the first one:

.. code-block:: bash

    python runners/tests/localization.py --data runners/tests/data/localization/localization_test_data_stairs_1.npy

The script will print the pose error estimated using point-only (hloc), and the pose error estimated by our hybrid point-line localization framework. In addition, two images will be created in the output folder (default to ``outputs/test_outputs/localization``) showing the inliers point and line correspondences in hybrid localization projected using the two estimated camera pose (by point-only and point+line) onto the query image with 2D point and line detections marked. 

An improved accuracy of the hybrid point-line method is expected to be observed.
