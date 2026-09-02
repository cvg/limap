Localization with points & lines
=================================

LIMAP provides a runner script to run visual localization integrating lines along with point features on the `7Scenes Dataset <https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/>`_.

Use ``runners/7scenes/localization.py`` to run the localization experiment; use the ``--help`` option and take a look at the ``cfgs/localization`` folder for all the possible options and configurations.

Alternatively, take a look at the :py:meth:`limap.runners.point_line_localization` runner or the :py:mod:`limap.estimators.absolute_pose` API to run localization with points and lines, using 2D-3D point and line correspondences directly.

------------------------------------
Example on 7Scenes
------------------------------------

Here we provide a tutorial for the visual localization experiment from the paper `3D Line Mapping Revisited <https://arxiv.org/abs/2303.17504>`_ (in CVPR 2023), specifically on the *Stairs* scene of the `7Scenes <https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/>`_ dataset.

Follow `hloc <https://github.com/cvg/Hierarchical-Localization/tree/master/hloc/pipelines/7Scenes>`_, download the images from the project page:

.. code-block:: bash

    export dataset=datasets/7scenes
    for scene in stairs; \
    do wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/$scene.zip -P $dataset \
    && unzip $dataset/$scene.zip -d $dataset && unzip $dataset/$scene/'*.zip' -d $dataset/$scene; done

Download the SIFT SfM models and DenseVLAD image pairs, courtesy of Torsten Sattler:

.. code-block:: bash

    function download {
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf /tmp/cookies.txt
    unzip $2 -d $dataset && rm $2;
    }
    download 1cu6KUR7WHO7G4EO49Qi3HEKU6n_yYDjb $dataset/7scenes_sfm_triangulated.zip
    download 1IbS2vLmxr1N0f3CEnd_wsYlgclwTyvB1 $dataset/7scenes_densevlad_retrieval_top_10.zip

Download the rendered depth maps, courtesy of Eric Brachmann for `DSAC* <https://github.com/vislearn/dsacstar>`_:

.. code-block:: bash

    wget https://heidata.uni-heidelberg.de/api/access/datafile/4037 -O $dataset/7scenes_rendered_depth.tar.gz
    mkdir $dataset/depth/
    tar xzf $dataset/7scenes_rendered_depth.tar.gz -C $dataset/depth/ && rm $dataset/7scenes_rendered_depth.tar.gz

The download could take some time as the compressed data files contain all 7Scenes. You could delete the other scenes since for this example we are only using the Stairs scene.

Now, run the localization pipeline with points and lines:

.. code-block:: bash

    python runners/7scenes/localization.py --dataset $dataset -s stairs --skip_exists

It is also possible to use the rendered depth with the ``--use_dense_depth`` flag, in which case the 3D line map is built from the depth maps instead of triangulation:

.. code-block:: bash

    python runners/7scenes/localization.py --dataset $dataset -s stairs --skip_exists --use_dense_depth

Add ``--use_points_only`` to run the point-only baseline. The runner also runs `hloc <https://github.com/cvg/Hierarchical-Localization/tree/master/hloc/pipelines/7Scenes>`_ for extracting and matching the feature points and for comparing the results. The evaluation result is printed in the terminal after localization is finished. You can also evaluate an existing result ``.txt`` file with the ``--eval_file`` option.

Uncalibrated queries
--------------------

We also support localization without knowing the query intrinsics, by adding
the ``--uncalibrated`` flag:

.. code-block:: bash

    python runners/7scenes/localization.py --dataset $dataset -s stairs --skip_exists --uncalibrated

The focal length is then estimated jointly with the pose from the same point
and line correspondences, while the map keeps its own calibration. The
principal point is taken from the input camera, and the runner reports the
distribution of the estimated focal lengths once localization is finished.
