Quickstart
=================================

Some examples are prepared for users to quickly try out LIMAP for mapping and localization with lines.

------------------
Line Mapping
------------------

For this example we are using the first scene `ai_001_001` from `Hypersim <https://github.com/apple/ml-hypersim>`_ dataset. Download the test scene **(100 images)** with the following command:

.. code-block:: bash

    bash scripts/quickstart.sh

To run line mapping using **Line Triangulation** on Hypersim:

.. code-block:: bash

    python runners/hypersim/triangulation.py --output_dir outputs/quickstart_triangulation

To run line mapping using **Fit&Merge** on Hypersim:

.. code-block:: bash

    python runners/hypersim/fitnmerge.py --output_dir outputs/quickstart_fitnmerge

[**Tips**] Options are stored in the config folder: ``cfgs``. You can easily change the options with the Python argument parser. Here's an example:

.. code-block:: bash

    python runners/hypersim/triangulation.py --line2d.detector.method lsd \
                                             --line2d.visualize --triangulation.IoU_threshold 0.2 \
                                             --skip_exists --n_visible_views 5

In particular, ``skip_exists`` is a very useful option to avoid running point-based SfM and line detection/description repeatedly in each pass.

------------------------------------
Localization with Points and Lines
------------------------------------

The improvement that could be achieved by integrating lines along with point features for visual localization is best demonstrated using the Stairs scene from the `7Scenes <https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/>`_ dataset, where traditionally point-based localization struggles in performance.

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

Now, to run the localization pipeline with points and lines. As shown above, the configs are passed in as command line arguments.

.. code-block:: bash

    python runners/7scenes/localization.py --dataset $dataset -s stairs --skip_exists \
                                           --localization.optimize.loss_func TrivialLoss \
                                           --localization.optimize.normalize_weight

It is also possible to use the rendered depth with the `--use_dense_depth` flag, in which case the 3D line map will be built using LIMAP's Fit&Merge (enable merging by adding `--merging.do_merging`) utilities instead of triangulation.

.. code-block:: bash

    python runners/7scenes/localization.py --dataset $dataset -s stairs --skip_exists \
                                           --use_dense_depth \
                                           --localization.optimize.loss_func TrivialLoss \
                                           --localization.optimize.normalize_weight

The runner scripts will also run `hloc <https://github.com/cvg/Hierarchical-Localization/tree/master/hloc/pipelines/7Scenes>`_ for extracting and matching the feature points and for compariing the results.