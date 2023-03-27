Localization with Points & Lines
=================================

Currently, runner scripts are provided to run visual localization integrating line along with point features on the following Datasets: 

* `7Scenes <https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/>`_
* `Cambridge Landmarks <https://www.repository.cam.ac.uk/handle/1810/251342>`_
* `InLoc <http://www.ok.sc.e.titech.ac.jp/INLOC/>`_

Please follow hloc's guide for downloading and preparing Cambridge and 7Scenes dataset:

* `7Scenes <https://github.com/cvg/Hierarchical-Localization/tree/master/hloc/pipelines/7Scenes>`_
* `Cambridge <https://github.com/cvg/Hierarchical-Localization/tree/master/hloc/pipelines/Cambridge>`_

Use ``runners/<dataset>/localization.py`` to run localization experiments on these supported datasets, use ``--help`` option and take a look at ``cfgs/localization`` folder for all the possible options and configurations.

Alternatively, take a look at the :py:meth:`limap.estimators.absolute_pose.pl_estimate_absolute_pose` API or the :py:meth:`limap.runners.line_localization.line_localization` runner to run localization with points and lines, using 2D-3D point and line correspondences directly.