from ._pl_estimate_absolute_pose import _pl_estimate_absolute_pose

def pl_estimate_absolute_pose(cfg, l3ds, l3d_ids, l2ds, p3ds, p2ds, camera, campose=None,
                     inliers_line=None, inliers_point=None, jointloc_cfg=None, silent=True, logger=None):
    """
    Estimate absolute camera pose of a image from matched 2D-3D line and point correspondences.

    :param cfg:             Localization config, fields refer to "localization" section in :file:`cfgs/localization/default.yaml`
    :type cfg:              dict
    :param l3ds:            List of :class:`limap.base.Line3d`
    :type l3ds:             list
    :param l3d_ids:         List of integer indices into `l3ds` for match of each :class:`limap.base.Line32` in `l2ds`, same length of `l2ds`
    :type l3d_ids:          list
    :param l2ds:            List of :class:`limap.base.Line2d`, matched 2d lines, same length of `l3d_ids`
    :type l2ds:             list
    :param p3ds:            List of :class:`np.array`, matched 3D points, same length of `p2ds`
    :type p3ds:             list
    :param p2ds:            List of :class:`np.array`, matched 2D points, same length of `p3ds`
    :type p2ds:             list
    :param camera:          Camera of the query image
    :type camera:           :class:`limap.base.Camera`
    :param campose:         Initial camera pose, only useful for pose refinement (when cfg["ransac"]["method"] is None)
    :type campose:          :class:`limap.base.CameraPose`, optional
    :param inliers_line:    List of integer indices of line inliers, only useful for pose refinement
    :type inliers_line:     list, optional
    :param inliers_point:   List of integer indices of point inliers, only useful for pose refinement
    :type inliers_point:    list, optional
    :param jointloc_cfg:    Config for joint optimization, fields refer to :class:`limap.optimize.LineLocConfig`, pass ``None`` for default
    :type jointloc_cfg:     dict, optional
    :param silent:          Turn off to print logs during Ceres optimization
    :type silent:           bool, optional
    :param logger:          Logger to print logs for information
    :type logger:           :class:`logging.Logger`, optional

    :rtype: Tuple <:class:`limap.base.CameraPose`, :class:`limap.estimators.RansacStatistics`>
    :return: Estimated pose and ransac statistics.
    """
    return _pl_estimate_absolute_pose(cfg, l3ds, l3d_ids, l2ds, p3ds, p2ds, camera, campose=campose,
                                      inliers_line=inliers_line, inliers_point=inliers_point, jointloc_cfg=jointloc_cfg, silent=silent, logger=logger)

