from ._pl_estimate_absolute_pose import _pl_estimate_absolute_pose

def pl_estimate_absolute_pose(cfg, l3ds, l3d_ids, l2ds, p3ds, p2ds, camera, campose=None,
                     inliers_line=None, inliers_point=None, jointloc_cfg=None, silent=True, logger=None):
    """
    Estimate absolute camera pose of a image from matched 2D-3D line and point correspondences.

    Args:
        cfg (dist): Localization config, fields refer to "localization" section in :file:`cfgs/localization/default.yaml`
        l3ds (list): List of :class:`limap.base.Line3d`
        l3d_ids (list): List of integer indices into `l3ds` for match of each :class:`limap.base.Line3d` in `l2ds`, same length of `l2ds`
        l2ds (list): List of :class:`limap.base.Line2d`, matched 2d lines, same length of `l3d_ids`
        p3ds (list): List of :class:`np.array`, matched 3D points, same length of `p2ds`
        p2ds (list): List of :class:`np.array`, matched 2D points, same length of `p3ds`
        camera (:class:`limap.base.Camera`): Camera of the query image
        campose (:class:`limap.base.CameraPose`, optional): Initial camera pose, only useful for pose refinement (when ``cfg["ransac"]["method"]`` is None)
        inliers_line (list, optional): List of integer indices of line inliers, only useful for pose refinement
        inliers_point (list, optional): List of integer indices of point inliers, only useful for pose refinement
        jointloc_cfg (dict, optional): Config for joint optimization, fields refer to :class:`limap.optimize.LineLocConfig`, pass ``None`` for default
        silent (bool, optional): Turn off to print logs during Ceres optimization
        logger (:class:`logging.Logger`): Logger to print logs for information

    Returns:
        Tuple <:class:`limap.base.CameraPose`, :class:`limap.estimators.RansacStatistics`>: Estimated pose and ransac statistics.
    """
    return _pl_estimate_absolute_pose(cfg, l3ds, l3d_ids, l2ds, p3ds, p2ds, camera, campose=campose,
                                      inliers_line=inliers_line, inliers_point=inliers_point, jointloc_cfg=jointloc_cfg, silent=silent, logger=logger)

