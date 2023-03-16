from _limap import _ceresbase
import limap.optimize as _optimize
import limap.estimators as _estimators
import limap.base as _base
import numpy as np

def pl_estimate_absolute_pose(cfg, l3ds, l3d_ids, l2ds, p3ds, p2ds, camera, campose=None, 
                     inliers_line=None, inliers_point=None, jointloc_cfg=None, silent=True, logger=None):
    """
    Estimate absolute camera pose of a image from matched 2D-3D line and point correspondences.

    :param cfg:             dict, fields refer to "localization" section in `cfgs/localization/default.yaml`
    :param l3ds:            iterable of limap.base.Line3d
    :param l3d_ids:         iterable of int, indices into l3ds for match of each Line2d in `l2ds`, same length of `l2ds`
    :param l2ds:            iterable of limap.base.Line2d, matched 2d lines, same length of `l3d_ids`
    :param p3ds:            iterable of np.array, matched 3D points, same length of `p2ds`
    :param p2ds:            iterable of np.array, matched 2D points, same length of `p3ds`
    :param camera:          limap.base.Camera, camera of the query image
    :param campose:         limap.base.CameraPose (optional), initial camera pose, only useful for pose refinement 
                            (when cfg["ransac"]["method"] is None)
    :param inliers_line:    iterable of int (optional), indices of line inliers, only useful for pose refinement
    :param inliers_point:   iterable of int (optional), indices of point inliers, only useful for pose refinement
    :param jointloc_cfg:    dict (optional), fields corresponding to limap.optimize.LineLocConfig, pass None for default
    :param silent:          boolean (optional), turn off to print logs during Ceres optimization
    :param logger:          logging.Logger (optional), print logs for information

    :return: tuple<limap.base.CameraPose, limap.estimators.RansacStatistics>, estimated pose and ransac statistics.
    """ 
    if jointloc_cfg is None:
        jointloc_cfg = {}
        jointloc_cfg['loss_function'] = _ceresbase.TrivialLoss()

    # Optimization weight, not for RANSAC scoring
    if 'line_weight' in cfg:
        jointloc_cfg['weight_point'] = 1.0
        jointloc_cfg['weight_line'] = cfg['line_weight']

    ransac_cfg = cfg['ransac']
    if ransac_cfg['method'] is None:
        if inliers_point is not None:
            p2ds = np.array(p2ds)[inliers_point]
            p3ds = np.array(p3ds)[inliers_point]
            if logger:
                logger.info(f'{len(p2ds)} inliers reserved from {len(inliers_point)} point matches')
        if inliers_line is not None:
            line_matches_2to3 = np.array(line_matches_2to3)[inliers_line]
            if logger:
                logger.info(f'{len(line_matches_2to3)} inliers reserved from {len(inliers_line)} line matches')
        jointloc_cfg['loss_function'] = _ceresbase.HuberLoss(1.0)
        jointloc = _optimize.solve_jointloc(cfg['line_cost_func'], jointloc_cfg, l3ds, l3d_ids, l2ds, p3ds, p2ds, camera.K(), campose.R(), campose.T(), silent=silent)
        final_t = jointloc.GetFinalT().copy()
        final_q = jointloc.GetFinalQ().copy()
        return _base.CameraPose(final_q, final_t), None

    jointloc_config = _optimize.LineLocConfig(jointloc_cfg)
    if 'solver_options' not in jointloc_cfg or 'minimizer_progress_to_stdout' not in jointloc_cfg['solver_options']:
        jointloc_config.solver_options.minimizer_progress_to_stdout = False
    if silent:
        jointloc_config.print_summary = False
        jointloc_config.solver_options.minimizer_progress_to_stdout = False
        jointloc_config.solver_options.logging_type = _ceresbase.LoggingType.SILENT
    func = _optimize.get_lineloc_cost_func(cfg['line_cost_func'])
    jointloc_config.cost_function = func

    if ransac_cfg['method'] == 'hybrid':
        options = _estimators.ExtendedHybridLORansacOptions()
        options.squared_inlier_thresholds_ = [pow(ransac_cfg['thres_point'], 2), pow(ransac_cfg['thres_line'], 2)]
        options.data_type_weights_ = np.array([ransac_cfg['weight_point'], ransac_cfg['weight_line']])
        options.data_type_weights_ *= np.array([options.squared_inlier_thresholds_[1], options.squared_inlier_thresholds_[0]]) / np.sum(options.squared_inlier_thresholds_)
        options.min_num_iterations_ = ransac_cfg['min_num_iterations']
        options.final_least_squares_ = ransac_cfg['final_least_squares']
        result = _estimators.EstimateAbsolutePose_PointLine_Hybrid(l3ds, l3d_ids, l2ds, p3ds, p2ds,
                                                camera, options, jointloc_config, ransac_cfg['solver_flags'])
    else:
        options = _estimators.LORansacOptions()
        options.squared_inlier_threshold_ = pow(ransac_cfg['thres'], 2)
        options.min_num_iterations_ = ransac_cfg['min_num_iterations']
        options.final_least_squares_ = ransac_cfg['final_least_squares']
        result = _estimators.EstimateAbsolutePose_PointLine(l3ds, l3d_ids, l2ds, p3ds, p2ds,
                                                camera, options, jointloc_config, ransac_cfg['method'] == 'solver')
    return result


