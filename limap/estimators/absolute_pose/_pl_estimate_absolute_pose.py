import numpy as np
from _limap import _ceresbase

import limap.base as base
import limap.estimators as _estimators
import limap.optimize as optimize


def _pl_estimate_absolute_pose(
    cfg,
    l3ds,
    l3d_ids,
    l2ds,
    p3ds,
    p2ds,
    camera,
    campose=None,
    inliers_line=None,
    inliers_point=None,
    jointloc_cfg=None,
    silent=True,
    logger=None,
):
    if jointloc_cfg is None:
        jointloc_cfg = {}
        if cfg.get("optimize"):
            jointloc_cfg = cfg.get("optimize").copy()
            loss_func_args = jointloc_cfg.get("loss_func_args")
            if jointloc_cfg.get("loss_func") == "TrivialLoss":
                loss_func_args = []
            elif jointloc_cfg.get("loss_func") == "HuberLoss":
                assert (
                    len(loss_func_args) == 1
                ), "HuberLoss requires one argument"
            jointloc_cfg["loss_function"] = getattr(
                _ceresbase, jointloc_cfg["loss_func"]
            )(*loss_func_args)
            del jointloc_cfg["loss_func"], jointloc_cfg["loss_func_args"]
        else:
            jointloc_cfg["loss_function"] = _ceresbase.TrivialLoss()

    # Optimization weight, not for RANSAC scoring
    if "line_weight" in cfg:
        jointloc_cfg["weight_point"] = 1.0
        jointloc_cfg["weight_line"] = cfg["line_weight"]

    ransac_cfg = cfg["ransac"]
    if ransac_cfg["method"] is None:
        if inliers_point is not None:
            original_len = len(p2ds)
            p2ds = np.array(p2ds)[inliers_point]
            p3ds = np.array(p3ds)[inliers_point]
            if logger:
                logger.info(
                    f"{len(p2ds)} inliers reserved from \
                      {original_len} point matches"
                )

        if inliers_line is not None:
            original_len = len(l3d_ids)
            l3d_ids = np.array(l3d_ids)[inliers_line]
            l2ds = np.array(l2ds)[inliers_line]
            if logger:
                logger.info(
                    f"{len(l3d_ids)} inliers reserved from \
                      {original_len} line matches"
                )

        jointloc = optimize.solve_jointloc(
            cfg["line_cost_func"],
            jointloc_cfg,
            l3ds,
            l3d_ids,
            l2ds,
            p3ds,
            p2ds,
            camera.K(),
            campose.R(),
            campose.T(),
            silent=silent,
        )
        final_t = jointloc.GetFinalT().copy()
        final_q = jointloc.GetFinalQ().copy()
        return base.CameraPose(final_q, final_t), None

    options = (
        _estimators.HybridPoseEstimatorOptions()
        if ransac_cfg["method"] == "hybrid"
        else _estimators.JointPoseEstimatorOptions()
    )
    options.lineloc_config = optimize.LineLocConfig(jointloc_cfg)
    if (
        "solver_options" not in jointloc_cfg
        or "minimizer_progress_to_stdout" not in jointloc_cfg["solver_options"]
    ):
        options.lineloc_config.solver_options.minimizer_progress_to_stdout = (
            False
        )
    if silent:
        options.lineloc_config.print_summary = False
        options.lineloc_config.solver_options.minimizer_progress_to_stdout = (
            False
        )
        options.lineloc_config.solver_options.logging_type = (
            _ceresbase.LoggingType.SILENT
        )
    func = optimize.get_lineloc_cost_func(cfg["line_cost_func"])
    options.lineloc_config.cost_function = func

    if ransac_cfg["method"] == "hybrid":
        options.solver_flags = ransac_cfg["solver_flags"]

        ransac_options = options.ransac_options
        ransac_options.squared_inlier_thresholds_ = [
            pow(ransac_cfg["thres_point"], 2),
            pow(ransac_cfg["thres_line"], 2),
        ]
        ransac_options.data_type_weights_ = np.array(
            [ransac_cfg["weight_point"], ransac_cfg["weight_line"]]
        )
        ransac_options.data_type_weights_ *= np.array(
            [
                ransac_options.squared_inlier_thresholds_[1],
                ransac_options.squared_inlier_thresholds_[0],
            ]
        ) / np.sum(ransac_options.squared_inlier_thresholds_)
        ransac_options.min_num_iterations_ = ransac_cfg["min_num_iterations"]
        ransac_options.final_least_squares_ = ransac_cfg["final_least_squares"]

        result = _estimators.EstimateAbsolutePose_PointLine_Hybrid(
            l3ds, l3d_ids, l2ds, p3ds, p2ds, camera, options
        )
    else:
        options.sample_solve_first = ransac_cfg["method"] == "solver"
        options.ransac_options.squared_inlier_threshold_ = pow(
            ransac_cfg["thres"], 2
        )
        options.ransac_options.min_num_iterations_ = ransac_cfg[
            "min_num_iterations"
        ]
        options.ransac_options.final_least_squares_ = ransac_cfg[
            "final_least_squares"
        ]
        result = _estimators.EstimateAbsolutePose_PointLine(
            l3ds, l3d_ids, l2ds, p3ds, p2ds, camera, options
        )
    return result
