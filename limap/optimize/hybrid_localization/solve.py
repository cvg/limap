from collections import defaultdict

from _limap import _ceresbase, _optimize


def get_lineloc_cost_func(func_name):
    if func_name in [
        "MidpointDist",
        "MidpointDist2",
        "2DMidpointDist",
        "2DMidpointDist2",
    ]:
        return _optimize.LineLocCostFunction.E2DMidpointDist2
    if func_name in [
        "MidpointAngle",
        "MidpointAngleDist",
        "2DMidpointAngleDist",
    ]:
        return _optimize.LineLocCostFunction.E2DMidpointAngleDist3
    if func_name in [
        "PerpendicularDist",
        "PerpendicularDist2" "2DPerpendicularDist",
        "2DPerpendicularDist2",
    ]:
        return _optimize.LineLocCostFunction.E2DPerpendicularDist2
    if func_name in ["PerpendicularDist4", "2DPerpendicularDist4"]:
        return _optimize.LineLocCostFunction.E2DPerpendicularDist4
    if func_name in ["3DLineLineDist", "3DLineLineDist2"]:
        return _optimize.LineLocCostFunction.E3DLineLineDist2
    if func_name in ["3DPlaneLineDist", "3DPlaneLineDist2"]:
        return _optimize.LineLocCostFunction.E3DPlaneLineDist2
    raise ValueError(
        f"[Error] Unknown line localization cost function: {func_name}"
    )


def get_lineloc_weight_func(func_name):
    if func_name is None:
        return _optimize.LineLocCostFunctionWeight.ENoneWeight
    func_name = func_name.lower()
    if func_name == "cosine":
        return _optimize.LineLocCostFunctionWeight.ECosineWeight
    if func_name == "line3dpp":
        return _optimize.LineLocCostFunctionWeight.ELine3dppWeight
    if func_name == "length":
        return _optimize.LineLocCostFunctionWeight.ELengthWeight
    if func_name == "invlength":
        return _optimize.LineLocCostFunctionWeight.EInvLengthWeight
    raise ValueError(
        f"[Error] Unknown line localization weight function: {func_name}"
    )


def get_line_pairs(l3ds, l3d_ids, l2ds):
    merged_matches = defaultdict(list)
    for l2d, l3d_id in zip(l2ds, l3d_ids):
        merged_matches[l3d_id].append(l2d)

    line_3ds = []
    line_2ds = []
    for l3d_id in merged_matches:
        line_3ds.append(l3ds[l3d_id])
        line_2ds.append(merged_matches[l3d_id])
    return line_2ds, line_3ds


def solve_lineloc_merged(
    cost_func, lineloc_cfg, line_pairs, K, R, T, weight_func=None, silent=False
):
    if lineloc_cfg is None:
        lineloc_cfg = {}
    lineloc_config = _optimize.LineLocConfig(lineloc_cfg)
    if (
        "solver_options" not in lineloc_cfg
        or "minimizer_progress_to_stdout" not in lineloc_cfg["solver_options"]
    ):
        lineloc_config.solver_options.minimizer_progress_to_stdout = False
    if silent:
        lineloc_config.print_summary = False
        lineloc_config.solver_options.minimizer_progress_to_stdout = False
        lineloc_config.solver_options.logging_type = (
            _ceresbase.LoggingType.SILENT
        )

    lineloc_config.cost_function = get_lineloc_cost_func(cost_func)
    lineloc_config.cost_function_weight = get_lineloc_weight_func(weight_func)

    l2ds, l3ds = line_pairs
    lineloc_engine = _optimize.LineLocEngine(lineloc_config)
    lineloc_engine.Initialize(l3ds, l2ds, K, R, T)

    # setup and solve
    lineloc_engine.SetUp()
    lineloc_engine.Solve()
    return lineloc_engine


def solve_lineloc(
    cost_func, l3ds, l3d_ids, l2ds, K, R, T, line_weight_func=None, silent=False
):
    line_pairs = get_line_pairs(l3ds, l3d_ids, l2ds)
    return solve_lineloc_merged(
        cost_func, line_pairs, K, R, T, line_weight_func, silent
    )


def solve_jointloc_merged(
    line_cost_func,
    jointloc_cfg,
    line_pairs,
    point_pairs,
    K,
    R,
    T,
    line_weight_func=None,
    silent=False,
):
    if jointloc_cfg is None:
        jointloc_cfg = {}
    jointloc_config = _optimize.LineLocConfig(jointloc_cfg)
    if (
        "solver_options" not in jointloc_cfg
        or "minimizer_progress_to_stdout" not in jointloc_cfg["solver_options"]
    ):
        jointloc_config.solver_options.minimizer_progress_to_stdout = False
    if silent:
        jointloc_config.print_summary = False
        jointloc_config.solver_options.minimizer_progress_to_stdout = False
        jointloc_config.solver_options.logging_type = (
            _ceresbase.LoggingType.SILENT
        )
    jointloc_config.cost_function = get_lineloc_cost_func(line_cost_func)
    jointloc_config.cost_function_weight = get_lineloc_weight_func(
        line_weight_func
    )

    p2ds, p3ds = point_pairs
    l2ds, l3ds = line_pairs
    jointloc_engine = _optimize.JointLocEngine(jointloc_config)
    jointloc_engine.Initialize(l3ds, l2ds, p3ds, p2ds, K, R, T)

    # setup and solve
    jointloc_engine.SetUp()
    jointloc_engine.Solve()
    return jointloc_engine


def solve_jointloc(
    line_cost_func,
    jointloc_cfg,
    l3ds,
    l3d_ids,
    l2ds,
    p3ds,
    p2ds,
    K,
    R,
    T,
    line_weight_func=None,
    silent=False,
):
    line_pairs = get_line_pairs(l3ds, l3d_ids, l2ds)
    point_pairs = (p2ds, p3ds)
    return solve_jointloc_merged(
        line_cost_func,
        jointloc_cfg,
        line_pairs,
        point_pairs,
        K,
        R,
        T,
        line_weight_func,
        silent,
    )
