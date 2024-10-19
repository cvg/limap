from _limap import _ceresbase, _optimize


def _init_bundle_adjustment_engine(cfg, imagecols, max_num_iterations=100):
    ba_config = _optimize.HybridBAConfig(cfg) if isinstance(cfg, dict) else cfg
    ba_config.solver_options.logging_type = _ceresbase.LoggingType.SILENT
    ba_config.solver_options.max_num_iterations = max_num_iterations
    ba_engine = _optimize.HybridBAEngine(ba_config)
    ba_engine.InitImagecols(imagecols)
    return ba_engine


def _solve_bundle_adjustment(ba_engine):
    # setup and solve
    ba_engine.SetUp()
    ba_engine.Solve()
    return ba_engine


def solve_point_bundle_adjustment(
    cfg, imagecols, pointtracks, max_num_iterations=100
):
    ba_engine = _init_bundle_adjustment_engine(
        cfg, imagecols, max_num_iterations=max_num_iterations
    )
    ba_engine.InitPointTracks(pointtracks)
    ba_engine = _solve_bundle_adjustment(ba_engine)
    return ba_engine


def solve_line_bundle_adjustment(
    cfg, imagecols, linetracks, max_num_iterations=100
):
    ba_engine = _init_bundle_adjustment_engine(
        cfg, imagecols, max_num_iterations=max_num_iterations
    )
    ba_engine.InitLineTracks(linetracks)
    ba_engine = _solve_bundle_adjustment(ba_engine)
    return ba_engine


def solve_hybrid_bundle_adjustment(
    cfg, imagecols, pointtracks, linetracks, max_num_iterations=100
):
    ba_engine = _init_bundle_adjustment_engine(
        cfg, imagecols, max_num_iterations=max_num_iterations
    )
    ba_engine.InitPointTracks(pointtracks)
    ba_engine.InitLineTracks(linetracks)
    ba_engine = _solve_bundle_adjustment(ba_engine)
    return ba_engine
