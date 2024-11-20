from _limap import _ceresbase, _optimize


def solve_line_refinement(
    cfg,
    track,
    p_camviews,
    p_vpresults=None,
    p_heatmaps=None,
    p_patches=None,
    p_features=None,
    dtype="float16",
):
    """
    p_patches: list of PatchInfo_f objects
    """
    rf_config = _optimize.RefinementConfig(cfg)
    rf_config.solver_options.logging_type = _ceresbase.LoggingType.SILENT

    # initialize refinement engine
    if track.count_images() < rf_config.min_num_images:
        return None
    if p_patches is not None:
        channels = p_patches[0].array.shape[2]
    elif p_features is not None:
        channels = p_features[0].shape[2]
    else:
        channels = 128
    rf_engine_name = f"RefinementEngine_f{dtype[-2:]}_c{channels}"
    rf_engine = getattr(_optimize, rf_engine_name)(rf_config)

    # initialize track and camview
    rf_engine.Initialize(track, p_camviews)

    # initialize data interpolator
    if p_vpresults is not None:
        rf_engine.InitializeVPs(p_vpresults)
    if p_heatmaps is not None:
        rf_engine.InitializeHeatmaps(p_heatmaps)
    if p_patches is not None:
        rf_engine.InitializeFeaturesAsPatches(p_patches)
    elif p_features is not None:
        rf_engine.InitializeFeatures(p_features)

    # setup and solve
    rf_engine.SetUp()
    rf_engine.Solve()
    return rf_engine
