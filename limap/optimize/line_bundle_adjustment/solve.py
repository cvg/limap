from _limap import _base, _ceresbase, _optimize
import numpy as np

def solve_line_bundle_adjustment(cfg, reconstruction, vpresults=None, heatmaps=None, patches_list=None, dtype="float16", max_num_iterations=100):
    '''
    patches_list: list of PatchInfo_f objects
    '''
    ba_config = _optimize.LineBAConfig(cfg)
    ba_config.solver_options.logging_type = _ceresbase.LoggingType.SILENT
    ba_config.solver_options.max_num_iterations = max_num_iterations

    # initialize engine
    if patches_list is not None:
        channels = patches_list[0][0].array.shape[2]
    else:
        channels = 128
    lineba_engine_name = "LineBAEngine_f{0}_c{1}".format(dtype[-2:], channels)
    # print("Refinement type: ", lineba_engine_name)
    lineba_engine = getattr(_optimize, lineba_engine_name)(ba_config)
    lineba_engine.InitializeReconstruction(reconstruction)

    # initialize data interpolator
    if vpresults is not None:
        lineba_engine.InitializeVPs(vpresults)
    if heatmaps is not None:
        lineba_engine.InitializeHeatmaps(heatmaps)
    if patches_list is not None:
        lineba_engine.InitializePatches(patches_list)

    # setup and solve
    lineba_engine.SetUp()
    lineba_engine.Solve()
    new_reconstruction = lineba_engine.GetOutputReconstruction()
    return lineba_engine

