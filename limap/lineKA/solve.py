from _limap import _base, _ceresbase, _lineKA
import numpy as np

def solve(cfg, all_lines, camviews, all_matches, neighbors, heatmaps=None, patches_list=None, dtype="float16"):
    '''
    patches_list: list of PatchInfo_f objects
    '''
    ka_config = _lineKA.LineKAConfig(cfg)
    ka_config.solver_options.logging_type = _ceresbase.LoggingType.STDOUT

    # initialize engine
    if patches_list is not None:
        channels = patches_list[0][0].array.shape[2]
    else:
        channels = 128
    lineka_engine_name = "LineKAEngine_f{0}_c{1}".format(dtype[-2:], channels)
    print("Refinement type: ", lineka_engine_name)
    lineka_engine = getattr(_lineKA, lineka_engine_name)(ka_config)
    lineka_engine.Initialize(all_lines, camviews)
    lineka_engine.InitializeMatches(all_matches, neighbors)

    # initialize data interpolator
    if heatmaps is not None:
        lineka_engine.InitializeHeatmaps(heatmaps)
    if patches_list is not None:
        lineka_engine.InitializePatches(patches_list)

    # setup and solve
    lineka_engine.SetUp()
    lineka_engine.Solve()
    new_lines = lineka_engine.GetOutputLines()
    return new_lines

