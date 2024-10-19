from _limap import _optimize


def solve_global_pl_association(cfg, imagecols, bpt3d, all_bpt2ds):
    cfg_associator = _optimize.GlobalAssociatorConfig(cfg)
    associator = _optimize.GlobalAssociator(cfg_associator)
    associator.InitCameras(imagecols)
    associator.InitBipartite_PointLine(bpt3d, all_bpt2ds)

    # soft association
    associator.SetUp()
    associator.Solve()
    res_bpt3d = associator.GetBipartite3d_PointLine()
    return res_bpt3d
