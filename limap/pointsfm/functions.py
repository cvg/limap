from _limap import _pointsfm
from .colmap_reader import ReadInfos, ComputeNeighbors

def filter_by_cam_id(cam_id, prev_camviews, prev_neighbors):
    '''
    Filter the list by camview id
    '''
    assert len(prev_camviews) == len(prev_neighbors)
    camviews, neighbors = [], []
    id_maps = {}
    # filter ids
    for idx in range(len(prev_neighbors)):
        cam_id_idx = prev_camviews[idx].cam.cam_id()
        if cam_id_idx != cam_id:
            id_maps[idx] = -1
            continue
        id_maps[idx] = len(camviews)
        camviews.append(prev_camviews[idx])
        neighbors.append(prev_neighbors[idx])
    # map ids for neighbors
    for idx in range(len(camviews)):
        n0 = neighbors[idx]
        n1 = [id_maps[k] for k in n0]
        n2 = [k for k in n1 if k != -1]
        neighbors[idx] = n2
    return camviews, neighbors

def read_infos_colmap(cfg, colmap_path, model_path="sparse", image_path="images", n_neighbors=20):
    '''
    Read all infos from colmap including camviews, neighbors, and ranges
    '''
    model = _pointsfm.SfmModel()
    model.ReadFromCOLMAP(colmap_path, model_path, image_path)

    # get camviews
    camviews = ReadInfos(model, colmap_path, model_path=model_path, image_path=image_path, check_undistorted=True)

    # get neighbors
    neighbors = ComputeNeighbors(model, n_neighbors, min_triangulation_angle=cfg["min_triangulation_angle"], neighbor_type=cfg["neighbor_type"])

    # get ranges
    ranges = model.ComputeRanges(cfg["ranges"]["range_robust"], cfg["ranges"]["k_stretch"])
    return camviews, neighbors, ranges

