from _limap import _pointsfm
from .colmap_reader import ReadInfos, ComputeNeighbors

def filter_by_cam_id(cam_id, prev_imname_list, prev_camviews, prev_neighbors):
    '''
    Filter the list by camview id
    '''
    assert len(prev_imname_list) == len(prev_neighbors)
    assert len(prev_camviews) == len(prev_neighbors)
    imname_list, camviews, neighbors = [], [], []
    id_maps = {}
    # filter ids
    for idx in range(len(prev_neighbors)):
        cam_id_idx = prev_camviews[idx].cam.cam_id()
        if cam_id_idx != cam_id:
            id_maps[idx] = -1
            continue
        id_maps[idx] = len(imname_list)
        imname_list.append(prev_imname_list[idx])
        camviews.append(prev_camviews[idx])
        neighbors.append(prev_neighbors[idx])
    # map ids for neighbors
    for idx in range(len(imname_list)):
        n0 = neighbors[idx]
        n1 = [id_maps[k] for k in n0]
        n2 = [k for k in n1 if k != -1]
        neighbors[idx] = n2
    return imname_list, camviews, neighbors

def read_infos_colmap(cfg, colmap_path, model_path="sparse", image_path="images", n_neighbors=20, max_image_dim=None):
    '''
    Read all infos from colmap including imname_list, camviews, neighbors, and ranges
    '''
    model = _pointsfm.SfmModel()
    model.ReadFromCOLMAP(colmap_path, model_path, image_path)

    # get imname_list and camviews
    imname_list, camviews = ReadInfos(model, colmap_path, model_path=model_path, image_path=image_path, max_image_dim=max_image_dim, check_undistorted=True)

    # get neighbors
    neighbors = ComputeNeighbors(model, n_neighbors, min_triangulation_angle=cfg["min_triangulation_angle"], neighbor_type=cfg["neighbor_type"])

    # get ranges
    ranges = model.ComputeRanges(cfg["ranges"]["range_robust"], cfg["ranges"]["k_stretch"])
    return imname_list, camviews, neighbors, ranges

