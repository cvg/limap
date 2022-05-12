from _limap import _base, _pointsfm
from .colmap_reader import ReadInfos, ComputeNeighbors

def filter_by_cam_id(cam_id, prev_imagecols, prev_neighbors):
    '''
    Filter _base.ImageCollection by cam id
    '''
    assert prev_imagecols.NumImages() == len(prev_neighbors)
    neighbors = []
    prev_cameras, prev_camimages = prev_imagecols.get_cameras(), prev_imagecols.get_images()
    camimages = []
    # filter ids
    id_maps = {}
    for idx in range(prev_imagecols.NumImages()):
        camimage = prev_imagecols.camimage(idx)
        cam_id_idx = camimage.cam_id
        if cam_id_idx != cam_id:
            id_maps[idx] = -1
            continue
        id_maps[idx] = len(camimages)
        camimages.append(camimage)
        neighbors.append(prev_neighbors[idx])
    # map ids for neighbors
    cameras = {}
    cameras[cam_id] = prev_imagecols.cam(cam_id);
    imagecols = _base.ImageCollection(cameras, camimages)
    for idx in range(len(camimages)):
        n0 = neighbors[idx]
        n1 = [id_maps[k] for k in n0]
        n2 = [k for k in n1 if k != -1]
        neighbors[idx] = n2
    return imagecols, neighbors

def read_infos_colmap(cfg, colmap_path, model_path="sparse", image_path="images", n_neighbors=20):
    '''
    Read all infos from colmap including imagecols, neighbors, and ranges
    '''
    model = _pointsfm.SfmModel()
    model.ReadFromCOLMAP(colmap_path, model_path, image_path)

    # get imagecols
    imagecols = ReadInfos(model, colmap_path, model_path=model_path, image_path=image_path, check_undistorted=True)

    # get neighbors
    neighbors = ComputeNeighbors(model, n_neighbors, min_triangulation_angle=cfg["min_triangulation_angle"], neighbor_type=cfg["neighbor_type"])

    # get ranges
    ranges = model.ComputeRanges(cfg["ranges"]["range_robust"], cfg["ranges"]["k_stretch"])
    return imagecols, neighbors, ranges

