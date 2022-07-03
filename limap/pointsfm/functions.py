from _limap import _base, _pointsfm

def filter_by_cam_id(cam_id, prev_imagecols, prev_neighbors):
    '''
    Filter _base.ImageCollection by cam id
    '''
    assert prev_imagecols.NumImages() == len(prev_neighbors)
    valid_image_ids = []
    for img_id in prev_imagecols.get_img_ids():
        if prev_imagecols.camimage(img_id).cam_id != cam_id:
            continue
        valid_image_ids.append(img_id)
    imagecols = prev_imagecols.subset_by_image_ids(valid_image_ids)
    neighbors = imagecols.update_neighbors(prev_neighbors)
    return imagecols, neighbors

def ComputeNeighborsVector(model, n_neighbors, min_triangulation_angle=1.0, neighbor_type="iou"):
    '''
    Returns: vector<vector<int>>
    '''
    if neighbor_type == "iou":
        neighbors_vec = model.GetMaxIoUImages(n_neighbors, min_triangulation_angle)
    elif neighbor_type == "overlap":
        neighbors_vec = model.GetMaxOverlappingImages(n_neighbors, min_triangulation_angle)
    elif neighbor_type == "dice":
        neighbors_vec = model.GetMaxDiceCoeffImages(n_neighbors, min_triangulation_angle)
    else:
        raise NotImplementedError
    return neighbors_vec

def ComputeNeighbors(model, n_neighbors, min_triangulation_angle=1.0, neighbor_type="iou"):
    '''
    Returns: map<int, vector<int>>
    '''
    neighbors_vec = ComputeNeighborsVector(model, n_neighbors, min_triangulation_angle=min_triangulation_angle, neighbor_type=neighbor_type)
    neighbors = {}
    for img_id in range(len(neighbors_vec)):
        neighbors[img_id] = neighbors_vec[img_id]
    return neighbors

# compute neighborhood for a image list sorted as 'image{0:08d}.png'
def ComputeNeighborsSorted(model, n_neighbors, min_triangulation_angle=1.0, neighbor_type="iou"):
    '''
    Returns: map<int, vector<int>>
    '''
    # get neighbors
    neighbors_vec = ComputeNeighborsVector(model, n_neighbors, min_triangulation_angle=min_triangulation_angle, neighbor_type=neighbor_type)

    # map indexes
    image_names = model.GetImageNames()
    image_id_list = [int(name[5:-4]) for name in image_names]
    neighbors = {}
    for idx, img_id in enumerate(image_id_list):
        neighbors[img_id] = [image_id_list[k] for k in neighbors_vec[idx]]
    return neighbors

def compute_metainfos(cfg, model, n_neighbors=20):
    # get neighbors
    print("Computing visual neighbors... (n_neighbors = {0})".format(n_neighbors))
    neighbors = ComputeNeighbors(model, n_neighbors, min_triangulation_angle=cfg["min_triangulation_angle"], neighbor_type=cfg["neighbor_type"])

    # get ranges
    ranges = model.ComputeRanges(cfg["ranges"]["range_robust"], cfg["ranges"]["k_stretch"])
    return neighbors, ranges

def read_infos_colmap(cfg, colmap_path, model_path="sparse", image_path="images", n_neighbors=20):
    '''
    Read all infos from colmap including imagecols, neighbors, and ranges
    '''
    from .colmap_reader import ReadInfos
    model = _pointsfm.SfmModel()
    model.ReadFromCOLMAP(colmap_path, model_path, image_path)

    # get imagecols
    imagecols = ReadInfos(model, colmap_path, model_path=model_path, image_path=image_path)

    # get metainfos
    neighbors, ranges = compute_metainfos(cfg, model, n_neighbors=n_neighbors)
    return imagecols, neighbors, ranges

def read_infos_bundler(cfg, bundler_path, list_path, model_path, n_neighbors=20):
    '''
    Read all infos from Bundler format including imagecols, neighbors, ranges
    '''
    from .bundler_reader import ReadModelBundler
    model, imagecols = ReadModelBundler(bundler_path, list_path, model_path)

    # get metainfos
    neighbors, ranges = compute_metainfos(cfg, model, n_neighbors=n_neighbors)
    return imagecols, neighbors, ranges

def read_infos_visualsfm(cfg, vsfm_path, nvm_file="reconstruction.nvm", n_neighbors=20):
    '''
    Read all infos from VisualSfM format including imagecols, neighbors, ranges
    '''
    from .visualsfm_reader import ReadModelVisualSfM
    model, imagecols = ReadModelVisualSfM(vsfm_path, nvm_file=nvm_file)

    # get metainfos
    neighbors, ranges = compute_metainfos(cfg, model, n_neighbors=n_neighbors)
    return imagecols, neighbors, ranges


