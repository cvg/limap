from _limap import _base, _pointsfm

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

def ComputeNeighbors(model, n_neighbors, min_triangulation_angle=1.0, neighbor_type="iou"):
    if neighbor_type == "iou":
        neighbors = model.GetMaxIoUImages(n_neighbors, min_triangulation_angle)
    elif neighbor_type == "overlap":
        neighbors = model.GetMaxOverlappingImages(n_neighbors, min_triangulation_angle)
    elif neighbor_type == "dice":
        neighbors = model.GetMaxDiceCoeffImages(n_neighbors, min_triangulation_angle)
    else:
        raise NotImplementedError
    return neighbors

# compute neighborhood for a image list sorted as 'image{0:08d}.png'
def ComputeNeighborsSorted(model, n_neighbors, min_triangulation_angle=1.0, neighbor_type="iou"):
    # get neighbors
    neighbors = ComputeNeighbors(model, n_neighbors, min_triangulation_angle=min_triangulation_angle, neighbor_type=neighbor_type)

    # map indexes
    image_names = model.GetImageNames()
    image_id_list = [int(name[5:-4]) for name in image_names]
    n1 = [neighbors[image_id_list.index(k)] for k in range(len(image_id_list))]
    n2 = [[image_id_list[val] for val in neighbor] for neighbor in n1]
    neighbors = n2
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


