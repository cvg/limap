from _limap import _pointsfm
from pycolmap import logging


def filter_by_cam_id(cam_id, prev_imagecols, prev_neighbors):
    """
    Filter _base.ImageCollection by cam id
    """
    assert prev_imagecols.NumImages() == len(prev_neighbors)
    valid_image_ids = []
    for img_id in prev_imagecols.get_img_ids():
        if prev_imagecols.camimage(img_id).cam_id != cam_id:
            continue
        valid_image_ids.append(img_id)
    imagecols = prev_imagecols.subset_by_image_ids(valid_image_ids)
    neighbors = imagecols.update_neighbors(prev_neighbors)
    return imagecols, neighbors


def compute_neighbors(
    model, n_neighbors, min_triangulation_angle=1.0, neighbor_type="iou"
):
    """
    Returns: map<int, vector<int>>
    """
    if neighbor_type == "iou":
        neighbors = model.GetMaxIoUImages(n_neighbors, min_triangulation_angle)
    elif neighbor_type == "overlap":
        neighbors = model.GetMaxOverlapImages(
            n_neighbors, min_triangulation_angle
        )
    elif neighbor_type == "dice":
        neighbors = model.GetMaxDiceCoeffImages(
            n_neighbors, min_triangulation_angle
        )
    else:
        raise NotImplementedError
    return neighbors


def compute_metainfos(cfg, model, n_neighbors=20):
    # get neighbors
    logging.info(f"Computing visual neighbors... (n_neighbors = {n_neighbors})")
    neighbors = compute_neighbors(
        model,
        n_neighbors,
        min_triangulation_angle=cfg["min_triangulation_angle"],
        neighbor_type=cfg["neighbor_type"],
    )

    # get ranges
    ranges = model.ComputeRanges(
        cfg["ranges"]["range_robust"], cfg["ranges"]["k_stretch"]
    )
    return neighbors, ranges


def read_infos_colmap(
    cfg,
    colmap_path,
    model_path="sparse",
    image_path="images",
    compute_neighbors=True,
    n_neighbors=20,
):
    """
    Read all infos from colmap including imagecols, neighbors, and ranges
    """
    # get imagecols
    from .colmap_reader import ReadInfos

    imagecols = ReadInfos(
        colmap_path, model_path=model_path, image_path=image_path
    )
    if not compute_neighbors:
        return imagecols

    # get metainfos
    model = _pointsfm.SfmModel()
    model.ReadFromCOLMAP(colmap_path, model_path, image_path)
    neighbors, ranges = compute_metainfos(cfg, model, n_neighbors=n_neighbors)
    return imagecols, neighbors, ranges


def read_infos_bundler(
    cfg,
    bundler_path,
    list_path,
    model_path,
    compute_neighbors=True,
    n_neighbors=20,
):
    """
    Read all infos from Bundler format including imagecols, neighbors, ranges
    """
    from .bundler_reader import ReadModelBundler

    model, imagecols = ReadModelBundler(bundler_path, list_path, model_path)
    if not compute_neighbors:
        return imagecols

    # get metainfos
    neighbors, ranges = compute_metainfos(cfg, model, n_neighbors=n_neighbors)
    return imagecols, neighbors, ranges


def read_infos_visualsfm(
    cfg,
    vsfm_path,
    nvm_file="reconstruction.nvm",
    compute_neighbors=True,
    n_neighbors=20,
):
    """
    Read all infos from VisualSfM format including imagecols, neighbors, ranges
    """
    from .visualsfm_reader import ReadModelVisualSfM

    model, imagecols = ReadModelVisualSfM(vsfm_path, nvm_file=nvm_file)
    if not compute_neighbors:
        return imagecols

    # get metainfos
    neighbors, ranges = compute_metainfos(cfg, model, n_neighbors=n_neighbors)
    return imagecols, neighbors, ranges
