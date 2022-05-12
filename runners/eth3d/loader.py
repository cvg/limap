import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import limap.base as _base
import limap.pointsfm as _psfm

def read_scene_eth3d(cfg, dataset, reso_type, scene_id, cam_id=0, load_depth=False):
    # set scene id
    dataset.set_scene_id(reso_type, scene_id, cam_id=cam_id)

    # get camviews, neighbors, and ranges
    if cfg["info_path"] is None:
        imagecols, neighbors, ranges = _psfm.read_infos_colmap(cfg["sfm"], dataset.scene_dir, model_path=dataset.sparse_folder, image_path=dataset.image_folder, n_neighbors=100000)
        with open(os.path.join("tmp", "infos_eth3d.npy"), 'wb') as f:
            np.savez(f, imagecols_np=imagecols.as_dict(), neighbors=neighbors, ranges=ranges)
    else:
        with open(cfg["info_path"], 'rb') as f:
            data = np.load(f, allow_pickle=True)
            imagecols_np, neighbors, ranges = data["imagecols_np"], data["neighbors"], data["ranges"]
            imagecols = _base.ImageCollection(imagecols_np)

    # filter by camera ids for eth3d
    if dataset.cam_id != -1:
        imagecols, neighbors = _psfm.filter_by_cam_id(dataset.cam_id, imagecols, neighbors)

    # resize cameras
    if cfg["max_image_dim"] != -1 and cfg["max_image_dim"] is not None:
        imagecols.set_max_image_dim(cfg["max_image_dim"])

    # get depths
    if load_depth:
        depths = []
        for img_id in range(imagecols.NumImages()):
            depth = dataset.get_depth(imagecols.camview(img_id).image_name())
            depths.append(depth)
        return imagecols, neighbors, ranges, depths
    else:
        return imagecols, neighbors, ranges

