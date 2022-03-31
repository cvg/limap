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
        camviews, neighbors, ranges = _psfm.read_infos_colmap(cfg["sfm"], dataset.scene_dir, model_path=dataset.sparse_folder, image_path=dataset.image_folder, n_neighbors=100000)
        with open(os.path.join("tmp", "infos_eth3d.npy"), 'wb') as f:
            camviews_np = [view.as_dict() for view in camviews]
            np.savez(f, camviews_np=camviews_np, neighbors=neighbors, ranges=ranges)
    else:
        with open(cfg["info_path"], 'rb') as f:
            data = np.load(f, allow_pickle=True)
            camviews_np, neighbors, ranges = data["camviews_np"], data["neighbors"], data["ranges"]
            camviews = [_base.CameraView(view_np) for view_np in camviews_np]

    # filter by camera ids for eth3d
    if dataset.cam_id != -1:
        camviews, neighbors = _psfm.filter_by_cam_id(dataset.cam_id, camviews, neighbors)

    # resize camera
    if cfg["max_image_dim"] != -1 and cfg["max_image_dim"] is not None:
        for camview in camviews:
            camview.cam.set_max_image_dim(cfg["max_image_dim"])

    # get depths
    if load_depth:
        depths = []
        for camview in camviews:
            depth = dataset.get_depth(camview.image_name())
            depths.append(depth)
        return camviews, neighbors, ranges, depths
    else:
        return camviews, neighbors, ranges

