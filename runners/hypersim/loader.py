import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import limap.base as _base

def read_scene_hypersim(cfg, dataset, scene_id, cam_id=0, load_depth=False):
    # set scene id
    dataset.set_scene_id(scene_id)
    dataset.set_max_dim(cfg["max_image_dim"])

    # generate image indexes
    index_list = np.arange(0, cfg["input_n_views"], cfg["input_stride"]).tolist()
    index_list = dataset.filter_index_list(index_list, cam_id=cam_id)

    # get imname_list
    imname_list = []
    for image_id in index_list:
        imname = dataset.load_imname(image_id, cam_id=cam_id)
        imname_list.append(imname)

    # get cameras
    K = dataset.K.astype(np.float32)
    img_hw = [dataset.h, dataset.w]
    Ts, Rs = dataset.load_cameras(cam_id=cam_id)
    cameras = [_base.Camera("SIMPLE_PINHOLE", K, cam_id=0, hw=img_hw)]
    camviews = [_base.CameraView(cameras[0], _base.CameraPose(Rs[idx], Ts[idx])) for idx in index_list]

    if load_depth:
        # get depths
        depths = []
        for image_id in index_list:
            depth = dataset.load_depth(image_id, cam_id=cam_id)
            depths.append(depth)
        return imname_list, camviews, depths
    else:
        return imname_list, camviews

    # fit and merge
    line_fitnmerge(cfg, imname_list, camviews, depths, max_image_dim=cfg["max_image_dim"])


