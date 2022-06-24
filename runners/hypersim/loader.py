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

    # get image collections
    K = dataset.K.astype(np.float32)
    img_hw = [dataset.h, dataset.w]
    Ts, Rs = dataset.load_cameras(cam_id=cam_id)
    cameras, camimages = {}, {}
    cameras[0] = _base.Camera("SIMPLE_PINHOLE", K, cam_id=0, hw=img_hw)
    for image_id in index_list:
        pose = _base.CameraPose(Rs[image_id], Ts[image_id])
        imname = dataset.load_imname(image_id, cam_id=cam_id)
        camimage = _base.CameraImage(0, pose, image_name=imname)
        camimages[image_id] = camimage
    imagecols = _base.ImageCollection(cameras, camimages)

    if load_depth:
        # get depths
        depths = {}
        for image_id in index_list:
            depth = dataset.load_depth(image_id, cam_id=cam_id)
            depths[image_id] = depth
        return imagecols, depths
    else:
        return imagecols

