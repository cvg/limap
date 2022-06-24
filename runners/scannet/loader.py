import os, sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import limap.base as _base

def read_scene_scannet(cfg, dataset, scene_id, load_depth=False):
    # set scene id
    dataset.set_scene_id(scene_id)
    if load_depth:
        dataset.set_img_hw_resized((480, 640))
    else:
        dataset.set_max_dim(cfg["max_image_dim"])

    # get imname_list and cameras
    dataset.set_stride(cfg["stride"])
    imname_list = dataset.load_imname_list()
    K = dataset.load_intrinsics()
    img_hw = dataset.get_img_hw()
    Ts, Rs = dataset.load_cameras()
    cameras = [_base.Camera("PINHOLE", K, cam_id=0, hw=img_hw)]
    camimages = [_base.CameraImage(0, _base.CameraPose(Rs[idx], Ts[idx]), image_name=imname_list[idx]) for idx in range(len(imname_list))]
    imagecols = _base.ImageCollection(cameras, camimages)

    # trivial neighbors
    index_list = np.arange(0, len(imname_list)).tolist()
    neighbors = []
    for idx, image_id in enumerate(index_list):
        val = np.abs(np.array(index_list) - image_id)
        val[idx] = val.max() + 1
        neighbor = np.array(index_list)[np.argsort(val)[:cfg["n_neighbors"]]]
        neighbors.append(neighbor.tolist())
    neighbors = np.array(neighbors)
    for idx, index in enumerate(index_list):
        neighbors[neighbors == index] = idx

    # get depth
    if load_depth:
        print("Start loading depth maps...")
        depths = {}
        for img_id, imname in enumerate(tqdm(imname_list)):
            depth = dataset.get_depth(imname)
            depths[img_id] = depth
        return imagecols, neighbors, depths
    else:
        return imagecols, neighbors


