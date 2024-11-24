import os
import sys

import cv2
import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import limap.base as base


class ScanNetDepthReader(base.BaseDepthReader):
    def __init__(self, filename):
        super().__init__(filename)

    def read(self, filename):
        ref_depth = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        ref_depth = ref_depth.astype(np.float32) / 1000.0
        return ref_depth


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
    cameras = [base.Camera("PINHOLE", K, cam_id=0, hw=img_hw)]
    camimages = [
        base.CameraImage(
            0, base.CameraPose(Rs[idx], Ts[idx]), image_name=imname_list[idx]
        )
        for idx in range(len(imname_list))
    ]
    imagecols = base.ImageCollection(cameras, camimages)

    # TODO: advanced implementation with the original ids
    # trivial neighbors
    index_list = np.arange(0, len(imname_list)).tolist()
    neighbors = {}
    for idx, image_id in enumerate(index_list):
        val = np.abs(np.array(index_list) - image_id)
        val[idx] = val.max() + 1
        neighbor = np.array(index_list)[np.argsort(val)[: cfg["n_neighbors"]]]
        neighbors[image_id] = neighbor.tolist()

    # get depth
    if load_depth:
        depths = {}
        for img_id, imname in enumerate(imname_list):
            depth_fname = dataset.get_depth_fname(imname)
            depth = ScanNetDepthReader(depth_fname)
            depths[img_id] = depth
        return imagecols, neighbors, depths
    else:
        return imagecols, neighbors
