import os

import cv2
import numpy as np


class ETH3D:
    # constants
    max_image_dim = -1

    image_folder = "images"
    scenes = {}
    scenes["dslr"] = [
        "train/courtyard",
        "train/electro",
        "train/kicker",
        "train/office",
        "train/playground",
        "train/relief_2",
        "train/terrains",
        "train/delivery_area",
        "train/facade",
        "train/meadow",
        "train/pipes",
        "train/relief",
        "train/terrace",
        "test/botanical_garden",
        "test/bridge",
        "test/exhibition_hall",
        "test/living_room",
        "test/observatory",
        "test/statue",
        "test/boulders",
        "test/door",
        "test/lecture_room",
        "test/lounge",
        "test/old_computer",
        "test/terrace_2",
    ]
    scenes["lowres"] = [
        "train/delivery_area",
        "train/electro",
        "train/forest",
        "train/playground",
        "train/terrains",
        "test/lakeside",
        "test/sand_box",
        "test/storage_room",
        "test_storage_room_2",
        "test/tunnel",
    ]

    def __init__(self, data_dir):
        self.data_dir = data_dir

        # scene to load
        self.reso_type = None
        self.sparse_folder = None
        self.scene_dir = None
        self.cam_id = -1

    @classmethod
    def set_max_dim(cls, max_image_dim):
        cls.max_image_dim = max_image_dim

    def set_scene_id(self, reso_type, scene_id, cam_id=-1):
        self.reso_type = reso_type
        if self.reso_type == "dslr":
            self.sparse_folder = "dslr_calibration_undistorted"
        elif self.reso_type == "lowres":
            self.sparse_folder = "rig_calibration_undistorted"
        else:
            raise NotImplementedError
        if scene_id not in self.scenes[reso_type]:
            raise ValueError(
                f"Scene {scene_id} does not exist in ETH3D {reso_type} data."
            )
        self.scene_dir = os.path.join(self.data_dir, self.reso_type, scene_id)
        self.cam_id = cam_id

    def read_depth(self, fname):
        ref_depth = cv2.imread(fname, cv2.IMREAD_ANYDEPTH)
        ref_depth = ref_depth.astype(np.float32) / 256
        ref_depth[ref_depth == 0] = np.inf
        return ref_depth

    def get_depth_fname(self, imname, use_inpainted=True):
        imname = os.path.basename(imname)
        if use_inpainted:
            fname_depth = os.path.join(
                self.scene_dir, "inpainted_depth", f"{imname}.png"
            )
        else:
            fname_depth = os.path.join(
                self.scene_dir, "ground_truth_depth", f"{imname}.png"
            )
        return fname_depth

    def get_depth(self, imname, use_inpainted=True):
        fname_depth = self.get_depth_fname(imname, use_inpainted=use_inpainted)
        depth = self.read_depth(fname_depth)
        return depth
