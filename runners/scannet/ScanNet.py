import copy
import os

import cv2
import numpy as np


class ScanNet:
    # constants
    max_image_dim = -1
    img_hw_resized = None

    def __init__(self, data_dir):
        self.data_dir = data_dir

        # scene to load
        self.scene_id = None
        self.scene_dir = None
        self.stride = 1

        # initialize empty infos
        self.imname_list = []
        self.K, self.img_hw = None, None
        self.Rs, self.Ts = [], []

    @classmethod
    def set_max_dim(cls, max_image_dim):
        cls.max_image_dim = max_image_dim

    @classmethod
    def set_img_hw_resized(cls, img_hw_resized):
        cls.img_hw_resized = img_hw_resized

    def set_scene_id(self, scene_id):
        self.scene_id = scene_id
        self.scene_dir = os.path.join(self.data_dir, scene_id)

    def set_stride(self, stride):
        self.stride = stride
        # load all infos here
        self.loadinfos()

    def read_intrinsics(self, fname, mode="color"):
        with open(fname) as f:
            lines = f.readlines()
        img_hw = [-1, -1]
        K = np.zeros((3, 3))
        K[2, 2] = 1.0
        for line in lines:
            if line[:11] == mode + "Height":
                img_hw[0] = float(line.strip("\n").split("=")[1])
                continue
            if line[:10] == mode + "Width":
                img_hw[1] = float(line.strip("\n").split("=")[1])
                continue
            if line[:8] == "fx_" + mode:
                K[0, 0] = float(line.strip("\n").split("=")[1])
                continue
            if line[:8] == "fy_" + mode:
                K[1, 1] = float(line.strip("\n").split("=")[1])
                continue
            if line[:8] == "mx_" + mode:
                K[0, 2] = float(line.strip("\n").split("=")[1])
                continue
            if line[:8] == "my_" + mode:
                K[1, 2] = float(line.strip("\n").split("=")[1])
                continue
        return K, img_hw

    def read_pose(self, pose_txt):
        with open(pose_txt) as f:
            lines = f.readlines()
        mat = []
        for line in lines:
            dd = line.strip("\n").split()
            dd = [float(k) for k in dd]
            mat.append(dd)
        mat = np.array(mat)
        R_cam2world, t_cam2world = mat[:3, :3], mat[:3, 3]
        R = R_cam2world.T
        t = -R @ t_cam2world
        return R, t

    def loadinfos(self):
        img_folder = os.path.join(self.scene_dir, "color")
        # pose_folder = os.path.join(self.scene_dir, "pose")
        # depth_folder = os.path.join(self.scene_dir, "depth")
        n_images = len(os.listdir(img_folder))
        index_list = np.arange(0, n_images, self.stride).tolist()

        # load intrinsic
        fname_meta = os.path.join(self.scene_dir, f"{self.scene_id}.txt")
        K_orig, img_hw_orig = self.read_intrinsics(fname_meta)
        h_orig, w_orig = img_hw_orig[0], img_hw_orig[1]
        # reshape w.r.t max_image_dim
        K = copy.deepcopy(K_orig)
        img_hw_orig = (h_orig, w_orig)
        max_image_dim = self.max_image_dim
        if (max_image_dim is not None) and max_image_dim != -1:
            ratio = max_image_dim / max(h_orig, w_orig)
            if ratio < 1.0:
                h_new = int(round(h_orig * ratio))
                w_new = int(round(w_orig * ratio))
                K[0, :] = K[0, :] * w_new / w_orig
                K[1, :] = K[1, :] * h_new / h_orig
                img_hw = (h_new, w_new)
        if self.img_hw_resized is not None:
            h_new, w_new = self.img_hw_resized[0], self.img_hw_resized[1]
            K[0, :] = K[0, :] * w_new / w_orig
            K[1, :] = K[1, :] * h_new / h_orig
            img_hw = (h_new, w_new)
        self.K, self.img_hw = K, img_hw

        # get imname_list and cameras
        self.imname_list, self.Rs, self.Ts = [], [], []
        for index in index_list:
            imname = os.path.join(self.scene_dir, "color", f"{index}.jpg")
            self.imname_list.append(imname)

            pose_txt = os.path.join(self.scene_dir, "pose", f"{index}.txt")
            R, T = self.read_pose(pose_txt)
            self.Rs.append(R)
            self.Ts.append(T)

    def get_depth_fname(self, imname):
        depth_folder = os.path.join(self.scene_dir, "depth")
        img_id = int(os.path.basename(imname)[:-4])
        depth_fname = os.path.join(depth_folder, f"{img_id}.png")
        return depth_fname

    def get_depth(self, imname):
        depth_fname = self.get_depth_fname(imname)
        depth = cv2.imread(depth_fname, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000.0
        return depth

    def get_img_hw(self):
        if self.img_hw is None:
            self.loadinfos()
        return self.img_hw

    def load_intrinsics(self):
        if self.K is None:
            self.loadinfos()
        return self.K

    def load_imname_list(self):
        if len(self.imname_list) == 0:
            self.loadinfos()
        return self.imname_list

    def load_cameras(self):
        if len(self.Rs) == 0:
            self.loadinfos()
        return self.Ts, self.Rs
