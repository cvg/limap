import os

import cv2
import h5py
import numpy as np
import pyvista as pv


def to_homogeneous(arr):
    # Adds a new column with ones
    return np.hstack([arr, np.ones((len(arr), 1))])


def to_homogeneous_t(arr):
    # Adds a new row with ones
    return np.vstack([arr, np.ones((1, arr.shape[1]))])


def to_cartesian(arr):
    return arr[..., :-1] / arr[..., -1].reshape((-1,) + (1,) * (arr.ndim - 1))


def to_cartesian_t(arr):
    return arr[:-1] / arr[-1]


def read_image(imname, resize_hw=None, max_image_dim=None, set_gray=False):
    img = cv2.imread(imname)
    if resize_hw is not None:
        img = cv2.resize(img, (resize_hw[1], resize_hw[0]))
    if (max_image_dim is not None) and max_image_dim != -1:
        hw_now = img.shape[:2]
        ratio = max_image_dim / max(hw_now[0], hw_now[1])
        if ratio < 1.0:
            h_new = int(round(hw_now[0] * ratio))
            w_new = int(round(hw_now[1] * ratio))
            img = cv2.resize(img, (w_new, h_new))
    if set_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def read_raydepth(raydepth_fname, resize_hw=None, max_image_dim=None):
    with h5py.File(raydepth_fname, "r") as f:
        raydepth = np.array(f["dataset"]).astype(np.float32)
    if resize_hw is not None and raydepth.shape != resize_hw:
        raydepth = cv2.resize(raydepth, (resize_hw[1], resize_hw[0]))
    if (max_image_dim is not None) and max_image_dim != -1:
        hw_now = raydepth.shape[:2]
        ratio = max_image_dim / max(hw_now[0], hw_now[1])
        if ratio < 1.0:
            h_new = int(round(hw_now[0] * ratio))
            w_new = int(round(hw_now[1] * ratio))
            raydepth = cv2.resize(raydepth, (w_new, h_new))
    return raydepth


def raydepth2depth(raydepth, K, img_hw):
    K_inv = np.linalg.inv(K)
    h, w = raydepth.shape[0], raydepth.shape[1]
    grids = np.meshgrid(np.arange(w), np.arange(h))
    coords_homo = [grids[0].reshape(-1), grids[1].reshape(-1), np.ones(h * w)]
    coords_homo = np.stack(coords_homo)
    coeffs = np.linalg.norm(K_inv @ coords_homo, axis=0)
    coeffs = coeffs.reshape(h, w)
    depth = raydepth / coeffs
    return depth


class Hypersim:
    # constants
    default_h, default_w = 768, 1024
    h, w = default_h, default_w
    # fov_x = 60 * np.pi / 180
    # set fov_x to pi/3 to match DIODE dataset (60 degrees)
    fov_x = np.pi / 3  # set fov_x to pi/3 to match DIODE dataset (60 degrees)
    f = w / (2 * np.tan(fov_x / 2))
    fov_y = 2 * np.arctan(h / (2 * f))
    default_K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    K = default_K
    R180x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    def __init__(self, data_dir):
        self.data_dir = data_dir

        # scene to load
        self.scene_dir = None
        # meters per asset unit, initialized when set_scene_id is called
        self.mpau = None
        self.cam_id = -1
        self.Tvecs, self.Rvecs = None, None

    @classmethod
    def set_max_dim(cls, max_dim):
        ratio = max_dim / max(cls.h, cls.w)
        if ratio < 1.0:
            cls.set_resize_ratio(ratio)

    @classmethod
    def set_resize_ratio(cls, ratio):
        cls.h, cls.w = (
            int(round(cls.default_h * ratio)),
            int(round(cls.default_w * ratio)),
        )
        cls.K[0, :] = cls.default_K[0, :] * cls.w / cls.default_w
        cls.K[1, :] = cls.default_K[1, :] * cls.h / cls.default_h

    def read_mpau(self, scene_dir):
        fname_metascene = os.path.join(
            scene_dir, "_detail", "metadata_scene.csv"
        )
        import csv

        param_dict = {}
        with open(fname_metascene) as f:
            reader = csv.DictReader(f)
            for row in reader:
                param_dict[row["parameter_name"]] = row["parameter_value"]
        key = "meters_per_asset_unit"
        if key in param_dict:
            return float(param_dict[key])
        else:
            raise ValueError(f"Key {key} not exists in {fname_metascene}")

    def set_scene_id(self, scene_id):
        self.scene_dir = os.path.join(self.data_dir, scene_id)
        self.mpau = self.read_mpau(self.scene_dir)

    def filter_index_list(self, index_list, cam_id=0):
        new_index_list = []
        for image_id in index_list:
            image_fname = os.path.join(
                self.scene_dir,
                "images",
                f"scene_cam_{cam_id:02d}_final_preview",
                f"frame.{image_id:04d}.color.jpg",
            )
            if os.path.exists(image_fname):
                new_index_list.append(image_id)
        return new_index_list

    def load_cameras(self, cam_id=0, scene_id=None):
        if scene_id is not None:
            self.set_scene_id(scene_id)
        if self.cam_id == cam_id:
            return self.Tvecs, self.Rvecs
        self.cam_id = cam_id
        scene_dir = self.scene_dir

        positions_fname = os.path.join(
            scene_dir,
            "_detail",
            f"cam_{cam_id:02d}",
            "camera_keyframe_positions.hdf5",
        )
        with h5py.File(positions_fname, "r") as f:
            self.Tvecs = np.array(f["dataset"]).astype(np.float32)

        orientations_fname = os.path.join(
            scene_dir,
            "_detail",
            f"cam_{cam_id:02d}",
            "camera_keyframe_orientations.hdf5",
        )
        with h5py.File(orientations_fname, "r") as f:
            self.Rvecs = np.array(f["dataset"]).astype(np.float32)

        # change to world-frame R and t following Iago Suarez
        # [LINK] https://github.com/iago-suarez/powerful-lines/blob/main/src/evaluation/consensus_3dsegs_detection.py
        N_images = self.Tvecs.shape[0]
        for image_id in range(N_images):
            self.Rvecs[image_id] = self.R180x @ self.Rvecs[image_id].T
            self.Tvecs[image_id] = self.Tvecs[image_id] * self.mpau
            self.Tvecs[image_id] = -self.Rvecs[image_id] @ self.Tvecs[image_id]
        return self.Tvecs, self.Rvecs

    def load_imname(self, image_id, cam_id=0, scene_id=None):
        if scene_id is not None:
            self.set_scene_id(scene_id)
        scene_dir = self.scene_dir
        image_fname = os.path.join(
            scene_dir,
            "images",
            f"scene_cam_{cam_id:02d}_final_preview",
            f"frame.{image_id:04d}.color.jpg",
        )
        return image_fname

    def load_image(self, image_id, set_gray=True, cam_id=0, scene_id=None):
        image_fname = self.load_imname(
            image_id, cam_id=cam_id, scene_id=scene_id
        )
        img = read_image(
            image_fname, resize_hw=(self.h, self.w), set_gray=set_gray
        )
        return img

    def load_raydepth_fname(self, image_id, cam_id=0, scene_id=None):
        if scene_id is not None:
            self.set_scene_id(scene_id)
        scene_dir = self.scene_dir
        raydepth_fname = os.path.join(
            scene_dir,
            "images",
            f"scene_cam_{cam_id:02d}_geometry_hdf5",
            f"frame.{image_id:04d}.depth_meters.hdf5",
        )
        return raydepth_fname

    def load_raydepth(self, image_id, cam_id=0, scene_id=None):
        raydepth_fname = self.load_raydepth_fname(
            image_id, cam_id=cam_id, scene_id=scene_id
        )
        raydepth = read_raydepth(raydepth_fname, resize_hw=(self.h, self.w))
        return raydepth

    def load_depth(self, image_id, cam_id=0, scene_id=None):
        raydepth_fname = self.load_raydepth_fname(
            image_id, cam_id=cam_id, scene_id=scene_id
        )
        raydepth = read_raydepth(raydepth_fname, resize_hw=(self.h, self.w))
        depth = raydepth2depth(raydepth, self.K, (self.h, self.w))
        return depth

    def get_point_cloud(self, img_id, cam_id=0, scene_id=None):
        depth = self.load_depth(img_id, cam_id=cam_id, scene_id=scene_id)
        K, R, T = self.K, self.Rvecs[img_id], self.Tvecs[img_id]
        xv, yv = np.meshgrid(np.arange(self.w), np.arange(self.h))
        homo_2d = to_homogeneous_t(np.array([xv.flatten(), yv.flatten()]))
        points = np.linalg.inv(K) @ (homo_2d * depth.flatten())
        points = R.T @ (points - T[:, None])
        return points.T

    def get_point_cloud_from_list(self, img_id_list, cam_id=0, scene_id=None):
        all_points = []
        for img_id in img_id_list:
            points = self.get_point_cloud(img_id, cam_id=0, scene_id=None)
            all_points.append(points)
        all_points = np.concatenate(all_points, 0)
        return all_points

    @classmethod
    def set_camera_properties(cls, cam, Rt):
        h, w = cls.h, cls.w
        cam.view_frustum(w / h)
        cam.SetViewAngle(cls.fov_y * 180 / np.pi)
        cam.SetViewUp(0.0, -1.0, 0.0)
        cam.SetPosition(0, 0, 0)
        cam.SetFocalPoint(0.0, 0.0, 2.0)
        cam.clipping_range = (0.001, 100)
        cam.SetModelTransformMatrix(Rt.ravel())

    @classmethod
    def get_camera_frustrum(cls, Rt):
        h, w = cls.h, cls.w
        view_camera = pv.Camera()
        Hypersim.set_camera_properties(view_camera, Rt)
        view_camera.clipping_range = (1e-5, 0.5)
        frustum = view_camera.view_frustum(w / h)
        return frustum
