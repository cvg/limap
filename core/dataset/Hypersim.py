import os, sys
import h5py
import cv2
import numpy as np
import pyvista as pv

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.geometry import to_homogeneous, to_homogeneous_t, to_cartesian_t
import utils

class Hypersim:
    # constants
    default_h, default_w = 768, 1024
    h, w = default_h, default_w
    # fov_x = 60 * np.pi / 180  # set fov_x to pi/3 to match DIODE dataset (60 degrees)
    fov_x = np.pi / 3  # set fov_x to pi/3 to match DIODE dataset (60 degrees)
    f = w / (2 * np.tan(fov_x / 2))
    fov_y = 2 * np.arctan(h / (2 * f))
    default_K = np.array([[f, 0, w / 2],
                  [0, f, h / 2],
                  [0, 0, 1]])
    K = default_K
    R180x = np.array(
        [[1, 0, 0],
         [0, -1, 0],
         [0, 0, -1]])

    def __init__(self, data_dir):
        self.data_dir = data_dir

        # scene to load
        self.scene_dir = None
        self.mpau = None # meters per asset unit, initialized when set_scene_id is called
        self.cam_id = -1
        self.Tvecs, self.Rvecs = None, None

    @classmethod
    def set_max_dim(cls, max_dim):
        ratio = max_dim / max(cls.h, cls.w)
        if ratio < 1.0:
            cls.set_resize_ratio(ratio)

    @classmethod
    def set_resize_ratio(cls, ratio):
        cls.h, cls.w = int(round(cls.default_h * ratio)), int(round(cls.default_w * ratio))
        cls.K[0,:] = cls.default_K[0,:] * cls.w / cls.default_w
        cls.K[1,:] = cls.default_K[1,:] * cls.h / cls.default_h

    def read_mpau(self, scene_dir):
        fname_metascene = os.path.join(scene_dir, '_detail', 'metadata_scene.csv')
        import csv
        param_dict = {}
        with open(fname_metascene, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                param_dict[row['parameter_name']] = row['parameter_value']
        key = 'meters_per_asset_unit'
        if key in param_dict:
            return float(param_dict[key])
        else:
            raise ValueError('Key {0} not exists in {1}'.format(key, fname_metascene))

    def set_scene_id(self, scene_id):
        self.scene_dir = os.path.join(self.data_dir, scene_id)
        self.mpau = self.read_mpau(self.scene_dir)

    def filter_index_list(self, index_list, cam_id=0):
        new_index_list = []
        for image_id in index_list:
            image_fname = os.path.join(self.scene_dir, 'images', 'scene_cam_{0:02d}_final_preview'.format(cam_id), 'frame.{0:04d}.color.jpg'.format(image_id))
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

        positions_fname = os.path.join(scene_dir, '_detail', 'cam_{0:02d}'.format(cam_id), 'camera_keyframe_positions.hdf5')
        with h5py.File(positions_fname, 'r') as f:
            self.Tvecs = np.array(f['dataset']).astype(np.float32)

        orientations_fname = os.path.join(scene_dir, '_detail', 'cam_{0:02d}'.format(cam_id), 'camera_keyframe_orientations.hdf5')
        with h5py.File(orientations_fname, 'r') as f:
            self.Rvecs= np.array(f['dataset']).astype(np.float32)

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
        image_fname = os.path.join(scene_dir, 'images', 'scene_cam_{0:02d}_final_preview'.format(cam_id), 'frame.{0:04d}.color.jpg'.format(image_id))
        return image_fname

    def load_image(self, image_id, set_gray=True, cam_id=0, scene_id=None):
        image_fname = self.load_imname(image_id, cam_id=cam_id, scene_id=scene_id)
        img = utils.read_image(image_fname, resize_hw=(self.h, self.w), set_gray=set_gray)
        return img

    def load_raydepth_fname(self, image_id, cam_id=0, scene_id=None):
        if scene_id is not None:
            self.set_scene_id(scene_id)
        scene_dir = self.scene_dir
        raydepth_fname = os.path.join(scene_dir, 'images', 'scene_cam_{0:02d}_geometry_hdf5'.format(cam_id), 'frame.{0:04d}.depth_meters.hdf5'.format(image_id))
        return raydepth_fname

    def load_raydepth(self, image_id, cam_id=0, scene_id=None):
        raydepth_fname = self.load_raydepth_fname(image_id, cam_id=cam_id, scene_id=scene_id)
        raydepth = utils.read_raydepth(raydepth_fname, resize_hw=(self.h, self.w))
        return raydepth

    def load_depth(self, image_id, cam_id=0, scene_id=None):
        raydepth_fname = self.load_raydepth_fname(image_id, cam_id=cam_id, scene_id=scene_id)
        raydepth = utils.read_raydepth(raydepth_fname, resize_hw=(self.h, self.w))
        depth = utils.raydepth2depth(raydepth, self.K, (self.h, self.w))
        return depth

    def get_point_cloud(self, img_id, cam_id=0, scene_id=None):
        depth = self.load_depth(img_id, cam_id=cam_id, scene_id=scene_id)
        K, R, T = self.K, self.Rvecs[img_id], self.Tvecs[img_id]
        xv, yv = np.meshgrid(np.arange(self.w), np.arange(self.h))
        homo_2d = to_homogeneous_t(np.array([xv.flatten(), yv.flatten()]))
        points = np.linalg.inv(K) @ (homo_2d * depth.flatten())
        points = R.T @ (points - T[:,None])
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
        set_camera_properties(view_camera, Rt)
        view_camera.clipping_range = (1e-5, 0.5)
        frustum = view_camera.view_frustum(w / h)
        return frustum


