import os
import sys

import numpy as np
import pycolmap
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Hypersim import raydepth2depth, read_raydepth

from limap.scene import BaseDepthReader

# COLMAP's guess for a camera with no known focal length, as a factor of the
# larger image side (ImageReaderOptions::default_focal_length_factor).
DEFAULT_FOCAL_LENGTH_FACTOR = 1.2


class HypersimDepthReader(BaseDepthReader):
    def __init__(self, filename, K, img_hw):
        super().__init__(filename)
        self.K = K
        self.img_hw = img_hw

    def read(self, filename):
        raydepth = read_raydepth(filename, resize_hw=self.img_hw)
        depth = raydepth2depth(raydepth, self.K, self.img_hw)
        return depth


def read_scene_hypersim(
    cfg,
    dataset,
    scene_id,
    cam_id=0,
    load_depth=False,
    load_poses=True,
    uncalibrated=False,
    per_image_cameras=False,
) -> tuple[Path, pycolmap.Reconstruction, HypersimDepthReader | None]:
    """Read a Hypersim scene into a pycolmap reconstruction.

    With per_image_cameras every image gets its own camera instead of
    sharing one. Combined with uncalibrated, that makes each camera
    unconstrained when its image is registered, which is what exercises
    per-image focal length estimation; a shared camera is pinned down by
    the first image and left alone afterwards.
    """
    # set scene id
    dataset.set_scene_id(scene_id)

    # generate image indexes
    index_list = np.arange(
        0, cfg["input_n_views"], cfg["input_stride"]
    ).tolist()
    index_list = dataset.filter_index_list(index_list, cam_id=cam_id)

    # build pycolmap reconstruction
    recon = pycolmap.Reconstruction()
    K = dataset.K.astype(np.float32)
    img_hw = [dataset.h, dataset.w]
    if load_poses:
        Ts, Rs = dataset.load_cameras(cam_id=cam_id)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if uncalibrated:
        # Start from COLMAP's guess for an unknown camera rather than from
        # the true focal, so that it actually has to be recovered.
        fx = fy = DEFAULT_FOCAL_LENGTH_FACTOR * max(img_hw[0], img_hw[1])

    def _make_camera(camera_id):
        cam = pycolmap.Camera(
            camera_id=camera_id,
            model="PINHOLE",
            width=img_hw[1],  # w
            height=img_hw[0],  # h
            params=[fx, fy, cx, cy],
        )
        # GT intrinsics: mark the focal as a prior, else COLMAP re-estimates
        # it.
        cam.has_prior_focal_length = not uncalibrated
        recon.add_camera(cam)
        rig = pycolmap.Rig(rig_id=camera_id)
        rig.add_ref_sensor(cam.sensor_id)
        recon.add_rig(rig)
        return camera_id

    if not per_image_cameras:
        _make_camera(cam_id)
    image_dir = None
    for image_id in index_list:
        # Camera and rig ids share the image id, offset past cam_id so the
        # shared-camera layout above cannot collide with them.
        image_camera_id = (
            _make_camera(image_id + cam_id + 1) if per_image_cameras else cam_id
        )
        frame_kwargs = dict(frame_id=image_id, rig_id=image_camera_id)
        if load_poses:
            pose_mat = np.concatenate([Rs[image_id], Ts[image_id][:, None]], 1)
            frame_kwargs["rig_from_world"] = pycolmap.Rigid3d(pose_mat)
        frame = pycolmap.Frame(**frame_kwargs)
        imname = Path(dataset.load_imname(image_id, cam_id=cam_id))
        if image_dir is None:
            image_dir = imname.parent
        else:
            assert image_dir == imname.parent
        image = pycolmap.Image(
            name=imname.name,
            camera_id=image_camera_id,
            image_id=image_id,
            frame_id=image_id,
        )
        frame.add_data_id(image.data_id)
        recon.add_frame(frame)
        recon.add_image(image)
        if load_poses:
            recon.register_frame(frame.frame_id)

    if load_depth:
        # get depths
        depths = {}
        for image_id in index_list:
            depth_fname = dataset.load_raydepth_fname(image_id, cam_id=cam_id)
            depth = HypersimDepthReader(depth_fname, K, img_hw)
            depths[image_id] = depth
        return Path(image_dir), recon, depths
    else:
        return Path(image_dir), recon, None
