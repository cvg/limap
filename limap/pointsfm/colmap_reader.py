import os
import sys

from _limap import _base
from pycolmap import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hloc.utils.read_write_model import (
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
    read_points3D_binary,
    read_points3D_text,
)


def check_exists_colmap_model(model_path):
    if (
        os.path.exists(os.path.join(model_path, "cameras.bin"))
        and os.path.exists(os.path.join(model_path, "images.bin"))
        and os.path.exists(os.path.join(model_path, "points3D.bin"))
    ):
        return True
    return (
        os.path.exists(os.path.join(model_path, "cameras.txt"))
        and os.path.exists(os.path.join(model_path, "images.txt"))
        and os.path.exists(os.path.join(model_path, "points3D.txt"))
    )


def ReadInfos(colmap_path, model_path="sparse", image_path="images"):
    logging.info("Start loading COLMAP sparse reconstruction.")
    model_path = os.path.join(colmap_path, model_path)
    image_path = os.path.join(colmap_path, image_path)
    if os.path.exists(os.path.join(model_path, "cameras.bin")):
        fname_cameras = os.path.join(model_path, "cameras.bin")
        fname_images = os.path.join(model_path, "images.bin")
        colmap_cameras = read_cameras_binary(fname_cameras)
        colmap_images = read_images_binary(fname_images)
    elif os.path.exists(os.path.join(model_path, "cameras.txt")):
        fname_cameras = os.path.join(model_path, "cameras.txt")
        fname_images = os.path.join(model_path, "images.txt")
        colmap_cameras = read_cameras_text(fname_cameras)
        colmap_images = read_images_text(fname_images)
    else:
        raise ValueError(
            f"Error! The model file does not exist at {model_path}"
        )
    logging.info(f"Reconstruction loaded. (n_images = {len(colmap_images)})")

    # read cameras
    cameras = {}
    for cam_id, colmap_cam in colmap_cameras.items():
        cameras[cam_id] = _base.Camera(
            colmap_cam.model,
            colmap_cam.params,
            cam_id=cam_id,
            hw=[colmap_cam.height, colmap_cam.width],
        )

    # read images
    camimages = {}
    for img_id, colmap_image in colmap_images.items():
        imname = colmap_image.name
        cam_id = colmap_image.camera_id
        pose = _base.CameraPose(colmap_image.qvec, colmap_image.tvec)
        camimage = _base.CameraImage(
            cam_id, pose, image_name=os.path.join(image_path, imname)
        )
        camimages[img_id] = camimage

    # get image collection
    imagecols = _base.ImageCollection(cameras, camimages)
    return imagecols


def PyReadCOLMAP(colmap_path, model_path=None):
    if model_path is not None:
        model_path = os.path.join(colmap_path, model_path)
    else:
        model_path = colmap_path
    if os.path.exists(os.path.join(model_path, "points3D.bin")):
        fname_cameras = os.path.join(model_path, "cameras.bin")
        fname_points = os.path.join(model_path, "points3D.bin")
        fname_images = os.path.join(model_path, "images.bin")
        colmap_cameras = read_cameras_binary(fname_cameras)
        colmap_images = read_images_binary(fname_images)
        colmap_points = read_points3D_binary(fname_points)
    elif os.path.exists(os.path.join(model_path, "points3D.txt")):
        fname_cameras = os.path.join(model_path, "cameras.txt")
        fname_points = os.path.join(model_path, "points3D.txt")
        fname_images = os.path.join(model_path, "images.txt")
        colmap_cameras = read_cameras_text(fname_cameras)
        colmap_images = read_images_text(fname_images)
        colmap_points = read_points3D_text(fname_points)
    else:
        raise ValueError(
            f"Error! The model file does not exist at {model_path}"
        )
    reconstruction = {}
    reconstruction["cameras"] = colmap_cameras
    reconstruction["images"] = colmap_images
    reconstruction["points"] = colmap_points
    return reconstruction


def ReadPointTracks(colmap_reconstruction):
    pointtracks = {}
    for point3d_id, p in colmap_reconstruction["points"].items():
        p_image_ids, point2d_ids = p.image_ids, p.point2D_idxs
        p2d_list = []
        for p_img_id, point2d_id in zip(
            p_image_ids.tolist(), point2d_ids.tolist()
        ):
            p2d_list.append(
                colmap_reconstruction["images"][p_img_id].xys[point2d_id]
            )
        ptrack = _base.PointTrack(p.xyz, p_image_ids, point2d_ids, p2d_list)
        pointtracks[point3d_id] = ptrack
    return pointtracks
