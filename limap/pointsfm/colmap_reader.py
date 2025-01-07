import os
import sys

from _limap import _base
import pycolmap
from pycolmap import logging

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

def convert_colmap_to_imagecols(reconstruction: pycolmap.Reconstruction, image_path=None):
    # read cameras
    cameras = {}
    for cam_id, colmap_cam in reconstruction.cameras.items():
        cameras[cam_id] = _base.Camera(
            colmap_cam.model,
            colmap_cam.params,
            cam_id=cam_id,
            hw=[colmap_cam.height, colmap_cam.width],
        )

    # read images
    camimages = {}
    for img_id, colmap_image in reconstruction.images.items():
        imname = colmap_image.name
        cam_id = colmap_image.camera_id
        qvec_xyzw = colmap_image.cam_from_world.rotation.quat
        qvec = [qvec_xyzw[3], qvec_xyzw[0], qvec_xyzw[1], qvec_xyzw[2]]
        tvec = colmap_image.cam_from_world.translation
        pose = _base.CameraPose(qvec, tvec)
        image_name = imname if image_path is None else os.path.join(image_path, imname)
        camimage = _base.CameraImage(cam_id, pose, image_name=image_name)
        camimages[img_id] = camimage

    # get image collection
    imagecols = _base.ImageCollection(cameras, camimages)
    return imagecols


def ReadInfos(colmap_path, model_path="sparse", image_path="images"):
    logging.info("Start loading COLMAP sparse reconstruction.")
    model_path = os.path.join(colmap_path, model_path)
    image_path = os.path.join(colmap_path, image_path)
    reconstruction = pycolmap.Reconstruction(model_path)
    logging.info(f"Reconstruction loaded. (n_images = {reconstruction.num_images()}")
    imagecols = convert_colmap_to_imagecols(reconstruction, image_path=image_path)
    return imagecols


def PyReadCOLMAP(colmap_path, model_path=None):
    if model_path is not None:
        model_path = os.path.join(colmap_path, model_path)
    else:
        model_path = colmap_path
    return pycolmap.Reconstruction(model_path)


def ReadPointTracks(reconstruction: pycolmap.Reconstruction):
    pointtracks = {}
    for point3d_id, p in reconstruction.points3D.items():
        p_image_ids = []
        point2d_ids = []
        p2d_list = []
        for elem in p.track.elements:
            p_image_ids.append(elem.image_id)
            point2d_ids.append(elem.point2D_idx)
            p2d_list.append(reconstruction.images[elem.image_id].points2D[elem.point2D_idx].xy)
        ptrack = _base.PointTrack(p.xyz, p_image_ids, point2d_ids, p2d_list)
        pointtracks[point3d_id] = ptrack
    return pointtracks
