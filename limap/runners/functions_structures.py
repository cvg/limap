import os

import numpy as np
from pycolmap import logging
from tqdm import tqdm

import limap.pointsfm as pointsfm
import limap.structures as structures


def compute_2d_feature_points_sp(imagecols, output_path="tmp/featurepoints"):
    from pathlib import Path

    import cv2
    import h5py

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    image_path = os.path.join(output_path, "images")
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    ### copy images to tmp folder
    for img_id in imagecols.get_img_ids():
        img = imagecols.read_image(img_id)
        fname_to_save = os.path.join(image_path, f"image{img_id:08d}.png")
        cv2.imwrite(fname_to_save, img)

    # run superpoint
    image_path = Path(image_path)
    from hloc import extract_features

    outputs = Path(os.path.join(image_path, "hloc_outputs"))
    feature_conf = extract_features.confs["superpoint_aachen"]
    from limap.point2d.superpoint import run_superpoint

    feature_path = run_superpoint(feature_conf, image_path, outputs)

    # read keypoints
    f = h5py.File(feature_path, "r")
    all_keypoints = {}
    for img_id in imagecols.get_img_ids():
        fname = f"image{img_id:08d}.png"
        keypoints = np.array(f[fname]["keypoints"])
        all_keypoints[img_id] = keypoints
    return all_keypoints


def compute_colmap_model_with_junctions(
    cfg_bpt2d,
    cfg_sfm,
    imagecols,
    all_2d_lines,
    neighbors,
    output_model_path,
    skip_exists=False,
):
    all_keypoints = compute_2d_feature_points_sp(imagecols)
    all_keypoints_updated = {}
    config_bpt2d = structures.PL_Bipartite2dConfig(cfg_bpt2d)
    for img_id in imagecols.get_img_ids():
        bpt2d = structures.PL_Bipartite2d(config_bpt2d)
        bpt2d.init_lines(all_2d_lines[img_id])
        keypoints = all_keypoints[img_id]
        bpt2d.compute_intersection_with_points(keypoints)
        intersections = [point2d.p for point2d in bpt2d.get_all_points()]
        new_keypoints = np.concatenate([keypoints, np.array(intersections)], 0)
        all_keypoints_updated[img_id] = new_keypoints
    pointsfm.run_colmap_sfm_with_known_poses(
        cfg_sfm,
        imagecols,
        output_path=output_model_path,
        skip_exists=skip_exists,
        keypoints=all_keypoints_updated,
        neighbors=neighbors,
    )
    return True


def compute_2d_bipartites_from_colmap(
    reconstruction, imagecols, all_2d_lines, cfg=None
):
    if cfg is None:
        cfg = dict()
    all_bpt2ds = {}
    cfg_bpt2d = structures.PL_Bipartite2dConfig(cfg)
    colmap_cameras, colmap_images, colmap_points = (
        reconstruction["cameras"],
        reconstruction["images"],
        reconstruction["points"],
    )
    logging.info("Start computing 2D bipartites...")
    for img_id, colmap_image in tqdm(colmap_images.items()):
        n_points = colmap_image.xys.shape[0]
        indexes = np.arange(0, n_points)
        xys = colmap_image.xys
        point3D_ids = colmap_image.point3D_ids
        mask = colmap_image.point3D_ids >= 0

        # resize xys if needed
        cam_id = imagecols.camimage(img_id).cam_id
        orig_size = (
            colmap_cameras[cam_id].width,
            colmap_cameras[cam_id].height,
        )
        cam = imagecols.cam(cam_id)
        new_size = (cam.w(), cam.h())
        if orig_size != new_size:
            xys[:, 0] = xys[:, 0] * new_size[0] / orig_size[0]
            xys[:, 1] = xys[:, 1] * new_size[1] / orig_size[1]

        # init bpt2d
        bpt2d = structures.PL_Bipartite2d(cfg_bpt2d)
        bpt2d.init_lines(all_2d_lines[img_id])
        bpt2d.add_keypoints_with_point3D_ids(
            xys[mask], point3D_ids[mask], indexes[mask]
        )
        all_bpt2ds[img_id] = bpt2d
    points = {}
    for point3d_id, p in tqdm(colmap_points.items()):
        points[point3d_id] = p.xyz
    return all_bpt2ds, points
