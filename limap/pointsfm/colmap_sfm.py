import copy
import os
import shutil
import sys
from pathlib import Path

import cv2
import pycolmap
from pycolmap import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import hloc.utils.database as database
import hloc.utils.read_write_model as colmap_utils
from model_converter import convert_imagecols_to_colmap


def import_images_with_known_cameras(image_dir, database_path, imagecols):
    image_ids = imagecols.get_img_ids()
    image_name_list = sorted(os.listdir(image_dir))
    assert len(image_name_list) == len(image_ids)

    # connect to the database
    db = database.COLMAPDatabase(database_path)
    db.create_tables()
    # add camera
    for cam_id in imagecols.get_cam_ids():
        cam = imagecols.cam(cam_id)
        db.add_camera(
            int(cam.model), cam.w(), cam.h(), cam.params, camera_id=cam_id
        )
    # add image
    for img_name, img_id in zip(image_name_list, image_ids):
        cam_id = imagecols.camimage(img_id).cam_id
        db.add_image(img_name, cam_id, image_id=img_id)
    db.commit()


def write_pairs_from_neighbors(output_path, image_path, neighbors, image_ids):
    image_names = sorted(os.listdir(image_path))
    assert len(image_names) == len(image_ids)

    m_img_id_to_img_name = {}
    for img_name, img_id in zip(image_names, image_ids):
        m_img_id_to_img_name[img_id] = img_name

    # insert each pair only once
    m_pairs = {}
    for img_id in image_ids:
        m_pairs[img_id] = []
    with open(output_path, "w") as f:
        for img_id1, neighbor in neighbors.items():
            name1 = m_img_id_to_img_name[img_id1]
            for img_id2 in neighbor:
                name2 = m_img_id_to_img_name[img_id2]
                if img_id1 < img_id2:
                    id1 = img_id1
                    id2 = img_id2
                else:
                    id1 = img_id2
                    id2 = img_id1
                if id2 in m_pairs[id1]:
                    continue
                m_pairs[id1].append(id2)
                f.write(f"{name1} {name2}\n")


def run_hloc_matches(
    cfg, image_path, db_path, keypoints=None, neighbors=None, imagecols=None
):
    """
    Inputs:
    - neighbors: map<int, std::vector<int>> to avoid exhaustive matches
    - imagecols: optionally use the id mapping from \
        _base.ImageCollection to do the match
    """
    image_path = Path(image_path)
    from hloc import (
        extract_features,
        match_features,
        pairs_from_exhaustive,
        reconstruction,
        triangulation,
    )

    outputs = Path(os.path.join(os.path.dirname(db_path), "hloc_outputs"))
    sfm_dir = Path(os.path.join(outputs, "sfm"))
    feature_conf = extract_features.confs[cfg["descriptor"]]
    if cfg["descriptor"] == "sift":
        # make sift consistent with colmap
        feature_conf["model"]["options"] = dict()
        feature_conf["model"]["options"]["first_octave"] = -1
        feature_conf["model"]["options"]["peak_threshold"] = 0.02 / 3
    matcher_conf = match_features.confs[cfg["matcher"]]

    # feature extraction
    if keypoints is not None and keypoints != []:
        if cfg["descriptor"][:10] != "superpoint":
            raise ValueError(
                "Error! Non-superpoint feature extraction is unfortunately \
                 not supported in the current implementation."
            )
        # run superpoint
        from limap.point2d.superpoint import run_superpoint

        feature_path = run_superpoint(
            feature_conf, image_path, outputs, keypoints=keypoints
        )
    else:
        feature_path = extract_features.main(feature_conf, image_path, outputs)

    # feature matching
    if neighbors is None or imagecols is None:
        # run exhaustive matches
        sfm_pairs = outputs / "pairs-exhaustive.txt"
        features_path = outputs / (feature_conf["output"] + ".h5")
        match_path = pairs_from_exhaustive.main(
            sfm_pairs, features=features_path
        )
    else:
        # run matches on neighbors
        sfm_pairs = outputs / "pairs-from-neighbors.txt"
        write_pairs_from_neighbors(
            sfm_pairs, image_path, neighbors, imagecols.get_img_ids()
        )
    match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )
    sfm_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.create_empty_db(db_path)
    if imagecols is None:
        reconstruction.import_images(
            image_dir=image_path, database_path=db_path
        )
    else:
        # use the id mapping from imagecols
        import_images_with_known_cameras(
            image_path, db_path, imagecols
        )  # use cameras and id mapping
    image_ids = reconstruction.get_image_ids(db_path)
    reconstruction.import_features(image_ids, db_path, feature_path)
    reconstruction.import_matches(
        image_ids, db_path, sfm_pairs, match_path, None, None
    )
    triangulation.estimation_and_geometric_verification(db_path, sfm_pairs)


def run_colmap_sfm(
    cfg,
    imagecols,
    output_path="tmp/tmp_colmap",
    keypoints=None,
    skip_exists=False,
    map_to_original_image_names=True,
    neighbors=None,
):
    ### set up path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    db_path = os.path.join(output_path, "db.db")
    image_path = os.path.join(output_path, "images")
    model_path = os.path.join(output_path, "sparse")

    ### initialize sparse folder
    if skip_exists and os.path.exists(output_path):
        logging.info("[COLMAP] Skipping mapping")
        return
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    ### copy images to tmp folder
    keypoints_in_order = []
    for img_id in imagecols.get_img_ids():
        img = imagecols.read_image(img_id)
        fname_to_save = os.path.join(image_path, f"image{img_id:08d}.png")
        cv2.imwrite(fname_to_save, img)
        if keypoints is not None:
            keypoints_in_order.append(keypoints[img_id])

    # feature extraction and matching
    run_hloc_matches(
        cfg["hloc"],
        image_path,
        Path(db_path),
        keypoints=keypoints_in_order,
        neighbors=neighbors,
        imagecols=imagecols,
    )

    ### [COLMAP] mapper
    pycolmap.incremental_mapping(db_path, image_path, model_path)

    # map to original image names
    if map_to_original_image_names:
        fname_images_bin = os.path.join(model_path, "0", "images.bin")
        colmap_images = colmap_utils.read_images_binary(fname_images_bin)
        for img_id in imagecols.get_img_ids():
            if img_id not in colmap_images:
                continue
            colmap_images[img_id] = colmap_images[img_id]._replace(
                name=imagecols.image_name(img_id)
            )
        colmap_utils.write_images_binary(colmap_images, fname_images_bin)


def run_colmap_sfm_with_known_poses(
    cfg,
    imagecols,
    output_path="tmp/tmp_colmap",
    keypoints=None,
    skip_exists=False,
    map_to_original_image_names=False,
    neighbors=None,
):
    ### set up path
    db_path = os.path.join(output_path, "db.db")
    image_path = os.path.join(output_path, "images")
    model_path = os.path.join(output_path, "sparse", "reference_model")
    point_triangulation_path = os.path.join(output_path, "sparse")

    ### initialize sparse folder
    if skip_exists and os.path.exists(point_triangulation_path):
        logging.info("[COLMAP] Skipping point triangulation")
        return Path(point_triangulation_path)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    ### copy images to tmp folder
    keypoints_in_order = []
    imagecols_tmp = copy.deepcopy(imagecols)
    for img_id in imagecols.get_img_ids():
        img = imagecols.read_image(img_id)
        fname_to_save = os.path.join(image_path, f"image{img_id:08d}.png")
        cv2.imwrite(fname_to_save, img)
        if keypoints is not None:
            keypoints_in_order.append(keypoints[img_id])
        imagecols_tmp.change_image_name(img_id, f"image{img_id:08d}.png")

    # feature extraction and matching
    run_hloc_matches(
        cfg["hloc"],
        image_path,
        Path(db_path),
        keypoints=keypoints_in_order,
        neighbors=neighbors,
        imagecols=imagecols_tmp,
    )

    # write colmap model from imagecols
    convert_imagecols_to_colmap(imagecols_tmp, model_path)

    ### [COLMAP] point triangulation
    # point triangulation
    input_reconstruction = pycolmap.Reconstruction(model_path)
    pycolmap.triangulate_points(
        input_reconstruction, db_path, image_path, point_triangulation_path
    )

    # map to original image names
    if map_to_original_image_names:
        fname_images_bin = os.path.join(point_triangulation_path, "images.bin")
        colmap_images = colmap_utils.read_images_binary(fname_images_bin)
        for img_id in imagecols.get_img_ids():
            colmap_images[img_id] = colmap_images[img_id]._replace(
                name=imagecols.image_name(img_id)
            )
        colmap_utils.write_images_binary(colmap_images, fname_images_bin)

    return Path(point_triangulation_path)
