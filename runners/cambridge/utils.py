import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from pathlib import Path

import imagesize
import pycolmap
from hloc import (
    extract_features,
    localize_sfm,
    match_features,
    pairs_from_retrieval,
)
from tqdm import tqdm

import limap.base as base
import limap.pointsfm as pointsfm
import limap.util.io as limapio


def read_scene_visualsfm(
    cfg, vsfm_path, nvm_file="reconstruction.nvm", n_neighbors=20
):
    metainfos_filename = "infos_visualsfm.npy"
    output_dir = "tmp" if cfg["output_dir"] is None else cfg["output_dir"]
    limapio.check_makedirs(output_dir)
    if cfg["skip_exists"] and os.path.exists(
        os.path.join(output_dir, metainfos_filename)
    ):
        cfg["info_path"] = os.path.join(output_dir, metainfos_filename)
    if cfg["info_path"] is None:
        imagecols, neighbors, ranges = pointsfm.read_infos_visualsfm(
            cfg["sfm"], vsfm_path, nvm_file=nvm_file, n_neighbors=n_neighbors
        )
        with open(os.path.join(output_dir, metainfos_filename), "wb") as f:
            np.savez(
                f,
                imagecols_np=imagecols.as_dict(),
                neighbors=neighbors,
                ranges=ranges,
            )
    else:
        with open(cfg["info_path"], "rb") as f:
            data = np.load(f, allow_pickle=True)
            imagecols_np, neighbors, ranges = (
                data["imagecols_np"].item(),
                data["neighbors"].item(),
                data["ranges"],
            )
            imagecols = base.ImageCollection(imagecols_np)
    return imagecols, neighbors, ranges


def get_scene_info(vsfm_path, imagecols, query_images):
    with open(os.path.join(vsfm_path, "dataset_train.txt")) as f:
        lines = f.readlines()[3:]
    train_names = [line.split()[0] for line in lines]

    query_start_idx = 0
    if query_images is None:
        query_images = os.path.join(vsfm_path, "dataset_test.txt")
        query_start_idx = 3
    with open(query_images) as f:
        lines = f.readlines()[query_start_idx:]
    query_names = [line.split()[0] for line in lines]

    train_ids = []
    query_ids = []
    id_to_origin_name = {}
    for id in imagecols.get_img_ids():
        image_name = "/".join(imagecols.image_name(id).split("/")[-2:])
        if image_name in train_names:
            train_ids.append(id)
        if image_name in query_names:
            query_ids.append(id)
        id_to_origin_name[id] = image_name
    return train_ids, query_ids, id_to_origin_name


def undistort_and_resize(cfg, imagecols, logger=None):
    import cv2

    import limap.runners as runners

    # undistort images
    logger.info("Performing undistortion...")
    if not imagecols.IsUndistorted():
        imagecols = runners.undistort_images(
            imagecols,
            os.path.join(cfg["output_dir"], cfg["undistortion_output_dir"]),
            skip_exists=cfg["load_undistort"] or cfg["skip_exists"],
            n_jobs=cfg["n_jobs"],
        )
    image_dir = cfg["undistortion_output_dir"]
    if cfg["max_image_dim"] != -1 and cfg["max_image_dim"] is not None:
        image_dir = cfg["resized_output_dir"]
        imagecols.set_max_image_dim(cfg["max_image_dim"])
        limapio.check_makedirs(
            os.path.join(cfg["output_dir"], cfg["resized_output_dir"])
        )
        logger.info("Images resizing...")
        for img_id in tqdm(imagecols.get_img_ids()):
            imname_out = imagecols.camimage(img_id).image_name()
            imname_out = imname_out.replace(
                cfg["undistortion_output_dir"], cfg["resized_output_dir"]
            )
            if (
                not os.path.exists(imname_out)
                or max(imagesize.get(imname_out)) != cfg["max_image_dim"]
            ):
                img = imagecols.read_image(img_id)
                cam = imagecols.cam(imagecols.camimage(img_id).cam_id)
                nsize = (cam.w(), cam.h())
                img = cv2.resize(img, nsize)
                cv2.imwrite(imname_out, img)
            imagecols.change_image_name(img_id, imname_out)
    return image_dir, imagecols


def create_query_list(imagecols, out):
    model_dict = {
        0: "SIMPLE_PINHOLE",
        1: "PINHOLE",
        2: "SIMPLE_RADIAL",
        3: "RADIAL",
        4: "OPENCV",
        5: "OPENCV_FISHEYE",
        6: "FULL_OPENCV",
        7: "FOV",
        8: "SIMPLE_RADIAL_FISHEYE",
        9: "RADIAL_FISHEYE",
        10: "THIN_PRISM_FISHEYE",
    }

    data = []
    for img_id in imagecols.get_img_ids():
        cam_id = imagecols.camimage(img_id).cam_id
        camera = imagecols.cam(cam_id)
        w, h, params = camera.w(), camera.h(), camera.params
        name = imagecols.camimage(img_id).image_name().split("/")[-1]
        p = [name, model_dict[camera.as_dict()["model_id"]], w, h] + params
        data.append(" ".join(map(str, p)))
    with open(out, "w") as f:
        f.write("\n".join(data))


def get_result_filenames(cfg, args):
    ransac_cfg = cfg["ransac"]
    ransac_postfix = ""
    if ransac_cfg["method"] is not None:
        if ransac_cfg["method"] in ["ransac", "hybrid"]:
            ransac_postfix = "_{}".format(ransac_cfg["method"])
        elif ransac_cfg["method"] == "solver":
            ransac_postfix = "_sfransac"
        else:
            raise ValueError(
                "Unsupported ransac method: {}".format(ransac_cfg["method"])
            )
        ransac_postfix += "_{}".format(
            ransac_cfg["thres"]
            if ransac_cfg["method"] != "hybrid"
            else "{}-{}".format(
                ransac_cfg["thres_point"], ransac_cfg["thres_line"]
            )
        )
    results_point = "results_point.txt"
    results_joint = "results_joint_{}{}{}{}{}.txt".format(
        "{}_".format(cfg["2d_matcher"]),
        (
            "{}_".format(cfg["reprojection_filter"])
            if cfg["reprojection_filter"] is not None
            else ""
        ),
        (
            "filtered_"
            if cfg["2d_matcher"] == "superglue_endpoints"
            and cfg["epipolar_filter"]
            else ""
        ),
        cfg["line_cost_func"],
        ransac_postfix,
    )
    return results_point, results_joint


def eval(filename, poses_gt, query_ids, id_to_name, logger):
    errors_t = []
    errors_R = []
    pose_results = {}
    with open(filename) as f:
        for data in f.read().rstrip().split("\n"):
            data = data.split()
            name = data[0]
            q, t = np.split(np.array(data[1:], float), [4])
            pose_results[name] = base.CameraPose(q, t)

    for qid in query_ids:
        name = id_to_name[qid]
        if name not in pose_results:
            e_t = np.inf
            e_R = 180.0
        else:
            R_gt, t_gt = poses_gt[qid].R(), poses_gt[qid].T()
            R, t = pose_results[name].R(), pose_results[name].T()
            e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
            cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1.0, 1.0)
            e_R = np.rad2deg(np.abs(np.arccos(cos)))
        errors_t.append(e_t)
        errors_R.append(e_R)

    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)

    med_t = np.median(errors_t)
    med_R = np.median(errors_R)
    out = f"Results for file {filename}:"
    out += f"\nMedian errors: {med_t:.3f}m, {med_R:.3f}deg"

    out += "\nPercentage of test images localized within:"
    threshs_t = [0.01, 0.02, 0.03, 0.05, 0.25, 0.5, 5.0]
    threshs_R = [1.0, 2.0, 3.0, 5.0, 2.0, 5.0, 10.0]
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        out += f"\n\t{th_t*100:.0f}cm, {th_R:.0f}deg : {ratio*100:.2f}%"
    logger.info(out)


def run_hloc_cambridge(
    cfg,
    image_dir,
    imagecols,
    neighbors,
    train_ids,
    query_ids,
    id_to_origin_name,
    results_file,
    num_loc=10,
    logger=None,
):
    feature_conf = {
        "output": "feats-superpoint-n4096-r1024",
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 4096,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
        },
    }
    retrieval_conf = extract_features.confs["netvlad"]
    matcher_conf = match_features.confs["superglue"]

    results_dir = results_file.parent
    query_list = results_dir / "query_list_with_intrinsics.txt"
    loc_pairs = results_dir / f"pairs-query-netvlad{num_loc}.txt"
    image_list = [
        f"image{img_id:08d}.png" for img_id in (train_ids + query_ids)
    ]
    img_name_to_id = {
        f"image{id:08d}.png": id for id in (train_ids + query_ids)
    }

    imagecols_train = imagecols.subset_by_image_ids(train_ids)
    imagecols_query = imagecols.subset_by_image_ids(query_ids)

    # create query list
    create_query_list(imagecols_query, query_list)
    if logger:
        logger.info(f"Query list created at {query_list}")

    # pairs for retrieval
    if logger:
        logger.info("Extract features for image retrieval...")
    global_descriptors = extract_features.main(
        retrieval_conf,
        Path(cfg["output_dir"]) / image_dir,
        results_dir,
        image_list=image_list,
    )
    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        num_loc,
        db_list=[f"image{img_id:08d}.png" for img_id in train_ids],
        query_list=[f"image{img_id:08d}.png" for img_id in query_ids],
    )

    # feature extraction
    if logger:
        logger.info("Feature Extraction...")
    features = extract_features.main(
        feature_conf,
        Path(cfg["output_dir"]) / image_dir,
        results_dir,
        as_half=True,
        image_list=image_list,
    )
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], results_dir
    )

    # run reference sfm
    if logger:
        logger.info("Running COLMAP for 3D points...")
    neighbors_train = imagecols_train.update_neighbors(neighbors)

    ref_sfm_path = pointsfm.run_colmap_sfm_with_known_poses(
        cfg["sfm"],
        imagecols_train,
        os.path.join(cfg["output_dir"], "tmp_colmap"),
        neighbors=neighbors_train,
        map_to_original_image_names=False,
        skip_exists=cfg["skip_exists"],
    )
    ref_sfm = pycolmap.Reconstruction(ref_sfm_path)

    if not (
        cfg["skip_exists"] or cfg["localization"]["hloc"]["skip_exists"]
    ) or not os.path.exists(results_file):
        # point only localization
        if logger:
            logger.info("Running Point-only localization...")
        localize_sfm.main(
            ref_sfm,
            query_list,
            loc_pairs,
            features,
            loc_matches,
            results_file,
            covisibility_clustering=False,
        )

        # Read coarse poses
        with open(results_file) as f:
            lines = []
            for data in f.read().rstrip().split("\n"):
                data = data.split()
                name = data[0]
                q, t = np.split(np.array(data[1:], float), [4])
                img_id = img_name_to_id[name]
                line = (
                    " ".join(
                        [id_to_origin_name[img_id]]
                        + [str(x) for x in q]
                        + [str(x) for x in t]
                    )
                    + "\n"
                )
                lines.append(line)

        # Change image names back
        with open(results_file, "w") as f:
            f.writelines(lines)

        if logger:
            logger.info(f"Coarse pose saved at {results_file}")
    else:
        if logger:
            logger.info("Point-only localization skipped.")

    # Read coarse poses
    poses = {}
    with open(results_file) as f:
        lines = []
        for data in f.read().rstrip().split("\n"):
            data = data.split()
            name = data[0]
            q, t = np.split(np.array(data[1:], float), [4])
            poses[name] = base.CameraPose(q, t)
    if logger:
        logger.info(f"Coarse pose read from {results_file}")
    hloc_log_file = f"{results_file}_logs.pkl"

    return ref_sfm, poses, hloc_log_file
