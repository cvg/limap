import os
from pathlib import Path

import imagesize
import numpy as np
from hloc import extract_features, localize_inloc, match_features
from hloc.utils.parsers import parse_retrieval
from scipy.io import loadmat
from tqdm import tqdm

import limap.base as base
import limap.util.io as limapio


class InLocP3DReader(base.BaseP3DReader):
    def __init__(self, filename):
        super().__init__(filename)

    def read(self, filename):
        scan = loadmat(str(filename) + ".mat")["XYZcut"]
        return scan


def read_dataset_inloc(
    cfg, dataset_dir, loc_pairs, exclude_CSE=True, logger=None
):
    retrieval_dict = parse_retrieval(loc_pairs)
    queries = list(retrieval_dict.keys())

    metainfos_filename = "infos_inloc.npy"
    output_dir = cfg["output_dir"]
    limapio.check_makedirs(output_dir)
    if cfg["skip_exists"] and os.path.exists(
        os.path.join(output_dir, metainfos_filename)
    ):
        cfg["info_path"] = os.path.join(output_dir, metainfos_filename)

    paths = []
    for g in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"]:
        paths += list(Path(dataset_dir).glob("**/" + g))
    if len(paths) == 0:
        raise ValueError(f"Could not find any image in {dataset_dir}.")
    paths = sorted(list(set(paths)))
    names = [i.relative_to(dataset_dir).as_posix() for i in paths]
    if exclude_CSE:
        names_ = []
        for name in names:
            if "CSE" not in name:
                names_.append(name)
        names = names_
    if logger:
        logger.info(
            f"Found {len(names)} images in {dataset_dir}, \
              excluding CSE scenes: {exclude_CSE}"
        )

    if cfg["info_path"] is None:
        cameras, camimages = [], []
        train_ids, query_ids = [], []
        scales = (
            {}
            if cfg["max_image_dim"] != -1 and cfg["max_image_dim"] is not None
            else None
        )
        for img_id, name in tqdm(enumerate(names)):
            width, height = imagesize.get(str(dataset_dir / name))
            cx = 0.5 * width
            cy = 0.5 * height
            focal_length = max(width, height) * 28.0 / 36.0
            cam_dict = {
                "model_id": 0,  # SIMPLE_PINHOLE
                "cam_id": img_id,
                "width": width,
                "height": height,
                "params": [focal_length, cx, cy],
            }
            cam = base.Camera(cam_dict)
            cameras.append(cam)
            campose = base.CameraPose()

            if name in queries:
                query_ids.append(img_id)
                if scales is not None:
                    scales[name] = cfg["max_image_dim"] / max(width, height)
            else:
                train_ids.append(img_id)
                Tr = localize_inloc.get_scan_pose(dataset_dir, name)
                # Cam2World -> World2Cam
                R = Tr[:3, :3].T
                T = -R @ Tr[:3, -1:]
                campose = base.CameraPose(R, T)
            camimages.append(
                base.CameraImage(cam, campose, str(dataset_dir / name))
            )

        imagecols = base.ImageCollection(cameras, camimages)
        with open(os.path.join(output_dir, metainfos_filename), "wb") as f:
            np.savez(
                f,
                imagecols_np=imagecols.as_dict(),
                train_ids=train_ids,
                query_ids=query_ids,
                scales=scales,
            )
    else:
        with open(cfg["info_path"], "rb") as f:
            data = np.load(f, allow_pickle=True)
            imagecols_np, train_ids, query_ids, scales = (
                data["imagecols_np"].item(),
                data["train_ids"],
                data["query_ids"],
                data["scales"].item(),
            )
            imagecols = base.ImageCollection(imagecols_np)
    return imagecols, train_ids, query_ids, names, scales


def get_result_filenames(cfg, use_temporal=True):
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
            else "{}_{}".format(
                ransac_cfg["thres_point"], ransac_cfg["thres_line"]
            )
        )
        ransac_postfix += (
            "_{}".format(ransac_cfg["weight_line"])
            if ransac_cfg["method"] == "hybrid"
            else ""
        )
    results_point = "results_{}point.txt".format(
        "temporal_" if use_temporal else ""
    )
    results_joint = "results_newlsq_{}joint_{}{}{}{}{}.txt".format(
        "temporal_" if use_temporal else "",
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


def run_hloc_inloc(
    cfg, dataset, loc_pairs, results_file, num_skip=15, logger=None
):
    feature_conf = extract_features.confs["superpoint_inloc"]
    feature_conf["model"]["nms_radius"] = 3
    matcher_conf = match_features.confs["superglue"]

    results_dir = results_file.parent
    feature_path = extract_features.main(feature_conf, dataset, results_dir)
    match_path = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], results_dir
    )

    if not (
        cfg["skip_exists"] or cfg["localization"]["hloc"]["skip_exists"]
    ) or not os.path.exists(results_file):
        # point only localization
        if logger:
            logger.info("Running Point-only localization...")
        localize_inloc.main(
            dataset,
            loc_pairs,
            feature_path,
            match_path,
            results_file,
            skip_matches=num_skip,
        )  # skip database images with too few matches
        if logger:
            logger.info(f"Coarse pose saved at {results_file}")
    else:
        logger.info("Point-only localization skipped.")

    # Read coarse poses and inliers
    poses = {}
    with open(results_file) as f:
        for data in f.read().rstrip().split("\n"):
            data = data.split()
            name = data[0]
            q, t = np.split(np.array(data[1:], float), [4])
            poses[name] = base.CameraPose(q, t)
    logger.info(f"Coarse pose read from {results_file}")
    hloc_log_file = f"{results_file}_logs.pkl"

    query_prefix = "query/iphone7/"
    poses = {query_prefix + name: poses[name] for name in poses}

    return poses, hloc_log_file
