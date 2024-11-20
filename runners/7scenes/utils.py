import logging
import os
from pathlib import Path

import numpy as np
import PIL
import PIL.Image
import pycolmap
import torch
from hloc import (
    extract_features,
    localize_sfm,
    match_features,
    pairs_from_covisibility,
    triangulation,
)
from hloc.pipelines.Cambridge.utils import (
    create_query_list_with_intrinsics,
    evaluate,
)
from hloc.utils.read_write_model import read_model, write_model
from tqdm import tqdm

import limap.base as base
import limap.pointsfm as pointsfm
import limap.util.io as limapio

###############################################################################
# The following utils functions are taken/modified from hloc.pipelines.7scenes
###############################################################################

logger = logging.getLogger("hloc")


def create_reference_sfm(full_model, ref_model, blacklist=None, ext=".bin"):
    """Create a new COLMAP model with only training images."""
    logger.info("Creating the reference model.")
    ref_model.mkdir(exist_ok=True)
    cameras, images, points3D = read_model(full_model, ext)

    if blacklist is not None:
        with open(blacklist) as f:
            blacklist = f.read().rstrip().split("\n")

    train_ids = []
    test_ids = []
    images_ref = dict()
    for id_, image in images.items():
        if blacklist and image.name in blacklist:
            test_ids.append(id_)
            continue
        train_ids.append(id_)
        images_ref[id_] = image

    points3D_ref = dict()
    for id_, point3D in points3D.items():
        ref_ids = [i for i in point3D.image_ids if i in images_ref]
        if len(ref_ids) == 0:
            continue
        points3D_ref[id_] = point3D._replace(image_ids=np.array(ref_ids))

    write_model(cameras, images_ref, points3D_ref, ref_model, ".bin")
    logger.info(f"Kept {len(images_ref)} images out of {len(images)}.")
    return train_ids, test_ids


def scene_coordinates(p2D, R_w2c, t_w2c, depth, camera):
    assert len(depth) == len(p2D)
    p2D_norm = np.stack(pycolmap.Camera(camera._asdict()).image_to_world(p2D))
    p2D_h = np.concatenate([p2D_norm, np.ones_like(p2D_norm[:, :1])], 1)
    p3D_c = p2D_h * depth[:, None]
    p3D_w = (p3D_c - t_w2c) @ R_w2c
    return p3D_w


def interpolate_depth(depth, kp):
    h, w = depth.shape
    kp = kp / np.array([[w - 1, h - 1]]) * 2 - 1
    assert np.all(kp > -1) and np.all(kp < 1)
    depth = torch.from_numpy(depth)[None, None]
    kp = torch.from_numpy(kp)[None, None]
    grid_sample = torch.nn.functional.grid_sample

    # To maximize the number of points that have depth:
    # do bilinear interpolation first and then nearest for the remaining points
    interp_lin = grid_sample(depth, kp, align_corners=True, mode="bilinear")[
        0, :, 0
    ]
    interp_nn = torch.nn.functional.grid_sample(
        depth, kp, align_corners=True, mode="nearest"
    )[0, :, 0]
    interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
    valid = ~torch.any(torch.isnan(interp), 0)

    interp_depth = interp.T.numpy().flatten()
    valid = valid.numpy()
    return interp_depth, valid


def image_path_to_rendered_depth_path(image_name):
    parts = image_name.split("/")
    name = "_".join(["".join(parts[0].split("-")), parts[1]])
    name = name.replace("color", "pose")
    name = name.replace("png", "depth.tiff")
    return name


def project_to_image(p3D, R, t, camera, eps: float = 1e-4, pad: int = 1):
    p3D = (p3D @ R.T) + t
    visible = p3D[:, -1] >= eps  # keep points in front of the camera
    p2D_norm = p3D[:, :-1] / p3D[:, -1:].clip(min=eps)
    p2D = np.stack(pycolmap.Camera(camera._asdict()).world_to_image(p2D_norm))
    size = np.array([camera.width - pad - 1, camera.height - pad - 1])
    valid = np.all((p2D >= pad) & (p2D <= size), -1)
    valid &= visible
    return p2D[valid], valid


def correct_sfm_with_gt_depth(sfm_path, depth_folder_path, output_path):
    cameras, images, points3D = read_model(sfm_path)
    logger.info("Correcting sfm using depth...")
    for imgid, img in tqdm(images.items()):
        image_name = img.name
        depth_name = image_path_to_rendered_depth_path(image_name)

        depth = PIL.Image.open(Path(depth_folder_path) / depth_name)
        depth = np.array(depth).astype("float64")
        depth = depth / 1000.0  # mm to meter
        depth[(depth == 0.0) | (depth > 1000.0)] = np.nan

        R_w2c, t_w2c = img.qvec2rotmat(), img.tvec
        camera = cameras[img.camera_id]
        p3D_ids = img.point3D_ids
        p3Ds = np.stack([points3D[i].xyz for i in p3D_ids[p3D_ids != -1]], 0)

        p2Ds, valids_projected = project_to_image(p3Ds, R_w2c, t_w2c, camera)
        invalid_p3D_ids = p3D_ids[p3D_ids != -1][~valids_projected]
        interp_depth, valids_backprojected = interpolate_depth(depth, p2Ds)
        scs = scene_coordinates(
            p2Ds[valids_backprojected],
            R_w2c,
            t_w2c,
            interp_depth[valids_backprojected],
            camera,
        )
        invalid_p3D_ids = np.append(
            invalid_p3D_ids,
            p3D_ids[p3D_ids != -1][valids_projected][~valids_backprojected],
        )
        for p3did in invalid_p3D_ids:
            if p3did == -1:
                continue
            else:
                obs_imgids = points3D[p3did].image_ids
                invalid_imgids = list(np.where(obs_imgids == img.id)[0])
                points3D[p3did] = points3D[p3did]._replace(
                    image_ids=np.delete(obs_imgids, invalid_imgids),
                    point2D_idxs=np.delete(
                        points3D[p3did].point2D_idxs, invalid_imgids
                    ),
                )

        new_p3D_ids = p3D_ids.copy()
        sub_p3D_ids = new_p3D_ids[new_p3D_ids != -1]
        valids = np.ones(np.count_nonzero(new_p3D_ids != -1), dtype=bool)
        valids[~valids_projected] = False
        valids[valids_projected] = valids_backprojected
        sub_p3D_ids[~valids] = -1
        new_p3D_ids[new_p3D_ids != -1] = sub_p3D_ids
        img = img._replace(point3D_ids=new_p3D_ids)

        assert len(img.point3D_ids[img.point3D_ids != -1]) == len(
            scs
        ), f"{len(scs)}, {len(img.point3D_ids[img.point3D_ids != -1])}"
        for i, p3did in enumerate(img.point3D_ids[img.point3D_ids != -1]):
            points3D[p3did] = points3D[p3did]._replace(xyz=scs[i])
        images[imgid] = img

    output_path.mkdir(parents=True, exist_ok=True)
    write_model(cameras, images, points3D, output_path)


###############################################################################
# End of hloc utils
###############################################################################


class DepthReader(base.BaseDepthReader):
    def __init__(self, filename, depth_folder):
        super().__init__(filename)
        self.depth_folder = depth_folder

    def read(self, filename):
        depth = PIL.Image.open(Path(self.depth_folder) / filename)
        depth = np.array(depth).astype("float64")
        depth = depth / 1000.0  # mm to meter
        depth[(depth == 0.0) | (depth > 1000.0)] = np.inf
        return depth


def read_scene_7scenes(cfg, root_path, model_path, image_path, n_neighbors=20):
    metainfos_filename = "infos_7scenes.npy"
    output_dir = "tmp" if cfg["output_dir"] is None else cfg["output_dir"]
    limapio.check_makedirs(output_dir)
    if cfg["skip_exists"] and os.path.exists(
        os.path.join(output_dir, metainfos_filename)
    ):
        cfg["info_path"] = os.path.join(output_dir, metainfos_filename)
    if cfg["info_path"] is None:
        imagecols, neighbors, ranges = pointsfm.read_infos_colmap(
            cfg["sfm"],
            root_path,
            model_path,
            image_path,
            n_neighbors=n_neighbors,
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


def get_result_filenames(cfg, use_dense_depth=False):
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
    results_point = "results_{}_point.txt".format(
        "dense" if use_dense_depth else "sparse"
    )
    results_joint = "results_{}_joint_{}{}{}{}{}.txt".format(
        "dense" if use_dense_depth else "sparse",
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
    if cfg["2d_matcher"] == "gluestick":
        results_point = results_point.replace("point", "point_gluestick")
        results_joint = results_joint.replace("gluestick", "gluestickp+l")
    return results_point, results_joint


def get_train_test_ids_from_sfm(full_model, blacklist=None, ext=".bin"):
    cameras, images, points3D = read_model(full_model, ext)

    if blacklist is not None:
        with open(blacklist) as f:
            blacklist = f.read().rstrip().split("\n")

    train_ids, test_ids = [], []
    for id_, image in images.items():
        if blacklist and image.name in blacklist:
            test_ids.append(id_)
        else:
            train_ids.append(id_)

    return train_ids, test_ids


def run_hloc_7scenes(
    cfg,
    dataset,
    scene,
    results_file,
    test_list,
    num_covis=30,
    use_dense_depth=False,
    logger=None,
):
    results_dir = results_file.parent
    gt_dir = dataset / f"7scenes_sfm_triangulated/{scene}/triangulated"

    ref_sfm_sift = results_dir / "sfm_sift"
    ref_sfm = results_dir / "sfm_superpoint+superglue"
    query_list = results_dir / "query_list_with_intrinsics.txt"
    sfm_pairs = results_dir / f"pairs-db-covis{num_covis}.txt"
    depth_dir = dataset / f"depth/7scenes_{scene}/train/depth"
    retrieval_path = (
        dataset / "7scenes_densevlad_retrieval_top_10" / f"{scene}_top10.txt"
    )
    feature_conf = {
        "output": "feats-superpoint-n4096-r1024",
        "model": {"name": "superpoint", "nms_radius": 3, "max_keypoints": 4096},
        "preprocessing": {
            "globs": ["*.color.png"],
            "grayscale": True,
            "resize_max": 1024,
        },
    }
    if cfg["localization"]["2d_matcher"] == "gluestick":
        raise ValueError("GlueStick not yet supported in HLoc.")
        # matcher_conf = match_features.confs['gluestick']
    else:
        matcher_conf = match_features.confs["superglue"]
        matcher_conf["model"]["sinkhorn_iterations"] = 5

    # feature extraction
    features = extract_features.main(
        feature_conf, dataset / scene, results_dir, as_half=True
    )

    train_ids, query_ids = get_train_test_ids_from_sfm(gt_dir, test_list)
    create_reference_sfm(gt_dir, ref_sfm_sift, test_list)
    create_query_list_with_intrinsics(gt_dir, query_list, test_list)
    if not sfm_pairs.exists():
        pairs_from_covisibility.main(
            ref_sfm_sift, sfm_pairs, num_matched=num_covis
        )
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], results_dir
    )
    loc_matches = match_features.main(
        matcher_conf, retrieval_path, feature_conf["output"], results_dir
    )
    if not ref_sfm.exists():
        triangulation.main(
            ref_sfm,
            ref_sfm_sift,
            dataset / scene,
            sfm_pairs,
            features,
            sfm_matches,
        )

    if use_dense_depth:
        assert depth_dir is not None
        ref_sfm_fix = results_dir / "sfm_superpoint+superglue+depth"
        if not cfg["skip_exists"] or not ref_sfm_fix.exists():
            correct_sfm_with_gt_depth(ref_sfm, depth_dir, ref_sfm_fix)
        ref_sfm = ref_sfm_fix

    ref_sfm = pycolmap.Reconstruction(ref_sfm)

    if not (
        cfg["skip_exists"] or cfg["localization"]["hloc"]["skip_exists"]
    ) or not os.path.exists(results_file):
        # point only localization
        if logger:
            logger.info("Running Point-only localization...")
        localize_sfm.main(
            ref_sfm,
            query_list,
            retrieval_path,
            features,
            loc_matches,
            results_file,
            covisibility_clustering=False,
            prepend_camera_name=True,
        )
        if logger:
            logger.info(f"Coarse pose saved at {results_file}")
        evaluate(gt_dir, results_file, test_list)
    else:
        if logger:
            logger.info("Point-only localization skipped.")

    # Read coarse poses
    poses = {}
    with open(results_file) as f:
        for data in f.read().rstrip().split("\n"):
            data = data.split()
            name = data[0]
            q, t = np.split(np.array(data[1:], float), [4])
            poses[name] = base.CameraPose(q, t)
    if logger:
        logger.info(f"Coarse pose read from {results_file}")
    hloc_log_file = f"{results_file}_logs.pkl"

    return poses, hloc_log_file, {"train": train_ids, "query": query_ids}
