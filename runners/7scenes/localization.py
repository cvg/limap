import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import argparse
import logging
import pickle
from pathlib import Path

import hloc.utils.read_write_model as colmap_utils
import pycolmap
from hloc.utils.parsers import parse_retrieval
from utils import (
    DepthReader,
    evaluate,
    get_result_filenames,
    image_path_to_rendered_depth_path,
    read_scene_7scenes,
    run_hloc_7scenes,
)

import limap.runners as runners
import limap.util.config as cfgutils
import limap.util.io as limapio

formatter = logging.Formatter(
    fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger("JointLoc")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


def parse_config():
    arg_parser = argparse.ArgumentParser(
        description="run localization with point and lines"
    )
    arg_parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="cfgs/localization/7scenes.yaml",
        help="config file",
    )
    arg_parser.add_argument(
        "--default_config_file",
        type=str,
        default="cfgs/localization/default.yaml",
        help="default config file",
    )
    arg_parser.add_argument(
        "--dataset", type=Path, required=True, help="7scenes root path"
    )
    arg_parser.add_argument(
        "-s", "--scene", type=str, required=True, help="scene name(s)"
    )
    arg_parser.add_argument(
        "--info_path", type=str, default=None, help="load precomputed info"
    )

    arg_parser.add_argument(
        "--query_images",
        default=None,
        type=Path,
        help="Path to the file listing query images",
    )
    arg_parser.add_argument(
        "--eval", default=None, type=Path, help="Path to the result file"
    )

    arg_parser.add_argument(
        "--num_covis",
        type=int,
        default=30,
        help="Number of image pairs for SfM, default: %(default)s",
    )
    arg_parser.add_argument(
        "--num_loc",
        type=int,
        default=10,
        help="Number of image pairs for loc, default: %(default)s",
    )
    arg_parser.add_argument("--use_dense_depth", action="store_true")

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(
        args.config_file, default_path=args.default_config_file
    )
    shortcuts = dict()
    shortcuts["-nv"] = "--n_visible_views"
    shortcuts["-nn"] = "--n_neighbors"
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["info_path"] = args.info_path
    cfg["n_neighbors"] = args.num_covis
    cfg["n_neighbors_loc"] = args.num_loc
    if cfg["merging"]["do_merging"] and "--refinement.disable" not in unknown:
        # disable refinement for fitnmerge
        cfg["refinement"]["disable"] = True

    # Output path for LIMAP results (tmp)
    if cfg["output_dir"] is None:
        cfg["output_dir"] = f"tmp/7scenes/{args.scene}"
    # Output folder for LIMAP linetracks (in tmp)
    if cfg["output_folder"] is None:
        cfg["output_folder"] = "finaltracks"
    cfg["output_folder"] += "_{}".format(
        "dense" if args.use_dense_depth else "sparse"
    )
    return cfg, args


def main():
    cfg, args = parse_config()
    cfg = runners.setup(cfg)

    # outputs is for localization-related results
    outputs = Path(cfg["output_dir"]) / "localization"
    outputs.mkdir(exist_ok=True, parents=True)

    logger.info(f'Working on scene "{args.scene}".')
    gt_dir = f"7scenes_sfm_triangulated/{args.scene}/triangulated"
    imagecols, neighbors, ranges = read_scene_7scenes(
        cfg, str(args.dataset), gt_dir, args.scene, n_neighbors=args.num_covis
    )

    gt_dir = args.dataset / gt_dir
    test_list = args.query_images or gt_dir / "list_test.txt"
    if args.eval is not None:
        evaluate(gt_dir, args.eval, test_list)
        return

    results_point, results_joint = get_result_filenames(
        cfg["localization"], args.use_dense_depth
    )
    if args.use_dense_depth and cfg["merging"]["do_merging"]:
        results_joint = results_joint.replace("dense", "dense_fnm")
    results_point, results_joint = (
        outputs / results_point,
        outputs / results_joint,
    )

    ##########################################################
    # [A] hloc point-based localization
    ##########################################################
    poses, hloc_log_file, ids = run_hloc_7scenes(
        cfg,
        args.dataset,
        args.scene,
        results_point,
        test_list,
        args.num_covis,
        args.use_dense_depth,
        logger,
    )
    train_ids, query_ids = ids["train"], ids["query"]

    # Some paths useful for LIMAP localization too
    ref_sfm_path = outputs / (
        "sfm_superpoint+superglue" + ("+depth" if args.use_dense_depth else "")
    )
    depth_dir = args.dataset / f"depth/7scenes_{args.scene}/train/depth"
    retrieval_path = (
        args.dataset
        / "7scenes_densevlad_retrieval_top_10"
        / f"{args.scene}_top10.txt"
    )

    ##########################################################
    # [B] LIMAP triangulation/fitnmerge for database line tracks
    ##########################################################
    all_images = colmap_utils.read_images_binary(gt_dir / "images.bin")
    imagecols_train = imagecols.subset_by_image_ids(train_ids)
    if not args.use_dense_depth:
        finaltracks_dir = os.path.join(cfg["output_dir"], cfg["output_folder"])
        if not cfg["skip_exists"] or not os.path.exists(finaltracks_dir):
            logger.info("Running LIMAP triangulation...")
            linetracks_db = runners.line_triangulation(
                cfg, imagecols_train, neighbors=neighbors, ranges=ranges
            )
        else:
            linetracks_db = limapio.read_folder_linetracks(finaltracks_dir)
            logger.info(
                f"Loaded LIMAP triangulation result from {finaltracks_dir}"
            )
    else:
        finaltracks_dir = os.path.join(cfg["output_dir"], cfg["output_folder"])
        if (
            not cfg["skip_exists"]
            or not os.path.exists(finaltracks_dir)
            or not cfg["merging"]["do_merging"]
        ):
            logger.info("Running LIMAP fit&merge")
            depths = {
                id: DepthReader(
                    image_path_to_rendered_depth_path(all_images[id].name),
                    depth_dir,
                )
                for id in train_ids
            }
            linetracks_db = runners.line_fitnmerge(
                cfg, imagecols_train, depths, neighbors, ranges
            )
        else:
            linetracks_db = limapio.read_folder_linetracks(finaltracks_dir)
            logger.info(
                f"Loaded LIMAP triangulation result from {finaltracks_dir}"
            )

    ##########################################################
    # [C] Localization with points and lines
    ##########################################################
    retrieval = parse_retrieval(retrieval_path)
    img_id_to_name = {img_id: all_images[img_id].name for img_id in all_images}
    imagecols_query = imagecols.subset_by_image_ids(query_ids)

    # Instantiate ref_sfm
    ref_sfm = pycolmap.Reconstruction(ref_sfm_path)

    # Update coarse poses for epipolar methods
    if (
        cfg["localization"]["2d_matcher"] == "epipolar"
        or cfg["localization"]["epipolar_filter"]
    ):
        name_to_id = {img_id_to_name[img_id]: img_id for img_id in query_ids}
        for qname in poses:
            qid = name_to_id[qname]
            imagecols_query.set_camera_pose(qid, poses[qname])

    # Retrieve point correspondences from hloc
    with open(hloc_log_file, "rb") as f:
        hloc_logs = pickle.load(f)
    point_correspondences = {}
    for qid in query_ids:
        p2ds, p3ds, inliers = runners.get_hloc_keypoints_from_log(
            hloc_logs, img_id_to_name[qid], ref_sfm
        )
        point_correspondences[qid] = {
            "p2ds": p2ds,
            "p3ds": p3ds,
            "inliers": inliers,
        }

    # can return final_poses
    runners.hybrid_localization(
        cfg,
        imagecols_train,
        imagecols_query,
        point_correspondences,
        linetracks_db,
        retrieval,
        results_joint,
        img_name_dict=img_id_to_name,
    )

    evaluate(gt_dir, results_joint, test_list, only_localized=True)


if __name__ == "__main__":
    main()
