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

from hloc.utils.parsers import parse_retrieval
from utils import (
    eval,
    get_result_filenames,
    get_scene_info,
    read_scene_visualsfm,
    run_hloc_cambridge,
    undistort_and_resize,
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
        default="cfgs/localization/cambridge.yaml",
        help="config file",
    )
    arg_parser.add_argument(
        "--default_config_file",
        type=str,
        default="cfgs/localization/default.yaml",
        help="default config file",
    )
    arg_parser.add_argument(
        "-a", "--vsfm_path", type=str, required=True, help="visualsfm path"
    )
    arg_parser.add_argument(
        "--nvm_file",
        type=str,
        default="reconstruction.nvm",
        help="nvm filename",
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
        default=20,
        help="Number of image pairs for SfM, default: %(default)s",
    )
    arg_parser.add_argument(
        "--num_loc",
        type=int,
        default=10,
        help="Number of image pairs for loc, default: %(default)s",
    )

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(
        args.config_file, default_path=args.default_config_file
    )
    shortcuts = dict()
    shortcuts["-nv"] = "--n_visible_views"
    shortcuts["-nn"] = "--n_neighbors"
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["vsfm_path"] = args.vsfm_path
    cfg["nvm_file"] = args.nvm_file
    cfg["info_path"] = args.info_path
    cfg["n_neighbors"] = args.num_covis
    cfg["n_neighbors_loc"] = args.num_loc
    # Output path for LIMAP results (tmp)
    if cfg["output_dir"] is None:
        scene_id = os.path.basename(cfg["vsfm_path"])
        cfg["output_dir"] = f"tmp/cambridge/{scene_id}"
    # Output folder for LIMAP linetracks (in tmp)
    if cfg["output_folder"] is None:
        cfg["output_folder"] = "finaltracks"
    return cfg, args


def main():
    cfg, args = parse_config()
    cfg = runners.setup(cfg)
    scene_id = os.path.basename(cfg["vsfm_path"])

    # outputs is for localization-related results
    outputs = Path(cfg["output_dir"]) / "localization"
    outputs.mkdir(exist_ok=True, parents=True)

    logger.info(f'Working on scene "{scene_id}".')
    imagecols, neighbors, ranges = read_scene_visualsfm(
        cfg,
        cfg["vsfm_path"],
        nvm_file=cfg["nvm_file"],
        n_neighbors=args.num_covis,
    )
    train_ids, query_ids, id_to_origin_name = get_scene_info(
        cfg["vsfm_path"], imagecols, args.query_images
    )

    # GT for queries
    poses_gt = {
        img_id: imagecols.camimage(img_id).pose
        for img_id in imagecols.get_img_ids()
    }

    if args.eval is not None:
        eval(args.eval, poses_gt, query_ids, id_to_origin_name, logger)
        return

    image_dir, imagecols = undistort_and_resize(cfg, imagecols, logger)
    imagecols_train = imagecols.subset_by_image_ids(train_ids)

    results_point, results_joint = get_result_filenames(
        cfg["localization"], args
    )
    results_point, results_joint = (
        outputs / results_point,
        outputs / results_joint,
    )

    img_name_to_id = {
        f"image{id:08d}.png": id for id in (train_ids + query_ids)
    }

    ##########################################################
    # [A] hloc point-based localization
    ##########################################################
    ref_sfm, poses, hloc_log_file = run_hloc_cambridge(
        cfg,
        image_dir,
        imagecols,
        neighbors,
        train_ids,
        query_ids,
        id_to_origin_name,
        results_point,
        args.num_loc,
        logger,
    )
    eval(results_point, poses_gt, query_ids, id_to_origin_name, logger)

    # Some paths useful for LIMAP localization too
    loc_pairs = outputs / f"pairs-query-netvlad{args.num_loc}.txt"

    ##########################################################
    # [B] LIMAP triangulation/fitnmerge for database line tracks
    ##########################################################
    finaltracks_dir = os.path.join(cfg["output_dir"], "finaltracks")
    if not cfg["skip_exists"] or not os.path.exists(finaltracks_dir):
        logger.info("Running LIMAP triangulation...")
        linetracks_db = runners.line_triangulation(
            cfg, imagecols_train, neighbors=neighbors, ranges=ranges
        )
    else:
        linetracks_db = limapio.read_folder_linetracks(finaltracks_dir)
        logger.info(f"Loaded LIMAP triangulation result from {finaltracks_dir}")

    ##########################################################
    # [C] Localization with points and lines
    ##########################################################
    _retrieval = parse_retrieval(loc_pairs)
    imagecols_query = imagecols.subset_by_image_ids(query_ids)

    retrieval = {}
    for name in _retrieval:
        qid = img_name_to_id[name]
        retrieval[id_to_origin_name[qid]] = [
            id_to_origin_name[img_name_to_id[n]] for n in _retrieval[name]
        ]
    hloc_name_dict = {
        id: f"image{id:08d}.png" for id in (train_ids + query_ids)
    }

    # Update coarse poses for epipolar methods
    if (
        cfg["localization"]["2d_matcher"] == "epipolar"
        or cfg["localization"]["epipolar_filter"]
    ):
        name_to_id = {hloc_name_dict[img_id]: img_id for img_id in query_ids}
        for qname in poses:
            qid = name_to_id[qname]
            imagecols_query.set_camera_pose(qid, poses[qname])

    with open(hloc_log_file, "rb") as f:
        hloc_logs = pickle.load(f)
    point_correspondences = {}
    for qid in query_ids:
        p2ds, p3ds, inliers = runners.get_hloc_keypoints_from_log(
            hloc_logs, hloc_name_dict[qid], ref_sfm
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
        img_name_dict=id_to_origin_name,
    )

    # Evaluate
    eval(results_joint, poses_gt, query_ids, id_to_origin_name, logger)


if __name__ == "__main__":
    main()
