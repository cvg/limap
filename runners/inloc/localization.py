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

from utils import (
    InLocP3DReader,
    get_result_filenames,
    parse_retrieval,
    read_dataset_inloc,
    run_hloc_inloc,
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
        default="cfgs/localization/inloc.yaml",
        help="config file",
    )
    arg_parser.add_argument(
        "--default_config_file",
        type=str,
        default="cfgs/localization/default.yaml",
        help="default config file",
    )
    arg_parser.add_argument(
        "--dataset", type=Path, required=True, help="inloc dataset path"
    )
    arg_parser.add_argument(
        "--info_path", type=str, default=None, help="load precomputed info"
    )

    arg_parser.add_argument(
        "--num_loc",
        type=int,
        default=40,
        help="Number of image pairs for loc, default: %(default)s",
    )
    arg_parser.add_argument(
        "--num_skip",
        type=int,
        default=15,
        help="skip database images with too few matches, default: %(default)s",
    )
    arg_parser.add_argument(
        "--no_temporal",
        action="store_false",
        dest="use_temporal",
        default=True,
        help="Whether use the temporal retrieved neighbor file of hloc",
    )

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(
        args.config_file, default_path=args.default_config_file
    )
    shortcuts = dict()
    shortcuts["-nv"] = "--n_visible_views"
    shortcuts["-nn"] = "--n_neighbors"
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    if cfg["merging"]["do_merging"] and "--refinement.disable" not in unknown:
        # disable refinement for fitnmerge
        cfg["refinement"]["disable"] = True
    cfg["info_path"] = args.info_path
    cfg["n_neighbors_loc"] = args.num_loc

    # Output path for LIMAP results (tmp)
    if cfg["output_dir"] is None:
        cfg["output_dir"] = "tmp/inloc"
    # Output folder for LIMAP linetracks (in tmp)
    if cfg["output_folder"] is None:
        cfg["output_folder"] = "finaltracks"
    cfg["inloc_dataset"] = (
        args.dataset
    )  # For reading camera poses for estimating 3D lines fron depth
    return cfg, args


def main():
    cfg, args = parse_config()
    cfg = runners.setup(cfg)

    # outputs is for localization-related results
    outputs = Path(cfg["output_dir"]) / "localization"
    outputs.mkdir(exist_ok=True, parents=True)

    logger.info("Working on InLoc.")
    pairs = Path("third-party/Hierarchical-Localization/pairs/inloc/")
    loc_pairs = pairs / "pairs-query-netvlad{}{}.txt".format(
        args.num_loc, "-temporal" if args.use_temporal else ""
    )  # top 40 retrieved by NetVLAD

    imagecols, train_ids, query_ids, img_rel_names, scales = read_dataset_inloc(
        cfg, args.dataset, loc_pairs
    )
    if cfg["max_image_dim"] != -1 and cfg["max_image_dim"] is not None:
        imagecols.set_max_image_dim(cfg["max_image_dim"])

    results_point, results_joint = get_result_filenames(
        cfg["localization"], args.use_temporal
    )
    results_point, results_joint = (
        outputs / results_point,
        outputs / results_joint,
    )

    ##########################################################
    # [A] hloc point-based localization
    ##########################################################
    poses, hloc_log_file = run_hloc_inloc(
        cfg, args.dataset, loc_pairs, results_point, args.num_skip, logger
    )

    ##########################################################
    # [B] LIMAP fitting for database line tracks
    ##########################################################
    imagecols_train = imagecols.subset_by_image_ids(train_ids)

    # Only have tracks if we do fit&merge
    finaltracks_dir = os.path.join(cfg["output_dir"], cfg["output_folder"])
    if (
        not cfg["skip_exists"]
        or not os.path.exists(finaltracks_dir)
        or not cfg["merging"]["do_merging"]
    ):
        logger.info("Running LIMAP fit&merge")
        points_readers = {
            id: InLocP3DReader(imagecols.image_name(id)) for id in train_ids
        }
        linetracks_db = runners.line_fitting_with_points3d(
            cfg,
            imagecols_train,
            points_readers,
            inloc_read_transformations=True,
        )
    else:
        linetracks_db = limapio.read_folder_linetracks(finaltracks_dir)
        logger.info(f"Loaded LIMAP triangulation result from {finaltracks_dir}")

    ##########################################################
    # [C] Localization with points and lines
    ##########################################################
    retrieval = parse_retrieval(loc_pairs)
    img_id_to_name = {id: img_rel_names[id] for id in imagecols.get_img_ids()}
    imagecols_query = imagecols.subset_by_image_ids(query_ids)

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
            hloc_logs, img_id_to_name[qid], resize_scales=scales
        )
        point_correspondences[qid] = {
            "p2ds": p2ds,
            "p3ds": p3ds,
            "inliers": inliers,
        }

    final_poses = runners.hybrid_localization(
        cfg,
        imagecols_train,
        imagecols_query,
        point_correspondences,
        linetracks_db,
        retrieval,
        results_joint,
        img_name_dict=img_id_to_name,
    )

    # Overwrite results with filename without prefix path
    lines = []
    for qid, fpose in zip(query_ids, final_poses):
        name = img_id_to_name[qid]
        fq, ft = fpose.qvec, fpose.tvec
        line = (
            " ".join(
                [name.split("/")[-1]]
                + [str(x) for x in fq]
                + [str(x) for x in ft]
            )
            + "\n"
        )
        lines.append(line)
    with open(results_joint, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
