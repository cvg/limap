"""7scenes localization entry point.

This script runs point-line localization on the 7scenes dataset.
"""

import argparse
from pathlib import Path

import pycolmap
from pycolmap import logging
from dacite import from_dict, Config

import limap.runners as runners
import limap.estimators as estimators
import limap.estimators.bundle_adjustment as ba
import limap.sfm
import limap.util.config as cfgutils
from limap.scene import HolisticReconstruction

from hloc.utils.parsers import parse_retrieval

from utils import (
    DepthReader,
    build_db_model,
    correct_sfm_with_gt_depth,
    evaluate,
    get_train_test_ids,
    get_query_images,
    image_path_to_rendered_depth_path,
)


def parse_config():
    """Parse command line arguments and load config."""
    arg_parser = argparse.ArgumentParser(
        description="Run localization with point and lines"
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
        help="default config file (base config)",
    )
    arg_parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="7scenes root path",
    )
    arg_parser.add_argument(
        "-s",
        "--scene",
        type=str,
        required=True,
        help="scene name",
    )
    arg_parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="output directory",
    )
    arg_parser.add_argument(
        "--query_images",
        type=Path,
        default=None,
        help="Path to the file listing query images",
    )
    arg_parser.add_argument(
        "--eval_file",
        type=Path,
        default=None,
        help="Evaluate this results file instead of running localization",
    )
    arg_parser.add_argument(
        "--results_file",
        type=Path,
        default=None,
        help="Output results filename (default: results_{sparse,dense}.txt)",
    )
    arg_parser.add_argument(
        "--num_covis_pairs",
        type=int,
        default=30,
        help="Number of covisible pairs for triangulation",
    )
    arg_parser.add_argument(
        "--num_retrieved",
        type=int,
        default=10,
        help="Number of retrieved images for localization",
    )
    arg_parser.add_argument(
        "--db_model",
        type=Path,
        default=None,
        help=(
            "Prebuilt train-only COLMAP model to use as the database "
            "(default: derive it from the GT model into <output_dir>/db_model)"
        ),
    )
    arg_parser.add_argument(
        "--use_dense_depth",
        action="store_true",
        help="Use dense depth for line reconstruction",
    )
    arg_parser.add_argument(
        "--skip_exists",
        action="store_true",
        help="Skip existing results",
    )
    arg_parser.add_argument(
        "--use_points_only",
        action="store_true",
        help="Use only points for localization (no lines)",
    )
    arg_parser.add_argument(
        "--uncalibrated",
        action="store_true",
        help=(
            "Withhold the query focal length and estimate it along with the "
            "pose. The database model keeps its calibration."
        ),
    )
    arg_parser.add_argument(
        "--focal_init_factor",
        type=float,
        default=1.2,
        help=(
            "With --uncalibrated, the initial focal as a factor of the "
            "larger image side"
        ),
    )
    args = arg_parser.parse_args()

    # Load config: merge default.yaml with scene-specific yaml
    cfg = cfgutils.load_config(
        args.config_file, default_path=args.default_config_file
    )

    # Set output directory
    if args.output_dir is None:
        cfg["output_dir"] = Path(f"tmp/7scenes/{args.scene}")
    else:
        cfg["output_dir"] = args.output_dir

    cfg["skip_exists"] = args.skip_exists
    cfg["use_groups"] = False
    cfg["use_points_only"] = args.use_points_only
    cfg["pose_estimation"]["estimate_focal_length"] = args.uncalibrated

    # Set n_neighbors for triangulation/reconstruction
    cfg["n_neighbors"] = args.num_covis_pairs
    if "metainfo" not in cfg:
        cfg["metainfo"] = {}
    if "pair_generation" not in cfg["metainfo"]:
        cfg["metainfo"]["pair_generation"] = {}
    cfg["metainfo"]["pair_generation"]["n_neighbors"] = args.num_covis_pairs

    # Set n_retrieval for localization
    cfg["n_retrieval"] = args.num_retrieved

    return cfg, args


def main():
    cfg, args = parse_config()

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    logging.info(f'Working on scene "{args.scene}".')

    # Paths
    gt_dir = (
        args.dataset / f"7scenes_sfm_triangulated/{args.scene}/triangulated"
    )
    image_dir = args.dataset / args.scene
    depth_dir = args.dataset / f"depth/7scenes_{args.scene}/train/depth"
    retrieval_path = (
        args.dataset
        / "7scenes_densevlad_retrieval_top_10"
        / f"{args.scene}_top10.txt"
    )
    test_list = args.query_images or gt_dir / "list_test.txt"

    # Just evaluate if requested
    if args.eval_file is not None:
        evaluate(gt_dir, args.eval_file, test_list)
        return

    # Load full reconstruction and split train/test
    full_recon = pycolmap.Reconstruction(gt_dir)
    train_ids, query_ids = get_train_test_ids(full_recon, test_list)
    logging.info(
        f"Train images: {len(train_ids)}, Query images: {len(query_ids)}"
    )

    # Create query images dict
    query_images = get_query_images(
        full_recon,
        query_ids,
        image_dir,
        uncalibrated=args.uncalibrated,
        focal_init_factor=args.focal_init_factor,
    )
    if args.uncalibrated:
        init_focal = next(iter(query_images.values()))[1].focal_length
        logging.info(
            f"Uncalibrated queries: focal initialized to {init_focal:.1f} "
            "and estimated per image"
        )

    # Localization must not use the query images to build the map, so
    # the database model holds only the database images.
    db_model_dir = args.db_model or build_db_model(
        gt_dir, test_list, output_dir / "db_model"
    )

    # Type hooks for C++ classes that support dict construction
    type_hooks = {
        ba.GroupBundleAdjustmentOptions: ba.GroupBundleAdjustmentOptions,
        # Only pybind option types need a hook; dataclasses recurse.
        limap.sfm.GlobalLineTriangulationOptions: (
            limap.sfm.GlobalLineTriangulationOptions
        ),
        limap.sfm.GlobalGroupTriangulationOptions: (
            limap.sfm.GlobalGroupTriangulationOptions
        ),
        limap.sfm.GroupVerificationOptions: limap.sfm.GroupVerificationOptions,
        estimators.absolute_pose.PointLineAbsolutePoseOptions: (
            estimators.absolute_pose.PointLineAbsolutePoseOptions
        ),
    }

    # Results path
    if args.results_file is not None:
        results_path = args.results_file
    else:
        depth_suffix = "_dense" if args.use_dense_depth else "_sparse"
        results_path = output_dir / f"results{depth_suffix}.txt"

    # Database triangulation -> HolisticReconstruction
    if not args.use_dense_depth:
        # Sparse: automatic_structure_triangulation
        holistic_model_dir = output_dir / "triangulation" / "final_model"
        if cfg["skip_exists"] and HolisticReconstruction.exists_model(
            str(holistic_model_dir)
        ):
            logging.info(
                f"Skipping reconstruction - output exists: {holistic_model_dir}"
            )
        else:
            logging.info("Running automatic structure triangulation...")
            tri_options = from_dict(
                data_class=runners.AutomaticStructureTriangulationOptions,
                data=cfg,
                config=Config(strict=False, cast=[Path], type_hooks=type_hooks),
            )
            tri_options.skip_frontend_if_exists_database = True
            # Query images are matched against the database images'
            # descriptors, so keep them instead of clearing them after
            # triangulation.
            tri_options.cleanup_frontend_workspace = False
            # Feature config set via _image_description/_image_association
            holistic_model_dir = runners.automatic_structure_triangulation(
                tri_options,
                image_dir,
                db_model_dir,
                output_dir / "triangulation",
            )
        holistic_recon = HolisticReconstruction(str(holistic_model_dir))
    else:
        # Dense: correct_sfm + geometry_guided
        holistic_model_dir = output_dir / "geometry_guided" / "final_model"
        if cfg["skip_exists"] and HolisticReconstruction.exists_model(
            str(holistic_model_dir)
        ):
            logging.info(
                f"Skipping reconstruction - output exists: {holistic_model_dir}"
            )
        else:
            logging.info("Correcting SfM with ground truth depth...")
            corrected_dir = output_dir / "corrected_sfm"
            if not cfg["skip_exists"] or not corrected_dir.exists():
                correct_sfm_with_gt_depth(
                    db_model_dir, depth_dir, corrected_dir
                )

            logging.info("Running geometry-guided line reconstruction...")
            # Create depth readers for train images
            depth_readers = {
                img_id: DepthReader(
                    image_path_to_rendered_depth_path(
                        full_recon.images[img_id].name
                    ),
                    depth_dir,
                )
                for img_id in train_ids
            }

            geo_options = from_dict(
                data_class=runners.GeometryGuidedLineReconstructionOptions,
                data=cfg,
                config=Config(strict=False, cast=[Path], type_hooks=type_hooks),
            )
            holistic_model_dir = runners.line_reconstruction_with_depth_maps(
                geo_options,
                image_dir,
                corrected_dir,
                output_dir / "geometry_guided",
                depth_readers,
            )
        holistic_recon = HolisticReconstruction(str(holistic_model_dir))

    logging.info(f"Loaded reconstruction: {holistic_recon}")

    # Point-line localization
    retrieval = parse_retrieval(retrieval_path)

    loc_options = from_dict(
        data_class=runners.PointLineLocalizationOptions,
        data=cfg,
        config=Config(strict=False, cast=[Path], type_hooks=type_hooks),
    )
    # Set workspace_path to triangulation output (contains frontend/)
    loc_options.workspace_path = holistic_model_dir.parent

    # Build id_to_name mapping using COLMAP's image names
    id_to_name = {
        img_id: full_recon.images[img_id].name for img_id in full_recon.images
    }

    runners.point_line_localization(
        loc_options,
        holistic_recon,
        query_images,
        retrieval,
        results_path,
        id_to_name=id_to_name,
    )

    # Evaluate
    evaluate(gt_dir, results_path, test_list)


if __name__ == "__main__":
    main()
