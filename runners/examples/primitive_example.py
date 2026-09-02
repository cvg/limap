"""
Holistic reconstruction with cylinders and spheres from SAM3 segmentation.

This runner builds a holistic 3D reconstruction that includes points, lines,
vanishing points, planes, cylinders, and spheres. Cylinders and spheres are
detected via SAM3 instance segmentation masks for known object categories.

Usage:
    python runners/examples/primitive_example.py \
        --image_dir ~/data/primitive_samples/tmp_images/ \
        --model_dir ~/data/primitive_samples/primitive_video_outputs/sparse/0/ \
        --sam3_dir ~/data/primitive_samples/sam3_output/ \
        --output_dir ~/outputs/primitive_recon/
"""

import argparse
from pathlib import Path

import pycolmap
from pycolmap import logging

import limap.geometry
import limap.runners
import limap.sfm
import limap.util.io as limapio
from limap.image.dense_matcher import associate_via_dense_matching
from limap.image.dense_matcher.specs import DenseMatchingOptions
from limap.image.group_voting import vote_unmatched_groups
from limap.image.groups import sam3_group_description
from limap.runners.pipeline_steps import MetaInfoComputerOptions
from limap.runners.structure_frontend import (
    StructureFrontendOptions,
    structure_frontend_from_model,
)


DEFAULT_CATEGORY_TO_TYPE = {
    "can": limap.geometry.GroupType.CYLINDER,
    "football": limap.geometry.GroupType.SPHERE,
    "postcard": limap.geometry.GroupType.PLANE,
    "laptop": limap.geometry.GroupType.PLANE,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Holistic reconstruction with SAM3-based primitives"
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Path to image directory",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Path to COLMAP model directory (sparse/0/)",
    )
    parser.add_argument(
        "--sam3_dir",
        type=Path,
        required=True,
        help="Path to SAM3 output directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for reconstruction",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=50,
        help="Number of covisible neighbors per image",
    )
    parser.add_argument(
        "--max_image_dim",
        type=int,
        default=800,
        help="Maximum image dimension (resize if larger)",
    )
    parser.add_argument(
        "--skip_frontend",
        action="store_true",
        help="Skip Step 1 (point/line detection and matching)",
    )
    parser.add_argument(
        "--skip_groups_from_sam3",
        action="store_true",
        help="Skip SAM3 group injection and group voting",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help=(
            "SAM3 category:type pairs, e.g. 'can:CYLINDER football:SPHERE'. "
            "Defaults to can→CYLINDER, football→SPHERE"
        ),
    )
    # Visualization flags
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Launch Open3D visualization after reconstruction",
    )
    parser.add_argument("--show_planes", action="store_true")
    parser.add_argument("--show_spheres", action="store_true")
    parser.add_argument("--show_cylinders", action="store_true")
    return parser.parse_args()


def parse_category_to_type(
    categories: list[str] | None,
) -> dict[str, limap.geometry.GroupType]:
    """Parse category:type pairs from CLI args."""
    if categories is None:
        return dict(DEFAULT_CATEGORY_TO_TYPE)

    type_map = {
        "CYLINDER": limap.geometry.GroupType.CYLINDER,
        "SPHERE": limap.geometry.GroupType.SPHERE,
        "PLANE": limap.geometry.GroupType.PLANE,
    }
    result = {}
    for pair in categories:
        parts = pair.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid category:type pair: '{pair}'. "
                f"Expected format: 'category:TYPE'"
            )
        cat_name, type_name = parts
        if type_name.upper() not in type_map:
            raise ValueError(
                f"Unknown group type: '{type_name}'. "
                f"Valid types: {list(type_map.keys())}"
            )
        result[cat_name] = type_map[type_name.upper()]
    return result


def main():
    args = parse_args()
    category_to_type = parse_category_to_type(args.categories)
    logging.info(
        "Category mapping: "
        + ", ".join(f"{k}→{v.name}" for k, v in category_to_type.items())
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ###########################################################################
    # Step 1: Structure frontend (detection + matching for points, lines, VPs).
    #         Group matching is deferred to after SAM3 injection (Step 3).
    ###########################################################################
    logging.info("=" * 60)
    logging.info("Step 1: Structure frontend")
    logging.info("=" * 60)

    frontend_options = StructureFrontendOptions(
        max_image_dim=args.max_image_dim,
        cleanup_workspace=False,  # keep files for SAM3 injection
    )
    frontend_options.metainfo.pair_generation.n_neighbors = args.n_neighbors
    # Skip MoGe plane detection — all groups come from SAM3
    frontend_options.image_description.group_description.detect_plane = False
    # Skip group matching in frontend — we'll match all groups after SAM3
    frontend_options.image_association.skip_group_matching = True

    frontend_outputs = structure_frontend_from_model(
        frontend_options,
        args.image_dir,
        args.model_dir,
        output_dir,
        skip_if_exists=args.skip_frontend,
    )

    db_path = frontend_outputs.db_path
    structure_db_path = frontend_outputs.structure_db_path
    image_dir = frontend_outputs.image_dir
    model_dir = frontend_outputs.model_dir
    neighbors = frontend_outputs.neighbors
    ranges = frontend_outputs.ranges

    recon = pycolmap.Reconstruction(model_dir)
    image_names = {
        img_id: image_dir / img.name for img_id, img in recon.images.items()
    }
    group_workspace = output_dir / "frontend" / "group_description"

    if not args.skip_groups_from_sam3:
        #######################################################################
        # Step 2: Inject SAM3 groups (CYLINDER, SPHERE, extra PLANEs)
        #######################################################################
        logging.info("=" * 60)
        logging.info("Step 2: Inject SAM3 groups")
        logging.info("=" * 60)

        # Load existing start_ids from VP/plane detection
        existing_start_ids_path = group_workspace / "start_ids.npy"
        existing_start_ids = None
        if existing_start_ids_path.exists():
            existing_start_ids = limapio.read_npy(
                existing_start_ids_path
            ).item()

        # Don't pass recon — the recon has COLMAP's original keypoints which
        # differ from the frontend's SuperPoint detections stored in the DB.
        # Using recon.points2D would produce indices exceeding structure
        # num_points.
        start_ids = sam3_group_description(
            image_names=image_names,
            sam3_base_dir=args.sam3_dir,
            category_to_type=category_to_type,
            output_dir=group_workspace,
            db_path=db_path,
            structure_db_path=structure_db_path,
            existing_start_ids=existing_start_ids,
        )
        logging.info(
            f"Start IDs after SAM3 injection: {list(start_ids.keys())}"
        )

        #######################################################################
        # Step 3: Match groups via dense matching + voting fallback.
        #######################################################################
        logging.info("=" * 60)
        logging.info("Step 3: Group matching (dense + voting)")
        logging.info("=" * 60)

        dense_opts = DenseMatchingOptions()
        dense_opts.group_matching.overlap_thresh = 0.4
        dense_opts.group_matching.min_num_pixels = 200
        associate_via_dense_matching(
            dense_opts,
            image_names,
            neighbors,
            group_workspace,
            db_path,
            structure_db_path,
        )

        # Voting fallback for groups without dense matches (e.g. VPs)
        from limap._limap._image._groups import GroupVotingOptions

        vote_unmatched_groups(
            GroupVotingOptions(), neighbors, db_path, structure_db_path
        )

    ###########################################################################
    # Step 4: Multi-view point triangulation
    ###########################################################################
    logging.info("=" * 60)
    logging.info("Step 4: Point triangulation")
    logging.info("=" * 60)

    recon = pycolmap.Reconstruction(model_dir)
    triangulated_model = output_dir / "point_triangulation"
    point_tri_options = pycolmap.IncrementalPipelineOptions()
    point_tri_options.min_num_matches = 1
    pycolmap.triangulate_points(
        recon,
        db_path,
        image_dir,
        triangulated_model,
        options=point_tri_options,
        clear_points=True,
    )
    if ranges is None:
        ranges = limap.runners.compute_ranges(
            triangulated_model, MetaInfoComputerOptions().range_calculator
        )
    recon = pycolmap.Reconstruction(triangulated_model)

    ###########################################################################
    # Step 5: Global structure triangulation (lines + ALL group types)
    ###########################################################################
    logging.info("=" * 60)
    logging.info("Step 5: Global structure triangulation")
    logging.info("=" * 60)

    tri_options = limap.sfm.GlobalStructureTriangulationOptions()
    holistic_recon = limap.sfm.global_structure_triangulation(
        recon, structure_db_path, tri_options
    )

    ###########################################################################
    # Step 6: Write final model
    ###########################################################################
    final_model_dir = output_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    holistic_recon.write(final_model_dir)
    logging.info(f"Final model written to: {final_model_dir}")
    logging.info(f"Reconstruction summary: {holistic_recon}")

    # Print group summary
    srecon = holistic_recon.structure_recon
    type_counts = {}
    for g in srecon.groups3D.values():
        name = g.type.name
        type_counts[name] = type_counts.get(name, 0) + 1
    for type_name, count in sorted(type_counts.items()):
        logging.info(f"  {type_name}: {count} groups")

    ###########################################################################
    # Step 8: Visualization (optional)
    ###########################################################################
    if args.visualize:
        import sys

        viz_args = [
            sys.argv[0],
            "-i",
            str(final_model_dir),
        ]
        if args.show_planes:
            viz_args.append("--show_planes")
        if args.show_spheres:
            viz_args.append("--show_spheres")
        if args.show_cylinders:
            viz_args.append("--show_cylinders")

        # Import and run the visualization script
        sys.argv = viz_args
        import visualize_holistic_recon

        viz_args_parsed = visualize_holistic_recon.parse_args()
        visualize_holistic_recon.args = viz_args_parsed
        visualize_holistic_recon.main(viz_args_parsed)

    return final_model_dir


if __name__ == "__main__":
    main()
