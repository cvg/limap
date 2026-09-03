r"""
Runner for global structure triangulation.

This runner runs global structure triangulation (lines + groups) using a
pre-computed structure database and COLMAP model.

Usage::

    python -m limap.cli.global_structure_triangulation \
        --structure_db_path <path_to_structure_database.db> \
        --model_path <path_to_colmap_model> \
        --output_dir <output_directory>
"""

import argparse
from pathlib import Path

import pycolmap
from pycolmap import logging

import limap.estimators.bundle_adjustment
import limap.sfm
import limap.util.config as cfgutils
import limap.util.io as limapio
from limap._limap import _sfm
from limap.scene import (
    HolisticReconstruction,
    StructureDatabase,
    StructureDatabaseCache,
)


def run_global_structure_triangulation(
    structure_db_path: Path,
    model_path: Path,
    output_dir: Path,
    cfg: dict,
):
    """Run global structure triangulation using pre-computed databases.

    Args:
        structure_db_path: Path to structure_database.db
        model_path: Path to COLMAP model directory
        output_dir: Output directory for results
        cfg: Configuration dictionary with triangulation options
    """
    if not structure_db_path.exists():
        raise FileNotFoundError(
            f"Structure database not found: {structure_db_path}"
        )
    if not model_path.exists():
        raise FileNotFoundError(f"COLMAP model not found: {model_path}")

    # Load point reconstruction and structure database
    logging.info(f"Loading point reconstruction from {model_path}")
    point_recon = pycolmap.Reconstruction(model_path)

    logging.info(f"Loading structure database from {structure_db_path}")
    with StructureDatabase.open(structure_db_path) as structure_db:
        structure_db_cache = StructureDatabaseCache.create(structure_db)

    # Create holistic reconstruction
    holistic_recon = HolisticReconstruction(point_recon)

    # Configure options
    line_options = _sfm.GlobalLineTriangulationOptions()
    for key, value in cfg.get("global_line_triangulation", {}).items():
        if hasattr(line_options, key):
            setattr(line_options, key, value)

    group_options = _sfm.GlobalGroupTriangulationOptions()
    for key, value in cfg.get("global_group_triangulation", {}).items():
        if hasattr(group_options, key):
            setattr(group_options, key, value)

    ba = limap.estimators.bundle_adjustment
    ba_options = ba.StructureBundleAdjustmentOptions()
    for key, value in cfg.get("bundle_adjustment", {}).items():
        if hasattr(ba_options, key):
            setattr(ba_options, key, value)

    filter_options = limap.sfm.GroupVerificationOptions()
    for key, value in cfg.get("group_verification", {}).items():
        if hasattr(filter_options, key):
            setattr(filter_options, key, value)

    logging.info("Running global structure triangulation...")
    logging.info(f"  - Number of images: {len(point_recon.images)}")

    _sfm.global_triangulate_structure(
        structure_db_cache,
        holistic_recon,
        line_options,
        group_options,
        ba_options=ba_options,
        filter_options=filter_options,
    )

    # Report results
    srec = holistic_recon.structure_recon
    logging.info("Global structure triangulation complete.")
    logging.info(f"  - Number of 3D lines: {srec.num_lines3D()}")
    logging.info(f"  - Number of 3D groups: {srec.num_groups3D()}")

    # Save result
    limapio.check_makedirs(output_dir)
    holistic_recon.write(output_dir)
    logging.info(f"  - Saved to: {output_dir}")

    return holistic_recon


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="Run global structure triangulation"
    )
    arg_parser.add_argument(
        "--structure_db_path",
        type=Path,
        required=True,
        help="Path to structure_database.db",
    )
    arg_parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to COLMAP model directory",
    )
    arg_parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    arg_parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default=None,
        help="config file (optional)",
    )

    args, unknown = arg_parser.parse_known_args()
    cfg = {}
    if args.config_file:
        cfg = cfgutils.load_config(args.config_file)
    cfg = cfgutils.update_config(cfg, unknown, {})
    return args, cfg


def main():
    args, cfg = parse_args()
    run_global_structure_triangulation(
        args.structure_db_path,
        args.model_path,
        args.output_dir,
        cfg,
    )


if __name__ == "__main__":
    main()
