"""
Run structure-aware incremental SfM.

This pipeline performs full incremental SfM with hybrid point+line registration,
structure triangulation (points, lines, groups), and structure BA.

Usage:
    python -m limap.cli.structure_incremental_sfm \
        --db_path <path_to_colmap_database.db> \
        --structure_db_path <path_to_structure_database.db> \
        --image_path <path_to_images> \
        --output_dir <output_directory>
"""

import argparse
from pathlib import Path

import pycolmap
from pycolmap import logging

from limap._limap import _sfm
from limap.scene import (
    HolisticReconstructionManager,
    StructureDatabase,
    StructureDatabaseCache,
)
import limap.util.config as cfgutils
import limap.util.io as limapio


def run_structure_incremental_sfm(
    db_path: Path,
    structure_db_path: Path,
    image_path: Path,
    output_dir: Path,
    cfg: dict,
):
    """Run structure-aware incremental SfM.

    Args:
        db_path: Path to COLMAP database.db
        structure_db_path: Path to structure_database.db
        image_path: Path to image directory
        output_dir: Output directory for results
        cfg: Configuration dictionary
    """
    if not db_path.exists():
        raise FileNotFoundError(f"COLMAP database not found: {db_path}")
    if not structure_db_path.exists():
        raise FileNotFoundError(
            f"Structure database not found: {structure_db_path}"
        )

    # Load COLMAP database
    logging.info(f"Loading COLMAP database from {db_path}")
    with pycolmap.Database.open(db_path) as colmap_db:
        colmap_db_cache = pycolmap.DatabaseCache.create(
            colmap_db, pycolmap.DatabaseCacheOptions()
        )

    # Load structure database
    logging.info(f"Loading structure database from {structure_db_path}")
    with StructureDatabase.open(structure_db_path) as structure_db:
        structure_db_cache = StructureDatabaseCache.create(structure_db)

    # Create reconstruction manager
    reconstruction_manager = HolisticReconstructionManager()

    # Configure options
    options = _sfm.StructureIncrementalPipelineOptions()
    options.colmap_options.image_path = str(image_path)
    for key, value in cfg.get("colmap_options", {}).items():
        if hasattr(options.colmap_options, key):
            setattr(options.colmap_options, key, value)
    for key, value in cfg.get("structure_options", {}).items():
        if hasattr(options.structure_options, key):
            setattr(options.structure_options, key, value)

    # Run pipeline
    logging.info("Running structure-aware incremental SfM...")
    pipeline = _sfm.StructureIncrementalPipeline(
        options, colmap_db_cache, structure_db_cache, reconstruction_manager
    )
    pipeline.run()

    # Report and save
    for i in range(reconstruction_manager.size()):
        recon = reconstruction_manager.get(i)
        prec = recon.point_recon
        srec = recon.structure_recon
        logging.info(
            f"Model {i}: {prec.num_reg_images()} images, "
            f"{prec.num_points3D()} points, "
            f"{srec.num_lines3D()} lines, "
            f"{srec.num_groups3D()} groups"
        )

    limapio.check_makedirs(output_dir)
    reconstruction_manager.write(output_dir)
    logging.info(f"Saved to: {output_dir}")
    return reconstruction_manager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run structure-aware incremental SfM"
    )
    parser.add_argument(
        "--db_path",
        type=Path,
        required=True,
        help="Path to COLMAP database.db",
    )
    parser.add_argument(
        "--structure_db_path",
        type=Path,
        required=True,
        help="Path to structure_database.db",
    )
    parser.add_argument(
        "--image_path",
        type=Path,
        required=True,
        help="Path to image directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default=None,
        help="Config file (optional)",
    )

    args, unknown = parser.parse_known_args()
    cfg = {}
    if args.config_file:
        cfg = cfgutils.load_config(args.config_file)
    cfg = cfgutils.update_config(cfg, unknown, {})
    return args, cfg


def main():
    args, cfg = parse_args()
    run_structure_incremental_sfm(
        args.db_path,
        args.structure_db_path,
        args.image_path,
        args.output_dir,
        cfg,
    )


if __name__ == "__main__":
    main()
