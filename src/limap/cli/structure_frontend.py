"""
Runner for structure frontend (detection + matching).

This runner takes a COLMAP model and images, runs line/point detection and
matching, and outputs:
- database.db: COLMAP database with point features and matches
- structure_database.db: Structure database with line features and matches

Usage:
    python -m limap.cli.structure_frontend \
        --image_dir <path_to_images> \
        --model_path <path_to_colmap_model> \
        --output_dir <output_directory>

The output can then be used with global_line_triangulation.py or
incremental_line_triangulation.py.
"""

import argparse
from pathlib import Path
from dacite import from_dict, Config

import limap.runners
import limap.util.config as cfgutils


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="Run structure frontend (detection + matching)"
    )
    arg_parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Path to image directory",
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
        help="Output directory for databases",
    )
    arg_parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="cfgs/structure_triangulation/default.yaml",
        help="config file",
    )
    arg_parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip frontend if databases already exist",
    )

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(args.config_file)
    cfg = cfgutils.update_config(cfg, unknown, {})
    return args, cfg


def main():
    args, cfg = parse_args()

    # Build options from config
    options = from_dict(
        data_class=limap.runners.StructureFrontendOptions,
        data=cfg,
        config=Config(strict=False, cast=[Path]),
    )

    # Run frontend
    outputs = limap.runners.structure_frontend_from_model(
        options,
        args.image_dir,
        args.model_path,
        args.output_dir,
        skip_if_exists=args.skip_if_exists,
    )

    print("\nOutputs:")
    print(f"  - COLMAP database: {outputs.db_path}")
    print(f"  - Structure database: {outputs.structure_db_path}")
    print(f"  - Image directory: {outputs.image_dir}")
    print(f"  - Model directory: {outputs.model_dir}")


if __name__ == "__main__":
    main()
