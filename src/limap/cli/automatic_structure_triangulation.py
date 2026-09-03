r"""
Runner for structure triangulation from COLMAP.

Triangulates 3D points, lines, and groups (planes) from a COLMAP model.
Runs the full pipeline: frontend (detection + matching), point triangulation,
line/group triangulation, and optional bundle adjustment.

Usage::

    python -m limap.cli.automatic_structure_triangulation \
        -m <path_to_colmap_model> \
        -i <path_to_images> \
        -o <output_directory> \
        -c cfgs/structure_triangulation/default.yaml
"""

from pathlib import Path
from dacite import from_dict, Config

import limap.runners
import limap.util.config as cfgutils
import limap.estimators.bundle_adjustment as ba
import limap.sfm


def run_structure_triangulation_from_colmap(cfg, model_path, image_path):
    output_dir = Path(cfg["output_dir"])
    type_hooks = {
        ba.StructureBundleAdjustmentOptions: (
            ba.StructureBundleAdjustmentOptions
        ),
        ba.GroupBundleAdjustmentOptions: ba.GroupBundleAdjustmentOptions,
        # Only pybind option types need a hook; dataclasses recurse.
        limap.sfm.GlobalLineTriangulationOptions: (
            limap.sfm.GlobalLineTriangulationOptions
        ),
        limap.sfm.GlobalGroupTriangulationOptions: (
            limap.sfm.GlobalGroupTriangulationOptions
        ),
        limap.sfm.GroupVerificationOptions: limap.sfm.GroupVerificationOptions,
    }
    options = from_dict(
        data_class=limap.runners.AutomaticStructureTriangulationOptions,
        data=cfg,
        config=Config(strict=False, cast=[Path], type_hooks=type_hooks),
    )
    output_model_dir = limap.runners.automatic_structure_triangulation(
        options, Path(image_path), Path(model_path), output_dir
    )
    return output_model_dir


def parse_config():
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="triangulate 3d structures from COLMAP"
    )
    arg_parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="cfgs/structure_triangulation/default.yaml",
        help="config file",
    )
    arg_parser.add_argument(
        "--default_config_file",
        type=str,
        default=None,
        help="default config file",
    )
    arg_parser.add_argument(
        "-m", "--model_path", type=str, required=True, help="model path"
    )
    arg_parser.add_argument(
        "-i", "--image_path", type=str, required=True, help="image path"
    )
    arg_parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="output directory"
    )
    arg_parser.add_argument(
        "--max_image_dim", type=int, default=None, help="max image dim"
    )
    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(
        args.config_file, default_path=args.default_config_file
    )
    shortcuts = dict()
    shortcuts["-nv"] = "--n_visible_views"
    shortcuts["-nn"] = "--metainfo.pair_generation.n_neighbors"
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["model_path"] = args.model_path
    cfg["image_path"] = args.image_path
    cfg["output_dir"] = args.output_dir
    if ("max_image_dim" not in cfg) or args.max_image_dim is not None:
        cfg["max_image_dim"] = args.max_image_dim
    return cfg


def main():
    cfg = parse_config()
    run_structure_triangulation_from_colmap(
        cfg, cfg["model_path"], cfg["image_path"]
    )


if __name__ == "__main__":
    main()
