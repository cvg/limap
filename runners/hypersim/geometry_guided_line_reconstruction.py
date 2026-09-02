import os
import sys
import argparse
from pathlib import Path
from dacite import from_dict, Config

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Hypersim import Hypersim
from loader import read_scene_hypersim

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import limap.runners
import limap.util.config as cfgutils
import limap.util.io as limapio
import limap.estimators.bundle_adjustment as ba


def run_scene_hypersim(cfg, dataset, scene_id, cam_id=0):
    image_dir, recon, depth_readers = read_scene_hypersim(
        cfg, dataset, scene_id, cam_id=cam_id, load_depth=True
    )
    output_dir = Path(cfg["output_dir"])
    model_dir = output_dir / "init_model"
    limapio.check_makedirs(model_dir)
    recon.write(model_dir)

    # Type hooks for C++ classes that support dict construction
    BAOptions = ba.PointLineBundleAdjustmentOptions
    type_hooks = {BAOptions: BAOptions}
    options = from_dict(
        data_class=limap.runners.GeometryGuidedLineReconstructionOptions,
        data=cfg,
        config=Config(strict=False, cast=[Path], type_hooks=type_hooks),
    )
    output_model_dir = limap.runners.line_reconstruction_with_depth_maps(
        options, image_dir, model_dir, output_dir, depth_readers
    )
    return output_model_dir


def parse_config():
    arg_parser = argparse.ArgumentParser(
        description="geometry-guided 3D line reconstruction"
    )
    arg_parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="cfgs/geometry_guided_line_reconstruction/hypersim.yaml",
        help="config file",
    )
    arg_parser.add_argument(
        "--default_config_file",
        type=str,
        default="cfgs/geometry_guided_line_reconstruction/default.yaml",
        help="default config file",
    )

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(
        args.config_file, default_path=args.default_config_file
    )
    shortcuts = dict()
    shortcuts["-nv"] = "--n_visible_views"
    shortcuts["-nn"] = "--n_neighbors"
    shortcuts["-sid"] = "--scene_id"
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    return cfg


def main():
    cfg = parse_config()
    dataset = Hypersim(cfg["data_dir"])
    run_scene_hypersim(cfg, dataset, cfg["scene_id"], cam_id=cfg["cam_id"])


if __name__ == "__main__":
    main()
