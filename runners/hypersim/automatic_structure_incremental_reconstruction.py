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


def run_scene_hypersim(cfg, dataset, scene_id, cam_id=0):
    image_dir, recon, _ = read_scene_hypersim(
        cfg,
        dataset,
        scene_id,
        cam_id=cam_id,
        load_depth=False,
        load_poses=False,
    )
    output_dir = Path(cfg["output_dir"])

    options = from_dict(
        data_class=limap.runners.AutomaticStructureIncrementalReconstructionOptions,
        data=cfg,
        config=Config(strict=False, cast=[Path]),
    )
    output_path = limap.runners.automatic_structure_incremental_reconstruction(
        options, image_dir, output_dir, recon
    )
    return output_path


def parse_config():
    arg_parser = argparse.ArgumentParser(
        description="automatic structure-aware incremental "
        "reconstruction (hypersim)"
    )
    arg_parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="cfgs/structure_incremental_reconstruction/hypersim.yaml",
        help="config file",
    )
    arg_parser.add_argument(
        "--default_config_file",
        type=str,
        default="cfgs/structure_incremental_reconstruction/default.yaml",
        help="default config file",
    )

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(
        args.config_file, default_path=args.default_config_file
    )
    shortcuts = dict()
    shortcuts["-nn"] = "--metainfo.pair_generation.n_neighbors"
    shortcuts["-sid"] = "--scene_id"
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    return cfg


def main():
    cfg = parse_config()
    dataset = Hypersim(cfg["data_dir"])
    run_scene_hypersim(cfg, dataset, cfg["scene_id"], cam_id=cfg["cam_id"])


if __name__ == "__main__":
    main()
