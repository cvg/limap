import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from loader import read_scene_scannet
from ScanNet import ScanNet

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import limap.runners
import limap.util.config as cfgutils


def run_scene_scannet(cfg, dataset, scene_id):
    imagecols, neighbors, depths = read_scene_scannet(
        cfg, dataset, scene_id, load_depth=True
    )
    linetracks = limap.runners.line_fitnmerge(
        cfg, imagecols, depths, neighbors=neighbors
    )
    return linetracks


def parse_config():
    import argparse

    arg_parser = argparse.ArgumentParser(description="fitnmerge 3d lines")
    arg_parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="cfgs/fitnmerge/scannet.yaml",
        help="config file",
    )
    arg_parser.add_argument(
        "--default_config_file",
        type=str,
        default="cfgs/fitnmerge/default.yaml",
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
    cfg["folder_to_load"] = os.path.join(
        "precomputed", "scannet", cfg["scene_id"]
    )
    return cfg


def main():
    cfg = parse_config()
    dataset = ScanNet(cfg["data_dir"])
    run_scene_scannet(cfg, dataset, cfg["scene_id"])


if __name__ == "__main__":
    main()
