import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ETH3D import ETH3D
from loader import read_scene_eth3d

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import limap.runners
import limap.util.config as cfgutils


def run_scene_eth3d(cfg, dataset, reso_type, scene_id, cam_id=0):
    imagecols, neighbors, ranges, depths = read_scene_eth3d(
        cfg, dataset, reso_type, scene_id, cam_id=cam_id, load_depth=True
    )
    linetracks = limap.runners.line_fitnmerge(
        cfg, imagecols, depths, neighbors=neighbors, ranges=ranges
    )
    return linetracks


def parse_config():
    import argparse

    arg_parser = argparse.ArgumentParser(description="fitnmerge 3d lines")
    arg_parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="cfgs/fitnmerge/eth3d.yaml",
        help="config file",
    )
    arg_parser.add_argument(
        "--default_config_file",
        type=str,
        default="cfgs/fitnmerge/default.yaml",
        help="default config file",
    )
    arg_parser.add_argument(
        "--info_reuse",
        action="store_true",
        help="whether to use infonpy at tmp/infos_eth3d.npy",
    )
    arg_parser.add_argument(
        "--info_path", type=str, default=None, help="load precomputed info"
    )

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(
        args.config_file, default_path=args.default_config_file
    )
    shortcuts = dict()
    shortcuts["-nv"] = "--n_visible_views"
    shortcuts["-sid"] = "--scene_id"
    if args.info_reuse:
        cfg["info_path"] = "tmp/infos_eth3d.npy"
    cfg["info_path"] = args.info_path
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["folder_to_load"] = os.path.join(
        "precomputed",
        "eth3d",
        cfg["reso_type"],
        "{}_cam{}".format(cfg["scene_id"], cfg["cam_id"]),
    )
    return cfg


def main():
    cfg = parse_config()
    dataset = ETH3D(cfg["data_dir"])
    run_scene_eth3d(
        cfg, dataset, cfg["reso_type"], cfg["scene_id"], cfg["cam_id"]
    )


if __name__ == "__main__":
    main()
