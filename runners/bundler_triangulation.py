import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import limap.base as base
import limap.pointsfm as pointsfm
import limap.runners
import limap.util.config as cfgutils
import limap.util.io as limapio


def read_scene_bundler(
    cfg, bundler_path, list_path, model_path, n_neighbors=20
):
    metainfos_filename = "infos_bundler.npy"
    output_dir = "tmp" if cfg["output_dir"] is None else cfg["output_dir"]
    limapio.check_makedirs(output_dir)
    if cfg["skip_exists"] and os.path.exists(
        os.path.join(output_dir, metainfos_filename)
    ):
        cfg["info_path"] = os.path.join(output_dir, metainfos_filename)
    if cfg["info_path"] is None:
        imagecols, neighbors, ranges = pointsfm.read_infos_bundler(
            cfg["sfm"],
            bundler_path,
            list_path,
            model_path,
            n_neighbors=n_neighbors,
        )
        with open(os.path.join(output_dir, metainfos_filename), "wb") as f:
            np.savez(
                f,
                imagecols_np=imagecols.as_dict(),
                neighbors=neighbors,
                ranges=ranges,
            )
    else:
        with open(cfg["info_path"], "rb") as f:
            data = np.load(f, allow_pickle=True)
            imagecols_np, neighbors, ranges = (
                data["imagecols_np"].item(),
                data["neighbors"].item(),
                data["ranges"],
            )
            imagecols = base.ImageCollection(imagecols_np)
    return imagecols, neighbors, ranges


def run_bundler_triangulation(cfg, bundler_path, list_path, model_path):
    """
    Run triangulation from Bundler input
    """
    imagecols, neighbors, ranges = read_scene_bundler(
        cfg, bundler_path, list_path, model_path, n_neighbors=cfg["n_neighbors"]
    )

    # run triangulation
    linetracks = limap.runners.line_triangulation(
        cfg, imagecols, neighbors=neighbors, ranges=ranges
    )
    return linetracks


def parse_config():
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="triangulate 3d lines from bundler"
    )
    arg_parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="cfgs/triangulation/default.yaml",
        help="config file",
    )
    arg_parser.add_argument(
        "--default_config_file",
        type=str,
        default="cfgs/triangulation/default.yaml",
        help="default config file",
    )
    arg_parser.add_argument(
        "-a", "--bundler_path", type=str, required=True, help="bundler path"
    )
    arg_parser.add_argument(
        "-l",
        "--list_path",
        type=str,
        default="bundle/list.orig.txt",
        help="image list path",
    )
    arg_parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="bundle/bundle.orig.out",
        help="model path",
    )
    arg_parser.add_argument(
        "--max_image_dim", type=int, default=None, help="max image dim"
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
    shortcuts["-nn"] = "--n_neighbors"
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["bundler_path"] = args.bundler_path
    cfg["list_path"] = args.list_path
    cfg["model_path"] = args.model_path
    cfg["info_path"] = args.info_path
    if ("max_image_dim" not in cfg) or args.max_image_dim is not None:
        cfg["max_image_dim"] = args.max_image_dim
    return cfg


def main():
    cfg = parse_config()
    run_bundler_triangulation(
        cfg, cfg["bundler_path"], cfg["list_path"], cfg["model_path"]
    )


if __name__ == "__main__":
    main()
