import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import limap.base as base
import limap.pointsfm as pointsfm
import limap.runners
import limap.util.config as cfgutils
import limap.util.io as limapio


def read_scene_colmap(
    cfg, colmap_path, model_path="sparse", image_path="images", n_neighbors=20
):
    metainfos_filename = "infos_colmap.npy"
    output_dir = "tmp" if cfg["output_dir"] is None else cfg["output_dir"]
    limapio.check_makedirs(output_dir)
    if cfg["skip_exists"] and os.path.exists(
        os.path.join(output_dir, metainfos_filename)
    ):
        cfg["info_path"] = os.path.join(output_dir, metainfos_filename)
    if cfg["info_path"] is None:
        imagecols, neighbors, ranges = pointsfm.read_infos_colmap(
            cfg["sfm"],
            colmap_path,
            model_path=model_path,
            image_path=image_path,
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


def run_colmap_triangulation(
    cfg, colmap_path, model_path="sparse", image_path="images"
):
    """
    Run triangulation from COLMAP input
    """
    imagecols, neighbors, ranges = read_scene_colmap(
        cfg,
        colmap_path,
        model_path=model_path,
        image_path=image_path,
        n_neighbors=cfg["n_neighbors"],
    )

    # run triangulation
    linetracks = limap.runners.line_triangulation(
        cfg, imagecols, neighbors=neighbors, ranges=ranges
    )
    return linetracks


def parse_config():
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="triangulate 3d lines from COLMAP"
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
        "-a", "--colmap_path", type=str, default=None, help="colmap path"
    )
    arg_parser.add_argument(
        "-m", "--model_path", type=str, default="sparse", help="model path"
    )
    arg_parser.add_argument(
        "-i", "--image_path", type=str, default="images", help="image path"
    )
    arg_parser.add_argument(
        "--npyfolder",
        type=str,
        default="tmp",
        help="folder to load precomputed results",
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
    cfg["colmap_path"] = args.colmap_path
    cfg["image_path"] = args.image_path
    cfg["model_path"] = args.model_path
    cfg["folder_to_load"] = args.npyfolder
    cfg["info_path"] = args.info_path
    if cfg["colmap_path"] is None and cfg["info_path"] is None:
        raise ValueError("Error! colmap_path unspecified.")
    if ("max_image_dim" not in cfg) or args.max_image_dim is not None:
        cfg["max_image_dim"] = args.max_image_dim
    return cfg


def main():
    cfg = parse_config()
    run_colmap_triangulation(
        cfg, cfg["colmap_path"], cfg["model_path"], cfg["image_path"]
    )


if __name__ == "__main__":
    main()
