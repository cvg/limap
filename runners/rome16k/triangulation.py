import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Rome16K import Rome16K

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import limap.runners
import limap.util.config as cfgutils
from runners.bundler_triangulation import read_scene_bundler


def run_rome16k_triangulation(cfg, bundler_path, list_path, model_path):
    """
    Run triangulation from Rome16K input
    """
    imagecols, neighbors, ranges = read_scene_bundler(
        cfg, bundler_path, list_path, model_path, n_neighbors=cfg["n_neighbors"]
    )

    # Rome16K components
    if cfg["comp_id"] != -1:
        dataset = Rome16K(
            os.path.join(bundler_path, list_path),
            os.path.join(bundler_path, cfg["component_folder"]),
        )
        valid_image_ids = []
        for img_id in imagecols.get_img_ids():
            comp_id = dataset.get_component_id_for_image_id(img_id)
            if comp_id == cfg["comp_id"]:
                valid_image_ids.append(img_id)
        print(
            "[LOG] Get image subset from component {}: n_images = {}".format(
                cfg["comp_id"], len(valid_image_ids)
            )
        )
        imagecols = imagecols.subset_by_image_ids(valid_image_ids)

    # run triangulation
    linetracks = limap.runners.line_triangulation(
        cfg, imagecols, neighbors=neighbors, ranges=ranges
    )
    return linetracks


def parse_config():
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="triangulate 3d lines from specific component \
                     of Rome16k (bundler format)."
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
    arg_parser.add_argument(
        "--component_folder",
        type=str,
        default="bundle/components",
        help="component folder",
    )
    arg_parser.add_argument(
        "--comp_id", type=int, default=-1, help="component id"
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
    # components
    cfg["component_folder"] = args.component_folder
    cfg["comp_id"] = args.comp_id
    return cfg


def main():
    cfg = parse_config()
    run_rome16k_triangulation(
        cfg, cfg["bundler_path"], cfg["list_path"], cfg["model_path"]
    )


if __name__ == "__main__":
    main()
