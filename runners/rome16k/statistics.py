import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Rome16K import Rome16K

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import limap.util.config as cfgutils


def report_rome16k_statistics(cfg, bundler_path, list_path, model_path):
    """
    Run triangulation from Rome16K input
    """
    dataset = Rome16K(
        os.path.join(bundler_path, list_path),
        os.path.join(bundler_path, cfg["component_folder"]),
    )
    counts = []
    for comp_id in range(dataset.count_components()):
        count = dataset.count_images_in_component(comp_id)
        counts.append(count)
    indexes = np.argsort(counts)[::-1]
    for index in indexes.tolist():
        print(index, counts[index])


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
        "--component_folder",
        type=str,
        default="bundle/components",
        help="component folder",
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
    # components
    cfg["component_folder"] = args.component_folder
    return cfg


def main():
    cfg = parse_config()
    report_rome16k_statistics(
        cfg, cfg["bundler_path"], cfg["list_path"], cfg["model_path"]
    )


if __name__ == "__main__":
    main()
