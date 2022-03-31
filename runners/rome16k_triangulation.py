import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import core.utils as utils

from bundler_triangulation import run_bundler_undistortion, load_all_infos_bundler
from line_triangulation import line_triangulation

def run_rome16k_triangulation(cfg, bundler_path, list_path, model_path):
    imname_list, camviews, neighbors, ranges = load_all_infos_bundler(cfg, bundler_path, list_path, model_path)

    # components
    valid_index_list = np.arange(0, len(imname_list)).tolist()
    if cfg["comp_id"] != -1:
        from core.dataset import Rome
        dataset = Rome(os.path.join(bundler_path, list_path), os.path.join(bundler_path, cfg["component_folder"]))
        valid_index_list = []
        for img_id, imname in enumerate(imname_list):
            comp_id = dataset.get_component_id_for_image_id(img_id)
            if comp_id == cfg["comp_id"]:
                valid_index_list.append(img_id)
        new_imname_list, new_camviews, new_neighbors = [], [], []
        for img_id in valid_index_list:
            new_imname_list.append(imname_list[img_id])
            new_camviews.append(camviews[img_id])
            neighbor = neighbors[img_id]
            new_neighbor = [valid_index_list.index(k) for k in neighbor if k in valid_index_list]
            new_neighbors.append(new_neighbor)
        imname_list, camviews, neighbors = new_imname_list, new_camviews, new_neighbors

    # run triangulation
    line_triangulation(cfg, imname_list, camviews, neighbors=neighbors, ranges=ranges, max_image_dim=cfg["max_image_dim"], valid_index_list=valid_index_list)

def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(description='triangulate 3d lines from bundler')
    arg_parser.add_argument('-c', '--config_file', type=str, default='cfgs/triangulation/default.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str, default='cfgs/triangulation/default.yaml', help='default config file')
    arg_parser.add_argument('-a', '--bundler_path', type=str, required=True, help='bundler path')
    arg_parser.add_argument('-l', '--list_path', type=str, default='bundle/list.orig.txt', help='image list path')
    arg_parser.add_argument('-m', '--model_path', type=str, default='bundle/bundle.orig.out', help='model path')
    arg_parser.add_argument('--npyfolder', type=str, default="tmp", help='folder to load precomputed results')
    arg_parser.add_argument('--max_image_dim', type=int, default=None, help='max image dim')
    arg_parser.add_argument('--info_reuse', action='store_true', help="whether to use infonpy at tmp/infos_bundler.npy")
    arg_parser.add_argument('--info_path', type=str, default=None, help='load precomputed info')
    arg_parser.add_argument('--undistortion', action='store_true', help="whether to perform undistortion")
    arg_parser.add_argument('--undist_folder', type=str, default="undistorted", help='folder to save undistorted results')
    arg_parser.add_argument('--use_undist', action='store_true', help="whether to load undistorted results")
    arg_parser.add_argument('--component_folder', type=str, default='bundle/components', help='component folder')
    arg_parser.add_argument('--comp_id', type=int, default=-1, help="component id")

    args, unknown = arg_parser.parse_known_args()
    cfg = utils.load_config(args.config_file, default_path=args.default_config_file)
    shortcuts = dict()
    shortcuts['-nv'] = '--n_visible_views'
    shortcuts['-nn'] = '--n_neighbors'
    cfg = utils.update_config(cfg, unknown, shortcuts)
    cfg["bundler_path"] = args.bundler_path
    cfg["list_path"] = args.list_path
    cfg["model_path"] = args.model_path
    cfg["folder_to_load"] = args.npyfolder
    # undistortion
    cfg["undistortion"] = args.undistortion
    cfg["undist_folder"] = args.undist_folder
    cfg["use_undist"] = args.use_undist
    if args.info_reuse:
        cfg["info_path"] = "tmp/infos_bundler.npy"
    cfg["info_path"] = args.info_path
    if ("max_image_dim" not in cfg.keys()) or args.max_image_dim is not None:
        cfg["max_image_dim"] = args.max_image_dim
    # components
    cfg["component_folder"] = args.component_folder
    cfg["comp_id"] = args.comp_id
    return cfg

def main():
    cfg = parse_config()
    if cfg["undistortion"]:
        run_bundler_undistortion(cfg, cfg["bundler_path"], cfg["list_path"], cfg["model_path"])
    else:
        run_rome16k_triangulation(cfg, cfg["bundler_path"], cfg["list_path"], cfg["model_path"])

if __name__ == '__main__':
    main()

