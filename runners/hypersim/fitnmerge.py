import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from loader import read_scene_hypersim
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.dataset import Hypersim
import core.utils as utils
import limap.runners

def run_scene_hypersim(cfg, dataset, scene_id, cam_id=0):
    imname_list, camviews, depths = read_scene_hypersim(cfg, dataset, scene_id, cam_id=cam_id, load_depth=True)
    linetracks = limap.runners.line_fitnmerge(cfg, imname_list, camviews, depths, max_image_dim=cfg["max_image_dim"])
    return linetracks

def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(description='fit and merge 3d lines')
    arg_parser.add_argument('-c', '--config_file', type=str, default='cfgs/fitnmerge/hypersim.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str, default='cfgs/fitnmerge/default.yaml', help='default config file')
    arg_parser.add_argument('--npyfolder', type=str, default=None, help='folder to load precomputed results')

    args, unknown = arg_parser.parse_known_args()
    cfg = utils.load_config(args.config_file, default_path=args.default_config_file)
    shortcuts = dict()
    shortcuts['-nv'] = '--n_visible_views'
    shortcuts['-sid'] = '--scene_id'
    cfg = utils.update_config(cfg, unknown, shortcuts)
    cfg["folder_to_load"] = args.npyfolder
    if cfg["folder_to_load"] is None:
        cfg["folder_to_load"] = os.path.join("precomputed", "hypersim", cfg["scene_id"])
    return cfg

def main():
    cfg = parse_config()
    dataset = Hypersim(cfg["data_dir"])
    run_scene_hypersim(cfg, dataset, cfg["scene_id"], cam_id=cfg["cam_id"])

if __name__ == '__main__':
    main()

