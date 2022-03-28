import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from core.dataset import Hypersim
import core.utils as utils

import limap.base as _base
from line_fitnmerge import line_fitnmerge

def process_hypersim_scene(cfg, dataset_hypersim, scene_id, cam_id=0):
    # set scene id
    dataset_hypersim.set_scene_id(scene_id)
    dataset_hypersim.set_max_dim(cfg["max_image_dim"])

    # generate image indexes
    index_list = np.arange(0, cfg["input_n_views"], cfg["input_stride"]).tolist()
    index_list = dataset_hypersim.filter_index_list(index_list, cam_id=cam_id)

    # get imname_list
    imname_list = []
    for image_id in index_list:
        imname = dataset_hypersim.load_imname(image_id, cam_id=cam_id)
        imname_list.append(imname)

    # get cameras
    K = dataset_hypersim.K.astype(np.float32)
    img_hw = [dataset_hypersim.h, dataset_hypersim.w]
    Ts, Rs = dataset_hypersim.load_cameras(cam_id=cam_id)
    cameras = [_base.Camera("SIMPLE_PINHOLE", K, cam_id=0, hw=img_hw)]
    camviews = [_base.CameraView(cameras[0], _base.CameraPose(Rs[idx], Ts[idx])) for idx in index_list]

    # get depths
    depths = []
    for image_id in index_list:
        depth = dataset_hypersim.load_depth(image_id, cam_id=cam_id)
        depths.append(depth)

    # fit and merge
    line_fitnmerge(cfg, imname_list, camviews, depths, max_image_dim=cfg["max_image_dim"])

def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(description='fit and merge 3d lines')
    arg_parser.add_argument('-c', '--config_file', type=str, default='cfgs/fitnmerge/hypersim_fitnmerge.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str, default='cfgs/fitnmerge/default_fitnmerge.yaml', help='default config file')
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
    dataset_hypersim = Hypersim(cfg["data_dir"])
    process_hypersim_scene(cfg, dataset_hypersim, cfg["scene_id"], cam_id=cfg["cam_id"])

if __name__ == '__main__':
    main()

