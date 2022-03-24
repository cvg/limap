import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import numpy as np
from core.dataset import ETH3D
import core.utils as utils
import limap.base as _base
from colmap_triangulation import read_infos_colmap, filter_by_cam_id
from line_triangulation import line_triangulation

def process_eth3d_scene(cfg, dataset_eth3d, reso_type, scene_id, cam_id=0):
    # set scene id
    dataset_eth3d.set_scene_id(reso_type, scene_id, cam_id=cam_id)

    # get imname_list, cameras, neighbors, and ranges
    tmp_cfg = copy.deepcopy(cfg)
    tmp_cfg["n_neighbors"] = 100000 # collect enough neighbors for cam id filtering
    if cfg["info_path"] is None:
        imname_list, cameras, neighbors, ranges, cam_id_list = read_infos_colmap(tmp_cfg, dataset_eth3d.scene_dir, model_path=dataset_eth3d.sparse_folder, image_path=dataset_eth3d.image_folder, max_image_dim=cfg["max_image_dim"])
        with open(os.path.join("tmp", "infos_eth3d.npy"), 'wb') as f:
            cameras_np = [[cam.K, cam.R, cam.T[:,None].repeat(3, 1)] for cam in cameras]
            np.savez(f, imname_list=imname_list, cameras_np=cameras_np, neighbors=neighbors, ranges=ranges, cam_id_list=cam_id_list)
    else:
        with open(cfg["info_path"], 'rb') as f:
            data = np.load(f, allow_pickle=True)
            imname_list, cameras_np, neighbors, ranges, cam_id_list = data["imname_list"], data["cameras_np"], data["neighbors"], data["ranges"], data["cam_id_list"]
            cameras = [_base.Camera(cam[0], cam[1], cam[2][:,0]) for cam in cameras_np]

    # filter by camera ids for eth3d
    if dataset_eth3d.cam_id != -1:
        imname_list, cameras, neighbors = filter_by_cam_id(dataset_eth3d.cam_id, imname_list, cameras, neighbors, cam_id_list)
    neighbors = [neighbor[:cfg["n_neighbors"]] for neighbor in neighbors]

    # run triangulation
    line_triangulation(cfg, imname_list, cameras, neighbors=neighbors, ranges=ranges, max_image_dim=cfg["max_image_dim"])

def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(description='triangulate 3d lines')
    arg_parser.add_argument('-c', '--config_file', type=str, default='cfgs/eth3d_triangulation.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str, default='cfgs/default_triangulation.yaml', help='default config file')
    arg_parser.add_argument('--info_reuse', action='store_true', help="whether to use infonpy at tmp/infos_eth3d.npy")
    arg_parser.add_argument('--info_path', type=str, default=None, help='load precomputed info')

    args, unknown = arg_parser.parse_known_args()
    cfg = utils.load_config(args.config_file, default_path=args.default_config_file)
    shortcuts = dict()
    shortcuts['-nv'] = '--n_visible_views'
    shortcuts['-nn'] = '--n_neighbors'
    shortcuts['-sid'] = '--scene_id'
    if args.info_reuse:
        cfg["info_path"] = "tmp/infos_eth3d.npy"
    cfg["info_path"] = args.info_path
    cfg = utils.update_config(cfg, unknown, shortcuts)
    cfg["folder_to_load"] = os.path.join("precomputed", "eth3d", cfg["reso_type"], "{0}_cam{1}".format(cfg["scene_id"], cfg["cam_id"]))
    return cfg

def init_workspace():
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

def main():
    cfg = parse_config()
    init_workspace()
    dataset_eth3d = ETH3D(cfg["data_dir"])
    process_eth3d_scene(cfg, dataset_eth3d, cfg["reso_type"], cfg["scene_id"], cfg["cam_id"])

if __name__ == '__main__':
    main()

