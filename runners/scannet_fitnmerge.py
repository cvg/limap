import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from core.dataset import ScanNet
import core.utils as utils
import lineopt.base
from line_fitnmerge import line_fitnmerge
from tqdm import tqdm

def process_scannet_scene(cfg, dataset_scannet, scene_id):
    # set scene id
    dataset_scannet.set_scene_id(scene_id)
    dataset_scannet.set_img_hw_resized((480, 640))
    # dataset_scannet.set_max_dim(cfg["max_image_dim"])

    # get imname_list and cameras
    dataset_scannet.set_stride(cfg["stride"])
    imname_list = dataset_scannet.load_imname_list()
    K = dataset_scannet.load_intrinsics()
    img_hw = dataset_scannet.get_img_hw()
    Ts, Rs = dataset_scannet.load_cameras()
    cameras = []
    for idx in range(len(imname_list)):
        cam = lineopt.base.Camera(K, Rs[idx], Ts[idx])
        cam.set_hw(img_hw[0], img_hw[1])
        cameras.append(cam)

    # trivial neighbors
    index_list = np.arange(0, len(imname_list)).tolist()
    neighbors = []
    for idx, image_id in enumerate(index_list):
        val = np.abs(np.array(index_list) - image_id)
        val[idx] = val.max() + 1
        neighbor = np.array(index_list)[np.argsort(val)[:cfg["n_neighbors"]]]
        neighbors.append(neighbor.tolist())
    neighbors = np.array(neighbors)
    for idx, index in enumerate(index_list):
        neighbors[neighbors == index] = idx
    # neighbors = None

    # get depth
    depths = []
    for imname in tqdm(imname_list):
        depth = dataset_scannet.get_depth(imname)
        depths.append(depth)

    # run triangulation
    line_fitnmerge(cfg, imname_list, cameras, depths, neighbors=neighbors, resize_hw=(480, 640))

def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(description='fitnmerge 3d lines')
    arg_parser.add_argument('-c', '--config_file', type=str, default='cfgs/fitnmerge/scannet_fitnmerge.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str, default='cfgs/fitnmerge/default_fitnmerge.yaml', help='default config file')

    args, unknown = arg_parser.parse_known_args()
    cfg = utils.load_config(args.config_file, default_path=args.default_config_file)
    shortcuts = dict()
    shortcuts['-nv'] = '--n_visible_views'
    shortcuts['-nn'] = '--n_neighbors'
    shortcuts['-sid'] = '--scene_id'
    cfg = utils.update_config(cfg, unknown, shortcuts)
    cfg["folder_to_load"] = os.path.join("precomputed", "scannet", cfg["scene_id"])
    return cfg

def init_workspace():
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

def main():
    cfg = parse_config()
    init_workspace()
    dataset_scannet = ScanNet(cfg["data_dir"])
    process_scannet_scene(cfg, dataset_scannet, cfg["scene_id"])

if __name__ == '__main__':
    main()

