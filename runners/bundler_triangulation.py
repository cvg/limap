import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
import core.utils as utils

import limap.base as _base
import limap.pointsfm as _psfm
import limap.undistortion as _undist

from line_triangulation import line_triangulation
from tqdm import tqdm
from pathlib import Path

def read_infos_bundler(cfg, bundler_path, list_path, model_path, max_image_dim=None):
    '''
    Read all infos from bundler including camviews, neighbors, ranges
    '''
    model, camviews = _psfm.ReadModelBundler(bundler_path, list_path, model_path, max_image_dim=max_image_dim)

    # get neighbors
    neighbors = _psfm.ComputeNeighbors(model, cfg["n_neighbors"], min_triangulation_angle=cfg["sfm"]["min_triangulation_angle"], neighbor_type=cfg["sfm"]["neighbor_type"])

    # get ranges
    ranges = model.ComputeRanges(cfg["sfm"]["ranges"]["range_robust"], cfg["sfm"]["ranges"]["k_stretch"])
    return camviews, neighbors, ranges

def load_all_infos_bundler(cfg, bundler_path, list_path, model_path):
    if cfg["info_path"] is None:
        if cfg["use_undist"]:
            camviews, neighbors, ranges = read_infos_bundler(cfg, bundler_path, list_path, model_path, max_image_dim=-1)
        else:
            camviews, neighbors, ranges = read_infos_bundler(cfg, bundler_path, list_path, model_path, max_image_dim=cfg["max_image_dim"])
        with open(os.path.join("tmp", "infos_bundler.npy"), 'wb') as f:
            camviews_np = [view.as_dict() for view in camviews]
            np.savez(f, camviews_np=camviews_np, neighbors=neighbors, ranges=ranges)

    else:
        with open(cfg["info_path"], 'rb') as f:
            data = np.load(f, allow_pickle=True)
            camviews_np, neighbors, ranges = data["camviews_np"], data["neighbors"], data["ranges"]
            camviews = [_base.CameraView(view_np) for view_np in camviews_np]

    # TODO: load from undistortion folder
    if cfg["use_undist"]:
        n_images = len(imname_list)
        imname_list_new, camviews_new = [], []
        for image_id in tqdm(range(n_images)):
            imname = imname_list[image_id]
            relpath = Path(imname).relative_to(Path(bundler_path))
            imname_undistorted = os.path.join(bundler_path, cfg["undist_folder"], relpath)
            imname_list_new.append(imname_undistorted)
            fname_camera_undistorted = imname_undistorted[:-4] + '_cam.txt'
            camera_new = _base.Camera()
            camera_new.Read(fname_camera_undistorted)
            cameras_new.append(camera_new)
        imname_list, cameras = imname_list_new, cameras_new
        with open(os.path.join("tmp", "infos_bundler_undistorted.npy"), 'wb') as f:
            cameras_np = [[cam.K, cam.R, cam.T[:,None].repeat(3, 1), cam.dist_coeffs, [cam.h, cam.w]] for cam in cameras]
            np.savez(f, imname_list=imname_list, cameras_np=cameras_np, neighbors=neighbors, ranges=ranges)

    # return all infos
    return camviews, neighbors, ranges

def run_bundler_triangulation(cfg, bundler_path, list_path, model_path):
    # load all infos
    camviews, neighbors, ranges = load_all_infos_bundler(cfg, bundler_path, list_path, model_path)

    # run triangulation
    line_triangulation(cfg, camviews, neighbors=neighbors, ranges=ranges)

def run_bundler_undistortion(cfg, bundler_path, list_path, model_path):
    '''
    Run undistortion from bundler input
    '''
    if cfg["info_path"] is None:
        _, imname_list, cameras = _psfm.ReadModelBundler(bundler_path, list_path, model_path)
    else:
        with open(cfg["info_path"], 'rb') as f:
            data = np.load(f, allow_pickle=True)
            imname_list, cameras_np, _, _ = data["imname_list"], data["cameras_np"], data["neighbors"], data["ranges"]
            cameras = [_base.Camera(cam[0], cam[1], cam[2][:,0], cam[3], cam[4]) for cam in cameras_np]

    # undistort one by one
    print("Start undistorting images to {0}...".format(os.path.join(bundler_path, cfg["undist_folder"])))
    n_images = len(imname_list)
    for image_id in tqdm(range(n_images)):
        imname, camera = imname_list[image_id], cameras[image_id]
        relpath = Path(imname).relative_to(Path(bundler_path))

        imname_undistorted = os.path.join(bundler_path, cfg["undist_folder"], relpath)
        Path(imname_undistorted).parents[0].mkdir(parents=True, exist_ok=True)
        fname_camera_undistorted = imname_undistorted[:-4] + '_cam.txt'
        camera_undistorted = _undist.UndistortImageCamera(camera, imname, imname_undistorted)
        camera_undistorted.Write(fname_camera_undistorted)

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
    return cfg

def main():
    cfg = parse_config()
    if cfg["undistortion"]:
        run_bundler_undistortion(cfg, cfg["bundler_path"], cfg["list_path"], cfg["model_path"])
    else:
        run_bundler_triangulation(cfg, cfg["bundler_path"], cfg["list_path"], cfg["model_path"])

if __name__ == '__main__':
    main()

