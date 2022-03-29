import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import core.utils as utils
import limap.base as _base
import limap.pointsfm as _psfm
from line_triangulation import line_triangulation

def read_infos_colmap(cfg, colmap_path, model_path="sparse", image_path="images", max_image_dim=None):
    '''
    Read all infos from colmap including imname_list, camviews, neighbors, and ranges
    '''
    model = _psfm.SfmModel()
    model.ReadFromCOLMAP(colmap_path, model_path, image_path)

    # get imname_list and camviews
    imname_list, camviews = _psfm.ReadInfos(colmap_path, model_path=model_path, image_path=image_path, max_image_dim=max_image_dim, check_undistorted=True)

    # get neighbors
    neighbors = _psfm.compute_neighbors(model, cfg["n_neighbors"], min_triangulation_angle=cfg["sfm"]["min_triangulation_angle"], neighbor_type=cfg["sfm"]["neighbor_type"])

    # get ranges
    ranges = model.ComputeRanges(cfg["sfm"]["ranges"]["range_robust"], cfg["sfm"]["ranges"]["k_stretch"])
    return imname_list, camviews, neighbors, ranges

def filter_by_cam_id(cam_id, prev_imname_list, prev_camviews, prev_neighbors):
    '''
    Filter the list by camview id
    '''
    assert len(prev_imname_list) == len(prev_neighbors)
    assert len(prev_camviews) == len(prev_neighbors)
    imname_list, camviews, neighbors = [], [], []
    id_maps = {}
    # filter ids
    for idx in range(len(prev_neighbors)):
        cam_id_idx = prev_camviews[idx].cam.cam_id()
        if cam_id_idx != cam_id:
            id_maps[idx] = -1
            continue
        id_maps[idx] = len(imname_list)
        imname_list.append(prev_imname_list[idx])
        camviews.append(prev_camviews[idx])
        neighbors.append(prev_neighbors[idx])
    # map ids for neighbors
    for idx in range(len(imname_list)):
        n0 = neighbors[idx]
        n1 = [id_maps[k] for k in n0]
        n2 = [k for k in n1 if k != -1]
        neighbors[idx] = n2
    return imname_list, camviews, neighbors

def run_colmap_triangulation(cfg, colmap_path, model_path="sparse", image_path="images"):
    '''
    Run triangulation from COLMAP input
    '''
    if cfg["info_path"] is None:
        imname_list, camviews, neighbors, ranges = read_infos_colmap(cfg, colmap_path, model_path=model_path, image_path=image_path, max_image_dim=cfg["max_image_dim"])
        with open(os.path.join("tmp", "infos_colmap.npy"), 'wb') as f:
            camviews_np = [view.as_dict() for view in camviews]
            np.savez(f, imname_list=imname_list, camviews_np=camviews_np, neighbors=neighbors, ranges=ranges)
    else:
        with open(cfg["info_path"], 'rb') as f:
            data = np.load(f, allow_pickle=True)
            imname_list, camviews_np, neighbors, ranges = data["imname_list"], data["camviews_np"], data["neighbors"], data["ranges"]
            camviews = [_base.CameraView(view_np) for view_np in camviews_np]

    # run triangulation
    line_triangulation(cfg, imname_list, camviews, neighbors=neighbors, ranges=ranges, max_image_dim=cfg["max_image_dim"])

def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(description='triangulate 3d lines from COLMAP')
    arg_parser.add_argument('-c', '--config_file', type=str, default='cfgs/triangulation/default_triangulation.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str, default='cfgs/triangulation/default_triangulation.yaml', help='default config file')
    arg_parser.add_argument('-a', '--colmap_path', type=str, required=True, help='colmap path')
    arg_parser.add_argument('-m', '--model_path', type=str, default='sparse', help='model path')
    arg_parser.add_argument('-i', '--image_path', type=str, default='images', help='image path')
    arg_parser.add_argument('--npyfolder', type=str, default="tmp", help='folder to load precomputed results')
    arg_parser.add_argument('--max_image_dim', type=int, default=None, help='max image dim')
    arg_parser.add_argument('--info_reuse', action='store_true', help="whether to use infonpy at tmp/infos_colmap.npy")
    arg_parser.add_argument('--info_path', type=str, default=None, help='load precomputed info')

    args, unknown = arg_parser.parse_known_args()
    cfg = utils.load_config(args.config_file, default_path=args.default_config_file)
    shortcuts = dict()
    shortcuts['-nv'] = '--n_visible_views'
    shortcuts['-nn'] = '--n_neighbors'
    cfg = utils.update_config(cfg, unknown, shortcuts)
    cfg["colmap_path"] = args.colmap_path
    cfg["image_path"] = args.image_path
    cfg["model_path"] = args.model_path
    cfg["folder_to_load"] = args.npyfolder
    if args.info_reuse:
        cfg["info_path"] = "tmp/infos_colmap.npy"
    cfg["info_path"] = args.info_path
    if ("max_image_dim" not in cfg.keys()) or args.max_image_dim is not None:
        cfg["max_image_dim"] = args.max_image_dim
    return cfg

def main():
    cfg = parse_config()
    run_colmap_triangulation(cfg, cfg["colmap_path"], cfg["model_path"], cfg["image_path"])

if __name__ == '__main__':
    main()

