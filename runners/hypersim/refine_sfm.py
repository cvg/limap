import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Hypersim import Hypersim
from loader import read_scene_hypersim

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import limap.optimize as optimize
import limap.pointsfm as pointsfm
import limap.runners
import limap.util.config as cfgutils
import limap.util.evaluation as limapeval
import limap.util.io as limapio
from runners.colmap_triangulation import run_colmap_triangulation


def run_scene_hypersim(cfg, dataset, scene_id, cam_id=0):
    imagecols_gt = read_scene_hypersim(
        cfg, dataset, scene_id, cam_id=cam_id, load_depth=False
    )

    # run colmap
    cfg = limap.runners.setup(cfg)
    global_dir_save = cfg["dir_save"]
    limapio.save_npy(
        os.path.join(global_dir_save, "imagecols_gt.npy"), imagecols_gt
    )
    colmap_path = os.path.join(cfg["dir_save"], "colmap_sfm")
    # _psfm.run_colmap_sfm(cfg["sfm"], imagecols_gt, output_path=colmap_path,
    # skip_exists=cfg["skip_exists"], map_to_original_image_names=False)
    imagecols, _, _ = pointsfm.read_infos_colmap(
        cfg["sfm"], colmap_path, model_path="sparse/0", image_path="images"
    )
    limapio.save_npy(
        os.path.join(global_dir_save, "imagecols_sfm.npy"), imagecols
    )
    trans_errs, rot_errs = limapeval.eval_imagecols(imagecols, imagecols_gt)
    print(np.median(trans_errs), np.median(rot_errs))

    # run limap
    cfg_limap = cfg
    cfg_limap["output_dir"] = os.path.join(cfg["dir_save"], "limap_outputs")
    cfg_limap["visualize"] = False
    cfg_limap["info_path"] = None
    linetracks = run_colmap_triangulation(
        cfg_limap, colmap_path, model_path="sparse/0", image_path="images"
    )

    # run joint ba
    colmap_folder = os.path.join(colmap_path, "sparse/0")
    reconstruction = pointsfm.PyReadCOLMAP(colmap_folder)
    pointtracks = pointsfm.ReadPointTracks(reconstruction)
    cfg_ba = optimize.HybridBAConfig()
    ba_engine = optimize.solve_hybrid_bundle_adjustment(
        cfg_ba, imagecols, pointtracks, linetracks
    )
    new_imagecols = ba_engine.GetOutputImagecols()

    # evaluate
    limapio.save_npy(
        os.path.join(global_dir_save, "imagecols_optimized.npy"), new_imagecols
    )
    trans_errs_orig, rot_errs_orig = limapeval.eval_imagecols(
        imagecols, imagecols_gt
    )
    print(
        f"original: trans: {np.median(trans_errs_orig):.4f}, \
          rot: {np.median(rot_errs_orig):.4f}"
    )
    trans_errs, rot_errs = limapeval.eval_imagecols(new_imagecols, imagecols_gt)
    print(
        f"optimized: trans: {np.median(trans_errs):.4f}, \
          rot: {np.median(rot_errs):.4f}"
    )


def parse_config():
    import argparse

    arg_parser = argparse.ArgumentParser(description="fit and merge 3d lines")
    arg_parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="cfgs/triangulation/hypersim.yaml",
        help="config file",
    )
    arg_parser.add_argument(
        "--default_config_file",
        type=str,
        default="cfgs/triangulation/default.yaml",
        help="default config file",
    )
    arg_parser.add_argument(
        "--npyfolder",
        type=str,
        default=None,
        help="folder to load precomputed results",
    )

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(
        args.config_file, default_path=args.default_config_file
    )
    shortcuts = dict()
    shortcuts["-nv"] = "--n_visible_views"
    shortcuts["-sid"] = "--scene_id"
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["folder_to_load"] = args.npyfolder
    if cfg["folder_to_load"] is None:
        cfg["folder_to_load"] = os.path.join(
            "precomputed", "hypersim", cfg["scene_id"]
        )
    return cfg


def main():
    cfg = parse_config()
    dataset = Hypersim(cfg["data_dir"])
    run_scene_hypersim(cfg, dataset, cfg["scene_id"], cam_id=cfg["cam_id"])


if __name__ == "__main__":
    main()
