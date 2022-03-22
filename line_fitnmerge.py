import os, sys
import numpy as np
import cv2
import h5py
import core.detector.LSD as lsd
import core.detector.SOLD2 as sold2
import core.visualize as vis
import core.utils as utils
from core.estimators import *

import limap.base as _base
import limap.sfm as _sfm
import limap.merging as _mrg
import limap.lineBA as _lineBA
from tqdm import tqdm
import joblib

from line_triangulation import *

def fit_3d_segs(imname_list, all_2d_segs, cameras, depths, fitting_config, resize_hw=None, max_image_dim=None):
    '''
    Args:
    - all_2d_segs: list of 2d segs
    - cameras: list of limap.base.Camera
    - depths: list of depth images
    '''
    n_images = len(imname_list)
    seg3d_list = []
    def process(imname_list, all_2d_segs, cameras_np, depths, fitting_config, resize_hw, max_image_dim, idx):
        imname, segs, cam, depth = imname_list[idx], all_2d_segs[idx], cameras_np[idx], depths[idx]
        img_hw = utils.read_image(imname, resize_hw=resize_hw, max_image_dim=max_image_dim).shape[:2]
        seg3d_list_idx = []
        counter_seg3d = 0
        for seg_id, s in enumerate(segs):
            seg3d = estimate_seg3d_from_depth(s, depth, [cam[0], cam[1], cam[2]], [img_hw[0], img_hw[1]], ransac_th=fitting_config["ransac_th"], min_percentage_inliers=fitting_config["min_percentage_inliers"], var2d=fitting_config["var2d"])
            if seg3d is None:
                seg3d_list_idx.append((np.array([0., 0., 0.]), np.array([0., 0., 0.])))
            else:
                seg3d_list_idx.append(seg3d)
                counter_seg3d += 1
        return seg3d_list_idx
    cameras_np = [[cam.K, cam.R, cam.T] for cam in cameras]
    seg3d_list = joblib.Parallel(n_jobs=fitting_config["n_jobs"])(joblib.delayed(process)(imname_list, all_2d_segs, cameras_np, depths, fitting_config, resize_hw, max_image_dim, idx) for idx in tqdm(range(n_images)))
    return seg3d_list

def line_fitnmerge(cfg, imname_list, cameras, depths, neighbors=None, ranges=None, resize_hw=None, max_image_dim=None):
    '''
    Args:
    - image_names: list of imname
    - cameras: list of limap.base.Camera
    - depths: list of depth images
    '''
    # assertion check
    cfg = setup(cfg, imname_list, cameras, max_image_dim=None)
    if cfg["fitting"]["var2d"] == -1:
        cfg["fitting"]["var2d"] = cfg["var2d"][cfg["line2d"]["detector"]]
    if cfg["merging"]["var2d"] == -1:
        cfg["merging"]["var2d"] = cfg["var2d"][cfg["line2d"]["detector"]]

    ##########################################################
    # [A] sfm metainfos (neighbors, ranges)
    ##########################################################
    if neighbors is None:
        neighbors, ranges = compute_sfminfos(cfg, imname_list, cameras, resize_hw=resize_hw, max_image_dim=max_image_dim)

    ##########################################################
    # [B] get 2D line segments for each image
    ##########################################################
    all_2d_segs, _ = compute_2d_segs(cfg, imname_list, resize_hw=resize_hw, max_image_dim=max_image_dim, compute_descinfo=False)

    ##########################################################
    # [C] fit 3d segments
    ##########################################################
    fname_fit_segs = '{0}_fit_segs.npy'.format(cfg["line2d"]["detector"])
    if not cfg["load_fit"]:
        seg3d_list = fit_3d_segs(imname_list, all_2d_segs, cameras, depths, cfg["fitting"], resize_hw=resize_hw, max_image_dim=max_image_dim)
        with open(os.path.join(cfg["dir_save"], fname_fit_segs), 'wb') as f: np.savez(f, fit_segs=seg3d_list)
    else:
        with open(os.path.join(cfg["dir_load"], fname_fit_segs), 'rb') as f:
            dd = np.load(f, allow_pickle=True)
            seg3d_list = dd["fit_segs"]

    ##########################################################
    # [D] merge 3d segments
    ##########################################################
    img_hw = utils.read_image(imname_list[0], resize_hw=resize_hw, max_image_dim=max_image_dim).shape[:2]
    fname_all_3d_segs = '{0}_all_3d_segs_wv.npy'.format(cfg["line2d"]["detector"])
    linker = _base.LineLinker(cfg["merging"]["linker2d"], cfg["merging"]["linker3d"])
    graph, linetracks = _mrg.merging(linker, all_2d_segs, cameras, seg3d_list, neighbors, var2d=cfg["merging"]["var2d"])
    linetracks = _mrg.filtertracksbyreprojection(linetracks, cameras, cfg["filtering2d"]["th_angular_2d"], cfg["filtering2d"]["th_perp_2d"], num_outliers=0)
    if not cfg["remerging"]["disable"]:
        linker3d_remerge = _base.LineLinker3d(cfg["remerging"]["linker3d"])
        linetracks = _mrg.remerge(linker3d_remerge, linetracks, num_outliers=0)
        linetracks = _mrg.filtertracksbyreprojection(linetracks, cameras, cfg["filtering2d"]["th_angular_2d"], cfg["filtering2d"]["th_perp_2d"], num_outliers=0)

    ##########################################################
    # [E] geometric refinement
    ##########################################################
    if not cfg["refinement"]["disable"]:
        reconstruction = _base.LineReconstruction(linetracks, cameras)
        lineba_engine = _lineBA.solve(cfg["refinement"], reconstruction)
        new_reconstruction = lineba_engine.GetOutputReconstruction()
        linetracks = new_reconstruction.GetTracks()

    ##########################################################
    # [F] output and visualization
    ##########################################################
    final_output_dir = os.path.join(cfg["dir_save"], "fitnmerge_finaltracks")
    if not os.path.exists(final_output_dir): os.makedirs(final_output_dir)
    vis.save_linetracks_to_folder(linetracks, final_output_dir)

    VisTrack = vis.TrackVisualizer(linetracks)
    VisTrack.report()
    lines_np = VisTrack.get_lines_np()
    counts_np = VisTrack.get_counts_np()
    img_hw = utils.read_image(imname_list[0], resize_hw=resize_hw, max_image_dim=max_image_dim).shape[:2]
    with open(os.path.join(cfg["dir_save"], 'lines_to_vis.npy'), 'wb') as f: np.savez(f, lines=lines_np, counts=counts_np, img_hw=img_hw, ranges=None)
    vis.save_obj(os.path.join(cfg["dir_save"], 'lines_to_vis.obj'), lines_np, counts=counts_np, n_visible_views=cfg['n_visible_views'])

    if cfg["visualize"]:
        import pdb
        pdb.set_trace()
        VisTrack.vis_all_lines(img_hw=img_hw, n_visible_views=cfg["n_visible_views"])
        pdb.set_trace()

