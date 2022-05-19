import os, sys
import numpy as np
from tqdm import tqdm
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import limap.base as _base
import limap.fitting as _fit
import limap.merging as _mrg
import limap.lineBA as _lineBA
import limap.runners as _runners
import limap.util.io_utils as limapio
import limap.visualize as limapvis

def fit_3d_segs(all_2d_segs, imagecols, depths, fitting_config):
    '''
    Args:
    - all_2d_segs: list of 2d segs
    - imagecols: limap.base.ImageCollection
    - depths: list of depth images
    '''
    n_images = len(all_2d_segs)
    seg3d_list = []
    def process(all_2d_segs, camviews_np, depths, fitting_config, idx):
        segs, cam, depth = all_2d_segs[idx], camviews_np[idx], depths[idx]
        img_hw = [cam[3], cam[4]]
        seg3d_list_idx = []
        for seg_id, s in enumerate(segs):
            seg3d = _fit.estimate_seg3d_from_depth(s, depth, [cam[0], cam[1], cam[2]], [img_hw[0], img_hw[1]], ransac_th=fitting_config["ransac_th"], min_percentage_inliers=fitting_config["min_percentage_inliers"], var2d=fitting_config["var2d"])
            if seg3d is None:
                seg3d_list_idx.append((np.array([0., 0., 0.]), np.array([0., 0., 0.])))
            else:
                seg3d_list_idx.append(seg3d)
        return seg3d_list_idx
    camviews = [imagecols.camview(img_id) for img_id in range(imagecols.NumImages())]
    camviews_np = [[view.K(), view.R(), view.T(), view.h(), view.w()] for view in camviews]
    seg3d_list = joblib.Parallel(n_jobs=fitting_config["n_jobs"])(joblib.delayed(process)(all_2d_segs, camviews_np, depths, fitting_config, idx) for idx in tqdm(range(n_images)))
    return seg3d_list

def line_fitnmerge(cfg, imagecols, depths, neighbors=None, ranges=None):
    '''
    Args:
    - imagecols: limap.base.ImageCollection
    - depths: list of depth images
    '''
    # assertion check
    print("[LOG] Number of images: {0}".format(imagecols.NumImages()))
    cfg = _runners.setup(cfg)
    detector_name = cfg["line2d"]["detector"]["method"]
    if cfg["fitting"]["var2d"] == -1:
        cfg["fitting"]["var2d"] = cfg["var2d"][detector_name]
    if cfg["merging"]["var2d"] == -1:
        cfg["merging"]["var2d"] = cfg["var2d"][detector_name]
    limapio.save_txt_imname_list(os.path.join(cfg["dir_save"], 'image_list.txt'), imagecols.get_image_list())
    limapio.save_npy(os.path.join(cfg["dir_save"], 'image_collection.npy'), imagecols.as_dict())

    ##########################################################
    # [A] sfm metainfos (neighbors, ranges)
    ##########################################################
    if neighbors is None:
        neighbors, ranges = _runners.compute_sfminfos(cfg, imagecols)
    else:
        neighbors = [neighbor[:cfg["n_neighbors"]] for neighbor in neighbors]

    ##########################################################
    # [B] get 2D line segments for each image
    ##########################################################
    all_2d_segs, _ = _runners.compute_2d_segs(cfg, imagecols, compute_descinfo=cfg["line2d"]["compute_descinfo"])

    ##########################################################
    # [C] fit 3d segments
    ##########################################################
    fname_fit_segs = '{0}_fit_segs.npy'.format(cfg["line2d"]["detector"]["method"])
    if not cfg["load_fit"]:
        seg3d_list = fit_3d_segs(all_2d_segs, imagecols, depths, cfg["fitting"])
        limapio.save_npy(os.path.join(cfg["dir_save"], fname_fit_segs), seg3d_list)
    else:
        seg3d_list = limapio.read_npy(os.path.join(cfg["dir_load"], fname_fit_segs))

    ##########################################################
    # [D] merge 3d segments
    ##########################################################
    linker = _base.LineLinker(cfg["merging"]["linker2d"], cfg["merging"]["linker3d"])
    graph, linetracks = _mrg.merging(linker, all_2d_segs, imagecols, seg3d_list, neighbors, var2d=cfg["merging"]["var2d"])
    linetracks = _mrg.filtertracksbyreprojection(linetracks, imagecols, cfg["filtering2d"]["th_angular_2d"], cfg["filtering2d"]["th_perp_2d"], num_outliers=0)
    if not cfg["remerging"]["disable"]:
        linker3d_remerge = _base.LineLinker3d(cfg["remerging"]["linker3d"])
        linetracks = _mrg.remerge(linker3d_remerge, linetracks, num_outliers=0)
        linetracks = _mrg.filtertracksbyreprojection(linetracks, imagecols, cfg["filtering2d"]["th_angular_2d"], cfg["filtering2d"]["th_perp_2d"], num_outliers=0)

    ##########################################################
    # [E] geometric refinement
    ##########################################################
    if not cfg["refinement"]["disable"]:
        reconstruction = _base.LineReconstruction(linetracks, imagecols)
        lineba_engine = _lineBA.solve(cfg["refinement"], reconstruction)
        new_reconstruction = lineba_engine.GetOutputReconstruction()
        linetracks = new_reconstruction.GetTracks()

    ##########################################################
    # [F] output and visualization
    ##########################################################
    # save tracks
    limapio.save_folder_linetracks(os.path.join(cfg["dir_save"], "fitnmerge_finaltracks"), linetracks)
    limapio.save_txt_linetracks(os.path.join(cfg["dir_save"], "fitnmerge_alltracks.txt"), linetracks, n_visible_views=4)
    VisTrack = limapvis.PyVistaTrackVisualizer(linetracks, visualize=cfg["visualize"])
    VisTrack.report()
    limapio.save_obj(os.path.join(cfg["dir_save"], 'fitnmerge_lines_nv{0}.obj'.format(cfg["n_visible_views"])), VisTrack.get_lines_np(n_visible_views=cfg["n_visible_views"]))

    if cfg["visualize"]:
        import pdb
        pdb.set_trace()
        VisTrack.vis_all_lines(n_visible_views=cfg["n_visible_views"])
        pdb.set_trace()
    return linetracks

