import os
import numpy as np
from tqdm import tqdm
import joblib

import limap.base as _base
import limap.fitting as _fit
import limap.merging as _mrg
import limap.optimize as _optim
import limap.runners as _runners
import limap.util.io as limapio
import limap.visualize as limapvis

def fit_3d_segs(all_2d_segs, imagecols, depths, fitting_config):
    '''
    Args:
    - all_2d_segs: map<int, np.adarray>
    - imagecols: limap.base.ImageCollection
    - depths: map<int, CustomizedDepthReader>, where CustomizedDepthReader inherits _base.BaseDepthReader
    '''
    n_images = len(all_2d_segs)
    seg3d_list = []
    def process(all_2d_segs, imagecols, depths, fitting_config, img_id):
        segs, camview = all_2d_segs[img_id], imagecols.camview(img_id)
        depth = depths[img_id].read_depth(img_hw=[camview.h(), camview.w()])
        seg3d_list_idx = []
        for seg_id, s in enumerate(segs):
            seg3d = _fit.estimate_seg3d_from_depth(s, depth, camview, ransac_th=fitting_config["ransac_th"], min_percentage_inliers=fitting_config["min_percentage_inliers"], var2d=fitting_config["var2d"])
            if seg3d is None:
                seg3d_list_idx.append((np.array([0., 0., 0.]), np.array([0., 0., 0.])))
            else:
                seg3d_list_idx.append(seg3d)
        return seg3d_list_idx
    image_ids = imagecols.get_img_ids()
    seg3d_list = joblib.Parallel(n_jobs=fitting_config["n_jobs"])(joblib.delayed(process)(all_2d_segs, imagecols, depths, fitting_config, img_id) for img_id in tqdm(image_ids))
    output = {}
    for idx, seg3d_list_idx in enumerate(seg3d_list):
        output[image_ids[idx]] = seg3d_list_idx
    return output

def fit_3d_segs_with_points3d(all_2d_segs, imagecols, p3d_reader, fitting_config, inloc_dataset=None):
    '''
    Args:
    - all_2d_segs: map<int, np.adarray>
    - imagecols: limap.base.ImageCollection
    - p3d_reader: CustomizedP3Dreader inherits _base.BaseP3Dreader
    '''
    seg3d_list = []
    def process(all_2d_segs, imagecols, p3d_reader, fitting_config, img_id):
        segs, camview = all_2d_segs[img_id], imagecols.camview(img_id)
        p3ds = p3d_reader[img_id].read_p3ds()
        seg3d_list_idx = []
        for seg_id, s in enumerate(segs):
            seg3d = _fit.estimate_seg3d_from_points3d(s, p3ds, camview, imagecols.image_name(img_id), inloc_dataset, ransac_th=fitting_config["ransac_th"], min_percentage_inliers=fitting_config["min_percentage_inliers"], var2d=fitting_config["var2d"])
            if seg3d is None:
                seg3d_list_idx.append((np.array([0., 0., 0.]), np.array([0., 0., 0.])))
            else:
                seg3d_list_idx.append(seg3d)
        return seg3d_list_idx
    image_ids = imagecols.get_img_ids()
    seg3d_list = joblib.Parallel(n_jobs=fitting_config["n_jobs"])(joblib.delayed(process)(all_2d_segs, imagecols, p3d_reader, fitting_config, img_id) for img_id in tqdm(image_ids))
    output = {}
    for idx, seg3d_list_idx in enumerate(seg3d_list):
        output[image_ids[idx]] = seg3d_list_idx
    return output

def line_fitnmerge(cfg, imagecols, depths, neighbors=None, ranges=None):
    '''
    Args:
    - imagecols: limap.base.ImageCollection
    - depths: map<int, CustomizedDepthReader>, where CustomizedDepthReader inherits _base.BaseDepthReader
    '''
    # assertion check
    assert imagecols.IsUndistorted() == True
    print("[LOG] Number of images: {0}".format(imagecols.NumImages()))
    cfg = _runners.setup(cfg)
    detector_name = cfg["line2d"]["detector"]["method"]
    if cfg["fitting"]["var2d"] == -1:
        cfg["fitting"]["var2d"] = cfg["var2d"][detector_name]
    if cfg["merging"]["var2d"] == -1:
        cfg["merging"]["var2d"] = cfg["var2d"][detector_name]
    limapio.save_txt_imname_dict(os.path.join(cfg["dir_save"], 'image_list.txt'), imagecols.get_image_name_dict())
    limapio.save_npy(os.path.join(cfg["dir_save"], 'imagecols.npy'), imagecols.as_dict())

    ##########################################################
    # [A] sfm metainfos (neighbors, ranges)
    ##########################################################
    if neighbors is None:
        _, neighbors, ranges = _runners.compute_sfminfos(cfg, imagecols)
    else:
        neighbors = imagecols.update_neighbors(neighbors)
        for img_id, neighbor in neighbors.items():
            neighbors[img_id] = neighbors[img_id][:cfg["n_neighbors"]]

    ##########################################################
    # [B] get 2D line segments for each image
    ##########################################################
    all_2d_segs, _ = _runners.compute_2d_segs(cfg, imagecols, compute_descinfo=cfg["line2d"]["compute_descinfo"])

    ##########################################################
    # [C] fit 3d segments
    ##########################################################
    fname_fit_segs = '{0}_fit_segs.npy'.format(cfg["line2d"]["detector"]["method"])
    if (not cfg["load_fit"]) and (not (cfg["skip_exists"] and os.path.exists(os.path.join(cfg["dir_load"], fname_fit_segs)))):
        seg3d_list = fit_3d_segs(all_2d_segs, imagecols, depths, cfg["fitting"])
        limapio.save_npy(os.path.join(cfg["dir_save"], fname_fit_segs), seg3d_list)
    else:
        seg3d_list = limapio.read_npy(os.path.join(cfg["dir_load"], fname_fit_segs)).item()

    if "do_merging" in cfg["merging"] and not cfg["merging"]["do_merging"]:
        linetracks = []
        for img_id in all_2d_segs:
            for line_id, seg2d in enumerate(all_2d_segs[img_id]):
                seg3d = seg3d_list[img_id][line_id]
                l3d = _base.Line3d(seg3d[0], seg3d[1])
                l2d = _base.Line2d(seg2d[0:2], seg2d[2:4])
                if l3d.length() == 0:
                    continue
                track = _base.LineTrack(l3d, [img_id], [line_id], [l2d])
                linetracks.append(track)
        return linetracks

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
        cfg_ba = _optim.HybridBAConfig(cfg["refinement"])
        cfg_ba.set_constant_camera()
        ba_engine = _optim.solve_line_bundle_adjustment(cfg["refinement"], imagecols, linetracks, max_num_iterations=200)
        linetracks_map = ba_engine.GetOutputLineTracks(num_outliers=cfg["refinement"]["num_outliers_aggregator"])
        linetracks = [track for (track_id, track) in linetracks_map.items()]

    ### Filter out 0-length 3D lines
    linetracks = [track for track in linetracks if track.line.length() > 0]

    ##########################################################
    # [F] output and visualization
    ##########################################################
    # save tracks
    if "output_folder" not in cfg or cfg["output_folder"] is None:
        cfg["output_folder"] = "fitnmerge_finaltracks"
    limapio.save_folder_linetracks_with_info(os.path.join(cfg["dir_save"], cfg["output_folder"]), linetracks, config=cfg, imagecols=imagecols, all_2d_segs=all_2d_segs)
    limapio.save_txt_linetracks(os.path.join(cfg["dir_save"], "fitnmerge_alltracks.txt"), linetracks, n_visible_views=4)
    VisTrack = limapvis.Open3DTrackVisualizer(linetracks)
    VisTrack.report()
    limapio.save_obj(os.path.join(cfg["dir_save"], 'fitnmerge_lines_nv{0}.obj'.format(cfg["n_visible_views"])), VisTrack.get_lines_np(n_visible_views=cfg["n_visible_views"]))

    if cfg["visualize"]:
        import pdb
        pdb.set_trace()
        VisTrack.vis_reconstruction(imagecols, n_visible_views=cfg["n_visible_views"], width=2)
        pdb.set_trace()
    return linetracks

def line_fitting_with_3Dpoints(cfg, imagecols, p3d_readers, inloc_read_transformations=False):
    '''
    Args:
    - imagecols: limap.base.ImageCollection
    - p3d_readers: map<int, CustomizedP3DReader>, where CustomizedP3DReader inherits _base.P3DReader
    '''
    # assertion check
    assert imagecols.IsUndistorted() == True
    print("[LOG] Number of images: {0}".format(imagecols.NumImages()))
    cfg = _runners.setup(cfg)
    detector_name = cfg["line2d"]["detector"]["method"]
    if cfg["fitting"]["var2d"] == -1:
        cfg["fitting"]["var2d"] = cfg["var2d"][detector_name]
    if cfg["merging"]["var2d"] == -1:
        cfg["merging"]["var2d"] = cfg["var2d"][detector_name]
    limapio.save_txt_imname_dict(os.path.join(cfg["dir_save"], 'image_list.txt'), imagecols.get_image_name_dict())
    limapio.save_npy(os.path.join(cfg["dir_save"], 'imagecols.npy'), imagecols.as_dict())

    ##########################################################
    # [A] get 2D line segments for each image
    ##########################################################
    all_2d_segs, _ = _runners.compute_2d_segs(cfg, imagecols, compute_descinfo=cfg["line2d"]["compute_descinfo"])

    ##########################################################
    # [B] fit 3d segments
    ##########################################################
    fname_fit_segs = '{0}_fit_segs.npy'.format(cfg["line2d"]["detector"]["method"])
    if (not cfg["load_fit"]) and (not (cfg["skip_exists"] and os.path.exists(os.path.join(cfg["dir_load"], fname_fit_segs)))):
        if inloc_read_transformations: inloc_dataset = cfg['inloc_dataset']
        else: inloc_dataset = None
        seg3d_list = fit_3d_segs_with_points3d(all_2d_segs, imagecols, p3d_readers, cfg["fitting"], inloc_dataset)
        limapio.save_npy(os.path.join(cfg["dir_save"], fname_fit_segs), seg3d_list)
    else:
        seg3d_list = limapio.read_npy(os.path.join(cfg["dir_load"], fname_fit_segs)).item()

    linetracks = []
    for img_id in all_2d_segs:
        for line_id, seg2d in enumerate(all_2d_segs[img_id]):
            seg3d = seg3d_list[img_id][line_id]
            l3d = _base.Line3d(seg3d[0], seg3d[1])
            l2d = _base.Line2d(seg2d[0:2], seg2d[2:4])
            if l3d.length() == 0:
                continue
            track = _base.LineTrack(l3d, [img_id], [line_id], [l2d])
            linetracks.append(track)
    return linetracks
