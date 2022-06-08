import os, sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import limap.base as _base
import limap.merging as _mrg
import limap.triangulation as _tri
import limap.vpdetection as _vpdet
import limap.lineBA as _lineBA
import limap.runners as _runners
import limap.util.io_utils as limapio
import limap.visualize as limapvis

def line_triangulation(cfg, imagecols, neighbors=None, ranges=None):
    '''
    Args:
    - imagecols: limap.base.ImageCollection
    '''
    print("[LOG] Number of images: {0}".format(imagecols.NumImages()))
    cfg = _runners.setup(cfg)
    detector_name = cfg["line2d"]["detector"]["method"]
    if cfg["triangulation"]["var2d"] == -1:
        cfg["triangulation"]["var2d"] = cfg["var2d"][detector_name]
    # undistort images
    if not imagecols.IsUndistorted():
        imagecols = _runners.undistort_images(imagecols, os.path.join(cfg["output_dir"], cfg["undistortion_output_dir"]), load_undistort=cfg["load_undistort"] or cfg["skip_exists"])
    # resize cameras
    if cfg["max_image_dim"] != -1 and cfg["max_image_dim"] is not None:
        imagecols.set_max_image_dim(cfg["max_image_dim"])
    limapio.save_txt_imname_list(os.path.join(cfg["dir_save"], 'image_list.txt'), imagecols.get_image_list())
    limapio.save_npy(os.path.join(cfg["dir_save"], 'imagecols.npy'), imagecols.as_dict())

    ##########################################################
    # [A] sfm metainfos (neighbors, ranges)
    ##########################################################
    if neighbors is None:
        neighbors, ranges = _runners.compute_sfminfos(cfg, imagecols)
    else:
        neighbors = [neighbor[:cfg["n_neighbors"]] for neighbor in neighbors]

    ##########################################################
    # [B] get 2D line segments and line heatmaps for each image
    ##########################################################
    compute_descinfo = (not cfg["triangulation"]["use_exhaustive_matcher"])
    compute_descinfo = (compute_descinfo and (not cfg["load_match"]) and (not cfg["load_det"])) or cfg["line2d"]["compute_descinfo"]
    all_2d_segs, descinfo_folder = _runners.compute_2d_segs(cfg, imagecols, compute_descinfo=compute_descinfo)

    ##########################################################
    # [C] get line matches
    ##########################################################
    if not cfg["triangulation"]["use_exhaustive_matcher"]:
        matches_dir = _runners.compute_matches(cfg, descinfo_folder, neighbors)

    ##########################################################
    # [D] multi-view triangulation
    ##########################################################
    print('Start multi-view triangulation...')
    Triangulator = _tri.Triangulator(cfg["triangulation"])
    Triangulator.SetRanges(ranges)
    all_2d_lines = _base.GetAllLines2D(all_2d_segs)
    Triangulator.Init(all_2d_lines, imagecols)
    for img_id in range(imagecols.NumImages()):
        if cfg["triangulation"]["use_exhaustive_matcher"]:
            Triangulator.InitExhaustiveMatchImage(img_id, neighbors[img_id])
        else:
            matches = limapio.read_npy(os.path.join(matches_dir, "matches_{0}.npy".format(img_id)))
            Triangulator.InitMatchImage(img_id, matches, neighbors[img_id], triangulate=True, scoring=True)
    Triangulator.RunClustering()
    Triangulator.ComputeLineTracks()
    linetracks = Triangulator.GetTracks()

    # filtering 2d supports
    linetracks = _mrg.filtertracksbyreprojection(linetracks, imagecols, cfg["triangulation"]["filtering2d"]["th_angular_2d"], cfg["triangulation"]["filtering2d"]["th_perp_2d"])
    if not cfg["triangulation"]["remerging"]["disable"]:
        # remerging
        linker3d = _base.LineLinker3d(cfg["triangulation"]["remerging"]["linker3d"])
        linetracks = _mrg.remerge(linker3d, linetracks)
        linetracks = _mrg.filtertracksbyreprojection(linetracks, imagecols, cfg["triangulation"]["filtering2d"]["th_angular_2d"], cfg["triangulation"]["filtering2d"]["th_perp_2d"])
    linetracks = _mrg.filtertracksbysensitivity(linetracks, imagecols, cfg["triangulation"]["filtering2d"]["th_sv_angular_3d"], cfg["triangulation"]["filtering2d"]["th_sv_num_supports"])
    linetracks = _mrg.filtertracksbyoverlap(linetracks, imagecols, cfg["triangulation"]["filtering2d"]["th_overlap"], cfg["triangulation"]["filtering2d"]["th_overlap_num_supports"])
    validtracks = [track for track in linetracks if track.count_images() >= cfg["n_visible_views"]]

    ##########################################################
    # [E] geometric refinement
    ##########################################################
    if not cfg["refinement"]["disable"]:
        reconstruction = _base.LineReconstruction(linetracks, imagecols)
        vpresults = None
        if cfg["refinement"]["use_vp"]:
            vpresults = _vpdet.AssociateVPsParallel(all_2d_lines)
        lineba_engine = _lineBA.solve(cfg["refinement"], reconstruction, vpresults=vpresults, max_num_iterations=200)
        new_reconstruction = lineba_engine.GetOutputReconstruction()
        linetracks = new_reconstruction.GetTracks(num_outliers=cfg["refinement"]["num_outliers_aggregator"])

    ##########################################################
    # [F] output and visualization
    ##########################################################
    # save tracks
    limapio.save_txt_linetracks(os.path.join(cfg["dir_save"], "alltracks.txt"), linetracks, n_visible_views=4)
    limapio.save_folder_linetracks_with_info(os.path.join(cfg["dir_save"], "finaltracks"), linetracks, config=cfg, imagecols=imagecols, all_2d_segs=all_2d_segs)
    VisTrack = limapvis.PyVistaTrackVisualizer(linetracks, visualize=cfg["visualize"])
    VisTrack.report()
    limapio.save_obj(os.path.join(cfg["dir_save"], 'triangulated_lines_nv{0}.obj'.format(cfg["n_visible_views"])), VisTrack.get_lines_np(n_visible_views=cfg["n_visible_views"]))

    # visualize
    if cfg["visualize"]:
        validtracks = [track for track in linetracks if track.count_images() >= cfg["n_visible_views"]]
        def report_track(track_id):
            limapvis.visualize_line_track(imagecols, validtracks[track_id], prefix="track.{0}".format(track_id))
        import pdb
        pdb.set_trace()
        VisTrack.vis_all_lines(n_visible_views=cfg["n_visible_views"], width=2)
        pdb.set_trace()
    return linetracks

