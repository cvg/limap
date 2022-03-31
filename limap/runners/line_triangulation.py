import os, sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import core.visualize as vis
import core.utils as utils

import limap.base as _base
import limap.merging as _mrg
import limap.triangulation as _tri
import limap.vpdetection as _vpdet
import limap.lineBA as _lineBA
import limap.runners as _runners

def line_triangulation(cfg, imname_list, camviews, neighbors=None, ranges=None, resize_hw=None, max_image_dim=None, valid_index_list=None):
    '''
    Args:
    - image_names: list of imname
    - camviews: list of limap.base.CameraView
    '''
    cfg = _runners.setup(cfg, imname_list, camviews, max_image_dim=None)
    if cfg["triangulation"]["var2d"] == -1:
        cfg["triangulation"]["var2d"] = cfg["var2d"][cfg["line2d"]["detector"]]
    if (max_image_dim is not None) and max_image_dim != -1:
        for camview in camviews:
            camview.cam.set_max_image_dim(cfg["max_image_dim"])
    if (resize_hw is not None):
        for camview in camviews:
            camview.cam.resize(resize_hw[1], resize_hw[0])
    if valid_index_list is not None:
        assert len(valid_index_list) == len(imname_list)

    ##########################################################
    # [A] sfm metainfos (neighbors, ranges)
    ##########################################################
    if neighbors is None:
        neighbors, ranges = _runners.compute_sfminfos(cfg, imname_list, camviews, resize_hw=resize_hw, max_image_dim=max_image_dim)

    ##########################################################
    # [B] get 2D line segments and line heatmaps for each image
    ##########################################################
    compute_descinfo = (not cfg["triangulation"]["use_exhaustive_matcher"])
    all_2d_segs, descinfo_folder = _runners.compute_2d_segs(cfg, imname_list, resize_hw=resize_hw, max_image_dim=max_image_dim, compute_descinfo=compute_descinfo)
    if valid_index_list is not None:
        all_2d_segs = all_2d_segs[valid_index_list]

    # Writing out information for localization
    loc_dir = os.path.join(cfg["dir_save"], "localization")
    if not os.path.exists(loc_dir):
        os.makedirs(loc_dir)
    vis.txt_save_image_list(imname_list, os.path.join(loc_dir, "image_list.txt"))
    vis.txt_save_neighbors(neighbors, os.path.join(loc_dir, "neighbors.txt"))
    # vis.txt_save_detections(all_2d_segs, os.path.join(loc_dir, "detections.txt"))

    ##########################################################
    # [C] get line matches
    ##########################################################
    if cfg["triangulation"]["use_exhaustive_matcher"]:
        matches_dir = None
    else:
        matches_dir = _runners.compute_matches(cfg, imname_list, descinfo_folder, neighbors)
        print("Loading {0}".format(matches_dir))

    ##########################################################
    # [D] multi-view triangulation
    ##########################################################
    print('Start multi-view triangulation...')
    Triangulator = _tri.Triangulator(cfg["triangulation"])
    Triangulator.SetRanges(ranges)
    all_2d_lines = _tri.GetAllLines2D(all_2d_segs)
    Triangulator.Init(all_2d_lines, camviews)
    if valid_index_list is None:
        valid_index_list = np.arange(0, len(imname_list)).tolist()
    for img_id, index in enumerate(tqdm(valid_index_list)):
        if cfg["triangulation"]["use_exhaustive_matcher"]:
            Triangulator.InitExhaustiveMatchImage(img_id, neighbors[img_id])
        else:
            with open(os.path.join(matches_dir, "matches_{0}.npy".format(index)), 'rb') as f:
                data = np.load(f, allow_pickle=True)
                matches = data["data"]
            Triangulator.InitMatchImage(img_id, matches, neighbors[img_id], triangulate=True, scoring=True)
    Triangulator.RunClustering()
    Triangulator.ComputeLineTracks()
    linetracks = Triangulator.GetTracks()

    # filtering 2d supports
    linetracks = _mrg.filtertracksbyreprojection(linetracks, camviews, cfg["triangulation"]["filtering2d"]["th_angular_2d"], cfg["triangulation"]["filtering2d"]["th_perp_2d"])
    if not cfg["triangulation"]["remerging"]["disable"]:
        # remerging
        linker3d = _base.LineLinker3d(cfg["triangulation"]["remerging"]["linker3d"])
        linetracks = _mrg.remerge(linker3d, linetracks)
        linetracks = _mrg.filtertracksbyreprojection(linetracks, camviews, cfg["triangulation"]["filtering2d"]["th_angular_2d"], cfg["triangulation"]["filtering2d"]["th_perp_2d"])
    linetracks = _mrg.filtertracksbysensitivity(linetracks, camviews, cfg["triangulation"]["filtering2d"]["th_sv_angular_3d"], cfg["triangulation"]["filtering2d"]["th_sv_num_supports"])
    linetracks = _mrg.filtertracksbyoverlap(linetracks, camviews, cfg["triangulation"]["filtering2d"]["th_overlap"], cfg["triangulation"]["filtering2d"]["th_overlap_num_supports"])
    validtracks = [track for track in linetracks if track.count_images() >= cfg["n_visible_views"]]

    ##########################################################
    # [E] geometric refinement
    ##########################################################
    if not cfg["refinement"]["disable"]:
        reconstruction = _base.LineReconstruction(linetracks, camviews)
        vpresults = None
        if cfg["refinement"]["use_vp"]:
            vpresults = _vpdet.AssociateVPsParallel(all_2d_lines)
        lineba_engine = _lineBA.solve(cfg["refinement"], reconstruction, vpresults=vpresults, max_num_iterations=200)
        new_reconstruction = lineba_engine.GetOutputReconstruction()
        linetracks = new_reconstruction.GetTracks(num_outliers=cfg["refinement"]["num_outliers_aggregator"])

    ##########################################################
    # [F] output and visualization
    ##########################################################
    # save info
    final_output_dir = os.path.join(cfg["dir_save"], "finaltracks")
    if not os.path.exists(final_output_dir): os.makedirs(final_output_dir)
    vis.save_linetracks_to_folder(linetracks, final_output_dir)
    vis.txt_save_linetracks(linetracks, os.path.join(loc_dir, "alltracks.txt"), n_visible_views=4)
    camviews_np = [[view.K(), view.R(), view.T()[:,None].repeat(3, 1), [view.h(), view.w()]] for view in camviews]
    with open(os.path.join(final_output_dir, "all_infos.npy"), "wb") as f:
        np.savez(f, imname_list=imname_list, all_2d_segs=all_2d_segs, camviews_np=camviews_np, cfg=cfg, n_tracks=len(linetracks))

    VisTrack = vis.TrackVisualizer(linetracks, visualize=cfg["visualize"])
    VisTrack.report()
    lines_np = VisTrack.get_lines_np()
    counts_np = VisTrack.get_counts_np()
    img_hw = [camviews[0].h(), camviews[1].w()]
    with open(os.path.join(cfg["dir_save"], 'lines_to_vis.npy'), 'wb') as f: np.savez(f, lines=lines_np, counts=counts_np, img_hw=img_hw, ranges=None)
    vis.save_obj(os.path.join(cfg["dir_save"], 'lines_to_vis.obj'), lines_np, counts=counts_np, n_visible_views=cfg['n_visible_views'])
    # vis.save_obj(os.path.join(cfg["dir_save"], 'lines_nodes.obj'), Triangulator.GetAllValidBestTris())
    validtracks = [track for track in linetracks if track.count_images() >= cfg["n_visible_views"]]

    # visualize
    if cfg["visualize"]:
        def report_track(track_id):
            vis.visualize_line_track(imname_list, validtracks[track_id], max_image_dim=cfg["max_image_dim"], camviews=camviews, prefix="track.{0}".format(track_id))
        import pdb
        pdb.set_trace()
        VisTrack.vis_all_lines(img_hw, n_visible_views=cfg["n_visible_views"], width=2)
        pdb.set_trace()
    return linetracks

