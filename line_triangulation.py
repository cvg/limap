import os, sys
import numpy as np
from tqdm import tqdm
import core.detector.LSD as lsd
import core.detector.SOLD2 as sold2
import core.visualize as vis
import core.utils as utils

import limap.base as _base
import limap.merging as _mrg
import limap.triangulation as _tri
import limap.vpdetection as _vpdet
import limap.lineBA as _lineBA

def setup(cfg, imname_list, camviews, max_image_dim=None):
    # assertion check
    print("number of images: {0}".format(len(imname_list)))
    assert len(imname_list) == len(camviews), "number of images should match number of camviews"
    folder_to_save = cfg["folder_to_save"]
    if cfg["folder_to_save"] is None:
        folder_to_save = 'tmp'
    if not os.path.exists(folder_to_save): os.makedirs(folder_to_save)
    folder_to_load = cfg["folder_to_load"]
    if cfg["use_tmp"]: folder_to_load = "tmp"
    cfg["dir_save"] = folder_to_save
    cfg["dir_load"] = folder_to_load
    return cfg

def compute_sfminfos(cfg, imname_list, camviews, fname="sfm_metainfos.npy", resize_hw=None, max_image_dim=None):
    import limap.pointsfm as _psfm
    if not cfg["load_meta"]:
        # run colmap sfm and compute neighbors, ranges
        colmap_output_path = cfg["sfm"]["colmap_output_path"]
        if not cfg["sfm"]["reuse"]:
            _psfm.run_colmap_sfm_with_known_poses(cfg["sfm"], imname_list, camviews, resize_hw=resize_hw, max_image_dim=max_image_dim, output_path=colmap_output_path, use_cuda=cfg["use_cuda"])
        neighbors = _psfm.ComputeNeighborsSorted(colmap_output_path, cfg["n_neighbors"], min_triangulation_angle=cfg["sfm"]["min_triangulation_angle"], neighbor_type=cfg["sfm"]["neighbor_type"])
        ranges = _psfm.ComputeRanges(colmap_output_path, range_robust=cfg["sfm"]["ranges"]["range_robust"], k_stretch=cfg["sfm"]["ranges"]["k_stretch"])
    else:
        fname_load = os.path.join(cfg["dir_load"], fname)
        with open(fname_load, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            neighbors = data['neighbors']
            neighbors = [neighbor[:cfg["n_neighbors"]] for neighbor in neighbors]
            ranges = data['ranges']
    with open(os.path.join(cfg["dir_save"], fname), 'wb') as f: np.savez(f, imname_list=imname_list, neighbors=neighbors, ranges=ranges)
    return neighbors, ranges

def compute_2d_segs(cfg, imname_list, resize_hw=None, max_image_dim=None, compute_descinfo=True):
    descinfo_folder = None
    compute_descinfo = (compute_descinfo and (not cfg["load_match"]) and (not cfg["load_det"])) or cfg["line2d"]["compute_descinfo"]
    if not cfg["load_det"]:
        descinfo_folder = os.path.join(cfg["dir_save"], "{0}_descinfos".format(cfg["line2d"]["detector"]))
        heatmap_dir = os.path.join(cfg["dir_save"], 'sold2_heatmaps')
        if cfg["line2d"]["detector"] == "sold2":
            all_2d_segs, descinfos = sold2.sold2_detect_2d_segs_on_images(imname_list, resize_hw=resize_hw, max_image_dim=max_image_dim, heatmap_dir=heatmap_dir, max_num_2d_segs=cfg["line2d"]["max_num_2d_segs"])
            vis.save_datalist_to_folder(descinfo_folder, 'descinfo', imname_list, descinfos, is_descinfo=True)
            del descinfos
        elif cfg["line2d"]["detector"] == "lsd":
            all_2d_segs = lsd.lsd_detect_2d_segs_on_images(imname_list, resize_hw=resize_hw, max_image_dim=max_image_dim, max_num_2d_segs=cfg["line2d"]["max_num_2d_segs"])
        with open(os.path.join(cfg["dir_save"], '{0}_all_2d_segs.npy'.format(cfg["line2d"]["detector"])), 'wb') as f: np.savez(f, imname_list=imname_list, all_2d_segs=all_2d_segs)
        if cfg["line2d"]["detector"] != "sold2" and compute_descinfo:
            # we use the sold2 descriptors for all detectors for now
            sold2.sold2_compute_descinfos(imname_list, all_2d_segs, resize_hw=resize_hw, max_image_dim=max_image_dim, descinfo_dir=descinfo_folder)
    else:
        descinfo_folder = os.path.join(cfg["dir_load"], "{0}_descinfos".format(cfg["line2d"]["detector"]))
        fname_all_2d_segs = os.path.join(cfg["dir_load"], "{0}_all_2d_segs.npy".format(cfg["line2d"]["detector"]))
        print("Loading {0}...".format(fname_all_2d_segs))
        with open(fname_all_2d_segs, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            all_2d_segs = data['all_2d_segs']
        if compute_descinfo:
            descinfo_folder = os.path.join(cfg["dir_save"], "{0}_descinfos".format(cfg["line2d"]["detector"]))
            sold2.sold2_compute_descinfos(imname_list, all_2d_segs, resize_hw=resize_hw, max_image_dim=max_image_dim, descinfo_dir=descinfo_folder)
    # visualize
    if cfg["line2d"]["visualize"]:
        vis.tmp_visualize_2d_segs(imname_list, all_2d_segs, resize_hw=resize_hw, max_image_dim=max_image_dim)
    if cfg["line2d"]["save_l3dpp"]:
        img_hw = utils.read_image(imname_list[0], resize_hw=resize_hw, max_image_dim=max_image_dim).shape[:2]
        vis.tmp_save_all_2d_segs_for_l3dpp(imname_list, all_2d_segs, img_hw, folder=os.path.join(cfg["dir_save"], "l3dpp"))
    return all_2d_segs, descinfo_folder

def compute_matches(cfg, imname_list, descinfo_folder, neighbors):
    fname_all_matches = '{0}_all_matches_n{1}_top{2}.npy'.format(cfg["line2d"]["detector"], cfg["n_neighbors"], cfg["line2d"]["topk"])
    matches_dir = '{0}_all_matches_n{1}_top{2}'.format(cfg["line2d"]["detector"], cfg["n_neighbors"], cfg["line2d"]["topk"])
    if not cfg['load_match']:
        if descinfo_folder is None:
            descinfo_folder = os.path.join(cfg["dir_load"], "{0}_descinfos".format(cfg["line2d"]["detector"]))
        matches_folder = os.path.join(cfg["dir_save"], matches_dir)
        if cfg["line2d"]["topk"] == 0:
            all_matches = sold2.sold2_match_2d_segs_with_descinfo_by_folder(descinfo_folder, neighbors, n_jobs=cfg["line2d"]["n_jobs"], matches_dir=matches_folder)
        else:
            all_matches = sold2.sold2_match_2d_segs_with_descinfo_topk_by_folder(descinfo_folder, neighbors, topk=cfg["line2d"]["topk"], n_jobs=cfg["line2d"]["n_jobs"], matches_dir=matches_folder)
        with open(os.path.join(matches_folder, 'imname_list.npy'), 'wb') as f:
            np.savez(f, imname_list=imname_list)
        return matches_folder
    else:
        folder = os.path.join(cfg["dir_load"], matches_dir)
        if not os.path.exists(folder):
            raise ValueError("Folder {0} not found.".format(folder))
        return folder

def line_triangulation(cfg, imname_list, camviews, neighbors=None, ranges=None, resize_hw=None, max_image_dim=None, valid_index_list=None):
    '''
    Args:
    - image_names: list of imname
    - camviews: list of limap.base.CameraView
    '''
    cfg = setup(cfg, imname_list, camviews, max_image_dim=None)
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
        neighbors, ranges = compute_sfminfos(cfg, imname_list, camviews, resize_hw=resize_hw, max_image_dim=max_image_dim)

    ##########################################################
    # [B] get 2D line segments and line heatmaps for each image
    ##########################################################
    compute_descinfo = (not cfg["triangulation"]["use_exhaustive_matcher"])
    all_2d_segs, descinfo_folder = compute_2d_segs(cfg, imname_list, resize_hw=resize_hw, max_image_dim=max_image_dim, compute_descinfo=compute_descinfo)
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
        matches_dir = compute_matches(cfg, imname_list, descinfo_folder, neighbors)
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

