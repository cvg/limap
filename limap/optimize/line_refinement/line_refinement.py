import os
import numpy as np
from tqdm import tqdm

import limap.base as _base
import limap.features as _features
import limap.evaluation as _eval
import limap.optimize as _optim
import limap.util.io as limapio
import limap.visualize as limapvis

def line_refinement(cfg, tracks, imagecols, heatmap_dir=None, patch_dir=None, featuremap_dir=None, vpresults=None, n_visible_views=4):
    '''
    refine each line one by one with fixed cameras
    '''
    evaluator = None
    if cfg["mesh_dir"] is not None:
        MPAU = 0.02539999969303608 # hypersim
        evaluator = _eval.MeshEvaluator(cfg["mesh_dir"], MPAU)

    ids = [k for k in range(len(tracks)) if tracks[k].count_images() >= n_visible_views]
    opttracks = []
    for t_id, track_id in enumerate(tqdm(ids)):
        # initialize data
        track = tracks[track_id]
        sorted_ids = track.GetSortedImageIds()
        p_cameras = []
        p_vpresults = None
        p_heatmaps, p_patches, p_features = None, None, None
        if cfg["use_vp"]: p_vpresults = []
        if cfg["use_heatmap"]: p_heatmaps = []
        if cfg["use_feature"]:
            if patch_dir is not None:
                p_patches = []
            else:
                p_features = []
        for img_id in sorted_ids: # add data for each supporting image
            p_cameras.append(imagecols.camview(img_id))
            if cfg["use_vp"]:
                p_vpresults.append(vpresults[img_id])
            if cfg["use_heatmap"]:
                heatmap = limapio.read_npy(os.path.join(heatmap_dir, "heatmap_{0}.npy".format(img_id)))
                p_heatmaps.append(heatmap)
            if cfg["use_feature"]:
                if patch_dir is not None:
                    fname = os.path.join(patch_dir, "track{0}".format(track_id), "track{0}_img{1}.npy".format(track_id, img_id))
                    patch = _features.load_patch(fname, dtype=cfg["dtype"])
                    p_patches.append(patch)
                else:
                    with open(os.path.join(featuremap_dir, "feature_{}.npy").format(img_id), 'rb') as f:
                        featuremap = np.load(f, allow_pickle=True)
                    p_features.append(featuremap.transpose(1,2,0))

        # refine
        rf_engine = _optim.solve_line_refinement(cfg, track, p_cameras, p_vpresults=p_vpresults, p_heatmaps=p_heatmaps, p_patches=p_patches, p_features=p_features, dtype=cfg["dtype"])
        newtrack = _base.LineTrack(track)
        if rf_engine is None:
            opttracks.append(newtrack)
            continue
        newtrack.line = rf_engine.GetLine3d()
        opttracks.append(newtrack)

        # evaluation
        if evaluator is not None:
            dist = evaluator.ComputeDistLine(track.line, n_samples=1000)
            ratio = evaluator.ComputeInlierRatio(track.line, 0.001)
            newdist = evaluator.ComputeDistLine(newtrack.line, n_samples=1000)
            newratio = evaluator.ComputeInlierRatio(newtrack.line, 0.001)
            if newdist > dist and newratio < ratio:
                print("[DEBUG] t_id = {0}, original: dist = {1:.4f}, ratio = {2:.4f}".format(t_id, dist * 1000, ratio))
                print("[DEBUG] t_id = {0}, optimized: dist = {1:.4f}, ratio = {2:.4f}".format(t_id, newdist * 1000, newratio))

    # output
    newtracks = []
    counter = 0
    for idx, track in enumerate(tracks):
        if track.count_images() < n_visible_views:
            newtracks.append(track)
        else:
            newtracks.append(opttracks[counter])
            counter += 1

    # debug
    if cfg["visualize"]:
        def report_track(track_id):
            limapvis.visualize_line_track(imname_list, tracks[track_id], max_image_dim=-1, cameras=cameras, prefix="track.{0}".format(track_id))
        def report_newtrack(track_id):
            limapvis.visualize_line_track(imname_list, opttracks[track_id], max_image_dim=-1, cameras=cameras, prefix="newtrack.{0}".format(track_id))
        import pdb
        pdb.set_trace()
        VisTrack = limapvis.Open3DTrackVisualizer(newtracks)
        VisTrack.vis_reconstruction(imagecols, n_visible_views=n_visible_views, width=2)
    return newtracks


