import os

import numpy as np
from pycolmap import logging
from tqdm import tqdm

import limap.base as base
import limap.evaluation as limap_eval
import limap.features as limap_features
import limap.optimize as optimize
import limap.util.io as limapio
import limap.visualize as limapvis


def line_refinement(
    cfg,
    tracks,
    imagecols,
    heatmap_dir=None,
    patch_dir=None,
    featuremap_dir=None,
    vpresults=None,
    n_visible_views=4,
):
    """
    refine each line one by one with fixed cameras
    """
    evaluator = None
    if cfg["mesh_dir"] is not None:
        MPAU = 0.02539999969303608  # hypersim
        evaluator = limap_eval.MeshEvaluator(cfg["mesh_dir"], MPAU)

    ids = [
        k
        for k in range(len(tracks))
        if tracks[k].count_images() >= n_visible_views
    ]
    opttracks = []
    for t_id, track_id in enumerate(tqdm(ids)):
        # initialize data
        track = tracks[track_id]
        sorted_ids = track.GetSortedImageIds()
        p_cameras = []
        p_vpresults = None
        p_heatmaps, p_patches, p_features = None, None, None
        if cfg["use_vp"]:
            p_vpresults = []
        if cfg["use_heatmap"]:
            p_heatmaps = []
        if cfg["use_feature"]:
            if patch_dir is not None:
                p_patches = []
            else:
                p_features = []
        for img_id in sorted_ids:  # add data for each supporting image
            p_cameras.append(imagecols.camview(img_id))
            if cfg["use_vp"]:
                p_vpresults.append(vpresults[img_id])
            if cfg["use_heatmap"]:
                heatmap = limapio.read_npy(
                    os.path.join(heatmap_dir, f"heatmap_{img_id}.npy")
                )
                p_heatmaps.append(heatmap)
            if cfg["use_feature"]:
                if patch_dir is not None:
                    fname = os.path.join(
                        patch_dir,
                        f"track{track_id}",
                        f"track{track_id}_img{img_id}.npy",
                    )
                    patch = limap_features.load_patch(fname, dtype=cfg["dtype"])
                    p_patches.append(patch)
                else:
                    with open(
                        os.path.join(featuremap_dir, "feature_{}.npy").format(
                            img_id
                        ),
                        "rb",
                    ) as f:
                        featuremap = np.load(f, allow_pickle=True)
                    p_features.append(featuremap.transpose(1, 2, 0))

        # refine
        rf_engine = optimize.solve_line_refinement(
            cfg,
            track,
            p_cameras,
            p_vpresults=p_vpresults,
            p_heatmaps=p_heatmaps,
            p_patches=p_patches,
            p_features=p_features,
            dtype=cfg["dtype"],
        )
        newtrack = base.LineTrack(track)
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
                logging.info(
                    f"[DEBUG] t_id = {t_id}, \
                      original: dist = {dist * 1000:.4f}, ratio = {ratio:.4f}"
                )
                logging.info(
                    f"[DEBUG] t_id = {t_id}, optimized: \
                      dist = {newdist * 1000:.4f}, ratio = {newratio:.4f}"
                )

    # output
    newtracks = []
    counter = 0
    for track in tracks:
        if track.count_images() < n_visible_views:
            newtracks.append(track)
        else:
            newtracks.append(opttracks[counter])
            counter += 1

    # debug
    if cfg["visualize"]:
        import pdb

        pdb.set_trace()
        VisTrack = limapvis.Open3DTrackVisualizer(newtracks)
        VisTrack.vis_reconstruction(
            imagecols, n_visible_views=n_visible_views, width=2
        )
    return newtracks
