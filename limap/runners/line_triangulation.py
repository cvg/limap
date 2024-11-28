import os

from pycolmap import logging
from tqdm import tqdm

import limap.base as base
import limap.merging as merging
import limap.optimize as optimize
import limap.pointsfm as pointsfm
import limap.runners as runners
import limap.triangulation as triangulation
import limap.util.io as limapio
import limap.visualize as limapvis
import limap.vplib as vplib


def line_triangulation(cfg, imagecols, neighbors=None, ranges=None):
    """
    Main interface of line triangulation over multi-view images.

    Args:
        cfg (dict): Configuration. \
            Refer to :file:`cfgs/triangulation/default.yaml` as an example
        imagecols (:class:`limap.base.ImageCollection`): \
            The image collection corresponding to all the images of interest
        neighbors (dict[int -> list[int]], optional): \
            visual neighbors for each image. By default we compute \
            neighbor information from the covisibility of COLMAP triangulation.
        ranges (pair of :class:`np.array` each of shape (3,), optional): \
            robust 3D ranges for the scene. By default we compute \
            range information from the COLMAP triangulation.
    Returns:
        list[:class:`limap.base.LineTrack`]: list of output 3D line tracks
    """
    logging.info(f"[LOG] Number of images: {imagecols.NumImages()}")
    cfg = runners.setup(cfg)
    detector_name = cfg["line2d"]["detector"]["method"]
    if cfg["triangulation"]["var2d"] == -1:
        cfg["triangulation"]["var2d"] = cfg["var2d"][detector_name]
    # undistort images
    if not imagecols.IsUndistorted():
        imagecols = runners.undistort_images(
            imagecols,
            os.path.join(cfg["dir_save"], cfg["undistortion_output_dir"]),
            skip_exists=cfg["load_undistort"] or cfg["skip_exists"],
            n_jobs=cfg["n_jobs"],
        )
    # resize cameras
    assert imagecols.IsUndistorted()
    if cfg["max_image_dim"] != -1 and cfg["max_image_dim"] is not None:
        imagecols.set_max_image_dim(cfg["max_image_dim"])
    limapio.save_txt_imname_dict(
        os.path.join(cfg["dir_save"], "image_list.txt"),
        imagecols.get_image_name_dict(),
    )
    limapio.save_npy(
        os.path.join(cfg["dir_save"], "imagecols.npy"), imagecols.as_dict()
    )

    ##########################################################
    # [A] sfm metainfos (neighbors, ranges)
    ##########################################################
    sfminfos_colmap_folder = None
    if neighbors is None:
        sfminfos_colmap_folder, neighbors, ranges = runners.compute_sfminfos(
            cfg, imagecols
        )
    else:
        limapio.save_txt_metainfos(
            os.path.join(cfg["dir_save"], "metainfos.txt"), neighbors, ranges
        )
        neighbors = imagecols.update_neighbors(neighbors)
        for img_id, _ in neighbors.items():
            neighbors[img_id] = neighbors[img_id][: cfg["n_neighbors"]]
    limapio.save_txt_metainfos(
        os.path.join(cfg["dir_save"], "metainfos.txt"), neighbors, ranges
    )

    ##########################################################
    # [B] get 2D line segments for each image
    ##########################################################
    compute_descinfo = not cfg["triangulation"]["use_exhaustive_matcher"]
    compute_descinfo = (
        compute_descinfo and (not cfg["load_match"]) and (not cfg["load_det"])
    ) or cfg["line2d"]["compute_descinfo"]
    all_2d_segs, descinfo_folder = runners.compute_2d_segs(
        cfg, imagecols, compute_descinfo=compute_descinfo
    )

    ##########################################################
    # [C] get line matches
    ##########################################################
    if not cfg["triangulation"]["use_exhaustive_matcher"]:
        matches_dir = runners.compute_matches(
            cfg, descinfo_folder, imagecols.get_img_ids(), neighbors
        )

    ##########################################################
    # [D] multi-view triangulation
    ##########################################################
    Triangulator = triangulation.GlobalLineTriangulator(cfg["triangulation"])
    Triangulator.SetRanges(ranges)
    all_2d_lines = base.get_all_lines_2d(all_2d_segs)
    Triangulator.Init(all_2d_lines, imagecols)
    if cfg["triangulation"]["use_vp"]:
        vpdetector = vplib.get_vp_detector(
            cfg["triangulation"]["vpdet_config"],
            n_jobs=cfg["triangulation"]["vpdet_config"]["n_jobs"],
        )
        vpresults = vpdetector.detect_vp_all_images(
            all_2d_lines, imagecols.get_map_camviews()
        )
        Triangulator.InitVPResults(vpresults)
    # get 2d bipartites from pointsfm model
    if cfg["triangulation"]["use_pointsfm"]["enable"]:
        if cfg["triangulation"]["use_pointsfm"]["colmap_folder"] is None:
            colmap_model_path = None
            # check if colmap model exists from sfminfos computation
            if (
                cfg["triangulation"]["use_pointsfm"]["reuse_sfminfos_colmap"]
                and sfminfos_colmap_folder is not None
            ):
                colmap_model_path = os.path.join(
                    sfminfos_colmap_folder, "sparse"
                )
                if not pointsfm.check_exists_colmap_model(colmap_model_path):
                    colmap_model_path = None
            # retriangulate
            if colmap_model_path is None:
                colmap_output_path = os.path.join(
                    cfg["dir_save"], "colmap_outputs_junctions"
                )
                input_neighbors = None
                if cfg["triangulation"]["use_pointsfm"]["use_neighbors"]:
                    input_neighbors = neighbors
                pointsfm.run_colmap_sfm_with_known_poses(
                    cfg["sfm"],
                    imagecols,
                    output_path=colmap_output_path,
                    skip_exists=cfg["skip_exists"],
                    neighbors=input_neighbors,
                )
                colmap_model_path = os.path.join(colmap_output_path, "sparse")
        else:
            colmap_model_path = cfg["triangulation"]["use_pointsfm"][
                "colmap_folder"
            ]
        reconstruction = pointsfm.PyReadCOLMAP(colmap_model_path)
        all_bpt2ds, sfm_points = runners.compute_2d_bipartites_from_colmap(
            reconstruction, imagecols, all_2d_lines, cfg["structures"]["bpt2d"]
        )
        Triangulator.SetBipartites2d(all_bpt2ds)
        if cfg["triangulation"]["use_pointsfm"]["use_triangulated_points"]:
            Triangulator.SetSfMPoints(sfm_points)
    # triangulate
    logging.info("Start multi-view triangulation...")
    for img_id in tqdm(imagecols.get_img_ids()):
        if cfg["triangulation"]["use_exhaustive_matcher"]:
            Triangulator.TriangulateImageExhaustiveMatch(
                img_id, neighbors[img_id]
            )
        else:
            matches = limapio.read_npy(
                os.path.join(matches_dir, f"matches_{img_id}.npy")
            ).item()
            Triangulator.TriangulateImage(img_id, matches)
    linetracks = Triangulator.ComputeLineTracks()

    # filtering 2d supports
    linetracks = merging.filter_tracks_by_reprojection(
        linetracks,
        imagecols,
        cfg["triangulation"]["filtering2d"]["th_angular_2d"],
        cfg["triangulation"]["filtering2d"]["th_perp_2d"],
    )
    if not cfg["triangulation"]["remerging"]["disable"]:
        # remerging
        linker3d = base.LineLinker3d(
            cfg["triangulation"]["remerging"]["linker3d"]
        )
        linetracks = merging.remerge(linker3d, linetracks)
        linetracks = merging.filter_tracks_by_reprojection(
            linetracks,
            imagecols,
            cfg["triangulation"]["filtering2d"]["th_angular_2d"],
            cfg["triangulation"]["filtering2d"]["th_perp_2d"],
        )
    linetracks = merging.filter_tracks_by_sensitivity(
        linetracks,
        imagecols,
        cfg["triangulation"]["filtering2d"]["th_sv_angular_3d"],
        cfg["triangulation"]["filtering2d"]["th_sv_num_supports"],
    )
    linetracks = merging.filter_tracks_by_overlap(
        linetracks,
        imagecols,
        cfg["triangulation"]["filtering2d"]["th_overlap"],
        cfg["triangulation"]["filtering2d"]["th_overlap_num_supports"],
    )
    validtracks = [
        track
        for track in linetracks
        if track.count_images() >= cfg["n_visible_views"]
    ]

    ##########################################################
    # [E] geometric refinement
    ##########################################################
    if not cfg["refinement"]["disable"]:
        cfg_ba = optimize.HybridBAConfig(cfg["refinement"])
        cfg_ba.set_constant_camera()
        ba_engine = optimize.solve_line_bundle_adjustment(
            cfg["refinement"], imagecols, linetracks, max_num_iterations=200
        )
        linetracks_map = ba_engine.GetOutputLineTracks(
            num_outliers=cfg["refinement"]["num_outliers_aggregator"]
        )
        linetracks = [track for (track_id, track) in linetracks_map.items()]

    ##########################################################
    # [F] output and visualization
    ##########################################################
    # save tracks
    limapio.save_txt_linetracks(
        os.path.join(cfg["dir_save"], "alltracks.txt"),
        linetracks,
        n_visible_views=4,
    )
    limapio.save_folder_linetracks_with_info(
        os.path.join(cfg["dir_save"], cfg["output_folder"]),
        linetracks,
        config=cfg,
        imagecols=imagecols,
        all_2d_segs=all_2d_segs,
    )
    VisTrack = limapvis.Open3DTrackVisualizer(linetracks)
    VisTrack.report()
    limapio.save_obj(
        os.path.join(
            cfg["dir_save"],
            "triangulated_lines_nv{}.obj".format(cfg["n_visible_views"]),
        ),
        VisTrack.get_lines_np(n_visible_views=cfg["n_visible_views"]),
    )

    # visualize
    if cfg["visualize"]:
        validtracks = [
            track
            for track in linetracks
            if track.count_images() >= cfg["n_visible_views"]
        ]

        def report_track(track_id):
            limapvis.visualize_line_track(
                imagecols,
                validtracks[track_id],
                prefix=f"track.{track_id}",
            )

        import pdb

        pdb.set_trace()
        VisTrack.vis_reconstruction(
            imagecols, n_visible_views=cfg["n_visible_views"]
        )
        pdb.set_trace()
    return linetracks
