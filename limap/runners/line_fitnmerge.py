import os

import joblib
import numpy as np
from pycolmap import logging
from tqdm import tqdm

import limap.base as base
import limap.fitting as fitting
import limap.merging as merging
import limap.optimize as optimize
import limap.runners as runners
import limap.util.io as limapio
import limap.visualize as limapvis


def fit_3d_segs(all_2d_segs, imagecols, depths, fitting_config):
    """
    Fit 3D line segments over points produced by depth unprojection

    Args:
        all_2d_segs (dict[int -> :class:`np.adarray`]): \
            All the 2D line segments for each image
        imagecols (:class:`limap.base.ImageCollection`): \
            The image collection of all images of interest
        depths (dict[int -> :class:`CustomizedDepthReader`], \
        where :class:`CustomizedDepthReader` inherits \
        :class:`limap.base.depth_reader_base.BaseDepthReader`): \
            The depth map readers for each image
        fitting_config (dict): Configuration, \
            fields refer to :file:`cfgs/examples/fitting_3Dline.yaml`
    Returns:
        output (dict[int -> list[(:class:`np.array`, :class:`np.array`)]]): \
        for each image, output a list of :class:`np.array` pair, \
        representing two endpoints
    """
    seg3d_list = []

    def process(all_2d_segs, imagecols, depths, fitting_config, img_id):
        segs, camview = all_2d_segs[img_id], imagecols.camview(img_id)
        depth = depths[img_id].read_depth(img_hw=[camview.h(), camview.w()])
        seg3d_list_idx = []
        for s in segs:
            seg3d = fitting.estimate_seg3d_from_depth(
                s,
                depth,
                camview,
                ransac_th=fitting_config["ransac_th"],
                min_percentage_inliers=fitting_config["min_percentage_inliers"],
                var2d=fitting_config["var2d"],
            )
            if seg3d is None:
                seg3d_list_idx.append(
                    (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
                )
            else:
                seg3d_list_idx.append(seg3d)
        return seg3d_list_idx

    image_ids = imagecols.get_img_ids()
    seg3d_list = joblib.Parallel(n_jobs=fitting_config["n_jobs"])(
        joblib.delayed(process)(
            all_2d_segs, imagecols, depths, fitting_config, img_id
        )
        for img_id in tqdm(image_ids)
    )
    output = {}
    for idx, seg3d_list_idx in enumerate(seg3d_list):
        output[image_ids[idx]] = seg3d_list_idx
    return output


def fit_3d_segs_with_points3d(
    all_2d_segs, imagecols, p3d_reader, fitting_config, inloc_dataset=None
):
    """
    Fit 3D line segments over a set of 3D points

    Args:
        all_2d_segs (dict[int -> :class:`np.adarray`]): \
            All the 2D line segments for each image
        imagecols (:class:`limap.base.ImageCollection`): \
            The image collection of all images of interest
        p3d_reader (dict[int -> :class:`CustomizedP3DReader`], \
        where :class:`CustomizedP3DReader` inherits \
        :class:`limap.base.p3d_reader_base.BaseP3DReader`): \
            The point cloud readers for each image
        fitting_config (dict): Configuration, \
            fields refer to :file:`cfgs/examples/fitting_3Dline.yaml`
    Returns:
        output (dict[int -> list[(:class:`np.array`, :class:`np.array`)]]): \
            for each image, output a list of :class:`np.array` pair, \
            representing two endpoints
    """
    seg3d_list = []

    def process(all_2d_segs, imagecols, p3d_reader, fitting_config, img_id):
        segs, camview = all_2d_segs[img_id], imagecols.camview(img_id)
        p3ds = p3d_reader[img_id].read_p3ds()
        seg3d_list_idx = []
        for s in segs:
            seg3d = fitting.estimate_seg3d_from_points3d(
                s,
                p3ds,
                camview,
                imagecols.image_name(img_id),
                inloc_dataset,
                ransac_th=fitting_config["ransac_th"],
                min_percentage_inliers=fitting_config["min_percentage_inliers"],
                var2d=fitting_config["var2d"],
            )
            if seg3d is None:
                seg3d_list_idx.append(
                    (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
                )
            else:
                seg3d_list_idx.append(seg3d)
        return seg3d_list_idx

    image_ids = imagecols.get_img_ids()
    seg3d_list = joblib.Parallel(n_jobs=fitting_config["n_jobs"])(
        joblib.delayed(process)(
            all_2d_segs, imagecols, p3d_reader, fitting_config, img_id
        )
        for img_id in tqdm(image_ids)
    )
    output = {}
    for idx, seg3d_list_idx in enumerate(seg3d_list):
        output[image_ids[idx]] = seg3d_list_idx
    return output


def line_fitnmerge(cfg, imagecols, depths, neighbors=None, ranges=None):
    """
    Line reconstruction over multi-view RGB images given depths

    Args:
        cfg (dict): Configuration. \
            Fields refer to :file:`cfgs/fitnmerge/default.yaml` as an example
        imagecols (:class:`limap.base.ImageCollection`): \
            The image collection corresponding to all the images of interest
        depths (dict[int -> :class:`CustomizedDepthReader`], \
        where :class:`CustomizedDepthReader` inherits \
        :class:`limap.base.depth_reader_base.BaseDepthReader`): \
            The depth map readers for each image
        neighbors (dict[int -> list[int]], optional): \
            visual neighbors for each image. By default we compute neighbor \
            information from the covisibility of COLMAP triangulation.
        ranges (pair of :class:`np.array` each of shape (3,), optional): \
            robust 3D ranges for the scene. By default we compute range \
            information from the COLMAP triangulation.
    Returns:
        list[:class:`limap.base.LineTrack`]: list of output 3D line tracks
    """
    # assertion check
    assert imagecols.IsUndistorted()
    logging.info(f"[LOG] Number of images: {imagecols.NumImages()}")
    cfg = runners.setup(cfg)
    detector_name = cfg["line2d"]["detector"]["method"]
    if cfg["fitting"]["var2d"] == -1:
        cfg["fitting"]["var2d"] = cfg["var2d"][detector_name]
    if cfg["merging"]["var2d"] == -1:
        cfg["merging"]["var2d"] = cfg["var2d"][detector_name]
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
    if neighbors is None:
        _, neighbors, ranges = runners.compute_sfminfos(cfg, imagecols)
    else:
        neighbors = imagecols.update_neighbors(neighbors)
        for img_id, _ in neighbors.items():
            neighbors[img_id] = neighbors[img_id][: cfg["n_neighbors"]]

    ##########################################################
    # [B] get 2D line segments for each image
    ##########################################################
    all_2d_segs, _ = runners.compute_2d_segs(
        cfg, imagecols, compute_descinfo=cfg["line2d"]["compute_descinfo"]
    )

    ##########################################################
    # [C] fit 3d segments
    ##########################################################
    fname_fit_segs = "{}_fit_segs.npy".format(
        cfg["line2d"]["detector"]["method"]
    )
    if (not cfg["load_fit"]) and (
        not (
            cfg["skip_exists"]
            and os.path.exists(os.path.join(cfg["dir_load"], fname_fit_segs))
        )
    ):
        seg3d_list = fit_3d_segs(all_2d_segs, imagecols, depths, cfg["fitting"])
        limapio.save_npy(
            os.path.join(cfg["dir_save"], fname_fit_segs), seg3d_list
        )
    else:
        seg3d_list = limapio.read_npy(
            os.path.join(cfg["dir_load"], fname_fit_segs)
        ).item()

    if "do_merging" in cfg["merging"] and not cfg["merging"]["do_merging"]:
        linetracks = []
        for img_id in all_2d_segs:
            for line_id, seg2d in enumerate(all_2d_segs[img_id]):
                seg3d = seg3d_list[img_id][line_id]
                l3d = base.Line3d(seg3d[0], seg3d[1])
                l2d = base.Line2d(seg2d[0:2], seg2d[2:4])
                if l3d.length() == 0:
                    continue
                track = base.LineTrack(l3d, [img_id], [line_id], [l2d])
                linetracks.append(track)
        return linetracks

    ##########################################################
    # [D] merge 3d segments
    ##########################################################
    linker = base.LineLinker(
        cfg["merging"]["linker2d"], cfg["merging"]["linker3d"]
    )
    graph, linetracks = merging.merging(
        linker,
        all_2d_segs,
        imagecols,
        seg3d_list,
        neighbors,
        var2d=cfg["merging"]["var2d"],
    )
    linetracks = merging.filter_tracks_by_reprojection(
        linetracks,
        imagecols,
        cfg["filtering2d"]["th_angular_2d"],
        cfg["filtering2d"]["th_perp_2d"],
        num_outliers=0,
    )
    if not cfg["remerging"]["disable"]:
        linker3d_remerge = base.LineLinker3d(cfg["remerging"]["linker3d"])
        linetracks = merging.remerge(
            linker3d_remerge, linetracks, num_outliers=0
        )
        linetracks = merging.filter_tracks_by_reprojection(
            linetracks,
            imagecols,
            cfg["filtering2d"]["th_angular_2d"],
            cfg["filtering2d"]["th_perp_2d"],
            num_outliers=0,
        )

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

    ### Filter out 0-length 3D lines
    linetracks = [track for track in linetracks if track.line.length() > 0]

    ##########################################################
    # [F] output and visualization
    ##########################################################
    # save tracks
    if "output_folder" not in cfg or cfg["output_folder"] is None:
        cfg["output_folder"] = "fitnmerge_finaltracks"
    limapio.save_folder_linetracks_with_info(
        os.path.join(cfg["dir_save"], cfg["output_folder"]),
        linetracks,
        config=cfg,
        imagecols=imagecols,
        all_2d_segs=all_2d_segs,
    )
    limapio.save_txt_linetracks(
        os.path.join(cfg["dir_save"], "fitnmerge_alltracks.txt"),
        linetracks,
        n_visible_views=4,
    )
    VisTrack = limapvis.Open3DTrackVisualizer(linetracks)
    VisTrack.report()
    limapio.save_obj(
        os.path.join(
            cfg["dir_save"],
            "fitnmerge_lines_nv{}.obj".format(cfg["n_visible_views"]),
        ),
        VisTrack.get_lines_np(n_visible_views=cfg["n_visible_views"]),
    )

    if cfg["visualize"]:
        import pdb

        pdb.set_trace()
        VisTrack.vis_reconstruction(
            imagecols, n_visible_views=cfg["n_visible_views"]
        )
        pdb.set_trace()
    return linetracks


def line_fitting_with_points3d(
    cfg, imagecols, p3d_readers, inloc_read_transformations=False
):
    """
    Line reconstruction over multi-view images with its point cloud

    Args:
        cfg (dict): Configuration. \
            Fields refer to :file:`cfgs/fitnmerge/default.yaml` as an example
        imagecols (:class:`limap.base.ImageCollection`): \
            The image collection corresponding to all the images of interest
        p3d_reader (dict[int -> :class:`CustomizedP3DReader`], \
        where :class:`CustomizedP3DReader` inherits \
        :class:`limap.base.p3d_reader_base.BaseP3DReader`): \
            The point cloud readers for each image
        neighbors (dict[int -> list[int]], optional): \
            visual neighbors for each image. By default \
            we compute neighbor information from \
            the covisibility of COLMAP triangulation.
        ranges (pair of :class:`np.array` each of shape (3,), optional): \
            robust 3D ranges for the scene. By default we compute range \
            information from the COLMAP triangulation.
    Returns:
        list[:class:`limap.base.LineTrack`]: list of output 3D line tracks
    """
    # assertion check
    assert imagecols.IsUndistorted()
    logging.info(f"[LOG] Number of images: {imagecols.NumImages()}")
    cfg = runners.setup(cfg)
    detector_name = cfg["line2d"]["detector"]["method"]
    if cfg["fitting"]["var2d"] == -1:
        cfg["fitting"]["var2d"] = cfg["var2d"][detector_name]
    if cfg["merging"]["var2d"] == -1:
        cfg["merging"]["var2d"] = cfg["var2d"][detector_name]
    limapio.save_txt_imname_dict(
        os.path.join(cfg["dir_save"], "image_list.txt"),
        imagecols.get_image_name_dict(),
    )
    limapio.save_npy(
        os.path.join(cfg["dir_save"], "imagecols.npy"), imagecols.as_dict()
    )

    ##########################################################
    # [A] get 2D line segments for each image
    ##########################################################
    all_2d_segs, _ = runners.compute_2d_segs(
        cfg, imagecols, compute_descinfo=cfg["line2d"]["compute_descinfo"]
    )

    ##########################################################
    # [B] fit 3d segments
    ##########################################################
    fname_fit_segs = "{}_fit_segs.npy".format(
        cfg["line2d"]["detector"]["method"]
    )
    if (not cfg["load_fit"]) and (
        not (
            cfg["skip_exists"]
            and os.path.exists(os.path.join(cfg["dir_load"], fname_fit_segs))
        )
    ):
        if inloc_read_transformations:
            inloc_dataset = cfg["inloc_dataset"]
        else:
            inloc_dataset = None
        seg3d_list = fit_3d_segs_with_points3d(
            all_2d_segs, imagecols, p3d_readers, cfg["fitting"], inloc_dataset
        )
        limapio.save_npy(
            os.path.join(cfg["dir_save"], fname_fit_segs), seg3d_list
        )
    else:
        seg3d_list = limapio.read_npy(
            os.path.join(cfg["dir_load"], fname_fit_segs)
        ).item()

    linetracks = []
    for img_id in all_2d_segs:
        for line_id, seg2d in enumerate(all_2d_segs[img_id]):
            seg3d = seg3d_list[img_id][line_id]
            l3d = base.Line3d(seg3d[0], seg3d[1])
            l2d = base.Line2d(seg2d[0:2], seg2d[2:4])
            if l3d.length() == 0:
                continue
            track = base.LineTrack(l3d, [img_id], [line_id], [l2d])
            linetracks.append(track)
    return linetracks
