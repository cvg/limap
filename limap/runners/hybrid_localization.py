import os
from collections import defaultdict

import numpy as np
from hloc.utils.io import get_keypoints, get_matches
from pycolmap import logging
from tqdm import tqdm

import limap.base as base
import limap.estimators as estimators
import limap.line2d
import limap.runners as runners
import limap.util.io as limapio
from limap.optimize.hybrid_localization.functions import (
    filter_line_2to2_epipolarIoU,
    get_reprojection_dist_func,
    match_line_2to2_epipolarIoU,
    match_line_2to3,
    reprojection_filter_matches_2to3,
)


def get_hloc_keypoints(
    ref_sfm,
    features_path,
    matches_path,
    query_img_name,
    target_img_ids,
    logger=None,
):
    if ref_sfm is None or features_path is None or matches_path is None:
        if logger:
            logger.debug(
                "Not retrieving keypoint correspondences because \
                 at least one parameter is not provided."
            )
        return np.array([]), np.array([])

    kpq = get_keypoints(features_path, query_img_name)
    kpq += 0.5  # COLMAP coordinates

    kp_idx_to_3D = defaultdict(list)
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    num_matches_point = 0

    for i, tgt_id in enumerate(target_img_ids):
        image = ref_sfm.images[tgt_id]
        if image.num_points3D() == 0 and logger:
            logger.debug(f"No 3D points found for {image.name}.")
            continue
        points3D_ids = np.array(
            [p.point3D_id if p.has_point3D() else -1 for p in image.points2D]
        )

        matches, _ = get_matches(matches_path, query_img_name, image.name)
        matches = matches[points3D_ids[matches[:, 1]] != -1]
        num_matches_point += len(matches)
        for idx, m in matches:
            id_3D = points3D_ids[m]
            kp_idx_to_3D_to_db[idx][id_3D].append(i)
            # avoid duplicate observations
            if id_3D not in kp_idx_to_3D[idx]:
                kp_idx_to_3D[idx].append(id_3D)

    idxs = list(kp_idx_to_3D.keys())
    p2ds = np.array([kpq[i] for i in idxs for j in kp_idx_to_3D[i]])
    p3ds = np.array(
        [ref_sfm.points3D[j].xyz for i in idxs for j in kp_idx_to_3D[i]]
    )

    return p2ds, p3ds


def get_hloc_keypoints_from_log(
    logs, query_img_name, ref_sfm=None, resize_scales=None
):
    if ref_sfm is None:  # inloc
        p2ds = logs["loc"][query_img_name]["keypoints_query"]
        p3ds = logs["loc"][query_img_name]["3d_points"]
    else:
        p2ds = logs["loc"][query_img_name]["keypoints_query"]
        p3d_ids = logs["loc"][query_img_name]["points3D_ids"]
        p3ds = [ref_sfm.points3D[j].xyz for j in p3d_ids]
    inliers = logs["loc"][query_img_name]["PnP_ret"]["inliers"]

    p2ds, p3ds = np.array(p2ds), np.array(p3ds)
    if resize_scales is not None and query_img_name in resize_scales:
        scale = resize_scales[query_img_name]
        p2ds = (p2ds + 0.5) * scale - 0.5

    return p2ds, p3ds, inliers


def hybrid_localization(
    cfg,
    imagecols_db,
    imagecols_query,
    point_corresp,
    linemap_db,
    retrieval,
    results_path,
    img_name_dict=None,
    logger=None,
):
    """
    Run visual localization on query images with `imagecols`, \
    it takes 2D-3D point correspondences from HLoc;
    runs line matching using 2D line matcher ("epipolar" for Gao et al. \
    "Pose Refinement with Joint Optimization of Visual Points and Lines");
    calls :func:`~limap.estimators.absolute_pose.pl_estimate_absolute_pose` \
    to estimate the absolute camera pose for all query images,
    and writes results in results file in `results_path`.

    Args:
        cfg (dict): Configuration, \
            fields refer to :file:`cfgs/localization/default.yaml`
        imagecols_db (:class:`limap.base.ImageCollection`): \
            Image collection of database images, with triangulated camera poses
        imagecols_query (:class:`limap.base.ImageCollection`): \
            Image collection of query images, camera poses only used for \
            epipolar matcher/filter as coarse poses, \
            can be left uninitialized otherwise
        linemap_db (list[:class:`limap.base.LineTrack`]): \
            LIMAP triangulated/fitted line tracks
        retrieval (dict): Mapping of query image file path to \
            list of neighbor image file paths, e.g. returned from \
            :func:`hloc.utils.parsers.parse_retrieval`
        results_path (str | Path): \
            File path to write the localization results
        img_name_dict(dict, optional): \
            Mapping of query image IDs to the image file path, by default \
            the image names from `imagecols`
        logger (:class:`logging.Logger`, optional): \
            Logger to print logs for information

    Returns:
        Dict[int -> :class:`limap.base.CameraPose`]: \
            Mapping of query image IDs to the localized camera poses \
            for all query images.
    """

    if cfg["localization"]["2d_matcher"] not in [
        "epipolar",
        "sold2",
        "superglue_endpoints",
        "gluestick",
        "linetr",
        "lbd",
        "l2d2",
    ]:
        raise ValueError(
            "Unknown 2d line matcher: {}".format(
                cfg["localization"]["2d_matcher"]
            )
        )

    train_ids = imagecols_db.get_img_ids()
    query_ids = imagecols_query.get_img_ids()

    if img_name_dict is None:
        img_name_dict = {
            img_id: imagecols_db.image_name(img_id) for img_id in train_ids
        }
        img_name_dict.update(
            {img_id: imagecols_query.image_name(img_id) for img_id in query_ids}
        )
    id_to_name = img_name_dict
    name_to_id = {
        img_name_dict[img_id]: img_id for img_id in train_ids + query_ids
    }

    # GT for queries
    poses_db = {
        img_id: imagecols_db.camimage(img_id).pose for img_id in train_ids
    }

    # line detection of query images, fetch detection of
    # db images (generally already be detected during triangulation)
    all_db_segs, _ = runners.compute_2d_segs(
        cfg, imagecols_db, compute_descinfo=False
    )
    all_query_segs, _ = runners.compute_2d_segs(
        cfg, imagecols_query, compute_descinfo=False
    )
    all_db_lines = base.get_all_lines_2d(all_db_segs)
    all_query_lines = base.get_all_lines_2d(all_query_segs)
    line2track = base.get_invert_idmap_from_linetracks(all_db_lines, linemap_db)

    # Do matches for query images and retrieved neighbors
    # for superglue endpoints matcher
    if cfg["localization"]["2d_matcher"] != "epipolar":
        weight_path = cfg.get("weight_path", None)
        if cfg["localization"]["2d_matcher"] == "superglue_endpoints":
            extractor_name = "superpoint_endpoints"
            matcher_name = "superglue_endpoints"
        else:
            extractor_name = matcher_name = cfg["localization"]["2d_matcher"]
        ex_cfg = {"method": extractor_name, "topk": 0, "n_jobs": cfg["n_jobs"]}
        ma_cfg = {
            "method": matcher_name,
            "topk": 0,
            "n_jobs": cfg["n_jobs"],
            "superglue": {
                "weights": cfg["line2d"]["matcher"]["superglue"]["weights"]
            },
        }
        basedir = os.path.join(
            "line_detections", cfg["line2d"]["detector"]["method"]
        )
        folder_save = os.path.join(cfg["dir_save"], basedir)
        extractor = limap.line2d.get_extractor(ex_cfg, weight_path=weight_path)
        se_descinfo_dir = extractor.extract_all_images(
            folder_save,
            imagecols_db,
            all_db_segs,
            skip_exists=cfg["skip_exists"],
        )
        se_descinfo_dir = extractor.extract_all_images(
            folder_save, imagecols_query, all_query_segs, skip_exists=True
        )

        basedir = os.path.join(
            "line_matchings",
            cfg["line2d"]["detector"]["method"],
            f"feats_{matcher_name}",
        )
        matcher = limap.line2d.get_matcher(
            ma_cfg,
            extractor,
            n_neighbors=cfg["n_neighbors_loc"],
            weight_path=weight_path,
        )
        folder_save = os.path.join(cfg["dir_save"], basedir)
        retrieved_neighbors = {
            qid: [name_to_id[n] for n in retrieval[id_to_name[qid]]]
            for qid in query_ids
        }
        se_matches_dir = matcher.match_all_neighbors(
            folder_save,
            query_ids,
            retrieved_neighbors,
            se_descinfo_dir,
            skip_exists=cfg["skip_exists"],
        )

    # Localization
    logging.info("[LOG] Starting localization with points+lines...")
    final_poses = {}
    pose_dir = results_path.parent / "poses_{}".format(
        cfg["localization"]["2d_matcher"]
    )
    for qid in tqdm(query_ids):
        if cfg["localization"]["skip_exists"]:
            limapio.check_makedirs(str(pose_dir))
            if os.path.exists(os.path.join(pose_dir, f"{qid}.txt")):
                with open(os.path.join(pose_dir, f"{qid}.txt")) as f:
                    data = f.read().rstrip().split("\n")[0].split()
                    q, t = np.split(np.array(data[1:], float), [4])
                    final_poses[qid] = base.CameraPose(q, t)
                    continue
        if logger:
            logger.info(f"Query Image ID: {qid}")

        query_lines = all_query_lines[qid]
        qname = id_to_name[qid]
        query_pose = imagecols_query.get_camera_pose(qid)
        query_cam = imagecols_query.cam(imagecols_query.camimage(qid).cam_id)
        query_camview = base.CameraView(query_cam, query_pose)
        targets = retrieval[qname]

        if cfg["localization"]["2d_matcher"] != "epipolar":
            # Read from the pre-computed matches
            all_line_pairs_2to2 = limapio.read_npy(
                os.path.join(se_matches_dir, f"matches_{qid}.npy")
            ).item()

        all_line_pairs_2to3 = defaultdict(list)
        for tgt_img_name in targets:
            tgt_id = name_to_id[tgt_img_name]
            tgt_lines = all_db_lines[tgt_id]
            tgt_pose = poses_db[tgt_id]
            tgt_cam = imagecols_db.cam(imagecols_db.camimage(tgt_id).cam_id)

            if cfg["localization"]["2d_matcher"] == "epipolar":
                line_pairs_2to2 = match_line_2to2_epipolarIoU(
                    query_lines,
                    tgt_lines,
                    query_cam,
                    query_pose,
                    tgt_cam,
                    tgt_pose,
                    cfg["localization"]["IoU_threshold"],
                )
                line_pairs_2to3 = match_line_2to3(
                    line_pairs_2to2, line2track, tgt_id
                )
                for pair in line_pairs_2to3:
                    query_line_id, track_id = pair
                    all_line_pairs_2to3[query_line_id].append(track_id)
            else:
                # Optionally filter matching results based on epipolar IoU
                if cfg["localization"]["epipolar_filter"]:
                    filtered_line_pairs_2to2 = filter_line_2to2_epipolarIoU(
                        all_line_pairs_2to2[tgt_id],
                        query_lines,
                        tgt_lines,
                        query_cam,
                        query_pose,
                        tgt_cam,
                        tgt_pose,
                        cfg["localization"]["IoU_threshold"],
                    )
                else:
                    filtered_line_pairs_2to2 = all_line_pairs_2to2[tgt_id]
                line_pairs_2to3 = match_line_2to3(
                    filtered_line_pairs_2to2, line2track, tgt_id
                )
                for pair in line_pairs_2to3:
                    query_line_id, track_id = pair
                    all_line_pairs_2to3[query_line_id].append(track_id)

        # filter based on reprojection distance (to 1-1 correspondences),
        # mainly for "OPPO method"
        if cfg["localization"]["reprojection_filter"] is not None:
            line_matches_2to3 = reprojection_filter_matches_2to3(
                query_lines,
                query_camview,
                all_line_pairs_2to3,
                linemap_db,
                dist_thres=2,
                dist_func=get_reprojection_dist_func(
                    cfg["localization"]["reprojection_filter"]
                ),
            )
        else:
            line_matches_2to3 = [
                (x, y)
                for x in all_line_pairs_2to3
                for y in all_line_pairs_2to3[x]
            ]

        num_matches_line = len(line_matches_2to3)
        if logger:
            logger.info(
                f"{num_matches_line} line matches found \
                  for {len(query_lines)} 2D lines"
            )

        l3ds = [track.line for track in linemap_db]
        l2ds = [query_lines[pair[0]] for pair in line_matches_2to3]
        l3d_ids = [pair[1] for pair in line_matches_2to3]

        p3ds, p2ds = point_corresp[qid]["p3ds"], point_corresp[qid]["p2ds"]
        inliers_point = point_corresp[qid].get("inliers")  # default None
        final_pose, ransac_stats = estimators.pl_estimate_absolute_pose(
            cfg["localization"],
            l3ds,
            l3d_ids,
            l2ds,
            p3ds,
            p2ds,
            query_cam,
            query_pose,  # query_pose not used for ransac methods
            inliers_point=inliers_point,
            silent=True,
            logger=logger,
        )

        if cfg["localization"]["skip_exists"]:
            with open(os.path.join(pose_dir, f"{qid}.txt"), "w") as f:
                name = id_to_name[qid]
                fq, ft = final_pose.qvec, final_pose.tvec
                line = (
                    " ".join(
                        [name] + [str(x) for x in fq] + [str(x) for x in ft]
                    )
                    + "\n"
                )
                f.writelines([line])

        final_poses[qid] = final_pose

    lines = []
    for qid in query_ids:
        name = id_to_name[qid]
        fpose = final_poses[qid]
        fq, ft = fpose.qvec, fpose.tvec
        line = (
            " ".join([name] + [str(x) for x in fq] + [str(x) for x in ft])
            + "\n"
        )
        lines.append(line)

    # write results
    with open(results_path, "w") as f:
        f.writelines(lines)

    return final_poses
