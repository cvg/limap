import numpy as np
from bresenham import bresenham
import logging

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from LinearModelEstimator import LinearModelEstimator
from RANSACRegressor3D import RANSACRegressor3D
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.geometry import to_homogeneous_t

def fit_3d_seg(pts, ransac_th=0.015, min_percentage_inliers=0.6):
    # logging.getLogger().setLevel(logging.DEBUG)
    try:
        # Robustly estimate the line
        ransac = RANSACRegressor3D(LinearModelEstimator(), residual_threshold=ransac_th).fit(pts)
    except ValueError as err:
        logging.warning("RANSAC error: {0}".format(err))
        return None

    # Discard bad solutions by the number of bad points
    n_inliers = np.sum(ransac.inlier_mask_)

    # Discard bad solutions by the percentage of bad points
    percentage_of_inliers = n_inliers / float(len(pts))
    if percentage_of_inliers < min_percentage_inliers:
        logging.debug("-->\t Not enough inliers found: Minimum percentage is: "
                      + str(min_percentage_inliers) + ", and current is: " + str(percentage_of_inliers))
        return None

    line = ransac.estimator_.line

    projections = (pts[ransac.inlier_mask_] - line.ptn).dot(line.vec)
    start_3dpt = line.ptn + line.vec * projections.min()
    end_3dpt = line.ptn + line.vec * projections.max()

    if np.linalg.norm(pts[0] - start_3dpt) > np.linalg.norm(pts[0] - end_3dpt):
        start_3dpt, end_3dpt = end_3dpt, start_3dpt

    return start_3dpt, end_3dpt, ransac.inlier_mask_

def estimate_seg3d_from_depth(seg2d, depth, cam, hw, ransac_th=0.75, min_percentage_inliers=0.6, var2d=5.0):
    K, R, T = cam[0], cam[1], cam[2]
    h, w = hw[0], hw[1]

    seg2d = seg2d.astype(int)
    seg1_pts = np.array(list(bresenham(seg2d[0], seg2d[1], seg2d[2], seg2d[3]))).T
    valid_pxs = (seg1_pts[0] >= 0) & (seg1_pts[1] >= 0) & (seg1_pts[0] < w) & (seg1_pts[1] < h)
    seg1_pts = seg1_pts[:, valid_pxs]
    seg1_depths = depth[seg1_pts[1], seg1_pts[0]]
    valid_mask = ~np.isinf(seg1_depths)
    seg1_pts, seg1_depths = seg1_pts[:, valid_mask], seg1_depths[valid_mask]
    uncertainty = var2d * np.median(seg1_depths) / ((K[0, 0] + K[1, 1]) / 2.0)
    ransac_th = ransac_th * uncertainty
    unprojected = np.linalg.inv(K) @ to_homogeneous_t(seg1_pts)
    unprojected = unprojected * seg1_depths
    segment_3d_pts = (R.T @ unprojected) - (R.T @ T)[:, None].repeat(unprojected.shape[1], 1)
    segment_3d_pts = segment_3d_pts.T
    if segment_3d_pts.shape[0] <= 2:
        return None
    result = fit_3d_seg(segment_3d_pts, ransac_th=ransac_th, min_percentage_inliers=min_percentage_inliers)
    if result is None:
        return None
    else:
        start_3dpt, end_3dpt, inlier_mask = result
        return start_3dpt, end_3dpt

