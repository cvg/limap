from _limap import _base
from _limap import _fitting as _fit

import os, sys
import numpy as np
from bresenham import bresenham

def estimate_seg3d_from_depth(seg2d, depth, camview, ransac_th=0.75, min_percentage_inliers=0.6, var2d=5.0):
    K, R, T = camview.K(), camview.R(), camview.T()
    h, w = camview.h(), camview.w()

    # get points and depths
    seg2d = seg2d.astype(int)
    seg1_pts = np.array(list(bresenham(seg2d[0], seg2d[1], seg2d[2], seg2d[3]))).T
    valid_pxs = (seg1_pts[0] >= 0) & (seg1_pts[1] >= 0) & (seg1_pts[0] < w) & (seg1_pts[1] < h)
    seg1_pts = seg1_pts[:, valid_pxs]
    seg1_depths = depth[seg1_pts[1], seg1_pts[0]]
    valid_mask = ~np.isinf(seg1_depths)
    seg1_pts, seg1_depths = seg1_pts[:, valid_mask], seg1_depths[valid_mask]
    seg1_pts_homo = np.vstack([seg1_pts, np.ones((1, seg1_pts.shape[1]))])
    unprojected = np.linalg.inv(K) @ seg1_pts_homo
    unprojected = unprojected * seg1_depths
    points = (R.T @ unprojected) - (R.T @ T)[:, None].repeat(unprojected.shape[1], 1)
    if points.shape[1] <= 6:
        return None

    # fitting
    uncertainty = var2d * np.median(seg1_depths) / ((K[0, 0] + K[1, 1]) / 2.0)
    ransac_th = ransac_th * uncertainty
    options = _fit.LORansacOptions()
    options.squared_inlier_threshold_ = ransac_th * ransac_th
    result = _fit.Fit3DPoints(points, options)
    line, stats = result[0], result[1]
    if stats.inlier_ratio < min_percentage_inliers:
        return None
    else:
        start_3dpt, end_3dpt = line.start, line.end
        return start_3dpt, end_3dpt

