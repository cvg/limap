import numpy as np
from _limap import _estimators, _fitting
from bresenham import bresenham
from hloc.localize_inloc import interpolate_scan


def estimate_seg3d(points, ransac_th=0.75, min_percentage_inliers=0.6):
    options = _estimators.LORansacOptions()
    options.squared_inlier_threshold_ = ransac_th * ransac_th
    result = _fitting.Fit3DPoints(points, options)
    line, stats = result[0], result[1]
    if stats.inlier_ratio < min_percentage_inliers:
        return None
    else:
        start_3dpt, end_3dpt = line.start, line.end
        return start_3dpt, end_3dpt


def estimate_seg3d_from_depth(
    seg2d, depth, camview, ransac_th=0.75, min_percentage_inliers=0.6, var2d=5.0
):
    K, R, T = camview.K(), camview.R(), camview.T()
    h, w = camview.h(), camview.w()

    # get points and depths
    seg2d = seg2d.astype(int)
    seg1_pts = np.array(
        list(bresenham(seg2d[0], seg2d[1], seg2d[2], seg2d[3]))
    ).T
    valid_pxs = (
        (seg1_pts[0] >= 0)
        & (seg1_pts[1] >= 0)
        & (seg1_pts[0] < w)
        & (seg1_pts[1] < h)
    )
    seg1_pts = seg1_pts[:, valid_pxs]
    seg1_depths = depth[seg1_pts[1], seg1_pts[0]]
    valid_mask = ~np.isinf(seg1_depths)
    seg1_pts, seg1_depths = seg1_pts[:, valid_mask], seg1_depths[valid_mask]
    seg1_pts_homo = np.vstack([seg1_pts, np.ones((1, seg1_pts.shape[1]))])
    unprojected = np.linalg.inv(K) @ seg1_pts_homo
    unprojected = unprojected * seg1_depths
    points = (R.T @ unprojected) - (R.T @ T)[:, None].repeat(
        unprojected.shape[1], 1
    )
    if points.shape[1] <= 6:
        return None

    # fitting
    uncertainty = var2d * np.median(seg1_depths) / ((K[0, 0] + K[1, 1]) / 2.0)
    ransac_th = ransac_th * uncertainty
    return estimate_seg3d(points, ransac_th, min_percentage_inliers)


def estimate_seg3d_from_points3d(
    seg2d,
    p3ds,
    camview,
    image_name,
    inloc_dataset=None,
    ransac_th=0.75,
    min_percentage_inliers=0.6,
    var2d=5.0,
):
    h, w = camview.h(), camview.w()
    R, T = camview.R(), camview.T()

    # get points and depths
    seg1_pts = np.linspace(
        seg2d[0:2], seg2d[2:4], int(np.linalg.norm(seg2d[2:4] - seg2d[0:2]) * 2)
    ).T
    valid_pxs = (
        (seg1_pts[0] > 0)
        & (seg1_pts[1] > 0)
        & (seg1_pts[0] < w - 1)
        & (seg1_pts[1] < h - 1)
    )
    seg1_pts = seg1_pts[:, valid_pxs].T
    seg1_p3ds, valid = interpolate_scan(p3ds, seg1_pts)
    seg1_p3ds = seg1_p3ds[valid]
    seg1_ray_depths = np.linalg.norm(seg1_p3ds, axis=1)

    if inloc_dataset is not None:
        from hloc.localize_inloc import get_scan_pose

        Tr = get_scan_pose(inloc_dataset, image_name)
        points = Tr[:3, :3] @ seg1_p3ds.T + Tr[:3, -1:]
    else:
        points = (R.T @ seg1_p3ds.T) - (R.T @ T)[:, None].repeat(
            seg1_p3ds.T.shape[1], 1
        )

    if points.shape[1] <= 6:
        return None

    # TODO: test the best way to estimate uncertainty here
    uncertainty = var2d * np.median(seg1_ray_depths) / (0.7 * max(h, w))
    ransac_th = ransac_th * uncertainty

    # fitting
    return estimate_seg3d(points, ransac_th, min_percentage_inliers)
