import numpy as np
import pycolmap
from bresenham import bresenham

from limap.estimators import PoseLibRansacOptions
from limap.estimators.line3d import estimate_line3d_robust
from limap.geometry import Line3d


def estimate_seg3d(
    points: np.ndarray, ransac_th: float = 0.75, min_inlier_ratio: float = 0.6
) -> Line3d | None:
    """Fit a 3D line segment to a set of 3D points using RANSAC.

    Args:
        points: 3D points as (3, N) array
        ransac_th: RANSAC inlier threshold
        min_inlier_ratio: Minimum inlier ratio for valid fit

    Returns:
        Line3d object, or None if fitting fails
    """
    options = PoseLibRansacOptions()
    # Convert (3, N) array to list of (3, 1) column vectors
    points_list = [points[:, i : i + 1] for i in range(points.shape[1])]
    line, stats = estimate_line3d_robust(points_list, ransac_th, options)
    if line is None or stats.inlier_ratio < min_inlier_ratio:
        return None
    return line


def estimate_seg3d_from_depth(
    seg2d: np.ndarray,
    depth: np.ndarray,
    image: pycolmap.Image,
    ransac_th: float = 0.75,
    min_inlier_ratio: float = 0.6,
    var2d: float = 5.0,
) -> Line3d | None:
    """Estimate 3D segment from 2D segment and depth map.

    Args:
        seg2d: 2D segment endpoints (x1, y1, x2, y2)
        depth: Depth map (H, W)
        image: pycolmap.Image with camera and pose
        ransac_th: RANSAC threshold multiplier
        min_inlier_ratio: Minimum inlier ratio for valid fit
        var2d: 2D variance for uncertainty estimation

    Returns:
        Line3d object, or None if fitting fails
    """
    K = image.camera.calibration_matrix()
    R = image.cam_from_world().rotation.matrix()
    T = image.cam_from_world().translation
    h, w = image.camera.height, image.camera.width

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
    return estimate_seg3d(points, ransac_th, min_inlier_ratio)


def estimate_seg3d_from_points3d(
    seg2d: np.ndarray,
    p3ds: np.ndarray,
    image: pycolmap.Image,
    ransac_th: float = 0.75,
    min_inlier_ratio: float = 0.6,
    var2d: float = 5.0,
) -> Line3d | None:
    """Estimate 3D segment from 2D segment and point cloud.

    Args:
        seg2d: 2D segment endpoints (x1, y1, x2, y2)
        p3ds: Point cloud in world coordinates (H, W, 3)
        image: pycolmap.Image with camera and pose
        ransac_th: RANSAC threshold multiplier
        min_inlier_ratio: Minimum inlier ratio for valid fit
        var2d: 2D variance for uncertainty estimation

    Returns:
        Line3d object, or None if fitting fails
    """
    h, w = image.camera.height, image.camera.width

    # Sample points along the 2D segment
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

    # Interpolate 3D points from point cloud (already in world coordinates).
    # hloc is an optional dependency (limap[line2d]); import it lazily so
    # that `import limap.estimators` does not require it.
    from hloc.localize_inloc import interpolate_scan

    seg1_p3ds, valid = interpolate_scan(p3ds, seg1_pts)
    points = seg1_p3ds[valid].T  # (3, N)
    seg1_ray_depths = np.linalg.norm(seg1_p3ds[valid], axis=1)

    if points.shape[1] <= 6:
        return None

    # Estimate uncertainty based on ray depths
    uncertainty = var2d * np.median(seg1_ray_depths) / (0.7 * max(h, w))
    ransac_th = ransac_th * uncertainty

    return estimate_seg3d(points, ransac_th, min_inlier_ratio)
