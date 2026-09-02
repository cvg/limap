from __future__ import annotations

import numpy as np
import pycolmap

import limap.estimators as _estimators
from limap.geometry import Line2d, Line3d


def estimate_absolute_pose(
    l3ds: list[Line3d],
    l2ds: list[Line2d],
    p3ds: list[np.ndarray],
    p2ds: list[np.ndarray],
    camera: pycolmap.Camera,
    options: _estimators.PointLineAbsolutePoseOptions | None = None,
) -> _estimators.PointLineAbsolutePoseResult:
    """
    Estimate absolute camera pose from point and line correspondences.

    Uses PoseLib's hybrid RANSAC with adaptive solver selection
    (P3P, P2P1LL, P1P2LL, P3LL).

    Args:
        l3ds: Matched 3D line segments (same size as l2ds)
        l2ds: Matched 2D lines (same size as l3ds)
        p3ds: Matched 3D points (same size as p2ds)
        p2ds: Matched 2D points (same size as p3ds)
        camera: Camera intrinsics
        options: Estimation options (uses defaults if None)

    Returns:
        PointLineAbsolutePoseResult containing pose and RANSAC statistics
    """
    if options is None:
        options = _estimators.PointLineAbsolutePoseOptions()

    return _estimators.estimate_point_line_absolute_pose(
        l3ds, l2ds, p3ds, p2ds, camera, options
    )
