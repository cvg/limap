from limap._limap import _fitting
from limap._limap._fitting import *  # noqa: F403

from .fitting import (
    estimate_seg3d,
    estimate_seg3d_from_depth,
    estimate_seg3d_from_points3d,
)

__all__ = [n for n in _fitting.__dict__ if not n.startswith("_")] + [
    "estimate_seg3d",
    "estimate_seg3d_from_depth",
    "estimate_seg3d_from_points3d",
]
