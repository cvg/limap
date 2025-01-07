import numpy as np
import open3d as o3d
import pytest


@pytest.mark.ci_workflow
def test_open3d():
    o3d.geometry.LineSet.create_camera_visualization(
        500,
        500,
        np.array([[500.0, 0.0, 250.0], [0.0, 500.0, 250.0], [0.0, 0.0, 1.0]]),
        np.eye(4),
        scale=1.0,
    )
