import numpy as np
import pycolmap
import pytest

# open3d publishes no wheels for Python 3.13+, where the viewers are unusable.
pytest.importorskip("open3d")

from limap.visualize.viz3d import open3d_get_camera_frustums  # noqa: E402


@pytest.mark.ci_workflow
def test_open3d_camera_frustums():
    """Guard the open3d and pycolmap APIs the 3D viewer depends on.

    Both are pinned, so this only fires when a pin is bumped -- which is what
    it is for: open3d has broken these interfaces across versions before.
    """
    camera = pycolmap.Camera(
        camera_id=1,
        model="PINHOLE",
        width=500,
        height=500,
        params=[500.0, 500.0, 250.0, 250.0],
    )
    rig = pycolmap.Rig()
    rig.rig_id = 1
    rig.add_ref_sensor(camera.sensor_id)
    frame = pycolmap.Frame()
    frame.frame_id = 1
    frame.rig_id = 1
    frame.rig_from_world = pycolmap.Rigid3d()
    image = pycolmap.Image(image_id=1, camera_id=1, name="image.png")
    image.frame_id = 1
    frame.add_data_id(image.data_id)

    recon = pycolmap.Reconstruction()
    recon.add_camera(camera)
    recon.add_rig(rig)
    recon.add_frame(frame)
    recon.add_image(image)

    frustums = open3d_get_camera_frustums(recon)
    assert len(np.asarray(frustums.points)) > 0
    assert len(np.asarray(frustums.lines)) > 0
