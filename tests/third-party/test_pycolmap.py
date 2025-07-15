import pycolmap
import pytest


@pytest.mark.ci_workflow
def test_pycolmap():
    syn_options = pycolmap.SyntheticDatasetOptions(
        num_rigs=1,
        num_cameras_per_rig=3,
        num_frames_per_rig=8,
        num_points3D=100,
    )
    recon = pycolmap.synthesize_dataset(syn_options)
    assert recon.num_cameras() == 3
    assert recon.num_images() == 24
    assert recon.num_points3D() == 100
