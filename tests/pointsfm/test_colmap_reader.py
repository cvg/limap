import numpy.testing as npt
import pycolmap
import pytest

from limap.pointsfm.colmap_reader import ReadInfos, ReadPointTracks


@pytest.mark.ci_workflow
def test_convert_colmap_to_imagecols(tmp_path):
    syn_options = pycolmap.SyntheticDatasetOptions()
    recon = pycolmap.synthesize_dataset(syn_options)
    recon.write(tmp_path)
    imagecols = ReadInfos(tmp_path, model_path=".")
    assert imagecols.NumCameras() == recon.num_cameras()
    assert imagecols.NumImages() == recon.num_images()
    npt.assert_allclose(
        recon.images[1].cam_from_world().inverse().translation,
        imagecols.camimage(1).pose.center(),
        rtol=1e-5,
        atol=1e-8,
    )


@pytest.mark.ci_workflow
def test_read_point_tracks(tmp_path):
    syn_options = pycolmap.SyntheticDatasetOptions()
    recon = pycolmap.synthesize_dataset(syn_options)
    point_tracks = ReadPointTracks(recon)
    assert len(point_tracks) == recon.num_points3D()
