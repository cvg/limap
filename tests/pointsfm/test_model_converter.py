import pytest
import pycolmap
from limap.pointsfm.model_converter import convert_colmap_to_visualsfm
from limap.pointsfm.visualsfm_reader import ReadModelVisualSfM
import numpy.testing as npt

@pytest.mark.ci_workflow
def test_convert_colmap_to_visualsfm(tmp_path):
    syn_options = pycolmap.SyntheticDatasetOptions()
    recon = pycolmap.synthesize_dataset(syn_options)
    recon.write(tmp_path)
    output_nvm_file = tmp_path / "test_output.nvm"
    convert_colmap_to_visualsfm(tmp_path, output_nvm_file)

