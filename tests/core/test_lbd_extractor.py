import numpy as np
import pytest

from limap.image.line.LBD import extractor as lbd


@pytest.mark.ci_workflow
def test_lbd_multiscale_lines_roundtrip(tmp_path, monkeypatch):
    segments = np.array([[0.0, 0.0, 10.0, 0.0], [2.0, 3.0, 2.0, 12.0]])
    multiscale_lines = lbd.to_multiscale_lines(segments)
    descriptors = np.zeros((len(segments), 72))

    monkeypatch.setattr(
        lbd, "process_pyramid", lambda *args, **kwargs: ([], [])
    )

    def describe(pyramid, lines, *args):
        assert isinstance(lines, list)
        assert len(lines) == len(multiscale_lines)
        return descriptors

    monkeypatch.setattr(lbd.pytlbd, "lbd_multiscale_pyr", describe)
    extractor = lbd.LBDExtractor()
    descinfo = extractor.compute_descinfo(np.zeros((16, 16)), segments)
    extractor.save_descinfo(tmp_path, 0, descinfo)

    with extractor.read_descinfo(tmp_path, 0) as loaded:
        assert loaded["ms_lines"].shape == (len(segments),)
        assert loaded["ms_lines"].dtype == object
        for saved, original in zip(
            loaded["ms_lines"].tolist(), multiscale_lines, strict=True
        ):
            for (saved_scale, saved_line), (
                original_scale,
                original_line,
            ) in zip(saved, original, strict=True):
                assert saved_scale == original_scale
                np.testing.assert_array_equal(saved_line, original_line)
        np.testing.assert_array_equal(loaded["line_descriptors"], descriptors)
