"""Tests for the UPAL joint point-line detector.

The registry-level tests are pure Python and run in CI. The ones that build the
network need torch, the ``upal`` package and the ``points_lsd`` extension, none
of which are installed on every supported Python, so they skip when absent.
"""

import numpy as np
import pytest

from limap.image.line import (
    DetectorOptions,
    ExtractorOptions,
    MatcherOptions,
    get_uncertainty2d,
)
from limap.image.line.register_detector import UPALOptions


@pytest.mark.ci_workflow
def test_upal_uncertainty_registered():
    assert get_uncertainty2d("upal") == 2.0


@pytest.mark.ci_workflow
def test_upal_options_reachable_from_config():
    # Both registries must carry the options, since UPAL is dual-functional.
    assert DetectorOptions().upal_options == UPALOptions()
    assert ExtractorOptions().upal_options == UPALOptions()


@pytest.mark.ci_workflow
def test_upal_keypoint_budget_is_a_hard_count():
    # Not a cap: UPAL is top-k with no threshold, so this many keypoints come
    # back from every image. Changing it changes point density directly.
    assert UPALOptions().max_num_keypoints == 4096


@pytest.fixture(scope="module")
def detector():
    pytest.importorskip("torch")
    pytest.importorskip("upal")
    pytest.importorskip("points_lsd")
    from limap.image.line import get_detector

    return get_detector("upal", DetectorOptions())


@pytest.fixture(scope="module")
def image_pair(tmp_path_factory):
    """Two views of a wireframe box, offset so the lines are matchable."""
    import cv2

    paths = []
    tmp = tmp_path_factory.mktemp("upal")
    for i, shift in enumerate((0, 12)):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (100 + shift, 80), (520 + shift, 400), (255,) * 3, 2)
        cv2.line(img, (100 + shift, 80), (520 + shift, 400), (255,) * 3, 2)
        cv2.line(img, (300 + shift, 40), (300 + shift, 440), (255,) * 3, 2)
        path = tmp / f"view_{i}.png"
        cv2.imwrite(str(path), img)
        paths.append(path)
    return paths


def test_detect_and_extract_shapes(detector, image_pair):
    segs, descinfo = detector.detect_and_extract(image_pair[0])

    assert segs.ndim == 2 and segs.shape[1] == 5
    assert len(segs) > 0, "no segments on a synthetic wireframe"
    assert np.isfinite(segs).all()

    n_lines = len(segs)
    assert descinfo["lines"].shape == (2 * n_lines, 2)
    assert descinfo["lines_score"].shape == (n_lines,)
    assert descinfo["endpoints_desc"].shape == (128, 2 * n_lines)
    # Endpoint descriptors come out of the network L2-normalized.
    norms = np.linalg.norm(descinfo["endpoints_desc"], axis=0)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-4, atol=1e-4)


def test_detect_matches_detect_and_extract(detector, image_pair):
    segs = detector.detect(image_pair[0])
    segs_joint, _ = detector.detect_and_extract(image_pair[0])
    np.testing.assert_array_equal(segs, segs_joint)


def test_single_pass_descriptors_match_upstream_api(detector, image_pair):
    """The cached feature map must describe endpoints identically to upstream.

    ``detect_and_extract`` reuses one encoder pass where
    ``UPAL.describe_keypoints`` re-encodes; the descriptors must not drift.
    """
    import torch

    segs, descinfo = detector.detect_and_extract(image_pair[0])
    endpoints = segs[:, :4].reshape(1, -1, 2)

    _, image = detector._load_image(image_pair[0])
    with torch.no_grad():
        reference = detector.net.describe_keypoints(
            image, torch.from_numpy(endpoints).float().to(detector.device)
        )[0]
    np.testing.assert_allclose(
        descinfo["endpoints_desc"],
        reference.t().cpu().numpy(),
        rtol=1e-4,
        atol=1e-5,
    )


def test_descinfo_roundtrip(detector, image_pair, tmp_path):
    _, descinfo = detector.detect_and_extract(image_pair[0])
    detector.save_descinfo(tmp_path, 0, descinfo)
    loaded = detector.read_descinfo(tmp_path, 0)
    np.testing.assert_allclose(
        loaded["endpoints_desc"], descinfo["endpoints_desc"]
    )
    np.testing.assert_allclose(loaded["lines"], descinfo["lines"])


def test_sample_descinfo_by_indexes(detector, image_pair):
    _, descinfo = detector.detect_and_extract(image_pair[0])
    if len(descinfo["lines_score"]) < 3:
        pytest.skip("too few segments to subsample")

    indexes = [0, 2]
    sub = detector.sample_descinfo_by_indexes(descinfo, indexes)
    assert sub["lines"].shape == (4, 2)
    assert sub["endpoints_desc"].shape == (128, 4)
    # Endpoint pairs must stay with their own line.
    np.testing.assert_allclose(sub["lines"][:2], descinfo["lines"][0:2])
    np.testing.assert_allclose(sub["lines"][2:], descinfo["lines"][4:6])
    np.testing.assert_allclose(
        sub["endpoints_desc"][:, 2:], descinfo["endpoints_desc"][:, 4:6]
    )


def test_matcher_is_mutual_and_in_range(detector, image_pair):
    from limap.image.line import get_matcher

    _, d0 = detector.detect_and_extract(image_pair[0])
    _, d1 = detector.detect_and_extract(image_pair[1])
    matcher = get_matcher("upal", MatcherOptions(), detector)
    matches = matcher.match_pair(d0, d1)

    assert matches.ndim == 2 and matches.shape[1] == 2
    assert len(matches) > 0, "no matches between two views of the same box"
    assert matches[:, 0].max() < len(d0["lines_score"])
    assert matches[:, 1].max() < len(d1["lines_score"])
    # Mutual nearest neighbour is one-to-one in both directions.
    assert len(set(matches[:, 0])) == len(matches)
    assert len(set(matches[:, 1])) == len(matches)


def test_matcher_handles_empty_descinfo(detector, image_pair):
    from limap.image.line import get_matcher

    _, d0 = detector.detect_and_extract(image_pair[0])
    empty = {
        "image_shape": d0["image_shape"],
        "lines": np.zeros((0, 2)),
        "lines_score": np.zeros((0,)),
        "endpoints_desc": np.zeros((128, 0)),
    }
    matcher = get_matcher("upal", MatcherOptions(), detector)
    assert matcher.match_pair(d0, empty).shape == (0, 2)
    assert matcher.match_pair(empty, d0).shape == (0, 2)


def test_joint_detection_writes_hloc_readable_features(
    detector, image_pair, tmp_path
):
    """The joint path must emit exactly what the separate paths do."""
    h5py = pytest.importorskip("h5py")
    from limap.image.joint_point_line import (
        JointPointLineDetectionOptions,
        joint_point_line_detection,
    )

    image_paths = {i: p for i, p in enumerate(image_pair)}
    image_names = {i: p.name for i, p in enumerate(image_pair)}
    feature_path, all_2d_segs, descinfo_folder = joint_point_line_detection(
        JointPointLineDetectionOptions(method="upal", skip_exists=False),
        image_paths,
        image_names,
        tmp_path,
        tmp_path / "features",
    )

    assert set(all_2d_segs) == set(image_paths)
    assert descinfo_folder.exists()
    with h5py.File(str(feature_path), "r") as fd:
        assert set(fd.keys()) == set(image_names.values())
        grp = fd[image_names[0]]
        n_kpts = grp["keypoints"].shape[0]
        assert grp["descriptors"].shape == (128, n_kpts)
        assert grp["scores"].shape == (n_kpts,)
        # hloc rebuilds a tensor shape from image_size, so a float dtype
        # raises inside its match_features dataloader.
        image_size = grp["image_size"][:]
        np.testing.assert_array_equal(image_size, [640, 480])
        assert np.issubdtype(image_size.dtype, np.integer)
        import torch

        torch.empty((1,) + tuple(image_size)[::-1])
        keypoints = grp["keypoints"][:]
        assert (keypoints >= 0).all()
        assert (keypoints[:, 0] < 640).all() and (keypoints[:, 1] < 480).all()


def test_joint_detection_agrees_with_line_only_path(
    detector, image_pair, tmp_path
):
    """Enabling the point half must not perturb the lines."""
    from limap.image.joint_point_line import (
        JointPointLineDetectionOptions,
        joint_point_line_detection,
    )

    segs_line_only, _ = detector.detect_and_extract(image_pair[0])
    segs_line_only, _ = detector.take_longest_k(
        segs_line_only, detector.max_num_2d_segs
    )

    _, all_2d_segs, _ = joint_point_line_detection(
        JointPointLineDetectionOptions(method="upal", skip_exists=False),
        {0: image_pair[0]},
        {0: image_pair[0].name},
        tmp_path,
        tmp_path / "features",
    )
    np.testing.assert_allclose(
        all_2d_segs[0], segs_line_only[:, :4], rtol=1e-5, atol=1e-4
    )


def test_joint_detection_rejects_line_only_detector(image_pair, tmp_path):
    """A detector that predicts no keypoints must fail before doing work."""
    pytest.importorskip("pytlsd")
    from limap.image.joint_point_line import (
        JointPointLineDetectionOptions,
        joint_point_line_detection,
    )

    with pytest.raises(NotImplementedError, match="lsd"):
        joint_point_line_detection(
            JointPointLineDetectionOptions(method="lsd"),
            {0: image_pair[0]},
            {0: image_pair[0].name},
            tmp_path,
            tmp_path / "features",
        )


def test_feature_file_lands_where_hloc_resolves_it(
    detector, image_pair, tmp_path
):
    """hloc gets only the stem and rebuilds the path from the export dir.

    ``match_features.main`` resolves ``Path(export_dir, features + ".h5")``, so
    writing the file anywhere else silently breaks point matching.
    """
    from limap.image.joint_point_line import (
        JointPointLineDetectionOptions,
        joint_point_line_detection,
    )

    feature_dir = tmp_path / "workspace"
    feature_path, _, _ = joint_point_line_detection(
        JointPointLineDetectionOptions(method="upal", skip_exists=False),
        {0: image_pair[0]},
        {0: image_pair[0].name},
        tmp_path / "joint_detections",
        feature_dir,
    )
    assert (feature_dir / f"{feature_path.stem}.h5").exists()


@pytest.mark.ci_workflow
def test_joint_options_drive_association():
    """Association must follow the joint method, not line_detection.

    Otherwise the matcher would build its extractor from whatever
    line_detection happens to name, silently mismatching the descriptors the
    joint pass actually wrote.
    """
    from limap.image.joint_point_line import JointPointLineDetectionOptions

    joint = JointPointLineDetectionOptions(method="upal")
    joint.detector_options = DetectorOptions(
        upal_options=UPALOptions(max_num_keypoints=1234)
    )
    line_options = joint.as_line_detection_options()

    assert line_options.detector_method == "upal"
    assert line_options.extractor_method == "upal"
    assert line_options.extractor_options.upal_options.max_num_keypoints == 1234
    assert line_options.weight_path == joint.weight_path


def test_joint_detection_resumes_a_partial_run(detector, image_pair, tmp_path):
    """Skipping must require the hloc entry too, not just the line artifacts.

    The three outputs are written per image but to three places, so a run
    interrupted mid-image leaves segments without their keypoints.
    """
    h5py = pytest.importorskip("h5py")
    from limap.image.joint_point_line import (
        JointPointLineDetectionOptions,
        joint_point_line_detection,
    )

    image_paths = {i: p for i, p in enumerate(image_pair)}
    image_names = {i: p.name for i, p in enumerate(image_pair)}
    args = (image_paths, image_names, tmp_path, tmp_path / "features")
    feature_path, _, _ = joint_point_line_detection(
        JointPointLineDetectionOptions(skip_exists=False), *args
    )

    # Simulate a crash after the segments were written but before the h5.
    with h5py.File(str(feature_path), "a") as fd:
        del fd[image_names[1]]

    feature_path, _, _ = joint_point_line_detection(
        JointPointLineDetectionOptions(skip_exists=True), *args
    )
    with h5py.File(str(feature_path), "r") as fd:
        assert set(fd.keys()) == set(image_names.values())


@pytest.mark.ci_workflow
def test_joint_options_use_the_shared_weight_root():
    """weight_path must stay None so weights_root() decides.

    Pinning a root here would send the joint path to a different directory
    than every other detector, downloading the same checkpoint twice.
    """
    from limap.image.joint_point_line import JointPointLineDetectionOptions

    joint = JointPointLineDetectionOptions()
    assert joint.weight_path is None
    assert joint.detector_options.base_options.weight_path is None
    assert joint.as_line_detection_options().weight_path is None
