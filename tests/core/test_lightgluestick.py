"""Tests for the LightGlueStick matcher.

The registry-level tests are pure Python and run in CI. The ones that build the
network need torch and the ``lightgluestick`` package, a separate manual
install, so they skip when it is absent.
"""

from pathlib import Path

import numpy as np
import pytest

from limap.image.joint_point_line import JointMatcherOptions
from limap.image.line import DetectorOptions, ExtractorOptions, MatcherOptions
from limap.image.line.register_matcher import LightGlueStickOptions

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.ci_workflow
def test_options_reachable_from_both_registries():
    # LightGlueStick runs from either the line-only or the joint path, so both
    # registries must carry its options.
    assert MatcherOptions().lightgluestick_options == LightGlueStickOptions()
    assert JointMatcherOptions().lightgluestick_options == (
        LightGlueStickOptions()
    )


@pytest.mark.ci_workflow
def test_early_exit_is_off_by_default():
    # Early exit trades matches for speed, so it stays opt-in.
    assert LightGlueStickOptions().depth_confidence == -1.0


@pytest.mark.ci_workflow
def test_shipped_config_selects_the_joint_matcher():
    import limap.util.config as cfgutils

    cfg = cfgutils.load_config(
        str(
            REPO_ROOT / "cfgs/structure_triangulation/lightgluestick_joint.yaml"
        ),
        default_path=None,
    )
    association = cfg["_image_association"]
    assert association["use_joint_point_line_matcher"] is True
    assert association["joint_point_line_matcher"]["method"] == (
        "lightgluestick"
    )
    # The wireframe descriptors are what the matcher describes from.
    assert cfg["_image_description"]["line_detection"]["extractor_method"] == (
        "wireframe"
    )


def test_an_image_without_lines_yields_no_matches():
    """The network reads its junction count off ``lines_junc_idx.max()``, so
    the wrapper has to answer for a wireframe that has no lines at all."""
    pytest.importorskip("torch")
    from limap.image.joint_point_line.LightGlueStick.common import (
        run_lightgluestick,
    )

    described = {
        "image_shape": (480, 640),
        "lines": np.zeros((3, 2, 2)),
        "junctions": np.zeros((10, 2)),
    }
    empty = {
        "image_shape": (480, 640),
        "lines": np.zeros((0, 2, 2)),
        "junctions": np.zeros((4, 2)),
    }
    # No network is built: the pair is answered before it would be used.
    matches0, scores0, line_matches0, raw = run_lightgluestick(
        None, described, empty, "cpu"
    )
    assert raw is None
    np.testing.assert_array_equal(matches0, np.full(10, -1))
    np.testing.assert_array_equal(scores0, np.zeros(10))
    np.testing.assert_array_equal(line_matches0, np.full(3, -1))


@pytest.fixture(scope="module")
def image_pair(tmp_path_factory):
    """Two views of a wireframe box, offset so the lines are matchable."""
    import cv2

    paths = []
    tmp = tmp_path_factory.mktemp("lightgluestick")
    for i, shift in enumerate((0, 12)):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (100 + shift, 80), (520 + shift, 400), (255,) * 3, 2)
        cv2.line(img, (100 + shift, 80), (520 + shift, 400), (255,) * 3, 2)
        cv2.line(img, (300 + shift, 40), (300 + shift, 440), (255,) * 3, 2)
        path = tmp / f"view_{i}.png"
        cv2.imwrite(str(path), img)
        paths.append(path)
    return paths


@pytest.fixture(scope="module")
def descinfos(image_pair):
    pytest.importorskip("torch")
    pytest.importorskip("pytlsd")
    from limap.image.line import get_detector, get_extractor

    detector = get_detector("lsd", DetectorOptions())
    extractor = get_extractor("wireframe", ExtractorOptions())
    return [
        extractor.extract(path, detector.detect(path)) for path in image_pair
    ]


@pytest.fixture(scope="module")
def matcher():
    pytest.importorskip("torch")
    pytest.importorskip("lightgluestick")
    from limap.image.line import get_matcher

    return get_matcher("lightgluestick", MatcherOptions(), None)


def test_matcher_reads_the_wireframe_description(matcher, descinfos):
    matches = matcher.match_pair(*descinfos)

    assert matches.ndim == 2 and matches.shape[1] == 2
    assert len(matches) > 0, "no matches between two views of the same box"
    assert matches[:, 0].max() < len(descinfos[0]["lines"])
    assert matches[:, 1].max() < len(descinfos[1]["lines"])
    # The assignment is one-to-one in both directions.
    assert len(set(matches[:, 0])) == len(matches)
    assert len(set(matches[:, 1])) == len(matches)


def test_matcher_topk_ranks_every_line(matcher, descinfos):
    from limap.image.line import get_matcher
    from limap.image.line.base_matcher import BaseMatcherOptions

    topk = 5
    options = MatcherOptions(base_options=BaseMatcherOptions(topk=topk))
    matches = get_matcher("lightgluestick", options, None).match_pair(
        *descinfos
    )
    # Capped by how many lines there are to rank in the second image.
    expected = min(topk, len(descinfos[1]["lines"]))
    assert len(matches) == expected * len(descinfos[0]["lines"])


def test_joint_matcher_keeps_both_halves(descinfos, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("lightgluestick")
    import limap.util.io as limapio
    from limap.image.joint_point_line import get_joint_matcher

    for img_id, descinfo in enumerate(descinfos):
        limapio.save_npz(tmp_path / f"descinfo_{img_id}.npz", descinfo)

    matcher = get_joint_matcher("lightgluestick", JointMatcherOptions())
    assert matcher.check_compatibility("wireframe")
    described = [
        matcher.describe(tmp_path, tmp_path / "features.h5", img_id, "img")
        for img_id in range(2)
    ]
    result = matcher.match_pair(*described)

    assert result.point_matches0.shape == (described[0]["num_keypoints"],)
    assert result.point_scores0.shape == result.point_matches0.shape
    assert result.point_matches0.max() < described[1]["num_keypoints"]
    assert result.line_matches.ndim == 2
    assert result.line_matches.shape[1] == 2
    assert len(result.line_matches) > 0


def test_depth_confidence_reaches_the_network():
    pytest.importorskip("torch")
    pytest.importorskip("lightgluestick")
    from limap.image.line import get_matcher

    options = MatcherOptions(
        lightgluestick_options=LightGlueStickOptions(depth_confidence=0.95)
    )
    matcher = get_matcher("lightgluestick", options, None)
    assert matcher.net.conf.depth_confidence == 0.95
