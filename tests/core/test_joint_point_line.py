"""Cover the joint point-line matching abstraction.

The matchers themselves need torch and their git-sourced weights, so what is
checked here is the part that is easy to get wrong and cheap to run: lifting
junction matches back onto the keypoints of the COLMAP database, and the two
output formats the association step then imports.
"""

import numpy as np
import numpy.testing as npt
import pytest

from limap.image.joint_point_line import (
    JointPointLineMatcherOptions,
    get_joint_matcher,
    remap_point_matches,
)
from limap.image.joint_point_line.base_joint_matcher import (
    BaseJointMatcher,
    JointMatchResult,
)


@pytest.mark.ci_workflow
def test_remap_point_matches_drops_line_junctions():
    # Junctions 0 and 1 are line endpoints; 2 and 3 are keypoints 0 and 2.
    junc_to_keypoint1 = np.array([-1, -1, 0, 2])
    junc_to_keypoint2 = np.array([-1, 5, 7])
    matches0 = np.array([1, -1, 2, 0])
    scores0 = np.array([0.9, 0.0, 0.8, 0.7])

    matches, scores = remap_point_matches(
        matches0, scores0, junc_to_keypoint1, junc_to_keypoint2, 4
    )
    # Junction 0 matches a line endpoint, junction 3 matches one too, and
    # junction 1 has no match; only junction 2 -> keypoint 0 survives.
    npt.assert_array_equal(matches, [7, -1, -1, -1])
    npt.assert_allclose(scores, [0.8, 0.0, 0.0, 0.0])


@pytest.mark.ci_workflow
def test_remap_point_matches_empty():
    matches, scores = remap_point_matches(
        np.empty(0, dtype=int),
        np.empty(0),
        np.empty(0, dtype=int),
        np.empty(0, dtype=int),
        3,
    )
    npt.assert_array_equal(matches, [-1, -1, -1])
    npt.assert_allclose(scores, np.zeros(3))


@pytest.mark.ci_workflow
def test_get_joint_matcher_rejects_unknown_method():
    with pytest.raises(NotImplementedError):
        get_joint_matcher(
            "not_a_matcher", JointPointLineMatcherOptions().matching_options
        )


@pytest.mark.ci_workflow
def test_weight_path_reaches_the_matcher_options(tmp_path):
    options = JointPointLineMatcherOptions(weight_path=tmp_path)
    assert options.matching_options.base_options.weight_path == tmp_path
    # The property hands out a copy, so the stored options stay untouched.
    assert (
        JointPointLineMatcherOptions().matching_options.base_options.weight_path
        is None
    )


class _CountingMatcher(BaseJointMatcher):
    """A joint matcher pairing keypoint i of one image with i of the other."""

    def __init__(self, num_keypoints):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.pairs = []

    def get_module_name(self):
        return "counting"

    def check_compatibility(self, extractor_method):
        return True

    def describe(self, descinfo_folder, feature_path, img_id, image_name):
        return img_id

    def match_pair(self, descinfo1, descinfo2):
        self.pairs.append((descinfo1, descinfo2))
        return JointMatchResult(
            point_matches0=np.arange(self.num_keypoints),
            point_scores0=np.ones(self.num_keypoints),
            line_matches=np.array([[0, 1], [1, 0]]),
        )


@pytest.mark.ci_workflow
def test_match_all_neighbors_writes_both_halves(tmp_path):
    h5py = pytest.importorskip("h5py")
    parsers = pytest.importorskip("hloc.utils.parsers")
    import limap.util.io as limapio

    matcher = _CountingMatcher(num_keypoints=3)
    image_names = {1: "a.png", 2: "b.png", 3: "c.png"}
    neighbors = {1: [2, 3], 2: [1, 3], 3: [1, 2]}

    match_path, line_folder = matcher.match_all_neighbors(
        tmp_path,
        image_names,
        neighbors,
        tmp_path / "features.h5",
        tmp_path / "descinfos",
    )

    # Each unordered pair is matched exactly once, not once per direction.
    assert len(matcher.pairs) == 3
    assert sorted(matcher.pairs) == [(1, 2), (1, 3), (2, 3)]

    # hloc's pair keys carry a "/", so h5py nests them; what matters is that
    # hloc.utils.io.get_matches resolves each one.
    with h5py.File(str(match_path), "r") as fd:
        for a, b in [
            ("a.png", "b.png"),
            ("a.png", "c.png"),
            ("b.png", "c.png"),
        ]:
            assert parsers.names_to_pair(a, b) in fd
        group = fd[parsers.names_to_pair("a.png", "b.png")]
        npt.assert_array_equal(group["matches0"].__array__(), [0, 1, 2])
        assert group["matching_scores0"].shape == (3,)

    # Every image gets a file, holding the pairs it was the first image of.
    written = {
        img_id: limapio.read_npy(line_folder / f"matches_{img_id}.npy").item()
        for img_id in image_names
    }
    assert sorted(written[1]) == [2, 3]
    assert sorted(written[2]) == [3]
    assert written[3] == {}
    npt.assert_array_equal(written[1][2], [[0, 1], [1, 0]])


@pytest.mark.ci_workflow
def test_match_all_neighbors_redoes_an_interrupted_run(tmp_path):
    """An interrupted run leaves the folder and a partial h5 behind. Reusing
    that would import a truncated set of point matches and no line matches."""
    pytest.importorskip("h5py")
    pytest.importorskip("hloc.utils.parsers")

    matcher = _CountingMatcher(num_keypoints=3)
    image_names = {1: "a.png", 2: "b.png", 3: "c.png"}
    neighbors = {1: [2, 3], 2: [1, 3], 3: [1, 2]}
    args = (tmp_path, image_names, neighbors, tmp_path / "f.h5", tmp_path / "d")

    _, line_folder = matcher.match_all_neighbors(*args, skip_exists=True)
    assert len(matcher.pairs) == 3

    # Simulate dying after the h5 was opened but before the line matches were
    # written: the folder is there, the files are not.
    matcher.get_line_match_filename(line_folder, 1).unlink()
    matcher.match_all_neighbors(*args, skip_exists=True)
    assert len(matcher.pairs) == 6, "partial output was reused"


@pytest.mark.ci_workflow
def test_match_all_neighbors_reuses_an_existing_output(tmp_path):
    pytest.importorskip("h5py")
    pytest.importorskip("hloc.utils.parsers")

    matcher = _CountingMatcher(num_keypoints=3)
    image_names = {1: "a.png", 2: "b.png"}
    neighbors = {1: [2], 2: [1]}
    args = (tmp_path, image_names, neighbors, tmp_path / "f.h5", tmp_path / "d")

    matcher.match_all_neighbors(*args, skip_exists=True)
    assert len(matcher.pairs) == 1
    matcher.match_all_neighbors(*args, skip_exists=True)
    assert len(matcher.pairs) == 1
    matcher.match_all_neighbors(*args, skip_exists=False)
    assert len(matcher.pairs) == 2
