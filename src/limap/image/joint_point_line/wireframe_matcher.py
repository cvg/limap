"""Matchers over the wireframe description.

GlueStick and LightGlueStick consume the same description -- the merged line
endpoints of
:class:`~limap.image.joint_point_line.GlueStick.WireframeExtractor` followed by
the keypoints of the same SuperPoint pass -- and differ only in the network
that runs over it. Everything around that network lives here.
"""

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from typeguard import typechecked

import limap.util.io as limapio

from ..line.base_matcher import BaseMatcher, DefaultMatcherOptions
from .base_joint_matcher import (
    BaseJointMatcher,
    DefaultJointMatcherOptions,
    JointMatchResult,
    remap_point_matches,
    write_hloc_features,
)

# SuperPoint's detection noise, recorded on the exported keypoints so that the
# pose-guided geometric verification scales its threshold the way it does for
# hloc's own SuperPoint features.
DETECTION_NOISE = 2.0


def _assigned_line_matches(line_matches0):
    """The (N, 2) index pairs of the network's own line assignment."""
    matched = line_matches0 != -1
    return np.stack([np.flatnonzero(matched), line_matches0[matched]], axis=1)


def _topk_line_matches(raw_line_scores, topk):
    """For each line of the first image, its topk candidates, best first."""
    topk = min(topk, raw_line_scores.shape[1])
    candidates = torch.argsort(raw_line_scores, dim=1)[:, -topk:]
    candidates = torch.flip(candidates, dims=(1,)).cpu().numpy()
    return np.stack(
        [np.arange(candidates.shape[0]).repeat(topk), candidates.flatten()],
        axis=1,
    )


class WireframeNetwork:
    """The network half of a wireframe matcher."""

    def _match(self, descinfo1, descinfo2):
        """
        Virtual method (need to be implemented) - run the network over two \
        wireframe descriptions

        Returns:
            The junction matches of the first image and their scores, the \
            line assignment of the first image, and the raw line scores -- \
            the last None when the pair has no lines to rank.
        """
        raise NotImplementedError


class WireframeLineMatcher(WireframeNetwork, BaseMatcher):
    """The line half of a wireframe matcher.

    Use the joint matcher of the same network instead to keep the point
    matches that the same forward pass produces.
    """

    def __init__(self, extractor, options=DefaultMatcherOptions, device=None):
        super().__init__(extractor, options)
        self.device = "cuda" if device is None else device

    def check_compatibility(self, extractor):
        return extractor.get_module_name() == "wireframe"

    def match_pair(self, descinfo1, descinfo2):
        _, _, line_matches0, raw_line_scores = self._match(descinfo1, descinfo2)
        if self.topk == 0 or raw_line_scores is None:
            return _assigned_line_matches(line_matches0)
        return _topk_line_matches(raw_line_scores, self.topk)


class WireframeJointMatcher(WireframeNetwork, BaseJointMatcher):
    """A wireframe matcher, keeping the point matches of the same pass.

    :class:`WireframeLineMatcher` runs the same network and reads only its
    line assignment, leaving the point half to a second matcher over the same
    images. This one reads both.

    The junctions it matches are the merged line endpoints plus the keypoints
    of the same SuperPoint pass. :meth:`export_point_features` puts that second
    half in the COLMAP database during the description step, so the point
    matches index keypoints that are actually there, and the network is not run
    a second time to produce them.
    """

    def __init__(self, options=DefaultJointMatcherOptions, device=None):
        super().__init__(options)
        self.device = "cuda" if device is None else device

    def check_compatibility(self, extractor_method):
        return extractor_method == "wireframe"

    @classmethod
    @typechecked
    def export_point_features(
        cls,
        descinfo_folder: Path,
        image_names: dict[int, str],
        feature_path: Path,
    ) -> Path:
        """Write the wireframe extractor's keypoints as an hloc feature file.

        The junctions it describes are the merged line endpoints followed by
        the keypoints of the same SuperPoint pass; only the second half is
        written, so every keypoint in the COLMAP database is one this matcher
        can match.
        """
        feature_path.unlink(missing_ok=True)
        for img_id, name in tqdm(image_names.items()):
            descinfo = cls._read_descinfo(descinfo_folder, img_id)
            n_line_junctions = cls._n_line_junctions(descinfo, descinfo_folder)
            height, width = descinfo["image_shape"][:2]
            write_hloc_features(
                feature_path,
                name,
                descinfo["junctions"][n_line_junctions:],
                descinfo["junc_scores"][n_line_junctions:],
                descinfo["junc_desc"][:, n_line_junctions:],
                (int(width), int(height)),
                DETECTION_NOISE,
            )
        return feature_path

    @staticmethod
    def _read_descinfo(descinfo_folder, img_id):
        # Named by WireframeExtractor.get_descinfo_fname; instantiating the
        # extractor here just to ask would load SuperPoint.
        return limapio.read_npz(descinfo_folder / f"descinfo_{img_id}.npz")

    @staticmethod
    def _n_line_junctions(descinfo, descinfo_folder):
        if "n_line_junctions" not in descinfo:
            raise ValueError(
                f"The line descriptors in {descinfo_folder} predate joint "
                "matching and cannot be split into line junctions and "
                "keypoints. Re-run the description step for them."
            )
        return int(descinfo["n_line_junctions"])

    @typechecked
    def describe(
        self,
        descinfo_folder: Path,
        feature_path: Path,
        img_id: int,
        image_name: str,
    ) -> dict:
        """Take the description the line step already wrote.

        Its junctions are exactly what the network needs, and its keypoints
        are the ones :meth:`export_point_features` put in the COLMAP database,
        so the mapping back to keypoint indices is just the offset between the
        two halves. ``feature_path`` is therefore unused.
        """
        descinfo = self._read_descinfo(descinfo_folder, img_id)
        n_line_junctions = self._n_line_junctions(descinfo, descinfo_folder)
        num_keypoints = descinfo["junctions"].shape[0] - n_line_junctions
        return {
            "image_shape": descinfo["image_shape"],
            "lines": descinfo["lines"],
            "line_scores": descinfo["line_scores"],
            "lines_junc_idx": descinfo["lines_junc_idx"],
            "junctions": descinfo["junctions"],
            "junc_scores": descinfo["junc_scores"],
            "junc_desc": descinfo["junc_desc"],
            "junc_to_keypoint": np.concatenate(
                [
                    np.full(n_line_junctions, -1, dtype=int),
                    np.arange(num_keypoints),
                ]
            ),
            "num_keypoints": num_keypoints,
        }

    @typechecked
    def match_pair(self, descinfo1, descinfo2) -> JointMatchResult:
        matches0, match_scores0, line_matches0, raw_line_scores = self._match(
            descinfo1, descinfo2
        )
        point_matches0, point_scores0 = remap_point_matches(
            matches0,
            match_scores0,
            descinfo1["junc_to_keypoint"],
            descinfo2["junc_to_keypoint"],
            descinfo1["num_keypoints"],
        )
        if self.topk == 0 or raw_line_scores is None:
            line_matches = _assigned_line_matches(line_matches0)
        else:
            line_matches = _topk_line_matches(raw_line_scores, self.topk)
        return JointMatchResult(point_matches0, point_scores0, line_matches)
