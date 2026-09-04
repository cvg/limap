from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
from typeguard import typechecked

import limap.util.io as limapio


@dataclass
class BaseJointMatcherOptions:
    """
    Base options for the joint point-line matcher

    :param topk: number of top line matches for each line \
        (if equal to 0, take the matcher's own assignment)
    :param n_neighbors: number of visual neighbors, \
        only for naming the output folder
    :param weight_path: root directory to load/store weights \
        (at default, ``~/.cache/limap``, overridable with the \
        ``LIMAP_WEIGHTS_PATH`` environment variable)
    """

    topk: int = 0
    n_neighbors: int = 20
    weight_path: Path | None = None


DefaultJointMatcherOptions = BaseJointMatcherOptions()

# hloc's match files store the matched index as an int16.
_MAX_KEYPOINTS = np.iinfo(np.short).max


@dataclass
class JointMatchResult:
    """Both halves of what one joint forward pass produces for an image pair.

    :param point_matches0: shape (n_keypoints1,), for each keypoint of the \
        first image the index of its match in the second, or -1. Indexes the \
        keypoints of the COLMAP database, so that it can be written in hloc's \
        match format and imported by the usual path.
    :param point_scores0: shape (n_keypoints1,), the matching score of each \
        entry of ``point_matches0``, 0 where there is no match.
    :param line_matches: shape (N, 2), pairs of line indices.
    """

    point_matches0: np.ndarray
    point_scores0: np.ndarray
    line_matches: np.ndarray


class BaseJointMatcher:
    """
    Virtual class for a matcher that produces point and line matches together

    GlueStick and its successors match points and lines in a single pass, so
    running a point matcher and a line matcher separately does the shared work
    twice. A joint matcher replaces both.
    """

    def __init__(self, options=DefaultJointMatcherOptions):
        self.topk = options.topk
        self.n_neighbors = options.n_neighbors
        self.weight_path = options.weight_path

    # The functions below are required for joint matchers
    @typechecked
    def get_module_name(self) -> str:
        """
        Virtual method (need to be implemented) - return the name of the module
        """
        raise NotImplementedError

    @typechecked
    def check_compatibility(self, extractor_method: str) -> bool:
        """
        Virtual method (need to be implemented) - whether the line \
        descriptors of that extractor are the ones this matcher describes from

        Args:
            extractor_method (str): The line extractor of the description step
        """
        raise NotImplementedError

    @typechecked
    def describe(
        self,
        descinfo_folder: Path,
        feature_path: Path,
        img_id: int,
        image_name: str,
    ) -> Any:
        """
        Virtual method (need to be implemented) - build the description of \
        one image from the line descriptors and the point features

        The point features are read from the hloc feature file that fed the \
        COLMAP database, so that the point matches this description leads to \
        index the keypoints already stored there.

        Args:
            descinfo_folder (pathlib.Path): The line descriptor folder
            feature_path (pathlib.Path): The hloc feature file
            img_id (int): The image id, as used by the descriptor folder
            image_name (str): The image name, as used by the feature file
        Returns:
            The joint description of the image
        """
        raise NotImplementedError

    @classmethod
    @typechecked
    def export_point_features(
        cls,
        descinfo_folder: Path,
        image_names: dict[int, str],
        feature_path: Path,
    ) -> Path:
        """
        Virtual method (need to be implemented) - write the keypoints the \
        line description already produced as an hloc feature file

        A joint matcher matches junctions, and its line descriptors are built
        from a pass over the image that detects keypoints anyway. Writing those
        keypoints is what puts them in the COLMAP database, so that the point
        matches can index them -- and it means the network runs once, not once
        for the line descriptors and again for the point features.

        A classmethod: the description step calls it before any weights are
        needed, so it must not require an instantiated network.

        Args:
            descinfo_folder (pathlib.Path): The line descriptor folder
            image_names (dict[int -> str]): image name for each image id
            feature_path (pathlib.Path): The hloc feature file to write
        Returns:
            feature_path
        """
        raise NotImplementedError

    @typechecked
    def match_pair(self, descinfo1, descinfo2) -> JointMatchResult:
        """
        Virtual method (need to be implemented) - match the points and the \
        lines of two images in a single pass

        Args:
            descinfo1: The description of the first image, from `describe`
            descinfo2: The description of the second image, from `describe`
        Returns:
            :class:`JointMatchResult`
        """
        raise NotImplementedError

    @typechecked
    def get_output_folder(self, output_folder: Path) -> Path:
        """
        Return the folder holding both halves of the output

        Args:
            output_folder (pathlib.Path): The output folder
        """
        return (
            output_folder
            / f"{self.get_module_name()}_n{self.n_neighbors}_top{self.topk}"
        )

    @typechecked
    def get_point_match_path(self, output_folder: Path) -> Path:
        """
        Return the path of the point matches, in hloc's match file format

        Args:
            output_folder (pathlib.Path): The output folder
        """
        return self.get_output_folder(output_folder) / "point_matches.h5"

    @typechecked
    def get_line_matches_folder(self, output_folder: Path) -> Path:
        """
        Return the folder of the line matches, one file per image

        Args:
            output_folder (pathlib.Path): The output folder
        """
        return self.get_output_folder(output_folder) / "line_matches"

    @typechecked
    def get_line_match_filename(
        self, line_matches_folder: Path, img_id: int
    ) -> Path:
        """
        Return the file holding one image's line matches to its neighbors

        Args:
            line_matches_folder (pathlib.Path): The line matches folder
            img_id (int): The image id
        """
        return line_matches_folder / f"matches_{img_id}.npy"

    @typechecked
    def match_all_neighbors(
        self,
        output_folder: Path,
        image_names: dict[int, str],
        neighbors: dict[int, list[int]],
        feature_path: Path,
        descinfo_folder: Path,
        skip_exists: bool = False,
    ) -> tuple[Path, Path]:
        """
        Match all images with their visual neighbors

        Each pair is matched once, in one pass for both modalities. The point
        matches go to an hloc match file so that the import and the geometric
        verification of the separate path apply unchanged; the line matches go
        to the per-image files :func:`limap.scene.import_line_matches` reads.

        Args:
            output_folder (pathlib.Path): The output folder
            image_names (dict[int -> str]): image name for each image id
            neighbors (dict[int -> list[int]]): visual neighbors of each image
            feature_path (pathlib.Path): The hloc feature file
            descinfo_folder (pathlib.Path): The line descriptor folder
            skip_exists (bool): Whether to reuse an existing output
        Returns:
            point_match_path, line_matches_folder
        """
        import h5py
        from hloc.utils.parsers import names_to_pair

        point_match_path = self.get_point_match_path(output_folder)
        line_matches_folder = self.get_line_matches_folder(output_folder)
        # The line matches are written in one go once every pair is matched,
        # so a full set of them is what says the previous run got to the end.
        # The folder alone does not: it is created before any matching.
        if (
            skip_exists
            and point_match_path.exists()
            and all(
                self.get_line_match_filename(
                    line_matches_folder, img_id
                ).exists()
                for img_id in neighbors
            )
        ):
            return point_match_path, line_matches_folder
        limapio.delete_folder(self.get_output_folder(output_folder))
        limapio.check_makedirs(line_matches_folder)

        # Written per image, but each pair is matched under whichever of its
        # two images comes first. The structure database canonicalizes the
        # pair, so recording one direction is enough for both.
        line_matches: dict[int, dict[int, np.ndarray]] = {
            img_id: {} for img_id in neighbors
        }
        done: set[tuple[int, int]] = set()

        # Descriptions run to a few MB each, so only the ones a pair is about
        # to need are held; the fan-out of one image sets the useful size.
        cache_size = 2 * max(len(ngs) for ngs in neighbors.values()) + 1
        descinfo_cache: OrderedDict[int, Any] = OrderedDict()

        def get_descinfo(img_id):
            if img_id in descinfo_cache:
                descinfo_cache.move_to_end(img_id)
            else:
                descinfo_cache[img_id] = self.describe(
                    descinfo_folder,
                    feature_path,
                    img_id,
                    image_names[img_id],
                )
                if len(descinfo_cache) > cache_size:
                    descinfo_cache.popitem(last=False)
            return descinfo_cache[img_id]

        with h5py.File(str(point_match_path), "a") as fd:
            for img_id in tqdm(list(neighbors.keys())):
                for ng_img_id in neighbors[img_id]:
                    pair = (min(img_id, ng_img_id), max(img_id, ng_img_id))
                    if ng_img_id == img_id or pair in done:
                        continue
                    done.add(pair)
                    descinfo2 = get_descinfo(ng_img_id)
                    descinfo1 = get_descinfo(img_id)
                    result = self.match_pair(descinfo1, descinfo2)
                    line_matches[img_id][ng_img_id] = result.line_matches
                    # hloc stores match indices as int16, so say so rather
                    # than silently wrapping around.
                    if result.point_matches0.max(initial=-1) > _MAX_KEYPOINTS:
                        raise ValueError(
                            f"Image {image_names[ng_img_id]} has more than "
                            f"{_MAX_KEYPOINTS} keypoints, which hloc's match "
                            "file format cannot index."
                        )
                    grp = fd.create_group(
                        names_to_pair(
                            image_names[img_id], image_names[ng_img_id]
                        )
                    )
                    grp.create_dataset(
                        "matches0", data=result.point_matches0.astype(np.short)
                    )
                    grp.create_dataset(
                        "matching_scores0",
                        data=result.point_scores0.astype(np.half),
                    )

        for img_id, matches in line_matches.items():
            limapio.save_npy(
                self.get_line_match_filename(line_matches_folder, img_id),
                matches,
            )
        return point_match_path, line_matches_folder


@typechecked
def write_hloc_features(
    feature_path: Path,
    name: str,
    keypoints: np.ndarray,
    scores: np.ndarray,
    descriptors: np.ndarray,
    image_size: tuple[int, int],
    uncertainty: float,
) -> None:
    """Append one image's keypoints to an hloc-format feature file.

    Written here rather than through ``hloc.extract_features`` because the
    predictions come from the pass that produced the line descriptors, and
    hloc's extractor loop would re-run the network to get them.

    Args:
        feature_path: the hloc feature file, appended to
        name (str): the image name, as used by the feature file
        keypoints: shape (K, 2), in the original image coordinates
        scores: shape (K,)
        descriptors: shape (D, K)
        image_size (tuple[int, int]): (width, height)
        uncertainty (float): detection noise, read back by the geometric \
            verification
    """
    import h5py

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(feature_path), "a") as fd:
        if name in fd:
            del fd[name]
        grp = fd.create_group(name)
        grp.create_dataset("keypoints", data=keypoints.astype(np.float32))
        grp.create_dataset("scores", data=scores.astype(np.float16))
        grp.create_dataset("descriptors", data=descriptors.astype(np.float16))
        grp.create_dataset("image_size", data=np.array(image_size))
        grp["keypoints"].attrs["uncertainty"] = uncertainty


@typechecked
def remap_point_matches(
    matches0: np.ndarray,
    scores0: np.ndarray,
    junc_to_keypoint1: np.ndarray,
    junc_to_keypoint2: np.ndarray,
    num_keypoints1: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift matches over junctions to matches over keypoints.

    A joint matcher matches junctions, which are the keypoints of the COLMAP
    database plus the endpoints of the lines. ``junc_to_keypoint`` gives the
    keypoint index of each junction, or -1 for a line endpoint; matches
    touching an endpoint are dropped, since the line half already carries them.

    Args:
        matches0: shape (n_junctions1,), matched junction index or -1
        scores0: shape (n_junctions1,), the score of each match
        junc_to_keypoint1: shape (n_junctions1,), keypoint index or -1
        junc_to_keypoint2: shape (n_junctions2,), keypoint index or -1
        num_keypoints1: number of keypoints of the first image
    Returns:
        matches0, scores0 over the keypoints of the first image
    """
    keypoint_matches = np.full(num_keypoints1, -1, dtype=int)
    keypoint_scores = np.zeros(num_keypoints1, dtype=float)
    if len(matches0) == 0 or len(junc_to_keypoint2) == 0:
        return keypoint_matches, keypoint_scores

    matched = matches0 >= 0
    src = junc_to_keypoint1
    dst = np.where(
        matched, junc_to_keypoint2[np.where(matched, matches0, 0)], -1
    )
    keep = matched & (src >= 0) & (dst >= 0)
    keypoint_matches[src[keep]] = dst[keep]
    keypoint_scores[src[keep]] = scores0[keep]
    return keypoint_matches, keypoint_scores
