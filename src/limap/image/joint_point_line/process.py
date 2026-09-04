from pathlib import Path

from pycolmap import logging
from typeguard import typechecked

from .register_joint_matcher import (
    get_joint_matcher,
    get_joint_matcher_class,
)
from .specs import JointPointLineMatcherOptions


@typechecked
def joint_point_line_description(
    options: JointPointLineMatcherOptions,
    descinfo_folder: Path,
    image_names: dict[int, str],
    feature_dir: Path,
) -> Path:
    """Export the keypoints the line description already produced.

    The joint matcher matches junctions -- merged line endpoints plus the
    keypoints of the same network pass -- so the keypoints have to reach the
    COLMAP database for its point matches to index anything. Taking them from
    the line descriptors rather than detecting them again is what keeps that
    pass single, and keeps the junctions the matcher sees identical to the ones
    the descriptors were built from.

    The feature file must go directly in ``feature_dir``: the geometric
    verification resolves it by path, and hloc's readers expect its layout.

    Args:
        options: the matcher options
        descinfo_folder: the line descriptor folder
        image_names (dict[int -> str]): image name for each image id
        feature_dir: where the feature file goes
    Returns:
        The hloc feature file
    """
    logging.info(
        "[LOG] Export 2D point features from the line descriptors "
        f"(matcher = {options.method}, n_images = {len(image_names)})..."
    )
    matcher_class = get_joint_matcher_class(options.method)
    return matcher_class.export_point_features(
        descinfo_folder,
        image_names,
        feature_dir / f"feats-{options.method}.h5",
    )


@typechecked
def joint_point_line_matching(
    options: JointPointLineMatcherOptions,
    image_names: dict[int, str],
    neighbors: dict[int, list[int]],
    point_descriptor_path: Path,
    line_descriptor_path: Path,
    line_extractor_method: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Match the points and the lines of every pair in a single pass.

    Returns the point matches in hloc's match file format and the line matches
    as the per-image files the separate paths produce, so that the import and
    the geometric verification downstream are unchanged.

    Args:
        options: the matcher options
        image_names (dict[int -> str]): image name for each image id
        neighbors (dict[int -> list[int]]): visual neighbors of each image
        point_descriptor_path: the hloc feature file
        line_descriptor_path: the line descriptor folder
        line_extractor_method (str): the extractor that wrote those descriptors
        output_dir: where both halves of the output go
    Returns:
        point_match_path, line_matches_folder
    """
    logging.info(
        "[LOG] Start joint 2D point and line matching... "
        f"(matcher = {options.method}, "
        f"n_images = {len(image_names)}, "
        f"n_neighbors = {max(len(v) for v in neighbors.values())})"
    )
    matcher = get_joint_matcher(options.method, options.matching_options)
    if not matcher.check_compatibility(line_extractor_method):
        raise ValueError(
            f"Joint matcher {options.method!r} cannot describe from the line "
            f"descriptors of extractor {line_extractor_method!r}."
        )
    return matcher.match_all_neighbors(
        output_dir,
        image_names,
        neighbors,
        point_descriptor_path,
        line_descriptor_path,
        skip_exists=options.skip_exists,
    )
