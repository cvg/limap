from pathlib import Path

import numpy as np
from pycolmap import logging
from tqdm import tqdm
from typeguard import typechecked

import limap.util.io as limapio

from ..line import get_detector
from ..line.base_detector import BaseDetector
from .register_joint_matcher import (
    get_joint_matcher,
    get_joint_matcher_class,
)
from .specs import (
    JointPointLineDetectionOptions,
    JointPointLineMatcherOptions,
)


@typechecked
def _write_hloc_features(
    feature_path: Path, name: str, points: dict, uncertainty: float
) -> None:
    """Append one image's keypoints to an hloc-format feature file.

    Written here rather than through ``hloc.extract_features`` because the
    predictions come from a pass that also produced the line segments, and
    hloc's extractor loop would re-run the network to get them.

    Stores the arrays as the network produced them, unlike the matcher path's
    :func:`write_hloc_features`: here they are the point descriptors.
    """
    import h5py

    with h5py.File(str(feature_path), "a") as fd:
        if name in fd:
            del fd[name]
        grp = fd.create_group(name)
        for key in ("keypoints", "scores", "descriptors", "image_size"):
            grp.create_dataset(key, data=points[key])
        grp["keypoints"].attrs["uncertainty"] = uncertainty


@typechecked
def joint_point_line_detection(
    options: JointPointLineDetectionOptions,
    image_paths: dict[int, Path],
    image_names: dict[int, str],
    output_dir: Path,
    feature_dir: Path,
) -> tuple[Path, dict[int, np.ndarray], Path]:
    """Detect points and lines for all images in one network pass each.

    A joint detector is a line detector that also predicts keypoints, so it
    implements ``detect_and_extract_joint``; ones that do not are rejected
    before any work is done.

    Returns the hloc feature file, the 2D segments per image, and the line
    descriptor folder -- the same three artifacts the separate point and line
    paths produce, so the callers downstream are unchanged.

    Segments and line descriptors go under ``output_dir``, but the feature file
    must go directly in ``feature_dir``: point matching passes only its stem to
    hloc, which resolves it as ``feature_dir / f"{stem}.h5"``.
    """
    logging.info(
        "[LOG] Start joint 2D point and line detection "
        f"(detector = {options.method}, "
        f"n_images = {len(image_paths)})..."
    )

    detector = get_detector(options.method, options.detector_options)
    if (
        type(detector).detect_and_extract_joint
        is BaseDetector.detect_and_extract_joint
    ):
        raise NotImplementedError(
            f"Line detector {options.method!r} does not predict "
            "keypoints, so it cannot drive joint point-line detection."
        )
    folder_save = output_dir / options.method
    seg_folder = detector.get_segments_folder(folder_save)
    descinfo_folder = detector.get_descinfo_folder(folder_save)
    feature_path = feature_dir / f"feats-{options.method}.h5"

    if not options.skip_exists:
        limapio.delete_folder(seg_folder)
        limapio.delete_folder(descinfo_folder)
        feature_path.unlink(missing_ok=True)
    limapio.check_makedirs(seg_folder)
    limapio.check_makedirs(descinfo_folder)
    feature_path.parent.mkdir(parents=True, exist_ok=True)

    uncertainty = getattr(
        getattr(detector, "net", None), "detection_noise", 1.0
    )
    # Segments, descriptors and features are written per image but to three
    # places, so a run interrupted mid-image can have some and not others.
    described = set()
    if options.skip_exists and feature_path.exists():
        import h5py

        with h5py.File(str(feature_path), "r") as fd:
            described = set(fd.keys())
    for img_id, image_path in tqdm(image_paths.items()):
        if (
            options.skip_exists
            and limapio.exists_txt_segments(seg_folder, img_id)
            and Path(
                detector.get_descinfo_fname(descinfo_folder, img_id)
            ).exists()
            and image_names[img_id] in described
        ):
            continue
        segs, descinfo, points = detector.detect_and_extract_joint(image_path)
        n_segs_orig = segs.shape[0]
        segs, indexes = detector.take_longest_k(
            segs, max_num_2d_segs=detector.max_num_2d_segs
        )
        if indexes.shape[0] < n_segs_orig:
            descinfo = detector.sample_descinfo_by_indexes(
                descinfo, indexes.tolist()
            )
        limapio.save_txt_segments(seg_folder, img_id, segs)
        detector.save_descinfo(descinfo_folder, img_id, descinfo)
        _write_hloc_features(
            feature_path, image_names[img_id], points, uncertainty
        )

    all_2d_segs = limapio.read_all_segments_from_folder(seg_folder)
    all_2d_segs = {img_id: all_2d_segs[img_id] for img_id in image_paths}
    return feature_path, all_2d_segs, descinfo_folder


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
