import numpy as np
from pathlib import Path
from pycolmap import logging
from tqdm import tqdm
from typeguard import typechecked

import limap.util.io as limapio

from ..line import get_detector
from ..line.base_detector import BaseDetector
from .specs import JointPointLineDetectionOptions


@typechecked
def _write_hloc_features(
    feature_path: Path, name: str, points: dict, uncertainty: float
) -> None:
    """Append one image's keypoints to an hloc-format feature file.

    Written here rather than through ``hloc.extract_features`` because the
    predictions come from a pass that also produced the line segments, and
    hloc's extractor loop would re-run the network to get them.
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
