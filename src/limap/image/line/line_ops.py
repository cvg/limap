import numpy as np
from pathlib import Path
from typeguard import typechecked
from pycolmap import logging

from .specs import LineDetectionOptions, LineMatcherOptions
from .register_detector import get_detector, get_extractor
from .register_matcher import get_matcher


@typechecked
def line_detection(
    image_paths: dict[int, Path], output_dir, options: LineDetectionOptions
) -> tuple[dict[int, np.ndarray], Path | None]:
    if options.extractor_method is not None:
        logging.info(
            "[LOG] Start 2D line detection and description "
            f"(detector = {options.detector_method}, "
            f"extractor = {options.extractor_method}, "
            f"n_images = {len(image_paths)})..."
        )
    else:
        logging.info(
            "[LOG] Start 2D line detection and description "
            f"(detector = {options.detector_method}, "
            f"n_images = {len(image_paths)})..."
        )

    folder_save = output_dir / options.detector_method
    descinfo_folder = None
    detector = get_detector(options.detector_method, options.detector_options)
    if (
        options.compute_descinfo
        and (options.detector_method == options.extractor_method)
        and (not detector.do_merge_lines)
    ):
        (
            all_2d_segs,
            descinfo_folder,
        ) = detector.detect_and_extract_all_images(
            folder_save, image_paths, skip_exists=options.skip_exists
        )
    else:
        all_2d_segs = detector.detect_all_images(
            folder_save, image_paths, skip_exists=options.skip_exists
        )
        if options.compute_descinfo:
            extractor = get_extractor(
                options.extractor_method, options.extractor_options
            )
            descinfo_folder = extractor.extract_all_images(
                folder_save,
                image_paths,
                all_2d_segs,
                skip_exists=options.skip_exists,
            )
    return all_2d_segs, descinfo_folder


@typechecked
def line_matching(
    descinfo_folder: Path,
    output_dir: Path,
    neighbors: dict[int, list[int]],
    det_options: LineDetectionOptions,
    options: LineMatcherOptions,
) -> Path:
    image_ids = list(neighbors.keys())
    logging.info(
        f"[LOG] Start matching 2D lines... "
        f"(extractor = {det_options.extractor_method}, "
        f"matcher = {options.method}, "
        f"n_images = {len(image_ids)}, "
        f"n_neighbors = {max(len(v) for v in neighbors.values())})"
    )
    extractor = get_extractor(
        det_options.extractor_method, det_options.extractor_options
    )
    matcher = get_matcher(options.method, options.matching_options, extractor)
    folder_save = (
        output_dir
        / det_options.detector_method
        / f"feats_{det_options.extractor_method}"
    )
    matches_folder = matcher.match_all_neighbors(
        folder_save,
        image_ids,
        neighbors,
        descinfo_folder,
        skip_exists=options.skip_exists,
    )
    return matches_folder


@typechecked
def exhaustive_line_matching(
    descinfo_folder: Path,
    output_dir: Path,
    image_ids: list[int],
    det_options: LineDetectionOptions,
    options: LineMatcherOptions,
) -> Path:
    logging.info(
        "[LOG] Start exhausive matching 2D lines..."
        f"(extractor = {det_options.extractor_method}, "
        f"matcher = {options.method}, "
        f"n_images = {len(image_ids)})"
    )

    extractor = get_extractor(
        det_options.extractor_method, det_options.extractor_options
    )
    matcher = get_matcher(options.method, options.matching_options, extractor)

    folder_save = (
        output_dir
        / det_options.detector_method
        / f"feats_{det_options.extractor_method}"
    )
    matches_folder = matcher.match_all_exhaustive_pairs(
        folder_save, image_ids, descinfo_folder, skip_exists=options.skip_exists
    )
    return matches_folder
