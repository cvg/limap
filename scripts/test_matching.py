"""Smoke-test the 2D line detection, description and matching stack.

Runs a detector, an extractor and a matcher from the registries in
``limap.image.line`` over an image pair, reports the matching time, and
writes visualizations of the detections and of the matched segments.

Usage:
    python scripts/test_matching.py IMG1 IMG2 -o outputs/test_matching
    python scripts/test_matching.py IMG1 IMG2 \
        --detector deeplsd --extractor wireframe --matcher gluestick
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import seaborn as sns
from pycolmap import logging

from limap.image.line import (
    DetectorOptions,
    ExtractorOptions,
    MatcherOptions,
    get_detector,
    get_extractor,
    get_matcher,
)
from limap.visualize import draw_2d_lines


def cuda_sync():
    """Synchronize CUDA, if torch is around, so the timing is meaningful."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def as_endpoints(seg):
    """One (5,) detection -> the (2, 2) array the drawing helpers take."""
    return np.asarray(seg[:4], dtype=float).reshape(2, 2)


def draw_detections(image, segs):
    if len(segs) == 0:
        return image.copy()
    return draw_2d_lines(image, [as_endpoints(s) for s in segs], thickness=2)


def draw_matches(image1, image2, segs1, segs2, matches):
    """Draw every matched pair in a shared color on both images."""
    draw1, draw2 = image1.copy(), image2.copy()
    if len(matches) == 0:
        return draw1, draw2
    palette = sns.color_palette("husl", n_colors=len(matches))
    for (idx1, idx2), rgb in zip(matches, palette, strict=True):
        color = [int(255 * c) for c in rgb[::-1]]  # RGB -> BGR
        for image, seg in ((draw1, segs1[idx1]), (draw2, segs2[idx2])):
            start, end = as_endpoints(seg).astype(int)
            cv2.line(image, tuple(start), tuple(end), color, 4)
    return draw1, draw2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image1", type=Path, help="first image")
    parser.add_argument("image2", type=Path, help="second image")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("outputs/test_matching"),
        help="directory for the visualizations",
    )
    parser.add_argument("--detector", default="lsd", help="line detector")
    parser.add_argument(
        "--extractor", default="dense_naive", help="line extractor"
    )
    parser.add_argument("--matcher", default="dense_roma", help="line matcher")
    parser.add_argument(
        "--weights",
        default="outdoor",
        help="weights for the dense / superglue matchers",
    )
    parser.add_argument(
        "--max_num_2d_segs",
        type=int,
        default=3000,
        help="cap on the detections per image",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for image_path in (args.image1, args.image2):
        if not image_path.exists():
            raise FileNotFoundError(image_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detector_options = DetectorOptions()
    detector_options.base_options.max_num_2d_segs = args.max_num_2d_segs
    detector = get_detector(args.detector, detector_options)
    extractor = get_extractor(args.extractor, ExtractorOptions())
    matcher_options = MatcherOptions()
    matcher_options.dense_options.weights = args.weights
    matcher_options.superglue_options.weights = args.weights
    matcher = get_matcher(args.matcher, matcher_options, extractor)

    segs1 = detector.detect(args.image1)
    desc1 = extractor.extract(args.image1, segs1)
    segs2 = detector.detect(args.image2)
    desc2 = extractor.extract(args.image2, segs2)
    logging.info(f"detected {len(segs1)} and {len(segs2)} segments")

    cuda_sync()
    start = time.perf_counter()
    matches = matcher.match_pair(desc1, desc2)
    cuda_sync()
    logging.info(
        f"{len(matches)} matches in {time.perf_counter() - start:.3f}s"
    )

    image1 = cv2.imread(str(args.image1))
    image2 = cv2.imread(str(args.image2))
    draw1, draw2 = draw_matches(image1, image2, segs1, segs2, matches)
    outputs = {
        "detections_1.png": draw_detections(image1, segs1),
        "detections_2.png": draw_detections(image2, segs2),
        "matches_1.png": draw1,
        "matches_2.png": draw2,
    }
    for name, image in outputs.items():
        cv2.imwrite(str(args.output_dir / name), image)
    logging.info(f"wrote {len(outputs)} visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
