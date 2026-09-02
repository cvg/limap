"""Utilities for loading SAM3 segmentation results and converting to Group2d."""

import json
import numpy as np
from pathlib import Path

import limap.geometry
import limap.scene

from .planelib.utils import convert_plane_mask_to_groups2d


def load_sam3_data_for_image(
    sam3_base_dir: Path,
    categories: dict[str, limap.geometry.GroupType],
    image_name: str,
) -> dict[limap.geometry.GroupType, tuple[np.ndarray, list[float]]]:
    """
    Load SAM3 masks and scores for a given image across all categories.

    Args:
        sam3_base_dir: Base directory of SAM3 output (contains category subdirs)
        categories: Mapping from category name to GroupType
            e.g. {"can": GroupType.CYLINDER, "football": GroupType.SPHERE}
        image_name: Image filename, e.g. "output_0001.png"

    Returns:
        Dict mapping GroupType to (binary_masks, scores) where
        binary_masks is (N_det, H, W) uint8 and scores is list[float].
        Only types with at least one detection are included.
    """
    stem = Path(image_name).stem  # e.g. "output_0001"
    result = {}

    for category, group_type in categories.items():
        cat_dir = sam3_base_dir / category
        mask_path = cat_dir / "masks" / f"{stem}.npy"
        results_path = cat_dir / "results.json"

        if not mask_path.exists():
            continue

        masks = np.load(mask_path)  # (N_det, H, W) uint8
        if masks.ndim == 2:
            masks = masks[np.newaxis]  # single detection → add batch dim
        if masks.shape[0] == 0:
            continue

        # Load scores from results.json
        scores = []
        if results_path.exists():
            with open(results_path) as f:
                results_data = json.load(f)
            if image_name in results_data:
                detections = results_data[image_name].get("detections", [])
                scores = [d.get("score", 1.0) for d in detections]

        # Ensure scores match number of masks
        if len(scores) != masks.shape[0]:
            scores = [1.0] * masks.shape[0]

        # Aggregate masks of the same GroupType
        if group_type in result:
            existing_masks, existing_scores = result[group_type]
            masks = np.concatenate([existing_masks, masks], axis=0)
            scores = existing_scores + scores
        result[group_type] = (masks, scores)

    return result


def convert_sam3_masks_to_groups2d(
    binary_masks: np.ndarray,
    group_type: limap.geometry.GroupType,
    points: np.ndarray,
    lines: list[limap.geometry.Line2d],
    scores: list[float] | None = None,
    min_line_overlap_length: float = 40.0,
    dilation_radius: float = 2.0,
    target_size: tuple[int, int] | None = None,
) -> tuple[list[limap.scene.Group2d], np.ndarray]:
    """
    Convert SAM3 binary masks to Group2d objects and a label mask.

    Generalized version of convert_plane_mask_to_groups2d that works with
    any group type and handles per-detection binary masks with overlap
    priority based on scores.

    Args:
        binary_masks: (N_det, H, W) uint8 binary masks
        group_type: GroupType for the created groups
        points: (N, 2) array of 2D keypoint coordinates
        lines: list of Line2d objects
        scores: Optional per-detection scores for overlap priority
            (higher score wins the pixel). If None, earlier index wins.
        min_line_overlap_length: minimum overlap length to associate a line
        dilation_radius: radius for dilating regions
        target_size: Optional (height, width) to resize masks to. Use when
            masks were generated at a different resolution than the working
            images (e.g. after --max_image_dim resizing).

    Returns:
        Tuple of (list[Group2d], label_mask) where label_mask is (H, W) int
        array with labels 1..N_det (0 = background).
    """
    import cv2

    if binary_masks.ndim == 2:
        binary_masks = binary_masks[np.newaxis]
    N_det, H, W = binary_masks.shape

    # Resize masks if target_size differs from mask resolution
    if target_size is not None and (target_size[0] != H or target_size[1] != W):
        tH, tW = target_size
        resized = np.empty((N_det, tH, tW), dtype=binary_masks.dtype)
        for i in range(N_det):
            resized[i] = cv2.resize(
                binary_masks[i],
                (tW, tH),
                interpolation=cv2.INTER_NEAREST,
            )
        binary_masks = resized
        H, W = tH, tW

    if N_det == 0:
        return [], np.zeros((H, W), dtype=np.int32)

    # Build (H, W) label mask from binary masks, resolving overlaps
    # by score priority (higher score wins the pixel)
    label_mask = np.zeros((H, W), dtype=np.int32)

    if scores is not None and len(scores) == N_det:
        # Sort by score ascending so that higher-score detections
        # overwrite lower-score ones
        order = np.argsort(scores)
    else:
        order = np.arange(N_det)

    for det_idx in order:
        mask_k = binary_masks[det_idx] > 0
        label_mask[mask_k] = det_idx + 1  # labels are 1-indexed

    # Reuse existing plane conversion logic with the label mask.
    # convert_plane_mask_to_groups2d creates PLANE-typed groups, but we
    # need groups of the requested type — so we create them with the
    # correct type and copy point/line associations.
    if group_type == limap.geometry.GroupType.PLANE:
        groups = convert_plane_mask_to_groups2d(
            label_mask,
            points,
            lines,
            min_line_overlap_length=min_line_overlap_length,
            dilation_radius=dilation_radius,
        )
        return groups, label_mask

    # For non-PLANE types (CYLINDER, SPHERE), only associate points —
    # lines don't have meaningful geometric relationships with curved surfaces.
    plane_groups = convert_plane_mask_to_groups2d(
        label_mask,
        points,
        [],
        min_line_overlap_length=min_line_overlap_length,
        dilation_radius=dilation_radius,
    )

    # Re-create groups with correct type, copying point associations only.
    # pg.points returns AssociatedFeature2d objects; extract .idx
    groups = []
    for pg in plane_groups:
        g = limap.scene.Group2d(group_type)
        point_ids = [af.idx for af in pg.points]
        if point_ids:
            g.add_points(point_ids)
        groups.append(g)

    return groups, label_mask
