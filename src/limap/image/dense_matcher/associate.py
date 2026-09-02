from pathlib import Path
from typeguard import typechecked
import cv2
import numpy as np
from tqdm import tqdm
import pycolmap
from pycolmap import logging

import limap.geometry
import limap.scene
import limap.util.io as limapio

from .specs import (
    PointDenseMatchingOptions,
    LineDenseMatchingOptions,
    GroupDenseMatchingOptions,
    DenseMatchingOptions,
)
from .metrics import (
    compute_point_distance_matrix,
    compute_line_distance_matrix,
    compute_mask_overlap_matrix,
)
from .base_dense_matcher import BiDenseMatchingResult
from .register_dense_matcher import get_dense_matcher


@typechecked
def match_points_via_dense_matching(
    options: PointDenseMatchingOptions,
    warp: BiDenseMatchingResult,
    points1: np.ndarray,
    points2: np.ndarray,
):
    d12 = compute_point_distance_matrix(warp.match_1to2, points1, points2)
    d21 = compute_point_distance_matrix(warp.match_2to1, points2, points1)

    # Nearest neighbor
    N1, N2 = d12.shape
    nn12 = d12.argmin(axis=1)  # shape (N1,)
    nn21 = d21.argmin(axis=1)  # shape (N2,)

    # Check mutual condition in vectorized form:
    mutual = nn21[nn12] == np.arange(N1)  # shape (N1,), boolean
    good_dist = d12[np.arange(N1), nn12] < options.pixel_thresh
    keep = mutual & good_dist

    # Final matches (vectorized)
    i_indices = np.where(keep)[0]
    j_indices = nn12[keep]

    matches = np.stack([i_indices, j_indices], axis=1)
    return matches


@typechecked
def match_lines_via_dense_matching(
    options: LineDenseMatchingOptions,
    warp: BiDenseMatchingResult,
    lines1: list[limap.geometry.Line2d],
    lines2: list[limap.geometry.Line2d],
):
    d12 = compute_line_distance_matrix(
        warp.match_1to2,
        lines1,
        lines2,
        n_samples=options.n_samples,
        min_segment_overlap=options.min_segment_overlap,
    )
    d21 = compute_line_distance_matrix(
        warp.match_2to1,
        lines2,
        lines1,
        n_samples=options.n_samples,
        min_segment_overlap=options.min_segment_overlap,
    )

    # Nearest neighbor
    N1, N2 = d12.shape
    nn12 = d12.argmin(axis=1)  # shape (N1,)
    nn21 = d21.argmin(axis=1)  # shape (N2,)

    # Check mutual condition in vectorized form:
    mutual = nn21[nn12] == np.arange(N1)  # shape (N1,), boolean
    good_dist = d12[np.arange(N1), nn12] < options.pixel_thresh
    keep = mutual & good_dist

    # Final matches (vectorized)
    i_indices = np.where(keep)[0]
    j_indices = nn12[keep]

    matches = np.stack([i_indices, j_indices], axis=1)
    return matches


@typechecked
def match_groups_via_dense_matching(
    options: GroupDenseMatchingOptions,
    warp: BiDenseMatchingResult,
    mask1: np.ndarray,
    mask2: np.ndarray,
):
    o12 = compute_mask_overlap_matrix(
        warp.match_1to2, mask1, mask2, min_num_pixels=options.min_num_pixels
    )
    o21 = compute_mask_overlap_matrix(
        warp.match_2to1, mask2, mask1, min_num_pixels=options.min_num_pixels
    )

    # Nearest neighbor
    N1, N2 = o12.shape
    if N1 == 0 or N2 == 0:
        return np.empty((0, 2), dtype=np.int64)

    nn12 = o12.argmax(axis=1)  # shape (N1,)
    nn21 = o21.argmax(axis=1)  # shape (N2,)

    # Check mutual condition in vectorized form:
    mutual = nn21[nn12] == np.arange(N1)  # shape (N1,), boolean
    good_overlap = o12[np.arange(N1), nn12] >= options.overlap_thresh
    keep = mutual & good_overlap

    # Final matches (vectorized)
    i_indices = np.where(keep)[0]
    j_indices = nn12[keep]

    matches = np.stack([i_indices, j_indices], axis=1)
    return matches


@typechecked
def associate_via_dense_matching(
    options: DenseMatchingOptions,
    image_names: dict[int, Path],
    neighbors: dict[int, list[int]],
    group_workspace_path: Path,
    db_path: Path,
    structure_db_path: Path,
    warp_cache_dir: Path | None = None,
) -> None:
    dense_matcher = get_dense_matcher(options.method)
    if warp_cache_dir is not None:
        warp_cache_dir.mkdir(parents=True, exist_ok=True)

    with (
        pycolmap.Database.open(db_path) as db,
        limap.scene.StructureDatabase.open(structure_db_path) as struct_db,
    ):
        for img_id, image_name in tqdm(image_names.items()):
            img = None  # lazy load
            for ng_img_id in neighbors[img_id]:
                # Try loading from cache
                warps = None
                if warp_cache_dir is not None:
                    cache_path = (
                        warp_cache_dir / f"warp_{img_id}_{ng_img_id}.npy"
                    )
                    if cache_path.exists():
                        warps = BiDenseMatchingResult.from_dict(
                            limapio.read_npy(cache_path).item()
                        )

                # Compute if not cached
                if warps is None:
                    if img is None:
                        img = cv2.imread(str(image_name))
                    ng_img = cv2.imread(str(image_names[ng_img_id]))
                    warps = dense_matcher.get_warping_symmetric(img, ng_img)
                    # Save to cache
                    if warp_cache_dir is not None and options.save_warps:
                        cache_path = (
                            warp_cache_dir / f"warp_{img_id}_{ng_img_id}.npy"
                        )
                        limapio.save_npy(cache_path, warps.to_dict())

                # point matching
                if not options.skip_point_matching:
                    points1 = db.read_keypoints(img_id)
                    points2 = db.read_keypoints(ng_img_id)
                    point_matches = match_points_via_dense_matching(
                        options.point_matching, warps, points1, points2
                    )
                else:
                    point_matches = np.empty((0, 2), dtype=np.uint32)

                # line matching
                if not options.skip_line_matching:
                    lines1 = struct_db.read_structure2d(img_id).lines
                    lines2 = struct_db.read_structure2d(ng_img_id).lines
                    line_matches = match_lines_via_dense_matching(
                        options.line_matching,
                        warps,
                        lines1,
                        lines2,
                    )
                else:
                    line_matches = np.empty((0, 2), dtype=np.uint32)

                # group matching (skip if disabled or groups not described)
                start_ids_path = group_workspace_path / "start_ids.npy"
                group_matches = []
                if not options.skip_group_matching and start_ids_path.exists():
                    start_ids = limapio.read_npy(start_ids_path).item()
                    _type_to_subdir = {
                        "PLANE": "plane_detections",
                        "CYLINDER": "cylinder_detections",
                        "SPHERE": "sphere_detections",
                    }
                    for group_name, start_ids_group in start_ids.items():
                        if group_name == "VP":
                            continue
                        if group_name not in _type_to_subdir:
                            logging.warning(
                                f"Skipping unknown group type in "
                                f"dense matching: {group_name}"
                            )
                            continue
                        mask_dir = (
                            group_workspace_path
                            / _type_to_subdir[group_name]
                            / "seg_labels"
                        )
                        if not mask_dir.exists():
                            continue
                        mask1 = limapio.read_npy(
                            mask_dir / f"mask_{img_id}.npy"
                        ).astype(int)
                        mask2 = limapio.read_npy(
                            mask_dir / f"mask_{ng_img_id}.npy"
                        ).astype(int)
                        curr_group_matches = match_groups_via_dense_matching(
                            options.group_matching, warps, mask1, mask2
                        )
                        curr_group_matches += [
                            start_ids_group[img_id],
                            start_ids_group[ng_img_id],
                        ]
                        group_matches.extend(curr_group_matches)
                group_matches = np.array(group_matches)

                # Import matching
                if len(point_matches) > 0 and not db.exists_matches(
                    img_id, ng_img_id
                ):
                    db.write_matches(
                        img_id, ng_img_id, point_matches.astype(np.uint32)
                    )
                    # skip geometric verification for now
                    db.write_two_view_geometry(
                        img_id,
                        ng_img_id,
                        pycolmap.TwoViewGeometry(
                            inlier_matches=point_matches.astype(np.uint32)
                        ),
                    )
                if len(line_matches) > 0 and not struct_db.exists_line_matches(
                    img_id, ng_img_id
                ):
                    struct_db.write_line_matches(
                        img_id, ng_img_id, line_matches.astype(np.uint32)
                    )
                if len(
                    group_matches
                ) > 0 and not struct_db.exists_group_matches(img_id, ng_img_id):
                    struct_db.write_group_matches(
                        img_id, ng_img_id, group_matches.astype(np.uint32)
                    )
