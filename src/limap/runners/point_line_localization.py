"""Point-line visual localization module.

This module provides functions for localizing query images using
both point and line correspondences against a database (HolisticReconstruction).
"""

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pycolmap
from pycolmap import logging
from tqdm import tqdm
from typeguard import typechecked

import hloc.extract_features
import hloc.match_features

from limap.image.specs import (
    PointDetectionOptions,
    PointMatcherOptions,
    ImageDescriptionOptions,
    ImageAssociationOptions,
)
from limap.scene import HolisticReconstruction
from limap.geometry import Line2d, Line3d
import limap.image.line
import limap.util.io as limapio
from limap.estimators.absolute_pose import estimate_point_line_absolute_pose
from limap.estimators import absolute_pose


@dataclass
class PointLineLocalizationOptions:
    """Options for point-line visual localization."""

    # Workspace path for intermediate outputs
    workspace_path: Path | str = "tmp"

    # Common options that propagate to nested options
    skip_exists: bool = False
    weight_path: Path = field(
        default_factory=lambda: Path.home() / ".limap" / "models"
    )
    max_image_dim: int | None = None

    # Image description (point/line detection) - same structure as triangulation
    _image_description: ImageDescriptionOptions = field(
        default_factory=ImageDescriptionOptions
    )

    # Image association (point/line matching) - same structure as triangulation
    _image_association: ImageAssociationOptions = field(
        default_factory=ImageAssociationOptions
    )

    pose_estimation: absolute_pose.PointLineAbsolutePoseOptions = field(
        default_factory=absolute_pose.PointLineAbsolutePoseOptions
    )

    # Localization-specific
    n_retrieval: int = 10
    use_points_only: bool = False

    @property
    def image_description(self) -> ImageDescriptionOptions:
        """Image description options with propagated skip_exists/weight_path."""
        options = deepcopy(self._image_description)
        options.line_detection.skip_exists = self.skip_exists
        options.line_detection.weight_path = self.weight_path
        options.skip_group_description = (
            True  # Never describe groups in localization
        )
        return options

    @image_description.setter
    def image_description(self, opts: ImageDescriptionOptions) -> None:
        self._image_description = deepcopy(opts)

    @property
    def image_association(self) -> ImageAssociationOptions:
        """Image association options with propagated skip_exists/weight_path."""
        options = deepcopy(self._image_association)
        options.line_matcher.skip_exists = self.skip_exists
        options.line_matcher.weight_path = self.weight_path
        return options

    @image_association.setter
    def image_association(self, opts: ImageAssociationOptions) -> None:
        self._image_association = deepcopy(opts)

    # Convenience properties for easier access
    @property
    def point_detection(self) -> PointDetectionOptions:
        """Point detection options."""
        return self._image_description.point_detection

    @property
    def point_matching(self) -> PointMatcherOptions:
        """Point matching options."""
        return self._image_association.point_matcher

    @property
    def line_detection(self):
        """Line detection options with propagated skip_exists/weight_path."""
        options = deepcopy(self._image_description.line_detection)
        options.skip_exists = self.skip_exists
        options.weight_path = self.weight_path
        return options

    @property
    def line_matching(self):
        """Line matching options with propagated skip_exists/weight_path."""
        options = deepcopy(self._image_association.line_matcher)
        options.skip_exists = self.skip_exists
        options.weight_path = self.weight_path
        return options


@typechecked
def point_line_localization(
    options: PointLineLocalizationOptions,
    holistic_recon: HolisticReconstruction,
    query_images: dict[int, tuple[Path, pycolmap.Camera]],
    retrieval: dict[str, list[str]],
    results_path: Path,
    id_to_name: dict[int, str] | None = None,
) -> dict[int, pycolmap.Rigid3d]:
    """
    Point-line visual localization.

    Takes a single HolisticReconstruction containing the database:
    - point_recon: COLMAP reconstruction with 3D points
    - structure_recon: Lines3D and 2D structures per image

    Computes point and line correspondences on the fly for each query image.
    Intermediate outputs (detections, descriptors, matches) are stored in
    options.workspace_path and can be reused when options.skip_exists=True.

    Args:
        options: Localization options containing:
            - workspace_path: Directory for intermediate outputs
            - skip_exists: Reuse cached detections/matches if they exist
            - weight_path: Path to neural network weights
            - point_detection/matching: Point detector/matcher configuration
            - line_detection/matching: Line detector/matcher configuration
            - pose_estimation: C++ PointLineAbsolutePoseOptions
        holistic_recon: Database reconstruction with both points and lines
        query_images: Dict mapping query image ID to (image_path, camera)
        retrieval: Query image name -> list of database image names
        results_path: Path to write final localization results file
        id_to_name: Optional dict mapping image IDs to image names
            (used for retrieval lookup and results output). If None,
            uses image_path.name which may not match retrieval format.

    Returns:
        Mapping of query image IDs to estimated poses (pycolmap.Rigid3d)
    """
    workspace_path = Path(options.workspace_path)
    workspace_path.mkdir(parents=True, exist_ok=True)

    point_recon = holistic_recon.point_recon
    structure_recon = holistic_recon.structure_recon

    # Build name to image_id mapping for database images
    db_name_to_id = {
        image.name: image_id for image_id, image in point_recon.images.items()
    }

    # Build id_to_name mapping for query images if not provided
    if id_to_name is None:
        id_to_name = {
            qid: image_path.name
            for qid, (image_path, _) in query_images.items()
        }

    # Collect all query-neighbor pairs for matching
    logging.info("Collecting query-neighbor pairs...")
    query_neighbors: dict[int, list[int]] = {}
    for qid, (_query_image_path, _query_camera) in query_images.items():
        query_name = id_to_name[qid]
        if query_name not in retrieval:
            continue
        neighbor_names = retrieval[query_name]
        neighbor_ids = [
            db_name_to_id[n] for n in neighbor_names if n in db_name_to_id
        ]
        if neighbor_ids:
            query_neighbors[qid] = neighbor_ids

    if not query_neighbors:
        logging.warning("No valid query-neighbor pairs found")
        return {}

    # Run all point matching upfront
    logging.info("Running point feature extraction and matching...")
    point_matches = _match_all_points(
        options,
        query_images,
        query_neighbors,
        point_recon,
        workspace_path,
        id_to_name,
    )

    # Run all line matching upfront (if not points-only mode)
    line_matches: dict[int, dict] = {}
    if not options.use_points_only:
        logging.info("Running line detection and matching...")
        line_matches = _match_all_lines(
            options,
            query_images,
            query_neighbors,
            structure_recon,
            workspace_path,
            id_to_name,
        )

    # Localize each query image using pre-computed matches
    n_pts, n_lines = len(point_matches), len(line_matches)
    logging.info(
        f"Localizing query images... (points: {n_pts}, lines: {n_lines})"
    )
    final_poses: dict[int, pycolmap.Rigid3d] = {}
    # Only filled when the focal length is estimated instead of given.
    estimated_cameras: dict[int, pycolmap.Camera] = {}
    estimate_focal = options.pose_estimation.estimate_focal_length

    for qid in tqdm(query_neighbors.keys()):
        _query_image_path, query_camera = query_images[qid]

        # Get point correspondences from pre-computed matches
        p2ds, p3ds = _get_point_correspondences_from_matches(
            point_matches.get(qid, {}), point_recon
        )

        # Point-only mode: use pycolmap
        if options.use_points_only:
            if len(p2ds) < 4:
                continue
            # pycolmap writes the estimated focal into the camera it is given.
            camera = deepcopy(query_camera) if estimate_focal else query_camera
            ret = pycolmap.estimate_and_refine_absolute_pose(
                p2ds,
                p3ds,
                camera,
                estimation_options={
                    "ransac": {
                        "max_error": options.pose_estimation.max_error_point
                    },
                    "estimate_focal_length": estimate_focal,
                },
                refinement_options={"refine_focal_length": estimate_focal},
            )
            if ret is not None and ret["cam_from_world"] is not None:
                final_poses[qid] = ret["cam_from_world"]
                if estimate_focal:
                    estimated_cameras[qid] = camera
            continue

        # Get line correspondences from pre-computed matches
        l2ds, l3ds = _get_line_correspondences_from_matches(
            line_matches.get(qid, {}), structure_recon
        )

        # Check if we have enough correspondences
        if len(p2ds) + len(l2ds) < 4:
            logging.warning(
                f"Query {qid}: {len(p2ds)} points, {len(l2ds)} lines - skipping"
            )
            continue

        # Estimate pose using hybrid RANSAC
        result = estimate_point_line_absolute_pose(
            l3ds, l2ds, p3ds, p2ds, query_camera, options.pose_estimation
        )
        if result.success:
            final_poses[qid] = result.pose  # Already a pycolmap.Rigid3d
            if result.camera is not None:
                estimated_cameras[qid] = result.camera

    # Fill in identity poses for queries without estimates
    identity_pose = pycolmap.Rigid3d()
    for qid in query_images:
        if qid not in final_poses:
            final_poses[qid] = identity_pose

    # Write results
    _write_results(final_poses, id_to_name, results_path)

    n_localized = len([p for p in final_poses.values() if p != identity_pose])
    logging.info(
        f"Localization completed: {n_localized}/{len(query_images)} images"
    )
    if estimated_cameras:
        # focal_length is only defined for models with a single focal
        # parameter, so read the first one directly (PINHOLE has fx and fy).
        focals = np.array(
            [
                cam.params[cam.focal_length_idxs()[0]]
                for cam in estimated_cameras.values()
            ]
        )
        logging.info(
            f"Estimated focal length over {len(focals)} images: median "
            f"{np.median(focals):.1f}, 10-90% "
            f"[{np.percentile(focals, 10):.1f}, "
            f"{np.percentile(focals, 90):.1f}]"
        )
    return final_poses


def _match_all_points(
    options: PointLineLocalizationOptions,
    query_images: dict[int, tuple[Path, pycolmap.Camera]],
    query_neighbors: dict[int, list[int]],
    point_recon: pycolmap.Reconstruction,
    workspace_path: Path,
    id_to_name: dict[int, str],
) -> dict[int, dict]:
    """
    Match all query images against their database neighbors for points.

    Returns dict mapping query_id -> {keypoints, kp_idx_to_3D}
    """
    point_det_opts = options.point_detection
    point_match_opts = options.point_matching

    # Feature extraction config
    feature_conf = hloc.extract_features.confs.get(point_det_opts.method)
    if feature_conf is None:
        logging.warning(f"Unknown point detector: {point_det_opts.method}")
        return {}

    # Path to database features (extracted during triangulation)
    db_features_path = (
        workspace_path / "frontend" / f"{feature_conf['output']}.h5"
    )
    if not db_features_path.exists():
        # Without these there are no correspondences at all, and every
        # query would silently receive an identity pose.
        raise FileNotFoundError(
            f"Database features not found: {db_features_path}. "
            "They are written by triangulation, so check that it used the "
            "same point detector and that its workspace was kept "
            "(cleanup_frontend_workspace=False)."
        )

    localization_dir = workspace_path / "localization"
    localization_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    cache_file = localization_dir / "point_matches.npy"
    if options.skip_exists and cache_file.exists():
        logging.info("Loading cached point matches...")
        return limapio.read_npy(cache_file).item()

    # Collect all query image names (relative paths matching id_to_name format)
    query_image_list = [id_to_name[qid] for qid in query_neighbors]

    # Compute image_dir: base directory from which id_to_name paths are relative
    # E.g., if full_path = /path/to/7scenes/stairs/seq-01/frame.png
    #       and id_to_name = seq-01/frame.png
    #       then image_dir = /path/to/7scenes/stairs
    first_qid = next(iter(query_neighbors))
    first_full_path = str(query_images[first_qid][0])
    first_rel_path = id_to_name[first_qid]
    # Strip the relative path from the full path to get base directory
    image_dir = Path(first_full_path[: -len(first_rel_path)].rstrip("/"))

    # Extract features for all query images at once
    query_features_path = localization_dir / f"{feature_conf['output']}.h5"
    hloc.extract_features.main(
        feature_conf,
        image_dir,
        localization_dir,
        image_list=query_image_list,
    )

    # Build pairs file with ALL query-neighbor pairs
    pairs_dir = localization_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    pairs_file = pairs_dir / "all_pairs.txt"

    with open(pairs_file, "w") as f:
        for qid, neighbor_ids in query_neighbors.items():
            query_name = id_to_name[qid]
            for db_img_id in neighbor_ids:
                db_image = point_recon.images[db_img_id]
                if db_image.num_points3D == 0:
                    continue
                f.write(f"{query_name} {db_image.name}\n")

    # Get matcher config
    matcher_conf = hloc.match_features.confs.get(point_match_opts.method)
    if matcher_conf is None:
        matcher_conf = hloc.match_features.confs["superglue"]

    # Run matching for all pairs at once
    matches_path = hloc.match_features.main(
        matcher_conf,
        pairs_file,
        feature_conf["output"],
        localization_dir,
        features_ref=db_features_path,
    )

    # Read all query keypoints and collect matches
    result: dict[int, dict] = {}

    with (
        h5py.File(str(query_features_path), "r") as feat_file,
        h5py.File(str(matches_path), "r") as match_file,
    ):
        for qid, neighbor_ids in query_neighbors.items():
            query_name = id_to_name[qid]
            try:
                query_group = feat_file[query_name]
            except KeyError:
                continue

            kpq = query_group["keypoints"].__array__()
            kpq = kpq + 0.5  # COLMAP coordinates

            kp_idx_to_3D: dict[int, list[int]] = {}

            for db_img_id in neighbor_ids:
                db_image = point_recon.images[db_img_id]
                if db_image.num_points3D == 0:
                    continue
                db_name = db_image.name

                # Get point3D IDs for database image
                # Note: has_point3D is a method, not a property
                points3D_ids = np.array(
                    [
                        p.point3D_id if p.has_point3D() else -1
                        for p in db_image.points2D
                    ]
                )

                # Find the pair in h5 file (hloc uses nested structure)
                # Structure: match_file[key0][key1]["matches0"]
                key0 = query_name.replace("/", "-")
                key1 = db_name.replace("/", "-")

                if key0 in match_file and key1 in match_file[key0]:
                    matches = match_file[key0][key1]["matches0"].__array__()
                elif key1 in match_file and key0 in match_file[key1]:
                    # Reversed order - need to swap match indices
                    matches = match_file[key1][key0]["matches0"].__array__()
                    # matches0 gives idx1 for each idx0, invert it
                    valid = matches != -1
                    inverted = -np.ones_like(matches)
                    inverted[matches[valid]] = np.where(valid)[0]
                    matches = inverted
                else:
                    continue

                idx = np.where(matches != -1)[0]
                if len(idx) == 0:
                    continue
                matches = np.stack([idx, matches[idx]], -1)

                # Filter to matches with valid 3D points
                matches = matches[points3D_ids[matches[:, 1]] != -1]

                for query_idx, db_idx in matches:
                    p3d_id = points3D_ids[db_idx]
                    if query_idx not in kp_idx_to_3D:
                        kp_idx_to_3D[query_idx] = []
                    if p3d_id not in kp_idx_to_3D[query_idx]:
                        kp_idx_to_3D[query_idx].append(p3d_id)

            result[qid] = {"keypoints": kpq, "kp_idx_to_3D": kp_idx_to_3D}

    # Save to cache
    limapio.save_npy(cache_file, result)
    return result


def _match_all_lines(
    options: PointLineLocalizationOptions,
    query_images: dict[int, tuple[Path, pycolmap.Camera]],
    query_neighbors: dict[int, list[int]],
    structure_recon,
    workspace_path: Path,
    id_to_name: dict[int, str],
) -> dict[int, dict]:
    """
    Match all query images against their database neighbors for lines.

    Returns dict mapping query_id -> {query_segs, matches}
    """
    localization_dir = workspace_path / "localization"
    localization_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    cache_file = localization_dir / "line_matches.npy"
    if options.skip_exists and cache_file.exists():
        logging.info("Loading cached line matches...")
        return limapio.read_npy(cache_file).item()

    line_det_opts = options.line_detection
    line_match_opts = options.line_matching

    # Get detector, extractor, and matcher
    detector = limap.image.line.get_detector(
        line_det_opts.detector_method, line_det_opts.detector_options
    )
    extractor = limap.image.line.get_extractor(
        line_det_opts.extractor_method, line_det_opts.extractor_options
    )
    matcher = limap.image.line.get_matcher(
        line_match_opts.method, line_match_opts.matching_options, extractor
    )

    # Database descriptor folder path
    descinfo_folder = (
        workspace_path
        / "frontend"
        / "line_detections"
        / line_det_opts.detector_method
        / "descinfos"
        / line_det_opts.extractor_method
    )

    if not descinfo_folder.exists():
        raise FileNotFoundError(
            f"Line descriptors not found: {descinfo_folder}. "
            "They are written by triangulation, so check that it used the "
            "same line detector and extractor and that its workspace was "
            "kept (cleanup_frontend_workspace=False)."
        )

    result: dict[int, dict] = {}

    # Process each query image
    for qid, neighbor_ids in tqdm(
        query_neighbors.items(), desc="Line matching"
    ):
        query_image_path = query_images[qid][0]

        # Detect and extract query lines
        query_segs = detector.detect(query_image_path)
        query_descinfo = extractor.extract(query_image_path, query_segs)

        # Match against each database neighbor
        all_matches: dict[int, np.ndarray] = {}
        for db_img_id in neighbor_ids:
            db_descinfo = matcher.read_descinfo(descinfo_folder, db_img_id)
            all_matches[db_img_id] = matcher.match_pair(
                query_descinfo, db_descinfo
            )

        result[qid] = {"query_segs": query_segs, "matches": all_matches}

    # Save to cache
    limapio.save_npy(cache_file, result)

    return result


def _get_point_correspondences_from_matches(
    point_match_data: dict,
    point_recon: pycolmap.Reconstruction,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert point match data to 2D-3D correspondences."""
    if not point_match_data:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3)

    kpq = point_match_data["keypoints"]
    kp_idx_to_3D = point_match_data["kp_idx_to_3D"]

    p2ds = []
    p3ds = []
    for kp_idx in kp_idx_to_3D:
        for p3d_id in kp_idx_to_3D[kp_idx]:
            # Convert to int (numpy saves as float64, but dict keys are int)
            p3d_id = int(p3d_id)
            if p3d_id in point_recon.points3D:
                p2ds.append(kpq[kp_idx])
                p3ds.append(point_recon.points3D[p3d_id].xyz)

    return np.array(p2ds).reshape(-1, 2), np.array(p3ds).reshape(-1, 3)


def _get_line_correspondences_from_matches(
    line_match_data: dict,
    structure_recon,
) -> tuple[list[Line2d], list[Line3d]]:
    """Convert line match data to 2D-3D correspondences.

    Unlike points, we keep ALL line correspondences including duplicates.
    The same query line can match multiple 3D lines, and the same 3D line
    can be matched by multiple query lines. This matches the reference
    implementation behavior.
    """
    if not line_match_data:
        return [], []

    query_segs = line_match_data["query_segs"]
    all_matches = line_match_data["matches"]

    l2ds: list[Line2d] = []
    l3ds: list[Line3d] = []

    for db_img_id, matches in all_matches.items():
        if len(matches) == 0:
            continue

        # Get database 2D lines for this image
        db_structure2d = structure_recon.structure2d(db_img_id)
        db_lines = db_structure2d.lines

        for query_idx, db_idx in matches:
            if db_idx >= len(db_lines):
                continue

            db_line2d = db_lines[db_idx]
            line3d_id = db_line2d.line3D_id

            # Skip if no associated 3D line
            if line3d_id < 0:
                continue

            if not structure_recon.exists_line3D(line3d_id):
                continue

            line3d = structure_recon.line(line3d_id)

            # Create query 2D line from detected segments
            query_seg = query_segs[query_idx][:4]
            query_line2d = Line2d(query_seg.reshape(2, 2).astype(np.float64))

            # Keep all correspondences (no deduplication)
            l2ds.append(query_line2d)
            l3ds.append(line3d)

    return l2ds, l3ds


def _write_results(
    final_poses: dict[int, pycolmap.Rigid3d],
    id_to_name: dict[int, str],
    results_path: Path,
) -> None:
    """Write localization results to file in COLMAP format."""
    results_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for qid, pose in final_poses.items():
        name = id_to_name[qid]
        # Results use the COLMAP/hloc convention QW QX QY QZ, while
        # pycolmap orders quaternions as [x, y, z, w].
        qx, qy, qz, qw = pose.rotation.quat
        q = [qw, qx, qy, qz]
        t = pose.translation
        line = (
            " ".join([name] + [str(x) for x in q] + [str(x) for x in t]) + "\n"
        )
        lines.append(line)

    with open(results_path, "w") as f:
        f.writelines(lines)

    logging.info(f"Results written to {results_path}")
