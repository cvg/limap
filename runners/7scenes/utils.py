"""Utility functions for 7scenes localization.

This module provides dataset-specific utilities for the 7scenes dataset,
including depth reading and SfM correction.
"""

from pathlib import Path

import numpy as np
import PIL.Image
import pycolmap
import torch
from pycolmap import logging


class DepthReader:
    """Reader for 7scenes rendered depth maps."""

    def __init__(self, filename: str, depth_folder: Path):
        self.filename = filename
        self.depth_folder = depth_folder

    def read(self) -> np.ndarray:
        """Read depth map and convert to meters."""
        depth = PIL.Image.open(self.depth_folder / self.filename)
        depth = np.array(depth).astype("float64")
        depth = depth / 1000.0  # mm to meter
        depth[(depth == 0.0) | (depth > 1000.0)] = np.inf
        return depth


def image_path_to_rendered_depth_path(image_name: str) -> str:
    """Convert image path to corresponding rendered depth path."""
    parts = image_name.split("/")
    name = "_".join(["".join(parts[0].split("-")), parts[1]])
    name = name.replace("color", "pose")
    name = name.replace("png", "depth.tiff")
    return name


def get_train_test_ids(
    recon: pycolmap.Reconstruction,
    test_list: Path,
) -> tuple[list[int], list[int]]:
    """
    Split image IDs into train and test sets based on test list file.

    Args:
        recon: COLMAP reconstruction containing all images
        test_list: Path to file listing test image names

    Returns:
        Tuple of (train_ids, test_ids)
    """
    with open(test_list) as f:
        test_names = set(f.read().rstrip().split("\n"))

    train_ids, test_ids = [], []
    for img_id, image in recon.images.items():
        if image.name in test_names:
            test_ids.append(img_id)
        else:
            train_ids.append(img_id)

    return train_ids, test_ids


def get_query_images(
    full_recon: pycolmap.Reconstruction,
    query_ids: list[int],
    image_dir: Path,
    uncalibrated: bool = False,
    focal_init_factor: float = 1.2,
) -> dict[int, tuple[Path, pycolmap.Camera]]:
    """
    Create query images dict: {image_id: (image_path, camera)}.

    With uncalibrated, the ground-truth focal length is withheld and
    replaced by focal_init_factor * max(width, height), the guess COLMAP
    makes for an unknown camera. The principal point is kept, since the
    estimator needs it to unproject, and the focal serves as the starting
    point for estimating the real one.

    Args:
        full_recon: COLMAP reconstruction containing all images
        query_ids: List of query image IDs
        image_dir: Directory containing images
        uncalibrated: Withhold the ground-truth focal length
        focal_init_factor: Initial focal as a factor of the larger side

    Returns:
        Dict mapping query image ID to (image_path, camera)
    """
    query_images = {}
    for img_id in query_ids:
        image = full_recon.images[img_id]
        camera = full_recon.cameras[image.camera_id]
        if uncalibrated:
            camera = pycolmap.Camera(
                model="SIMPLE_PINHOLE",
                width=camera.width,
                height=camera.height,
                params=[
                    focal_init_factor * max(camera.width, camera.height),
                    camera.principal_point_x,
                    camera.principal_point_y,
                ],
            )
        query_images[img_id] = (image_dir / image.name, camera)
    return query_images


def build_db_model(
    gt_dir: Path,
    test_list: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    """
    Build the localization database model: database images only.

    The 7Scenes ground-truth reconstruction covers every frame of a
    scene, queries included. Query poses must be estimated from a map
    built without them, so this writes a copy with the query images and
    their observations removed.

    Args:
        gt_dir: Ground-truth reconstruction of the scene
        test_list: File listing the query image names (``list_test.txt``)
        output_dir: Directory to write the database model into
        overwrite: Rebuild even if a model is already cached there

    Returns:
        Path to the database model (== output_dir)
    """
    output_dir = Path(output_dir)
    with open(test_list) as f:
        test_names = set(f.read().rstrip().split("\n"))

    if (output_dir / "images.bin").exists() and not overwrite:
        # A model cached for a different query list is not reusable.
        cached = pycolmap.Reconstruction(output_dir)
        stale = [
            im.name for im in cached.images.values() if im.name in test_names
        ]
        if not stale:
            logging.info(f"Reusing existing database model: {output_dir}")
            return output_dir
        logging.warning(
            f"Cached database model at {output_dir} contains {len(stale)} "
            "of the current query images; rebuilding."
        )

    recon = pycolmap.Reconstruction(gt_dir)
    _, query_ids = get_train_test_ids(recon, test_list)
    for img_id in query_ids:
        image = recon.images[img_id]
        # Images are removed a whole frame at a time.
        frame = recon.frames[image.frame_id]
        if len(frame.data_ids) > 1:
            raise NotImplementedError(
                f"query image {image.name} shares a frame with "
                f"{len(frame.data_ids) - 1} other image(s), which would be "
                "removed with it; multi-camera rigs are not supported."
            )
        for idx in range(len(image.points2D)):
            # Deleting an observation can drop the 3D point, clearing it
            # from other keypoints too, so re-check on every index.
            if image.point2D(idx).has_point3D():
                recon.delete_observation(img_id, idx)
        recon.deregister_frame(image.frame_id)
    recon.tear_down()

    left = [im.name for im in recon.images.values() if im.name in test_names]
    assert not left, f"{len(left)} query images left in the database model"

    output_dir.mkdir(parents=True, exist_ok=True)
    recon.write(output_dir)
    logging.info(
        f"Database model written to {output_dir}: "
        f"{recon.num_images()} images, {recon.num_points3D()} points"
    )
    return output_dir


def interpolate_depth(
    depth: np.ndarray, kp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate depth values at keypoint locations.

    Args:
        depth: Depth image (H, W)
        kp: Keypoint locations (N, 2) in pixel coordinates

    Returns:
        Tuple of (interpolated_depth, valid_mask)
    """
    h, w = depth.shape
    kp = kp / np.array([[w - 1, h - 1]]) * 2 - 1
    assert np.all(kp > -1) and np.all(kp < 1)

    depth_tensor = torch.from_numpy(depth)[None, None]
    kp_tensor = torch.from_numpy(kp)[None, None]

    # Bilinear interpolation first, then nearest for remaining points
    interp_lin = torch.nn.functional.grid_sample(
        depth_tensor, kp_tensor, align_corners=True, mode="bilinear"
    )[0, :, 0]
    interp_nn = torch.nn.functional.grid_sample(
        depth_tensor, kp_tensor, align_corners=True, mode="nearest"
    )[0, :, 0]

    interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
    valid = ~torch.any(torch.isnan(interp), 0)

    interp_depth = interp.T.numpy().flatten()
    valid = valid.numpy()
    return interp_depth, valid


def correct_sfm_with_gt_depth(
    sfm_path: Path,
    depth_folder_path: Path,
    output_path: Path,
) -> Path:
    """
    Correct SfM 3D points using ground truth depth maps.

    Args:
        sfm_path: Path to input COLMAP reconstruction
        depth_folder_path: Path to folder with depth images
        output_path: Path to write corrected reconstruction

    Returns:
        Path to corrected reconstruction
    """
    from tqdm import tqdm

    recon = pycolmap.Reconstruction(sfm_path)
    logging.info("Correcting SfM using depth...")

    for _img_id, image in tqdm(recon.images.items()):
        image_name = image.name
        depth_name = image_path_to_rendered_depth_path(image_name)

        depth = PIL.Image.open(depth_folder_path / depth_name)
        depth = np.array(depth).astype("float64")
        depth = depth / 1000.0  # mm to meter
        depth[(depth == 0.0) | (depth > 1000.0)] = np.nan

        camera = recon.cameras[image.camera_id]
        cam_from_world = image.cam_from_world
        world_from_cam = cam_from_world.inverse()

        # Get 3D point IDs observed in this image
        p3D_ids = []
        p3Ds = []
        for p2d in image.points2D:
            if p2d.has_point3D():
                p3D_ids.append(p2d.point3D_id)
                p3Ds.append(recon.points3D[p2d.point3D_id].xyz)

        if not p3Ds:
            continue

        p3Ds = np.stack(p3Ds, 0)

        # Project 3D points to camera frame and then to image
        p3Ds_cam = np.array([cam_from_world * p for p in p3Ds])
        visible = p3Ds_cam[:, 2] > 1e-4
        p2Ds_all = camera.img_from_cam(p3Ds_cam[:, :2] / p3Ds_cam[:, 2:])

        # Check bounds
        pad = 1
        in_bounds = (
            (p2Ds_all[:, 0] >= pad)
            & (p2Ds_all[:, 0] <= camera.width - pad - 1)
            & (p2Ds_all[:, 1] >= pad)
            & (p2Ds_all[:, 1] <= camera.height - pad - 1)
        )
        valids_projected = visible & in_bounds
        p2Ds = p2Ds_all[valids_projected]

        if len(p2Ds) == 0:
            continue

        # Interpolate depth and backproject
        interp_depth, valids_backprojected = interpolate_depth(depth, p2Ds)

        # Compute scene coordinates for valid points
        if np.any(valids_backprojected):
            valid_p2Ds = p2Ds[valids_backprojected]
            valid_depth = interp_depth[valids_backprojected]

            # Backproject: 2D -> normalized -> 3D in camera -> 3D in world
            p2D_norm = camera.cam_from_img(valid_p2Ds)
            p3D_cam = np.column_stack(
                [p2D_norm * valid_depth[:, None], valid_depth]
            )
            p3D_world = np.array([world_from_cam * p for p in p3D_cam])

            # Update 3D points
            valid_p3D_ids = np.array(p3D_ids)[valids_projected][
                valids_backprojected
            ]
            for i, p3d_id in enumerate(valid_p3D_ids):
                point = recon.points3D[p3d_id]
                point.xyz = p3D_world[i]

    output_path.mkdir(parents=True, exist_ok=True)
    recon.write(output_path)
    logging.info(f"Corrected SfM written to {output_path}")
    return output_path


def evaluate(
    gt_dir: Path,
    results_file: Path,
    test_list: Path,
    only_localized: bool = False,
) -> dict:
    """
    Evaluate localization results against ground truth.

    Args:
        gt_dir: Path to COLMAP ground truth reconstruction
        results_file: Path to results file (name qw qx qy qz tx ty tz)
        test_list: Path to file listing test image names
        only_localized: If True, only evaluate successfully localized images

    Returns:
        Dict with evaluation metrics
    """
    # Load ground truth
    gt_recon = pycolmap.Reconstruction(gt_dir)

    # Build name -> GT pose mapping
    gt_poses = {}
    for image in gt_recon.images.values():
        gt_poses[image.name] = image.cam_from_world()

    # Load test list
    with open(test_list) as f:
        test_names = set(f.read().rstrip().split("\n"))

    # Load results
    results = {}
    with open(results_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            name = parts[0]
            # File holds QW QX QY QZ; pycolmap wants [x, y, z, w].
            qw, qx, qy, qz = (float(x) for x in parts[1:5])
            tvec = np.array([float(x) for x in parts[5:8]])
            rotation = pycolmap.Rotation3d(np.array([qx, qy, qz, qw]))
            results[name] = pycolmap.Rigid3d(rotation, tvec)

    # Compute errors
    errors_t = []
    errors_r = []
    localized = 0
    total = 0

    for name in sorted(test_names):
        total += 1
        if name not in gt_poses:
            continue
        if name not in results:
            continue

        gt_pose = gt_poses[name]
        est_pose = results[name]

        # Check if identity pose (failed localization). pycolmap orders
        # quaternions as [x, y, z, w], so identity is [0, 0, 0, 1].
        is_identity = np.allclose(
            est_pose.rotation.quat, [0, 0, 0, 1], atol=1e-6
        ) and np.allclose(est_pose.translation, [0, 0, 0], atol=1e-6)

        if is_identity:
            if only_localized:
                continue
            continue

        localized += 1

        # Translation error (in meters)
        # GT and Est are both cam_from_world
        # Position = -R^T * t
        gt_pos = -(gt_pose.rotation.inverse().matrix() @ gt_pose.translation)
        est_pos = -(est_pose.rotation.inverse().matrix() @ est_pose.translation)
        error_t = np.linalg.norm(gt_pos - est_pos)

        # Rotation error (in degrees)
        R_rel = est_pose.rotation * gt_pose.rotation.inverse()
        trace = np.trace(R_rel.matrix())
        error_r = np.arccos(np.clip((trace - 1) / 2, -1, 1)) * 180 / np.pi

        errors_t.append(error_t)
        errors_r.append(error_r)

    if len(errors_t) == 0:
        logging.warning("No images were successfully localized!")
        return {"localized": 0, "total": total}

    errors_t = np.array(errors_t)
    errors_r = np.array(errors_r)

    # Compute statistics
    med_t = np.median(errors_t)
    med_r = np.median(errors_r)
    mean_t = np.mean(errors_t)
    mean_r = np.mean(errors_r)

    # Compute % within thresholds (typical 7scenes thresholds)
    thresholds = [(0.05, 5), (0.1, 5), (0.25, 10)]
    pct_within = []
    for th_t, th_r in thresholds:
        within = np.sum((errors_t < th_t) & (errors_r < th_r))
        pct = 100.0 * within / len(errors_t)
        pct_within.append((th_t, th_r, pct))

    logging.info("\nResults Summary:")
    logging.info(
        f"  Localized: {localized}/{total} ({100 * localized / total:.1f}%)"
    )
    logging.info(f"  Median error: {med_t:.4f}m, {med_r:.2f}deg")
    logging.info(f"  Mean error: {mean_t:.4f}m, {mean_r:.2f}deg")
    for th_t, th_r, pct in pct_within:
        logging.info(f"  Within ({th_t}m, {th_r}deg): {pct:.1f}%")

    return {
        "localized": localized,
        "total": total,
        "median_t": med_t,
        "median_r": med_r,
        "mean_t": mean_t,
        "mean_r": mean_r,
        "errors_t": errors_t,
        "errors_r": errors_r,
    }
