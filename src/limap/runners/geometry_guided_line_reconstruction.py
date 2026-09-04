"""Geometry-guided line reconstruction from depth maps or point clouds."""

import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pycolmap
from pycolmap import logging
from tqdm import tqdm
from typeguard import typechecked

import limap.estimators.bundle_adjustment as ba
import limap.estimators.line3d
import limap.geometry
import limap.image.line
import limap.runners
import limap.util.io as limapio
from limap.image.specs import LineDetectionOptions
from limap.scene import (
    BaseDepthReader,
    BasePointCloudReader,
    HolisticReconstruction,
    Structure2d,
    Wireframe2d,
)
from limap.sfm import LineMergingOptions
from limap.estimators.bundle_adjustment import PointLineBundleAdjustmentOptions
from limap.util.types import Ranges

from .pipeline_steps import MetaInfoComputerOptions


@dataclass
class LineFittingOptions:
    """Options for 3D line fitting from depth/point-cloud."""

    cache_file: Path | None = None
    var2d: float | None = None  # 2D variance; if None, derived from detector
    ransac_th: float = 0.75
    min_inlier_ratio: float = 0.9
    n_jobs: int = 4


@dataclass
class GeometryGuidedLineReconstructionOptions:
    """Top-level options for geometry-guided line reconstruction."""

    # Top-level options
    max_image_dim: int | None = None
    weight_path: Path | None = None
    skip_exists: bool = False
    n_visible_views: int = 4
    n_neighbors: int = 100
    run_bundle_adjustment: bool = True

    # Nested options
    metainfo: MetaInfoComputerOptions = field(
        default_factory=MetaInfoComputerOptions
    )
    _line_detection: LineDetectionOptions = field(
        default_factory=LineDetectionOptions
    )
    _fitting: LineFittingOptions = field(default_factory=LineFittingOptions)
    merging: LineMergingOptions = field(default_factory=LineMergingOptions)
    bundle_adjustment: PointLineBundleAdjustmentOptions = field(
        default_factory=PointLineBundleAdjustmentOptions
    )

    @property
    def line_detection(self) -> LineDetectionOptions:
        options = deepcopy(self._line_detection)
        options.skip_exists = self.skip_exists
        if self.weight_path is not None:
            options.weight_path = self.weight_path
        return options

    @line_detection.setter
    def line_detection(self, opts: LineDetectionOptions) -> None:
        self._line_detection = deepcopy(opts)

    @property
    def fitting(self) -> LineFittingOptions:
        from limap.image.line import get_uncertainty2d

        options = deepcopy(self._fitting)
        if options.var2d is None:
            options.var2d = get_uncertainty2d(
                self._line_detection.detector_method
            )
        return options

    @fitting.setter
    def fitting(self, opts: LineFittingOptions) -> None:
        self._fitting = deepcopy(opts)


def _setup_and_preprocess(
    max_image_dim: int | None,
    image_dir: Path,
    model_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Setup output directory, validate reconstruction, undistort and resize."""
    output_dir.mkdir(parents=True, exist_ok=True)
    recon = pycolmap.Reconstruction(model_dir)
    if not limap.runners.check_valid_reconstruction(image_dir, recon):
        logging.fatal("COLMAP reconstruction does not match the images")
    logging.info(f"[LOG] Number of images: {recon.num_images()}")

    image_dir, model_dir = limap.runners.undistort_images(
        image_dir, model_dir, output_dir / "undistorted"
    )

    if max_image_dim is not None:
        limap.runners.resize_images_to_max_dim(
            image_dir, model_dir, max_image_dim
        )

    return image_dir, model_dir


def _maybe_compute_metainfos(
    options: MetaInfoComputerOptions,
    image_dir: Path,
    model_dir: Path,
    output_dir: Path,
    neighbors: dict[int, list[int]] | None,
    ranges: Ranges | None,
) -> tuple[dict[int, list[int]], Ranges]:
    """Compute neighboring pairs and depth ranges if not provided."""
    if neighbors is not None:
        return neighbors, ranges

    logging.info(
        "Neighboring pairs not provided. "
        "Computing from COLMAP point triangulation..."
    )
    recon = pycolmap.Reconstruction(model_dir)
    point_triangulation_dir = (
        output_dir / "calculate_neighbors" / "point_triangulation"
    )
    target_image_dir = point_triangulation_dir / "images"
    target_model_dir = point_triangulation_dir / "sparse"
    if not target_model_dir.exists():
        if recon.num_points3D() <= 100:
            limap.runners.automatic_point_triangulation(
                image_dir,
                model_dir,
                point_triangulation_dir,
                options.point_triangulation,
            )
        else:
            shutil.copytree(image_dir, target_image_dir, dirs_exist_ok=True)
            shutil.copytree(model_dir, target_model_dir, dirs_exist_ok=True)
    metainfo_options = deepcopy(options)
    metainfo_options.cache_file = output_dir / "metainfos.txt"
    return limap.runners.compute_metainfos(
        point_triangulation_dir, metainfo_options
    )


def _detect_2d_lines(
    options: LineDetectionOptions,
    recon: pycolmap.Reconstruction,
    image_dir: Path,
    output_dir: Path,
) -> dict[int, list[limap.geometry.Line2d]]:
    """Detect 2D lines in all images and return as Line2d objects."""
    image_paths = {
        image.image_id: image_dir / image.name
        for image in recon.images.values()
    }
    workspace_path = output_dir / "frontend"
    all_2d_segs, _ = limap.image.line.line_detection(
        image_paths, workspace_path, options
    )
    # Convert numpy arrays to Line2d objects
    all_lines2d = {}
    for img_id, segs in all_2d_segs.items():
        all_lines2d[img_id] = limap.geometry.get_line2d_vector_from_array(segs)
    return all_lines2d


def _fit_3d_segs_from_depth(
    options: LineFittingOptions,
    all_lines2d: dict[int, list[limap.geometry.Line2d]],
    recon: pycolmap.Reconstruction,
    depth_readers: dict[int, BaseDepthReader],
) -> dict[int, list[limap.geometry.Line3d | None]]:
    """Fit 3D segments from 2D lines using depth maps."""
    if options.cache_file and options.cache_file.exists():
        logging.info(f"Loading cached fitted lines from {options.cache_file}")
        return limapio.read_npy(options.cache_file).item()

    def process(img_id):
        image = recon.images[img_id]
        camera = recon.cameras[image.camera_id]
        lines2d = all_lines2d[img_id]
        depth = depth_readers[img_id].read_depth(
            img_hw=[camera.height, camera.width]
        )
        lines3d = []
        for line2d in lines2d:
            seg2d = line2d.as_array().flatten()  # [x0, y0, x1, y1]
            line3d = limap.estimators.line3d.estimate_seg3d_from_depth(
                seg2d,
                depth,
                image,
                ransac_th=options.ransac_th,
                min_inlier_ratio=options.min_inlier_ratio,
                var2d=options.var2d,
            )
            lines3d.append(line3d)
        return lines3d

    image_ids = list(all_lines2d.keys())
    results = joblib.Parallel(n_jobs=options.n_jobs, prefer="threads")(
        joblib.delayed(process)(img_id) for img_id in tqdm(image_ids)
    )
    all_lines3d = {
        image_ids[idx]: lines3d for idx, lines3d in enumerate(results)
    }

    if options.cache_file:
        limapio.save_npy(options.cache_file, all_lines3d)
    return all_lines3d


def _fit_3d_segs_from_point_cloud(
    options: LineFittingOptions,
    all_lines2d: dict[int, list[limap.geometry.Line2d]],
    recon: pycolmap.Reconstruction,
    p3d_readers: dict[int, BasePointCloudReader],
) -> dict[int, list[limap.geometry.Line3d | None]]:
    """Fit 3D segments from 2D lines using point clouds."""
    if options.cache_file and options.cache_file.exists():
        logging.info(f"Loading cached fitted lines from {options.cache_file}")
        return limapio.read_npy(options.cache_file).item()

    def process(img_id):
        image = recon.images[img_id]
        lines2d = all_lines2d[img_id]
        p3ds = p3d_readers[img_id].read_point_cloud()
        lines3d = []
        for line2d in lines2d:
            seg2d = line2d.as_array().flatten()  # [x0, y0, x1, y1]
            line3d = limap.estimators.line3d.estimate_seg3d_from_points3d(
                seg2d,
                p3ds,
                image,
                ransac_th=options.ransac_th,
                min_inlier_ratio=options.min_inlier_ratio,
                var2d=options.var2d,
            )
            lines3d.append(line3d)
        return lines3d

    image_ids = list(all_lines2d.keys())
    results = joblib.Parallel(n_jobs=options.n_jobs, prefer="threads")(
        joblib.delayed(process)(img_id) for img_id in tqdm(image_ids)
    )
    all_lines3d = {
        image_ids[idx]: lines3d for idx, lines3d in enumerate(results)
    }

    if options.cache_file:
        limapio.save_npy(options.cache_file, all_lines3d)
    return all_lines3d


def _create_holistic_reconstruction(
    recon: pycolmap.Reconstruction,
    all_lines2d: dict[int, list[limap.geometry.Line2d]],
    all_lines3d: dict[int, list[limap.geometry.Line3d | None]],
) -> HolisticReconstruction:
    """Create HolisticReconstruction from fitted 2D/3D lines.

    Each valid 2D-3D line pair becomes a separate Line3d with a single
    observation in its track.
    """
    holistic_recon = HolisticReconstruction(recon)
    structure_recon = holistic_recon.structure_recon

    line3d_id = 0
    for img_id in all_lines2d:
        lines2d = all_lines2d[img_id]
        lines3d = all_lines3d[img_id]

        # Must precede add_line3D, which writes the 2D->3D link back here.
        structure2d = Structure2d(lines2d, [], Wireframe2d())
        structure_recon.add_structure2d(img_id, structure2d)

        for line_idx, (_, line3d) in enumerate(
            zip(lines2d, lines3d, strict=False)
        ):
            if line3d is None:
                continue

            # Add track element to Line3d
            track_element = pycolmap.TrackElement(img_id, line_idx)
            line3d.track.add_element(track_element)

            # Add Line3d to structure reconstruction
            structure_recon.add_line3D(line3d_id, line3d)
            line3d_id += 1

    logging.info(f"Created {line3d_id} Line3d from fitted segments")
    return holistic_recon


def _merge_lines3d(
    holistic_recon: HolisticReconstruction,
    neighbors: dict[int, list[int]],
    options: GeometryGuidedLineReconstructionOptions,
) -> HolisticReconstruction:
    """Merge Line3d observations across multiple views.

    Uses 2D+3D similarity checks via LineLinker to combine Line3d
    that correspond to the same physical line in 3D space.
    """
    from limap._limap import _sfm

    # Set min_visible_views from top-level options
    merge_opts = options.merging
    merge_opts.min_visible_views = options.n_visible_views

    # Call C++ merging function directly with the pybind-ed options
    _sfm.merge_fitted_lines_3d(holistic_recon, neighbors, merge_opts)

    return holistic_recon


def _run_bundle_adjustment(
    options: ba.PointLineBundleAdjustmentOptions,
    holistic_recon: HolisticReconstruction,
) -> HolisticReconstruction:
    """Run bundle adjustment for line optimization.

    For geometry-guided reconstruction, we only optimize lines since points
    come from the input reconstruction (COLMAP/depth) and cameras are fixed.
    """
    logging.info("Running bundle adjustment...")

    # Setup BA options - only refine lines, not points or cameras
    ba_options = deepcopy(options)
    ba_options.refine_focal_length = False
    ba_options.refine_principal_point = False
    ba_options.refine_extra_params = False
    ba_options.refine_points = False
    ba_options.refine_lines = True

    # Setup BA config - add images and lines
    ba_config = ba.PointLineBundleAdjustmentConfig()
    for line_id in holistic_recon.structure_recon.lines3D:
        ba_config.add_variable_line(line_id)

    # Run bundle adjustment
    adjuster = ba.create_point_line_bundle_adjuster(
        ba_options, ba_config, holistic_recon
    )
    summary = adjuster.solve()
    logging.info(
        f"Bundle adjustment completed: "
        f"initial_cost={summary.ceres_summary.initial_cost:.4f}, "
        f"final_cost={summary.ceres_summary.final_cost:.4f}"
    )
    return holistic_recon


@typechecked
def line_reconstruction_with_depth_maps(
    options: GeometryGuidedLineReconstructionOptions,
    image_dir: Path,
    model_dir: Path,
    output_dir: Path,
    depth_readers: dict[int, BaseDepthReader],
    neighbors: dict[int, list[int]] | None = None,
    ranges: Ranges | None = None,
) -> Path:
    """
    Line reconstruction from multi-view RGB images with depth maps.

    Args:
        options: Configuration options
        image_dir: Path to image directory
        model_dir: Path to COLMAP sparse model
        output_dir: Path for output
        depth_readers: Per-image depth map readers
        neighbors: Optional precomputed neighbors
        ranges: Optional precomputed 3D ranges

    Returns:
        Path to final output model directory
    """
    image_dir, model_dir = _setup_and_preprocess(
        options.max_image_dim, image_dir, model_dir, output_dir
    )
    neighbors, ranges = _maybe_compute_metainfos(
        options.metainfo, image_dir, model_dir, output_dir, neighbors, ranges
    )
    recon = pycolmap.Reconstruction(model_dir)
    all_lines2d = _detect_2d_lines(
        options.line_detection, recon, image_dir, output_dir
    )

    fitting_options = options.fitting
    if options.skip_exists and fitting_options.cache_file is None:
        fitting_options.cache_file = output_dir / "fitted_lines3d.npy"
    all_lines3d = _fit_3d_segs_from_depth(
        fitting_options, all_lines2d, recon, depth_readers
    )

    holistic_recon = _create_holistic_reconstruction(
        recon, all_lines2d, all_lines3d
    )
    holistic_recon = _merge_lines3d(holistic_recon, neighbors, options)

    if options.run_bundle_adjustment:
        holistic_recon = _run_bundle_adjustment(
            options.bundle_adjustment, holistic_recon
        )

    final_output_dir = output_dir / "geometry_guided_final_model"
    final_output_dir.mkdir(parents=True, exist_ok=True)
    holistic_recon.write(final_output_dir)

    logging.info(f"Line reconstruction completed: {holistic_recon}")
    return final_output_dir


@typechecked
def line_reconstruction_with_point_cloud(
    options: GeometryGuidedLineReconstructionOptions,
    image_dir: Path,
    model_dir: Path,
    output_dir: Path,
    p3d_readers: dict[int, BasePointCloudReader],
    neighbors: dict[int, list[int]] | None = None,
    ranges: Ranges | None = None,
) -> Path:
    """
    Line reconstruction from multi-view images with point clouds.

    Args:
        options: Configuration options
        image_dir: Path to image directory
        model_dir: Path to COLMAP sparse model
        output_dir: Path for output
        p3d_readers: Per-image point cloud readers
        neighbors: Optional precomputed neighbors
        ranges: Optional precomputed 3D ranges

    Returns:
        Path to final output model directory
    """
    image_dir, model_dir = _setup_and_preprocess(
        options.max_image_dim, image_dir, model_dir, output_dir
    )
    neighbors, ranges = _maybe_compute_metainfos(
        options.metainfo, image_dir, model_dir, output_dir, neighbors, ranges
    )
    recon = pycolmap.Reconstruction(model_dir)
    all_lines2d = _detect_2d_lines(
        options.line_detection, recon, image_dir, output_dir
    )

    fitting_options = options.fitting
    if options.skip_exists and fitting_options.cache_file is None:
        fitting_options.cache_file = output_dir / "fitted_lines3d.npy"
    all_lines3d = _fit_3d_segs_from_point_cloud(
        fitting_options, all_lines2d, recon, p3d_readers
    )

    holistic_recon = _create_holistic_reconstruction(
        recon, all_lines2d, all_lines3d
    )
    holistic_recon = _merge_lines3d(holistic_recon, neighbors, options)

    if options.run_bundle_adjustment:
        holistic_recon = _run_bundle_adjustment(
            options.bundle_adjustment, holistic_recon
        )

    final_output_dir = output_dir / "geometry_guided_final_model"
    final_output_dir.mkdir(parents=True, exist_ok=True)
    holistic_recon.write(final_output_dir)

    logging.info(f"Line reconstruction completed: {holistic_recon}")
    return final_output_dir
