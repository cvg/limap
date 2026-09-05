"""Structure frontend: detection and matching for lines and points."""

import shutil
import sqlite3
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import pycolmap
from pycolmap import logging
from typeguard import typechecked

import limap.image
import limap.runners
from limap.util.types import Ranges

from limap.image.specs import (
    ImageDescriptionOptions,
    ImageAssociationOptions,
)
from .pipeline_steps import MetaInfoComputerOptions


@dataclass
class StructureFrontendOptions:
    """Options for structure frontend."""

    max_image_dim: int | None = None
    # Camera initialization mode.
    # None: use intrinsics from the input model (create_db_from_model)
    # pycolmap.CameraMode: infer cameras from images/EXIF
    camera_mode: pycolmap.CameraMode | None = None
    metainfo: MetaInfoComputerOptions = field(
        default_factory=MetaInfoComputerOptions
    )
    image_description: ImageDescriptionOptions = field(
        default_factory=ImageDescriptionOptions
    )
    image_association: ImageAssociationOptions = field(
        default_factory=ImageAssociationOptions
    )
    # Remove intermediate frontend files (descriptors, detections, matchings)
    # after results are imported into databases. Saves significant disk space.
    cleanup_workspace: bool = True


@dataclass
class StructureFrontendOutputs:
    """Outputs from structure frontend."""

    db_path: Path
    structure_db_path: Path
    image_dir: Path
    neighbors: dict[int, list[int]]
    ranges: Ranges | None
    model_dir: Path | None = None


def databases_are_complete(
    db_path: Path,
    structure_db_path: Path,
    num_images: int,
) -> bool:
    """Check that the databases hold results for every image of the model.

    A run that was interrupted partway leaves both files on disk while
    still incomplete. Reusing one produces an empty reconstruction, so
    check the contents rather than only whether the files exist.
    """

    def count(path: Path, table: str) -> int:
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[
                    0
                ]
        except sqlite3.Error:
            return 0

    if count(db_path, "two_view_geometries") == 0:
        return False
    return count(structure_db_path, "structure2d") >= num_images


def cleanup_frontend_workspace(output_dir: Path) -> None:
    """Remove intermediate frontend files that are no longer needed.

    After detection and matching results are imported into database.db
    and structure_database.db, the intermediate files under
    output_dir/frontend/ (descriptors, detections, matchings) are no
    longer needed. This function deletes them to save disk space while
    preserving group description files needed for later stages.

    Args:
        output_dir: The output directory passed to the frontend.
    """
    frontend_dir = output_dir / "frontend"
    if not frontend_dir.exists():
        return

    # Directories to delete
    dirs_to_delete = [
        frontend_dir / "line_detections",
        frontend_dir / "line_matchings",
        frontend_dir / "joint_matchings",
        frontend_dir / "dense_warps",
        # Normal maps only needed for plane-to-group conversion
        frontend_dir / "group_description" / "plane_detections" / "normal_maps",
        frontend_dir
        / "group_description"
        / "plane_detections"
        / "moge_outputs",
    ]
    for d in dirs_to_delete:
        if d.exists():
            shutil.rmtree(d)
            logging.verbose(1, f"Removed {d}")

    # Files to delete: *.h5 descriptors/matches, pair lists
    for h5_file in frontend_dir.glob("*.h5"):
        h5_file.unlink()
        logging.verbose(1, f"Removed {h5_file}")

    pairs_file = frontend_dir / "pairs-from-neighbors.txt"
    if pairs_file.exists():
        pairs_file.unlink()
        logging.verbose(1, f"Removed {pairs_file}")

    logging.info("Frontend workspace cleaned up.")


def _run_detection_and_matching(
    options: StructureFrontendOptions,
    recon: pycolmap.Reconstruction,
    image_dir: Path,
    neighbors: dict[int, list[int]],
    output_dir: Path,
    db_path: Path,
    structure_db_path: Path,
) -> tuple[Path, Path]:
    """Shared detection and matching logic for both frontends."""
    workspace_path = output_dir / "frontend"

    limap.image.create_empty_databases(
        recon,
        db_path,
        structure_db_path,
        image_dir=image_dir,
        camera_mode=options.camera_mode,
    )
    point_descriptor_path, line_descriptor_path = limap.image.image_description(
        options.image_description,
        recon,
        image_dir,
        workspace_path,
        db_path,
        structure_db_path,
    )

    image_association_options = deepcopy(options.image_association)
    image_association_options.point_descriptor_path = point_descriptor_path
    image_association_options.line_descriptor_path = line_descriptor_path
    limap.image.image_association(
        image_association_options,
        options.image_description,
        recon,
        image_dir,
        neighbors,
        workspace_path,
        db_path,
        structure_db_path,
    )

    logging.info("Structure frontend complete.")
    logging.info(f"  - COLMAP database: {db_path}")
    logging.info(f"  - Structure database: {structure_db_path}")


@typechecked
def structure_frontend_from_model(
    options: StructureFrontendOptions,
    image_dir: Path,
    model_dir: Path,
    output_dir: Path,
    neighbors: dict[int, list[int]] | None = None,
    ranges: Ranges | None = None,
    skip_if_exists: bool = False,
) -> StructureFrontendOutputs:
    """Run structure frontend for a posed model on disk.

    Use this when images have known poses (e.g., for triangulation).
    The model is read from model_dir, undistorted, and optionally
    resized.

    Args:
        options: Frontend options
        image_dir: Path to image directory
        model_dir: Path to COLMAP model directory (with poses)
        output_dir: Output directory
        neighbors: Optional pre-computed neighbors
        ranges: Optional pre-computed ranges
        skip_if_exists: Skip if databases already exist

    Returns:
        StructureFrontendOutputs with paths to outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    recon = pycolmap.Reconstruction(model_dir)
    logging.info(f"[LOG] Number of images: {recon.num_images()}")

    # Undistort images
    image_dir, model_dir = limap.runners.undistort_images(
        image_dir,
        model_dir,
        output_dir / "undistorted",
    )
    # Resize images to maximum dimension
    if options.max_image_dim is not None:
        limap.runners.resize_images_to_max_dim(
            image_dir, model_dir, options.max_image_dim
        )

    # Validate after undistort + resize so dimensions match
    recon = pycolmap.Reconstruction(model_dir)
    if not limap.runners.check_valid_reconstruction(image_dir, recon):
        logging.fatal("COLMAP reconstruction does not match the images")

    # Get neighbors from covisibility
    if neighbors is None:
        logging.info(
            "Neighboring pairs not provided. "
            "Running exhaustive-pair point triangulation first "
            "to get accurate covisible pairs"
        )
        point_triangulation_dir = (
            output_dir / "calculate_neighbors" / "point_triangulation"
        )
        target_image_dir = point_triangulation_dir / "images"
        target_model_dir = point_triangulation_dir / "sparse"
        if not target_model_dir.exists():
            recon = pycolmap.Reconstruction(model_dir)
            if recon.num_points3D() <= 100:
                limap.runners.automatic_point_triangulation(
                    image_dir,
                    model_dir,
                    point_triangulation_dir,
                    options.metainfo.point_triangulation,
                )
            else:
                shutil.copytree(
                    image_dir,
                    target_image_dir,
                    dirs_exist_ok=True,
                )
                shutil.copytree(
                    model_dir,
                    target_model_dir,
                    dirs_exist_ok=True,
                )
        metainfo_options = deepcopy(options.metainfo)
        metainfo_options.cache_file = output_dir / "metainfos.txt"
        neighbors, ranges = limap.runners.compute_metainfos(
            point_triangulation_dir, metainfo_options
        )

    # Check if databases already exist
    recon = pycolmap.Reconstruction(model_dir)
    db_path = output_dir / "database.db"
    structure_db_path = output_dir / "structure_database.db"

    exists_db = db_path.exists() and structure_db_path.exists()
    if skip_if_exists and exists_db:
        if databases_are_complete(
            db_path, structure_db_path, recon.num_images()
        ):
            logging.info("Databases already exist, skipping frontend")
            return StructureFrontendOutputs(
                db_path=db_path,
                structure_db_path=structure_db_path,
                image_dir=image_dir,
                model_dir=model_dir,
                neighbors=neighbors,
                ranges=ranges,
            )
        logging.warning(
            "Existing databases do not cover every image of the model, "
            "which usually means an earlier run was interrupted. "
            "Rerunning detection and matching."
        )

    _run_detection_and_matching(
        options,
        recon,
        image_dir,
        neighbors,
        output_dir,
        db_path,
        structure_db_path,
    )

    if options.cleanup_workspace:
        cleanup_frontend_workspace(output_dir)

    return StructureFrontendOutputs(
        db_path=db_path,
        structure_db_path=structure_db_path,
        image_dir=image_dir,
        model_dir=model_dir,
        neighbors=neighbors,
        ranges=ranges,
    )


@typechecked
def structure_frontend_from_images(
    options: StructureFrontendOptions,
    image_dir: Path,
    recon: pycolmap.Reconstruction,
    output_dir: Path,
    neighbors: dict[int, list[int]] | None = None,
    skip_if_exists: bool = False,
) -> StructureFrontendOutputs:
    """Run structure frontend from an in-memory reconstruction.

    Use this when images do NOT have poses (e.g., for SfM from
    scratch). The reconstruction is passed in-memory to avoid
    COLMAP's binary format which drops unposed frames/images.

    Args:
        options: Frontend options
        image_dir: Path to image directory
        recon: In-memory Reconstruction with cameras and images
            (no poses required)
        output_dir: Output directory
        neighbors: Optional pre-computed neighbors
        skip_if_exists: Skip if databases already exist

    Returns:
        StructureFrontendOutputs with paths to outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"[LOG] Number of images: {recon.num_images()}")

    # Resize images before validation (in-memory, no disk model needed)
    if options.max_image_dim is not None:
        limap.runners.resize_images(image_dir, recon, options.max_image_dim)

    if not limap.runners.check_valid_reconstruction(image_dir, recon):
        logging.fatal("COLMAP reconstruction does not match the images")

    # Get neighbors: exhaustive pairs
    if neighbors is None:
        logging.info("No poses available. Using exhaustive matching.")
        image_ids = sorted(recon.images.keys())
        # Each undirected pair (a, b) is assigned to exactly
        # one side based on parity of (a+b), keeping the
        # per-image workload balanced.
        neighbors = {
            img_id: [
                other
                for other in image_ids
                if other != img_id
                and (other > img_id) == ((img_id + other) % 2 == 0)
            ]
            for img_id in image_ids
        }

    # Check if databases already exist
    db_path = output_dir / "database.db"
    structure_db_path = output_dir / "structure_database.db"

    exists_db = db_path.exists() and structure_db_path.exists()
    if (
        skip_if_exists
        and exists_db
        and not databases_are_complete(
            db_path, structure_db_path, recon.num_images()
        )
    ):
        logging.warning(
            "Existing databases do not cover every image of the model, "
            "which usually means an earlier run was interrupted. "
            "Rerunning detection and matching."
        )
        skip_if_exists = False
    if skip_if_exists and exists_db:
        logging.info("Databases already exist, skipping frontend")
        return StructureFrontendOutputs(
            db_path=db_path,
            structure_db_path=structure_db_path,
            image_dir=image_dir,
            neighbors=neighbors,
            ranges=None,
        )

    _run_detection_and_matching(
        options,
        recon,
        image_dir,
        neighbors,
        output_dir,
        db_path,
        structure_db_path,
    )

    if options.cleanup_workspace:
        cleanup_frontend_workspace(output_dir)

    return StructureFrontendOutputs(
        db_path=db_path,
        structure_db_path=structure_db_path,
        image_dir=image_dir,
        neighbors=neighbors,
        ranges=None,
    )
