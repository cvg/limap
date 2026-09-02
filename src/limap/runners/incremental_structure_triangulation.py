from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import pycolmap
from pycolmap import logging
from typeguard import typechecked

import limap.runners
import limap.sfm
import limap.estimators.bundle_adjustment
from limap.estimators.bundle_adjustment import (
    StructureBundleAdjustmentOptions,
)
from limap._limap import _sfm
from limap.scene import (
    HolisticReconstruction,
    StructureDatabase,
    StructureDatabaseCache,
)
from limap.util.types import Ranges

from limap.image.specs import ImageDescriptionOptions, ImageAssociationOptions
from .pipeline_steps import MetaInfoComputerOptions
from .structure_frontend import (
    StructureFrontendOptions,
    structure_frontend_from_model,
)


@dataclass
class IncrementalStructureTriangulationOptions:
    """Options for incremental structure triangulation."""

    max_image_dim: int | None = None
    skip_frontend_if_exists_database: bool = False
    run_bundle_adjustment: bool = True
    use_groups: bool = True

    metainfo: MetaInfoComputerOptions = field(
        default_factory=MetaInfoComputerOptions
    )
    _image_description: ImageDescriptionOptions = field(
        default_factory=ImageDescriptionOptions
    )
    _image_association: ImageAssociationOptions = field(
        default_factory=ImageAssociationOptions
    )

    # Triangulation options
    triangulate_points: bool = True
    triangulate_lines: bool = True
    triangulate_groups: bool = True
    point_triangulation: pycolmap.IncrementalTriangulatorOptions = field(
        default_factory=pycolmap.IncrementalTriangulatorOptions
    )
    line_triangulation: _sfm.IncrementalLineTriangulatorOptions = field(
        default_factory=_sfm.IncrementalLineTriangulatorOptions
    )
    group_triangulation: _sfm.IncrementalGroupTriangulatorOptions = field(
        default_factory=_sfm.IncrementalGroupTriangulatorOptions
    )

    # Post-triangulation operations
    complete_tracks: bool = True
    merge_tracks: bool = True

    # Bundle adjustment options
    bundle_adjustment: StructureBundleAdjustmentOptions = field(
        default_factory=StructureBundleAdjustmentOptions
    )
    group_filtering: limap.sfm.GroupVerificationOptions = field(
        default_factory=limap.sfm.GroupVerificationOptions
    )

    @property
    def image_description(self) -> ImageDescriptionOptions:
        return deepcopy(self._image_description)

    @image_description.setter
    def image_description(self, opts: ImageDescriptionOptions) -> None:
        self._image_description = deepcopy(opts)

    @property
    def image_association(self) -> ImageAssociationOptions:
        return deepcopy(self._image_association)

    @image_association.setter
    def image_association(self, opts: ImageAssociationOptions) -> None:
        self._image_association = deepcopy(opts)


@typechecked
def incremental_structure_triangulation(
    options: IncrementalStructureTriangulationOptions,
    image_dir: Path,
    model_dir: Path,
    output_dir: Path,
    db_path: Path | None = None,
    structure_db_path: Path | None = None,
    image_order: list[int] | None = None,
    neighbors: dict[int, list[int]] | None = None,
    ranges: Ranges | None = None,
) -> Path:
    """
    Incrementally triangulate points, lines, and groups for registered images.

    This function assumes images are already registered
    (poses known). It triangulates the structure (points,
    lines, groups) incrementally, one image at a time.

    Args:
        options: Configuration with triangulation options
        image_dir: Path to images
        model_dir: Path to COLMAP model (cameras.bin, images.bin with poses)
        output_dir: Output directory for results
        db_path: COLMAP database with point correspondences
            (default: model_dir/database.db)
        structure_db_path: Structure database with
            line/group matches (default: run frontend)
        image_order: Order to process images (default: sorted by image_id)
        neighbors: Optional image neighbors for frontend
        ranges: Optional depth ranges

    Returns:
        Path to final model directory containing HolisticReconstruction
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    #######################################
    # 1. Setup databases
    #######################################
    if db_path is None:
        db_path = model_dir / "database.db"

    # Run structure frontend if no structure database provided
    if structure_db_path is None:
        logging.info("Running structure frontend...")
        frontend_options = StructureFrontendOptions(
            max_image_dim=options.max_image_dim,
            metainfo=options.metainfo,
            image_description=options.image_description,
            image_association=options.image_association,
        )
        frontend_outputs = structure_frontend_from_model(
            frontend_options,
            image_dir,
            model_dir,
            output_dir,
            neighbors=neighbors,
            ranges=ranges,
            skip_if_exists=options.skip_frontend_if_exists_database,
        )
        db_path = frontend_outputs.db_path
        structure_db_path = frontend_outputs.structure_db_path
        image_dir = frontend_outputs.image_dir
        model_dir = frontend_outputs.model_dir
        ranges = frontend_outputs.ranges

    #######################################
    # 2. Load reconstruction and correspondence graphs
    #######################################
    recon = pycolmap.Reconstruction(model_dir)

    # Load COLMAP DatabaseCache (for points correspondence graph + keypoints)
    logging.info("Loading correspondence graphs...")
    with pycolmap.Database.open(db_path) as colmap_db:
        colmap_db_cache = pycolmap.DatabaseCache.create(
            colmap_db, pycolmap.DatabaseCacheOptions()
        )
    # Populate keypoints from database into reconstruction
    recon.load(colmap_db_cache)
    point_corr_graph = colmap_db_cache.correspondence_graph

    # Load structure database cache (for lines/groups)
    with StructureDatabase.open(structure_db_path) as structure_db:
        structure_db_cache = StructureDatabaseCache.create(structure_db)

    #######################################
    # 3. Create holistic reconstruction
    #######################################
    holistic_recon = HolisticReconstruction(recon)

    # Compute ranges if not provided
    if ranges is None:
        ranges = limap.runners.compute_ranges(
            model_dir, options.metainfo.range_calculator
        )

    #######################################
    # 4. Run incremental structure triangulation (single C++ call)
    #######################################
    tri_options = _sfm.IncrementalStructureTriangulatorOptions()
    tri_options.triangulate_points = options.triangulate_points
    tri_options.triangulate_lines = options.triangulate_lines
    tri_options.triangulate_groups = (
        options.triangulate_groups and options.use_groups
    )
    tri_options.complete_tracks = options.complete_tracks
    tri_options.merge_tracks = options.merge_tracks
    tri_options.point_options = options.point_triangulation
    tri_options.line_options = options.line_triangulation
    tri_options.group_options = options.group_triangulation

    _sfm.incremental_triangulate_structure(
        point_corr_graph, structure_db_cache, holistic_recon, tri_options
    )

    #######################################
    # 5. Optional: Bundle adjustment
    #######################################
    if options.run_bundle_adjustment:
        ba = limap.estimators.bundle_adjustment
        logging.info("Running bundle adjustment...")

        # Setup BA options - only refine points, lines, and groups
        ba_options = deepcopy(options.bundle_adjustment)
        ba_options.refine_focal_length = False
        ba_options.refine_principal_point = False
        ba_options.refine_sensor_from_rig = False
        ba_options.refine_rig_from_world = False
        # Disable group refinement if not using groups
        if not options.use_groups or not options.triangulate_groups:
            ba_options.refine_groups = False

        # Setup BA config - add all images and optionally groups
        ba_config = ba.StructureBundleAdjustmentConfig()
        for image_id in holistic_recon.point_recon.reg_image_ids():
            ba_config.add_image(image_id)
        if options.use_groups and options.triangulate_groups:
            for group_id in holistic_recon.structure_recon.groups3D:
                ba_config.add_variable_group(group_id)

        # Run bundle adjustment
        adjuster = ba.create_structure_bundle_adjuster(
            ba_options, ba_config, holistic_recon
        )
        summary = adjuster.solve()
        logging.info(
            f"Bundle adjustment completed: "
            f"initial_cost={summary.ceres_summary.initial_cost:.4f}, "
            f"final_cost={summary.ceres_summary.final_cost:.4f}"
        )

        # Post-BA group filtering
        if options.use_groups and options.triangulate_groups:
            filter_stats = limap.sfm.filter_group_associations(
                holistic_recon, options.group_filtering
            )
            num_deleted = limap.sfm.delete_supportless_groups(holistic_recon)
            logging.info(
                f"Post-BA group filtering: "
                f"{filter_stats.num_groups_passed} passed, "
                f"{filter_stats.num_groups_failed} failed, "
                f"{filter_stats.num_associations_marked} marked inactive, "
                f"{filter_stats.num_associations_purged} purged, "
                f"{num_deleted} groups deleted"
            )

    #######################################
    # 6. Write output
    #######################################
    final_output_model_dir = output_dir / "final_model"
    final_output_model_dir.mkdir(parents=True, exist_ok=True)
    holistic_recon.write(final_output_model_dir)

    logging.info(
        f"Incremental structure triangulation completed: {holistic_recon}"
    )
    return final_output_model_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Incremental structure triangulation"
    )
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--db_path",
        type=Path,
        default=None,
        help="COLMAP database (default: model_dir/database.db)",
    )
    parser.add_argument(
        "--structure_db_path",
        type=Path,
        default=None,
        help="Structure database (default: run frontend)",
    )
    parser.add_argument(
        "--no_groups",
        action="store_true",
        help="Disable group triangulation",
    )
    parser.add_argument(
        "--no_ba",
        action="store_true",
        help="Disable bundle adjustment",
    )

    args = parser.parse_args()

    options = IncrementalStructureTriangulationOptions(
        use_groups=not args.no_groups,
        run_bundle_adjustment=not args.no_ba,
    )

    incremental_structure_triangulation(
        options,
        args.image_dir,
        args.model_dir,
        args.output_dir,
        db_path=args.db_path,
        structure_db_path=args.structure_db_path,
    )
