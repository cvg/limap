from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import pycolmap
from pycolmap import logging
from typeguard import typechecked

import limap.runners
import limap.sfm
from limap.image.specs import ImageDescriptionOptions, ImageAssociationOptions
from limap.sfm import (
    GlobalStructureTriangulationOptions,
    GroupVerificationOptions,
)
from limap.estimators.bundle_adjustment import StructureBundleAdjustmentOptions
from limap.util.types import Ranges

from .pipeline_steps import MetaInfoComputerOptions
from .structure_frontend import (
    StructureFrontendOptions,
    structure_frontend_from_model,
)


@dataclass
class AutomaticStructureTriangulationOptions:
    max_image_dim: int | None = None
    weight_path: Path = Path.home() / ".limap" / "models"
    skip_exists: bool = False
    skip_frontend_if_exists_database: bool = False
    # Delete the detections and descriptors once they have been imported
    # into the databases, to save disk space. Disable this if a later stage
    # still needs them, as visual localization does.
    cleanup_frontend_workspace: bool = True
    n_visible_views: int = 4
    run_bundle_adjustment: bool = True
    use_groups: bool = True
    # Triangulate without line descriptors: skip line description and
    # matching, and let the triangulator pair each line with every line in
    # its neighbors instead.
    use_exhaustive_line_matching: bool = False

    metainfo: MetaInfoComputerOptions = field(
        default_factory=MetaInfoComputerOptions
    )
    _image_description: ImageDescriptionOptions = field(
        default_factory=ImageDescriptionOptions
    )
    _image_association: ImageAssociationOptions = field(
        default_factory=ImageAssociationOptions
    )
    point_triangulation: pycolmap.IncrementalPipelineOptions | None = None
    _structure_triangulation: GlobalStructureTriangulationOptions = field(
        default_factory=GlobalStructureTriangulationOptions
    )
    bundle_adjustment: StructureBundleAdjustmentOptions = field(
        default_factory=StructureBundleAdjustmentOptions
    )
    group_filtering: GroupVerificationOptions = field(
        default_factory=GroupVerificationOptions
    )

    @property
    def image_description(self) -> ImageDescriptionOptions:
        options = deepcopy(self._image_description)
        options.line_detection.compute_descinfo = (
            options.line_detection.compute_descinfo
            and (not self.use_exhaustive_line_matching)
            and (not self.image_association.use_dense_matching)
        )
        options.line_detection.skip_exists = self.skip_exists
        options.line_detection.weight_path = self.weight_path
        options.group_description.plane_detection.weight_path = self.weight_path
        # Disable group description when not using groups
        if not self.use_groups:
            options.skip_group_description = True
        return options

    @image_description.setter
    def image_description(self, opts: ImageDescriptionOptions) -> None:
        self._image_description = deepcopy(opts)

    @property
    def image_association(self) -> ImageAssociationOptions:
        options = deepcopy(self._image_association)
        options.skip_line_matching = self.use_exhaustive_line_matching
        options.line_matcher.skip_exists = self.skip_exists
        options.line_matcher.weight_path = self.weight_path
        # Disable group matching when not using groups
        if not self.use_groups:
            options.skip_group_matching = True
        return options

    @image_association.setter
    def image_association(self, opts: ImageAssociationOptions) -> None:
        self._image_association = deepcopy(opts)

    @property
    def structure_triangulation(self) -> GlobalStructureTriangulationOptions:
        from limap.image.line import get_uncertainty2d

        options = deepcopy(self._structure_triangulation)
        if options.line_triangulation.var2d < 0:
            options.line_triangulation.var2d = get_uncertainty2d(
                self._image_description.line_detection.method
            )
        return options

    @structure_triangulation.setter
    def structure_triangulation(
        self, opts: GlobalStructureTriangulationOptions
    ) -> None:
        self._structure_triangulation = deepcopy(opts)

    @property
    def line_triangulation(self):
        """Shortcut to structure_triangulation.line_triangulation."""
        return self._structure_triangulation.line_triangulation

    @property
    def group_triangulation(self):
        """Shortcut to structure_triangulation.group_triangulation."""
        return self._structure_triangulation.group_triangulation


@typechecked
def automatic_structure_triangulation(
    options: AutomaticStructureTriangulationOptions,
    image_dir: Path,
    model_dir: Path,
    output_dir: Path,
    neighbors: dict[int, list[int]] | None = None,
    ranges: Ranges | None = None,
) -> Path:
    #######################################
    # Run structure frontend
    #######################################
    frontend_options = StructureFrontendOptions(
        max_image_dim=options.max_image_dim,
        metainfo=options.metainfo,
        image_description=options.image_description,
        image_association=options.image_association,
        cleanup_workspace=options.cleanup_frontend_workspace,
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
    # Supplying these is what puts the triangulator in exhaustive mode.
    exhaustive_match_neighbors = (
        frontend_outputs.neighbors
        if options.use_exhaustive_line_matching
        else None
    )

    ######################################
    # multi-view point triangulation
    ######################################
    triangulated_model = output_dir / "point_triangulation"
    if not triangulated_model.exists():
        recon = pycolmap.Reconstruction(model_dir)
        point_tri_options = (
            options.point_triangulation
            if options.point_triangulation
            else pycolmap.IncrementalPipelineOptions()
        )
        point_tri_options.min_num_matches = 1
        pycolmap.triangulate_points(
            recon,
            db_path,
            image_dir,
            triangulated_model,
            options=point_tri_options,
            clear_points=True,
        )
    else:
        logging.info(
            "Skipping point triangulation "
            f"(output exists: {triangulated_model})"
        )
    if ranges is None:
        ranges = limap.runners.compute_ranges(
            triangulated_model, options.metainfo.range_calculator
        )
    recon = pycolmap.Reconstruction(triangulated_model)

    # Restore keypoints for images excluded by COLMAP's DatabaseCache during
    # point triangulation. These images have valid poses but 0 Point2D because
    # the DatabaseCache didn't load their features. Re-populate from the
    # database so that CheckValidity passes
    # (NumPoints2D == Structure2d.NumPoints).
    num_restored = 0
    with pycolmap.Database.open(db_path) as db:
        for image_id, image in recon.images.items():
            if image.num_points2D() == 0:
                kp_blob = db.read_keypoints(image_id)
                if len(kp_blob) > 0:
                    points2D = pycolmap.Point2DList()
                    for row in kp_blob:
                        p = pycolmap.Point2D()
                        p.xy = row[:2]
                        points2D.append(p)
                    image.points2D = points2D
                    num_restored += 1
    if num_restored > 0:
        logging.warning(
            f"Restored keypoints for {num_restored} images excluded by "
            f"COLMAP DatabaseCache during point triangulation"
        )

    ##########################################################
    # multi-view structure triangulation (includes BA + filtering)
    ##########################################################
    tri_options = options.structure_triangulation
    if options.run_bundle_adjustment:
        tri_options.ba_options = deepcopy(options.bundle_adjustment)
        if not options.use_groups:
            tri_options.ba_options.refine_groups = False
        tri_options.filter_options = deepcopy(options.group_filtering)
    else:
        tri_options.ba_options = None
        tri_options.filter_options = None

    if options.use_groups:
        holistic_recon = limap.sfm.global_structure_triangulation(
            recon,
            structure_db_path,
            tri_options,
            exhaustive_match_neighbors,
        )
    else:
        # Only triangulate lines, skip groups
        holistic_recon = limap.sfm.pipelines.global_line_triangulation(
            recon,
            structure_db_path,
            tri_options.line_triangulation,
            exhaustive_match_neighbors,
        )

    final_output_model_dir = output_dir / "final_model"
    final_output_model_dir.mkdir(parents=True, exist_ok=True)
    holistic_recon.write(final_output_model_dir)

    logging.info(f"Structure triangulation completed: {holistic_recon}")
    return final_output_model_dir
