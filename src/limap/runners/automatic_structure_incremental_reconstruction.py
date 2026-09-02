from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import pycolmap
from pycolmap import logging
from typeguard import typechecked

from limap._limap import _sfm
from limap.scene import (
    HolisticReconstructionManager,
    StructureDatabase,
    StructureDatabaseCache,
)

from limap.image.specs import ImageDescriptionOptions, ImageAssociationOptions
from .pipeline_steps import MetaInfoComputerOptions
from .structure_frontend import (
    StructureFrontendOptions,
    structure_frontend_from_images,
)


@dataclass
class AutomaticStructureIncrementalReconstructionOptions:
    """Options for automatic structure-aware incremental reconstruction."""

    max_image_dim: int | None = None
    skip_frontend_if_exists_database: bool = False

    metainfo: MetaInfoComputerOptions = field(
        default_factory=MetaInfoComputerOptions
    )
    _image_description: ImageDescriptionOptions = field(
        default_factory=ImageDescriptionOptions
    )
    _image_association: ImageAssociationOptions = field(
        default_factory=ImageAssociationOptions
    )

    # SfM pipeline options (dict-based, applied via setattr to C++ types)
    colmap_options: dict = field(default_factory=dict)
    structure_options: dict = field(default_factory=dict)

    # Triangulate without line descriptors: skip line description and
    # matching, and pair each line with every line in its neighbors instead.
    use_exhaustive_line_matching: bool = False

    @property
    def image_description(self) -> ImageDescriptionOptions:
        options = deepcopy(self._image_description)
        if self.use_exhaustive_line_matching:
            options.line_detection.compute_descinfo = False
        return options

    @image_description.setter
    def image_description(self, opts: ImageDescriptionOptions) -> None:
        self._image_description = deepcopy(opts)

    @property
    def image_association(self) -> ImageAssociationOptions:
        options = deepcopy(self._image_association)
        options.skip_line_matching = self.use_exhaustive_line_matching
        return options

    @image_association.setter
    def image_association(self, opts: ImageAssociationOptions) -> None:
        self._image_association = deepcopy(opts)


def _apply_dict_options(target, options_dict):
    """Apply a (possibly nested) dict of options to a pybind11 object.

    Flat keys are applied via setattr. Dict values are applied recursively
    to the corresponding sub-object, enabling config like::

        {"structure_ba": {"refine_wireframe": False}}
    """
    for key, value in options_dict.items():
        if not hasattr(target, key):
            continue
        if isinstance(value, dict):
            _apply_dict_options(getattr(target, key), value)
        else:
            setattr(target, key, value)


@typechecked
def automatic_structure_incremental_reconstruction(
    options: AutomaticStructureIncrementalReconstructionOptions,
    image_dir: Path,
    output_dir: Path,
    recon: pycolmap.Reconstruction,
    db_path: Path | None = None,
    structure_db_path: Path | None = None,
    neighbors: dict[int, list[int]] | None = None,
) -> Path:
    """Run automatic structure-aware incremental SfM reconstruction.

    This function runs the full SfM pipeline: feature extraction, matching,
    geometric verification, and incremental reconstruction with hybrid
    point+line registration and structure triangulation.

    The input reconstruction needs camera intrinsics and image list but
    NOT poses. The pipeline estimates poses from scratch (incremental SfM).
    The reconstruction is passed in-memory to avoid COLMAP's binary format
    which drops unposed frames/images on write.

    Args:
        options: Configuration options
        image_dir: Path to images
        output_dir: Output directory for results
        recon: In-memory Reconstruction with cameras and images
            (no poses required)
        db_path: Optional COLMAP database (default: run frontend)
        structure_db_path: Optional structure database (default: run frontend)
        neighbors: Optional image neighbors for frontend

    Returns:
        Path to output directory containing reconstructed models
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    #######################################
    # 1. Run structure frontend if needed
    #######################################
    if db_path is None or structure_db_path is None:
        logging.info("Running structure frontend...")
        frontend_options = StructureFrontendOptions(
            max_image_dim=options.max_image_dim,
            metainfo=options.metainfo,
            image_description=options.image_description,
            image_association=options.image_association,
        )
        frontend_outputs = structure_frontend_from_images(
            frontend_options,
            image_dir,
            recon,
            output_dir,
            neighbors=neighbors,
            skip_if_exists=options.skip_frontend_if_exists_database,
        )
        db_path = frontend_outputs.db_path
        structure_db_path = frontend_outputs.structure_db_path
        image_dir = frontend_outputs.image_dir

    #######################################
    # 2. Load databases
    #######################################
    logging.info(f"Loading COLMAP database from {db_path}")
    with pycolmap.Database.open(db_path) as colmap_db:
        colmap_db_cache = pycolmap.DatabaseCache.create(
            colmap_db, pycolmap.DatabaseCacheOptions()
        )

    logging.info(f"Loading structure database from {structure_db_path}")
    with StructureDatabase.open(structure_db_path) as structure_db:
        structure_db_cache = StructureDatabaseCache.create(structure_db)

    #######################################
    # 3. Create reconstruction manager
    #######################################
    reconstruction_manager = HolisticReconstructionManager()

    #######################################
    # 4. Configure pipeline options
    #######################################
    pipeline_options = _sfm.StructureIncrementalPipelineOptions()
    pipeline_options.colmap_options.image_path = str(image_dir)
    _apply_dict_options(pipeline_options.colmap_options, options.colmap_options)
    _apply_dict_options(
        pipeline_options.structure_options, options.structure_options
    )
    if options.use_exhaustive_line_matching:
        line_opts = (
            pipeline_options.structure_options.triangulation.line_options
        )
        line_opts.exhaustive_match_neighbors = frontend_outputs.neighbors

    #######################################
    # 5. Run pipeline
    #######################################
    logging.info("Running structure-aware incremental SfM...")
    pipeline = _sfm.StructureIncrementalPipeline(
        pipeline_options,
        colmap_db_cache,
        structure_db_cache,
        reconstruction_manager,
    )
    pipeline.run()

    #######################################
    # 6. Report and save
    #######################################
    for i in range(reconstruction_manager.size()):
        holistic_recon = reconstruction_manager.get(i)
        prec = holistic_recon.point_recon
        srec = holistic_recon.structure_recon
        # Count groups by type
        group_type_counts = Counter(g.type.name for g in srec.groups3D.values())
        group_type_str = ", ".join(
            f"{t}={c}" for t, c in sorted(group_type_counts.items())
        )
        logging.info(
            f"Model {i}: {prec.num_reg_images()} images, "
            f"{prec.num_points3D()} points, "
            f"{srec.num_lines3D()} lines, "
            f"{srec.num_groups3D()} groups"
            f"{' (' + group_type_str + ')' if group_type_str else ''}"
        )

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_manager.write(models_dir)
    logging.info(f"Saved reconstructions to: {models_dir}")
    return output_dir
