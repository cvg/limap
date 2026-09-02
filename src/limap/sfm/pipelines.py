from limap._limap import _sfm

from dataclasses import dataclass, field
from pathlib import Path
from typeguard import typechecked
import pycolmap

from limap.estimators.bundle_adjustment import (
    StructureBundleAdjustmentOptions,
)
from limap.scene import (
    HolisticReconstruction,
    StructureDatabase,
    StructureDatabaseCache,
)


@dataclass
class GlobalStructureTriangulationOptions:
    line_triangulation: _sfm.GlobalLineTriangulationOptions = field(
        default_factory=_sfm.GlobalLineTriangulationOptions
    )
    group_triangulation: _sfm.GlobalGroupTriangulationOptions = field(
        default_factory=_sfm.GlobalGroupTriangulationOptions
    )
    ba_options: StructureBundleAdjustmentOptions | None = field(
        default_factory=StructureBundleAdjustmentOptions
    )
    filter_options: _sfm.GroupVerificationOptions | None = field(
        default_factory=_sfm.GroupVerificationOptions
    )


@typechecked
def global_line_triangulation(
    recon: pycolmap.Reconstruction,
    structure_db_path: Path,
    options: _sfm.GlobalLineTriangulationOptions,
    exhaustive_match_neighbors: dict[int, list[int]] | None = None,
) -> HolisticReconstruction:
    holistic_recon = HolisticReconstruction(recon)
    with StructureDatabase.open(structure_db_path) as structure_db:
        structure_db_cache = StructureDatabaseCache.create(structure_db)
    holistic_recon.structure_recon.load(structure_db_cache)
    holistic_recon.structure_recon.initialize_all_wireframes(
        options.wireframe2d_th
    )
    assert holistic_recon.check_validity()

    _sfm.global_line_triangulation(
        options,
        holistic_recon,
        structure_db_cache.structure_correspondence_graph.line_graph,
        exhaustive_match_neighbors or {},
    )
    return holistic_recon


@typechecked
def global_group_triangulation(
    holistic_recon: HolisticReconstruction,
    structure_db_path: Path,
    options: _sfm.GlobalGroupTriangulationOptions,
) -> None:
    with StructureDatabase.open(structure_db_path) as structure_db:
        structure_db_cache = StructureDatabaseCache.create(structure_db)

    _sfm.global_group_triangulation(
        options,
        holistic_recon,
        structure_db_cache.structure_correspondence_graph.group_graph,
    )


@typechecked
def global_structure_triangulation(
    recon: pycolmap.Reconstruction,
    structure_db_path: Path,
    options: GlobalStructureTriangulationOptions,
    exhaustive_match_neighbors: dict[int, list[int]] | None = None,
) -> HolisticReconstruction:
    holistic_recon = HolisticReconstruction(recon)
    with StructureDatabase.open(structure_db_path) as structure_db:
        structure_db_cache = StructureDatabaseCache.create(structure_db)

    _sfm.global_triangulate_structure(
        structure_db_cache,
        holistic_recon,
        options.line_triangulation,
        options.group_triangulation,
        ba_options=options.ba_options,
        filter_options=options.filter_options,
        exhaustive_match_neighbors=exhaustive_match_neighbors or {},
    )
    return holistic_recon
