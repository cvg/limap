from pathlib import Path
import numpy as np
from typeguard import typechecked
import pycolmap
from pycolmap import logging

from limap.geometry import Line2d
from limap.scene import StructureDatabase, Structure2d, Group2d
from limap.util.io import read_npy


@typechecked
def create_structure_db(db_path: Path, delete_if_exist=True) -> None:
    if db_path.exists():
        if delete_if_exist:
            logging.warning(
                "The structure database already exists, deleting it."
            )
            db_path.unlink()
        else:
            return None
    with StructureDatabase.open(db_path) as _:
        pass


@typechecked
def initialize_structures(
    structure_db: StructureDatabase, image_ids: list[int], db: pycolmap.Database
) -> None:
    for img_id in image_ids:
        structure = Structure2d()
        structure.set_num_points(db.num_keypoints_for_image(img_id))
        structure_db.write_structure2d(img_id, structure)


@typechecked
def initialize_structures_from_reconstruction(
    structure_db: StructureDatabase,
    recon: pycolmap.Reconstruction,
) -> None:
    """Initialize structures with num_points from the reconstruction.

    Use this instead of initialize_structures() when skipping point
    detection — the COLMAP database has no imported keypoints, but the
    reconstruction's image.num_points2D() has the correct count.
    """
    for img_id, image in recon.images.items():
        structure = Structure2d()
        structure.set_num_points(image.num_points2D())
        structure_db.write_structure2d(img_id, structure)


@typechecked
def import_line_detections(
    structure_db: StructureDatabase, all_2d_lines: dict[int, list[Line2d]]
) -> None:
    for img_id, lines2d in all_2d_lines.items():
        structure = structure_db.read_structure2d(img_id)
        structure.set_lines(lines2d)
        structure_db.update_structure2d(img_id, structure)


@typechecked
def import_line_matches(
    structure_db: StructureDatabase, matches_dir: Path
) -> None:
    for file_path in matches_dir.iterdir():
        if file_path.suffix == ".npy":
            img_id = int(file_path.stem.split("_")[-1])
            ng_matches = read_npy(file_path).item()
            for ng_img_id, matches in ng_matches.items():
                if not structure_db.exists_line_matches(img_id, ng_img_id):
                    structure_db.write_line_matches(
                        img_id, ng_img_id, matches.astype(np.uint32)
                    )


@typechecked
def import_group_descriptions(
    structure_db: StructureDatabase, all_groups2d: dict[int, list[Group2d]]
) -> None:
    for img_id, groups2d in all_groups2d.items():
        structure = structure_db.read_structure2d(img_id)
        structure.add_groups(groups2d)
        structure_db.update_structure2d(img_id, structure)
