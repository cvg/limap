import numpy as np
import pycolmap
from pathlib import Path
from typeguard import typechecked

import limap.scene
from limap._limap._image._groups import (
    GroupVotingOptions,
    match_groups_by_voting,
)


@typechecked
def vote_unmatched_groups(
    options: GroupVotingOptions,
    neighbors: dict[int, list[int]],
    db_path: Path,
    structure_db_path: Path,
) -> None:
    """Vote for unmatched groups using matched points and lines."""
    with (
        pycolmap.Database.open(db_path) as db,
        limap.scene.StructureDatabase.open(structure_db_path) as struct_db,
    ):
        for img_id in neighbors:
            for ng_img_id in neighbors[img_id]:
                # Read existing data
                two_view = db.read_two_view_geometry(img_id, ng_img_id)
                point_matches = two_view.inlier_matches
                if not struct_db.exists_line_matches(img_id, ng_img_id):
                    line_matches = np.zeros((0, 2), dtype=np.int32)
                else:
                    line_matches = struct_db.read_line_matches_blob(
                        img_id, ng_img_id
                    )
                if not struct_db.exists_group_matches(img_id, ng_img_id):
                    group_matches = np.zeros((0, 2), dtype=np.int32)
                else:
                    group_matches = struct_db.read_group_matches_blob(
                        img_id, ng_img_id
                    )

                structure1 = struct_db.read_structure2d(img_id)
                structure2 = struct_db.read_structure2d(ng_img_id)

                # Skip if no groups
                if structure1.num_groups() == 0 or structure2.num_groups() == 0:
                    continue

                # Already matched groups (local indices)
                matched_g1 = (
                    set(group_matches[:, 0].tolist())
                    if len(group_matches) > 0
                    else set()
                )
                matched_g2 = (
                    set(group_matches[:, 1].tolist())
                    if len(group_matches) > 0
                    else set()
                )

                # C++ voting
                voting_matches = match_groups_by_voting(
                    options,
                    point_matches.astype(np.int32),
                    line_matches.astype(np.int32),
                    structure1,
                    structure2,
                    matched_g1,
                    matched_g2,
                )

                # Append new matches
                if len(voting_matches) > 0:
                    if len(group_matches) > 0:
                        # Delete existing matches first, then write combined
                        struct_db.delete_group_matches(img_id, ng_img_id)
                        all_matches = np.vstack(
                            [group_matches, voting_matches.astype(np.uint32)]
                        )
                    else:
                        all_matches = voting_matches.astype(np.uint32)
                    struct_db.write_group_matches(
                        img_id, ng_img_id, all_matches.astype(np.uint32)
                    )
