import numpy as np
import pycolmap
import pytest

import limap
from limap.image.process import _mark_pose_verified_geometries


@pytest.mark.ci_workflow
def test_hash_map_backend_matches_pycolmap():
    # The backend changes the layout of the COLMAP types limap shares with
    # pycolmap, and a mismatch corrupts memory instead of failing to link.
    assert limap.__hash_map_backend__ == pycolmap.__hash_map_backend__


@pytest.mark.ci_workflow
def test_pose_verified_matches_reach_database_cache(tmp_path):
    database_path = tmp_path / "database.db"
    matches = np.stack([np.arange(40), np.arange(40)], axis=1).astype(np.uint32)
    with pycolmap.Database.open(database_path) as db:
        camera = pycolmap.Camera.create_from_model_name(
            0, "SIMPLE_PINHOLE", 100.0, 100, 100
        )
        camera_id = db.write_camera(camera)
        image_ids = []
        for name in ("a.jpg", "b.jpg"):
            image_id = db.write_image(
                pycolmap.Image(name=name, camera_id=camera_id)
            )
            db.write_keypoints(image_id, np.random.rand(50, 2) * 100)
            image_ids.append(image_id)
        db.write_matches(*image_ids, matches)
        # What hloc's pose-guided verification writes.
        db.write_two_view_geometry(
            *image_ids, pycolmap.TwoViewGeometry(inlier_matches=matches)
        )
        _mark_pose_verified_geometries(db)

    with pycolmap.Database.open(database_path) as db:
        cache = pycolmap.DatabaseCache.create(
            db, pycolmap.DatabaseCacheOptions(min_num_matches=15)
        )
    graph = cache.correspondence_graph
    assert graph.num_matches_between_images(*image_ids) == len(matches)
