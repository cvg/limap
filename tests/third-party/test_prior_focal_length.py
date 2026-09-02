"""Guard for `prior_focal_length` reaching the COLMAP database.

Our benchmarks build cameras from GT calibration. If the focal is not marked
as a prior, COLMAP (>= f7b079f4) routes shared-camera pinhole pairs into the
6pt shared-focal solver and re-estimates it at initialization; with
`ba_refine_focal_length=false` it is never corrected, and pose accuracy at
sub-degree thresholds collapses while coarse thresholds still look fine.

The failure is silent, so these tests pin the two links in the chain:
the flag survives the write to `database.db`, and COLMAP still branches on it.
"""

import sqlite3

import pycolmap
import pytest


def _camera(has_prior):
    cam = pycolmap.Camera(
        camera_id=1,
        model="PINHOLE",
        width=800,
        height=600,
        params=[600.0, 601.0, 400.0, 300.0],
    )
    cam.has_prior_focal_length = has_prior
    return cam


@pytest.mark.ci_workflow
@pytest.mark.parametrize("has_prior", [True, False])
def test_prior_focal_length_survives_database_write(tmp_path, has_prior):
    """`db.write_camera` must persist the flag; the fix relies on it."""
    db_path = tmp_path / f"prior_{has_prior}.db"
    with pycolmap.Database.open(db_path) as db:
        db.write_camera(_camera(has_prior), use_camera_id=True)

    with sqlite3.connect(db_path) as con:
        (stored,) = con.execute(
            "SELECT prior_focal_length FROM cameras"
        ).fetchone()
    assert bool(stored) is has_prior


@pytest.mark.ci_workflow
def test_prior_focal_length_round_trips_through_reconstruction(tmp_path):
    """The path our loaders take: recon -> create_db_from_model -> db."""
    hloc_triangulation = pytest.importorskip("hloc.triangulation")

    recon = pycolmap.Reconstruction()
    cam = _camera(True)
    recon.add_camera(cam)
    rig = pycolmap.Rig(rig_id=1)
    rig.add_ref_sensor(cam.sensor_id)
    recon.add_rig(rig)

    db_path = tmp_path / "from_model.db"
    hloc_triangulation.create_db_from_model(recon, db_path)

    with sqlite3.connect(db_path) as con:
        (stored,) = con.execute(
            "SELECT prior_focal_length FROM cameras"
        ).fetchone()
    assert stored == 1, (
        "prior_focal_length was lost between the reconstruction and the "
        "database; COLMAP will re-estimate the focal at two-view "
        "initialization and sub-degree pose accuracy will silently degrade"
    )
