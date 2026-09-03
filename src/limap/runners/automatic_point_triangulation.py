from dataclasses import dataclass, field
from pathlib import Path
import shutil

import pycolmap
from typeguard import typechecked

from limap.image.specs import PointDetectionOptions, PointMatcherOptions


@dataclass
class AutomaticPointTriangulationOptions:
    point_detection: PointDetectionOptions = field(
        default_factory=PointDetectionOptions
    )
    point_matcher: PointMatcherOptions = field(
        default_factory=PointMatcherOptions
    )
    pycolmap_options: pycolmap.IncrementalPipelineOptions | None = None


@typechecked
def automatic_point_triangulation(
    image_path: Path,
    model_path: Path,
    output_path: Path,
    options: AutomaticPointTriangulationOptions,
) -> Path:
    # hloc is an optional, git-sourced dependency (see requirements.txt);
    # import it lazily so that `import limap.runners` does not require it.
    from hloc import (
        extract_features,
        match_features,
        reconstruction,
        pairs_from_exhaustive,
        triangulation,
    )

    # Paths
    triangulated_model = output_path / "sparse"
    db_path = output_path / "database.db"
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Feature extraction with resizing ---
    feature_conf = extract_features.confs[options.point_detection.method]
    feature_path = extract_features.main(feature_conf, image_path, output_path)

    # --- Step 2: Matching ---
    # TODO: improve this exhaustive matcher to support retrieval
    pairs = output_path / "pairs-exhaustive.txt"
    pairs_from_exhaustive.main(pairs, features=feature_path)

    matcher_conf = match_features.confs[options.point_matcher.method]
    match_path = match_features.main(
        matcher_conf, pairs, feature_conf["output"], output_path
    )

    # --- Step 3: Create database ---
    shutil.copytree(image_path, output_path / "images", dirs_exist_ok=True)
    recon = pycolmap.Reconstruction(model_path)
    triangulation.create_db_from_model(recon, db_path)
    image_ids = reconstruction.get_image_ids(db_path)
    with pycolmap.Database.open(db_path) as db:
        reconstruction.import_features(image_ids, db, feature_path)
        reconstruction.import_matches(
            image_ids, db, pairs, match_path, None, None
        )
    triangulation.estimation_and_geometric_verification(db_path, pairs)

    # --- Step 4: Triangulate ---
    input_reconstruction = pycolmap.Reconstruction(model_path)
    if options.pycolmap_options:
        pycolmap.triangulate_points(
            input_reconstruction,
            db_path,
            image_path,
            triangulated_model,
            options=options.pycolmap_options,
        )
    else:
        pycolmap.triangulate_points(
            input_reconstruction, db_path, image_path, triangulated_model
        )
    return triangulated_model
