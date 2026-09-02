import pycolmap
from pycolmap import logging
import limap.scene
import limap.sfm
from pathlib import Path
from typeguard import typechecked

import hloc.extract_features
import hloc.match_features
import hloc.triangulation
import hloc.reconstruction

from .group_voting import vote_unmatched_groups
from .specs import ImageDescriptionOptions, ImageAssociationOptions
from .line import line_detection, line_matching
from .groups import group_description
from .dense_matcher import associate_via_dense_matching


@typechecked
def create_empty_databases(
    recon: pycolmap.Reconstruction,
    database_path: Path,
    structure_database_path: Path,
    image_dir: Path | None = None,
    camera_mode: pycolmap.CameraMode | None = None,
) -> None:
    if camera_mode is not None:
        # Infer cameras from images (EXIF or default focal length)
        assert image_dir is not None
        hloc.reconstruction.import_images(image_dir, database_path, camera_mode)
    else:
        # Use intrinsics from the input model
        hloc.triangulation.create_db_from_model(recon, database_path)
    limap.scene.create_structure_db(structure_database_path)


@typechecked
def image_description(
    options: ImageDescriptionOptions,
    recon: pycolmap.Reconstruction,
    image_path: Path,
    workspace_path: Path,
    db_path: Path,
    structure_db_path: Path,
) -> tuple[Path | None, Path | None]:
    """
    Return point_descriptor_path and line_descriptor_path
    """
    image_names = {i: image_path / img.name for i, img in recon.images.items()}
    if options.use_joint_point_line_detection:
        # TODO: support in the future with UPAL
        raise NotImplementedError
    else:
        # Perform keypoint detection (unless skipped)
        if options.skip_point_detection:
            logging.info("Skipping point detection (using existing points)")
            feature_path = None
        else:
            logging.info(
                f"Perform keypoint detection (n_images = {recon.num_images()})"
            )
            feature_conf = hloc.extract_features.confs[
                options.point_detection.method
            ]
            feature_path = hloc.extract_features.main(
                feature_conf, image_path, workspace_path
            )
            with pycolmap.Database.open(db_path) as db:
                hloc.reconstruction.import_features(
                    hloc.reconstruction.get_image_ids(db_path), db, feature_path
                )

        # Perform line detection
        logging.info(
            f"Perform line detection (n_images = {recon.num_images()})"
        )
        all_2d_segs, descinfo_folder = line_detection(
            image_names,
            workspace_path / "line_detections",
            options.line_detection,
        )
        all_2d_lines = limap.geometry.get_all_lines_2d(all_2d_segs)
        del all_2d_segs
        with limap.scene.StructureDatabase.open(
            structure_db_path
        ) as structure_db:
            if options.skip_point_detection:
                limap.scene.initialize_structures_from_reconstruction(
                    structure_db, recon
                )
            else:
                image_ids = list(recon.images.keys())
                with pycolmap.Database.open(db_path) as db:
                    limap.scene.initialize_structures(
                        structure_db, image_ids, db
                    )
            limap.scene.import_line_detections(structure_db, all_2d_lines)
        del all_2d_lines

    # Group description reads points and lines from databases per-image
    if not options.skip_group_description:
        group_description(
            image_names,
            workspace_path / "group_description",
            db_path,
            structure_db_path,
            options.group_description,
            recon=recon if options.skip_point_detection else None,
        )
    return feature_path, descinfo_folder


@typechecked
def _hloc_pairs_from_neighbors(
    output_path: Path,
    neighbors: dict[int, list[int]],
    image_names: dict[int, str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        for head, locs in neighbors.items():
            for loc in locs:
                f.write(f"{image_names[head]} {image_names[loc]}\n")


@typechecked
def image_association(
    options: ImageAssociationOptions,
    desc_options: ImageDescriptionOptions,
    recon: pycolmap.Reconstruction,
    image_path: Path,
    neighbors: dict[int, list[int]],
    workspace_path: Path,
    db_path: Path,
    structure_db_path: Path,
) -> None:
    image_names = {i: image_path / img.name for i, img in recon.images.items()}
    # Use dense matching
    if options.use_dense_matching:
        logging.info(
            f"Associate via dense matching (n_images = {len(image_names)})"
        )
        options.dense_matching_options.skip_group_matching = (
            options.skip_group_matching
        )
        associate_via_dense_matching(
            options.dense_matching_options,
            image_names,
            neighbors,
            workspace_path / "group_description",
            db_path,
            structure_db_path,
            warp_cache_dir=workspace_path / "dense_warps",
        )
    # Use classical feature matching
    else:
        if options.use_joint_point_line_matcher and (
            not options.skip_line_matching
        ):
            # TODO: integrate GlueStick here
            # TODO: support in the future with LightGlueStick
            raise NotImplementedError
        else:
            # keypoint matching (skip when using existing points)
            if options.skip_point_matching:
                logging.info("Skipping point matching (using existing points)")
            else:
                logging.info(
                    f"Perform keypoint matching (n_images = {len(image_names)})"
                )
                assert options.point_descriptor_path is not None, (
                    "point_descriptor_path needs to be initialized "
                    "to be able to run point matching"
                )
                pairs_path = workspace_path / "pairs-from-neighbors.txt"
                image_names_hloc = {
                    i: img.name for i, img in recon.images.items()
                }
                _hloc_pairs_from_neighbors(
                    pairs_path, neighbors, image_names_hloc
                )
                matcher_conf = hloc.match_features.confs[
                    options.point_matcher.method
                ]
                match_path = hloc.match_features.main(
                    matcher_conf,
                    pairs_path,
                    options.point_descriptor_path.stem,
                    workspace_path,
                )
                hloc_image_ids = hloc.reconstruction.get_image_ids(db_path)
                with pycolmap.Database.open(db_path) as db:
                    hloc.reconstruction.import_matches(
                        hloc_image_ids,
                        db,
                        pairs_path,
                        match_path,
                        None,
                        None,
                    )
                if options.geometric_verification:
                    # Determine verification mode
                    if options.pose_guided_point_verification is None:
                        use_pose_guided = all(
                            image.has_pose for image in recon.images.values()
                        )
                    else:
                        use_pose_guided = options.pose_guided_point_verification

                    if use_pose_guided:
                        with pycolmap.Database.open(db_path) as db:
                            hloc.triangulation.geometric_verification(
                                hloc_image_ids,
                                recon,
                                db,
                                options.point_descriptor_path,
                                pairs_path,
                                match_path,
                            )
                    else:
                        hloc.triangulation.estimation_and_geometric_verification(
                            db_path, pairs_path
                        )

            # line matching
            if (
                not options.skip_line_matching
            ):  # skip when using exhaustive matches
                logging.info(
                    f"Perform line matching (n_images = {len(image_names)})"
                )
                assert options.line_descriptor_path is not None, (
                    "line_descriptor_path needs to be initialized "
                    "to be able to run line matching"
                )
                matches_dir = line_matching(
                    options.line_descriptor_path,
                    workspace_path / "line_matchings",
                    neighbors,
                    desc_options.line_detection,
                    options.line_matcher,
                )
                with limap.scene.StructureDatabase.open(
                    structure_db_path
                ) as structure_db:
                    limap.scene.import_line_matches(structure_db, matches_dir)

        # Hybrid mode: classical points+lines, dense groups
        if (
            options.use_dense_matching_for_groups
            and not options.skip_group_matching
        ):
            logging.info(
                "Associate groups via dense matching "
                f"(n_images = {len(image_names)})"
            )
            dense_opts = options.dense_matching_options
            dense_opts.skip_point_matching = True
            dense_opts.skip_line_matching = True
            dense_opts.skip_group_matching = False
            associate_via_dense_matching(
                dense_opts,
                image_names,
                neighbors,
                workspace_path / "group_description",
                db_path,
                structure_db_path,
                warp_cache_dir=workspace_path / "dense_warps",
            )

    # Voting for unmatched groups
    if not options.skip_group_voting:
        vote_unmatched_groups(
            options.group_voting,
            neighbors,
            db_path,
            structure_db_path,
        )

    # VP geometric verification (requires poses)
    if options.vp_geometric_verification:
        is_posed = all(image.has_pose for image in recon.images.values())
        if is_posed:
            logging.info("Running VP geometric verification...")
            with limap.scene.StructureDatabase.open(
                structure_db_path
            ) as structure_db:
                num_removed = limap.sfm.verify_vp_matches(
                    recon, structure_db, options.vp_verification_threshold
                )
            logging.info(f"VP verification removed {num_removed} matches")
        else:
            logging.warning(
                "Skipping VP geometric verification: poses not available"
            )

    # Plane geometric verification (requires poses)
    if options.plane_geometric_verification:
        is_posed = all(image.has_pose for image in recon.images.values())
        if is_posed:
            logging.info("Running plane geometric verification...")
            with limap.scene.StructureDatabase.open(
                structure_db_path
            ) as structure_db:
                num_removed = limap.sfm.verify_plane_matches(
                    recon, structure_db, options.plane_verification_threshold
                )
            logging.info(f"Plane verification removed {num_removed} matches")
        else:
            logging.warning(
                "Skipping plane geometric verification: poses not available"
            )
