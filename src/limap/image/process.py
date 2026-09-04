import pycolmap
from pycolmap import logging
import limap.scene
import limap.sfm
from pathlib import Path
from typeguard import typechecked

from .group_voting import vote_unmatched_groups
from .specs import ImageDescriptionOptions, ImageAssociationOptions
from .line import line_detection, line_matching
from .groups import group_description


def _hloc():
    """Import hloc on demand.

    hloc is git-sourced (see requirements.txt) and cannot be declared as a
    project dependency, so importing it at module level would make
    ``import limap.image`` fail wherever it is not installed.
    """
    import hloc.extract_features  # noqa: F401
    import hloc.match_features  # noqa: F401
    import hloc.reconstruction  # noqa: F401
    import hloc.triangulation  # noqa: F401

    return hloc


@typechecked
def create_empty_databases(
    recon: pycolmap.Reconstruction,
    database_path: Path,
    structure_database_path: Path,
    image_dir: Path | None = None,
    camera_mode: pycolmap.CameraMode | None = None,
) -> None:
    hloc = _hloc()

    if camera_mode is not None:
        # Infer cameras from images (EXIF or default focal length)
        assert image_dir is not None
        hloc.reconstruction.import_images(image_dir, database_path, camera_mode)
    else:
        # Use intrinsics from the input model
        hloc.triangulation.create_db_from_model(recon, database_path)
    limap.scene.create_structure_db(structure_database_path)


def _import_point_features(hloc, db_path: Path, feature_path: Path) -> None:
    """Import an hloc feature file's keypoints into the COLMAP database."""
    with pycolmap.Database.open(db_path) as db:
        hloc.reconstruction.import_features(
            hloc.reconstruction.get_image_ids(db_path), db, feature_path
        )


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
    hloc = _hloc()

    image_names = {i: image_path / img.name for i, img in recon.images.items()}
    if options.use_joint_point_line_detection:
        # TODO: support in the future with UPAL
        raise NotImplementedError
    else:
        # A joint matcher's keypoints come out of the same pass as its line
        # descriptors, so the lines are described first and the points are
        # taken from that, instead of a second detection over the same images.
        joint_options = options.joint_point_line_matcher
        joint_matching = (
            joint_options is not None and not options.skip_point_detection
        )
        if not joint_matching:
            # Perform keypoint detection (unless skipped)
            if options.skip_point_detection:
                logging.info("Skipping point detection (using existing points)")
                feature_path = None
            else:
                logging.info(
                    "Perform keypoint detection "
                    f"(n_images = {recon.num_images()})"
                )
                feature_conf = hloc.extract_features.confs[
                    options.point_detection.method
                ]
                feature_path = hloc.extract_features.main(
                    feature_conf, image_path, workspace_path
                )
                _import_point_features(hloc, db_path, feature_path)

        # Perform line detection
        logging.info(
            f"Perform line detection (n_images = {recon.num_images()})"
        )
        all_2d_segs, descinfo_folder = line_detection(
            image_names,
            workspace_path / "line_detections",
            options.line_detection,
        )

        if joint_matching:
            from .joint_point_line import joint_point_line_description

            # point_detection is usually inherited from a base config, so say
            # plainly that it is not what runs here.
            logging.warning(
                "Joint matching with "
                f"'{joint_options.method}' takes its "
                "keypoints from the line description: point_detection "
                f"('{options.point_detection.method}') is unused."
            )
            feature_path = joint_point_line_description(
                joint_options,
                descinfo_folder,
                {i: img.name for i, img in recon.images.items()},
                workspace_path,
            )
            _import_point_features(hloc, db_path, feature_path)
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


def _import_point_matches(
    hloc, options, recon, db_path, pairs_path, match_path
) -> None:
    """Import an hloc match file into the COLMAP database and verify it.

    Shared by the separate point matcher and the joint point-line matcher,
    which writes the same format for its point half.
    """
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
    if not options.geometric_verification:
        return

    # Determine verification mode
    if options.pose_guided_point_verification is None:
        use_pose_guided = all(image.has_pose for image in recon.images.values())
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
    hloc = _hloc()
    from .dense_matcher import associate_via_dense_matching

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
        pairs_path = workspace_path / "pairs-from-neighbors.txt"
        image_names_hloc = {i: img.name for i, img in recon.images.items()}

        if options.use_joint_point_line_matcher and (
            not options.skip_line_matching
        ):
            # One pass per pair yields both halves, in the formats the
            # separate matchers below write, so the imports are the same.
            from .joint_point_line import joint_point_line_matching

            if options.skip_point_matching:
                raise ValueError(
                    "skip_point_matching cannot be combined with "
                    "use_joint_point_line_matcher: the joint matcher produces "
                    "the point matches and the line matches from the same "
                    "pass."
                )
            assert options.point_descriptor_path is not None, (
                "point_descriptor_path needs to be initialized "
                "to be able to run joint point-line matching"
            )
            assert options.line_descriptor_path is not None, (
                "line_descriptor_path needs to be initialized "
                "to be able to run joint point-line matching"
            )
            if desc_options.joint_point_line_matcher is None:
                raise ValueError(
                    "The description step did not export the joint matcher's "
                    "keypoints, so the COLMAP database holds keypoints it "
                    "cannot match. Set joint_point_line_matcher on the "
                    "description options too (the runners derive it)."
                )
            # point_matcher and line_matcher are usually inherited from a base
            # config, so say plainly that neither is what runs here.
            logging.warning(
                "Joint matching with "
                f"'{options.joint_point_line_matcher.method}' supplies both "
                f"the point matches and the line matches: point_matcher "
                f"('{options.point_matcher.method}') and line_matcher "
                f"('{options.line_matcher.method}') are unused."
            )
            _hloc_pairs_from_neighbors(pairs_path, neighbors, image_names_hloc)
            match_path, matches_dir = joint_point_line_matching(
                options.joint_point_line_matcher,
                image_names_hloc,
                neighbors,
                options.point_descriptor_path,
                options.line_descriptor_path,
                desc_options.line_detection.extractor_method,
                workspace_path / "joint_matchings",
            )
            _import_point_matches(
                hloc, options, recon, db_path, pairs_path, match_path
            )
            with limap.scene.StructureDatabase.open(
                structure_db_path
            ) as structure_db:
                limap.scene.import_line_matches(structure_db, matches_dir)
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
                _import_point_matches(
                    hloc, options, recon, db_path, pairs_path, match_path
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
