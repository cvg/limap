import numpy as np
from tqdm import tqdm
from typeguard import typechecked
from pathlib import Path
from pycolmap import logging

import limap.geometry
import limap.scene
import limap.util.io as limapio

from .specs import (
    VPDetectionOptions,
    PlaneDetectionOptions,
    GroupDescriptionOptions,
)
from .vplib import get_vp_detector, VPResult, convert_vpresults_to_groups2d
from .planelib import get_plane_detector, convert_plane_mask_to_groups2d
from .group_io import get_group_mask_filename, read_group_mask, write_group_mask
from .sam3_utils import load_sam3_data_for_image, convert_sam3_masks_to_groups2d


@typechecked
def vp_detection(
    image_names: dict[int, Path],
    all_2d_lines: dict[int, list[limap.geometry.Line2d]],
    output_dir: Path,
    options: VPDetectionOptions,
) -> dict[int, VPResult]:
    assert image_names.keys() == all_2d_lines.keys(), (
        "image_names and all_2d_segs keys do not match"
    )
    path_to_save = output_dir / options.method / "vpresults.npy"
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    if options.skip_exists and path_to_save.exists():
        vpresults = limapio.read_npy(path_to_save).item()
    else:
        detector = get_vp_detector(options.method, options.options)
        vpresults = detector.detect_vp_all_images(
            all_2d_lines, image_paths=image_names
        )
        limapio.save_npy(path_to_save, vpresults)
    return vpresults


def plane_detection(
    image_names: dict[int, Path],
    output_dir: Path,
    options: PlaneDetectionOptions,
):
    """
    Detect planar regions in images, yielding results one at a time.

    This is a generator to avoid accumulating all plane masks and normal
    maps in memory (which can easily exceed 30 GB for large datasets).

    Args:
        output_dir: Base directory (plane_info). Outputs go to:
            - output_dir/seg_labels/ for segmentation masks
            - output_dir/normal_maps/ for normal maps
            - output_dir/moge_outputs/ for MoGe-specific visualizations
            - output_dir/viz/ for visualizations (if options.visualize)

    Yields:
        (img_id, plane_mask, normal_map) tuples per image
    """
    # seaborn lives in the `viz` extra; import on demand.
    from .planelib.viz import visualize_top_components

    import cv2

    detector = get_plane_detector(options.method, options.options)

    seg_labels_dir = output_dir / "seg_labels"
    normal_maps_dir = output_dir / "normal_maps"
    moge_outputs_dir = output_dir / "moge_outputs"
    seg_labels_dir.mkdir(parents=True, exist_ok=True)
    normal_maps_dir.mkdir(parents=True, exist_ok=True)
    moge_outputs_dir.mkdir(parents=True, exist_ok=True)

    if options.visualize:
        viz_dir = output_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

    for img_id, image_name in tqdm(image_names.items()):
        seg_path = get_group_mask_filename(seg_labels_dir, img_id)
        normal_path = normal_maps_dir / f"normal_{img_id}.npy"

        if options.skip_exists and seg_path.exists():
            plane_mask = read_group_mask(seg_labels_dir, img_id)
            # Load cached normal map if it exists
            if normal_path.exists():
                normal_map = limapio.read_npy(normal_path)
            else:
                normal_map = None
        else:
            # Detect with normals if supported
            result = detector.detect_plane_mask_with_normals(image_name)
            # Handle both 2-tuple (base) and 4-tuple (pxwplanar) returns
            if len(result) == 4:
                plane_mask, normal_map, depth_map, planarity_mask = result
                bg_image = cv2.imread(str(image_name))

                # Save planarity mask and overlay
                cv2.imwrite(
                    str(moge_outputs_dir / f"planarity_{img_id}.png"),
                    planarity_mask,
                )
                overlay = bg_image.copy()
                overlay[planarity_mask > 0] = [0, 255, 0]
                blended = cv2.addWeighted(bg_image, 0.6, overlay, 0.4, 0)
                cv2.imwrite(
                    str(moge_outputs_dir / f"planarity_overlay_{img_id}.png"),
                    blended,
                )

                # Save depth visualization (normalized to 0-255)
                depth_viz = depth_map.copy()
                depth_viz = (depth_viz - depth_viz.min()) / (
                    depth_viz.max() - depth_viz.min() + 1e-8
                )
                depth_viz = (depth_viz * 255).astype(np.uint8)
                depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_TURBO)
                cv2.imwrite(
                    str(moge_outputs_dir / f"depth_{img_id}.png"), depth_viz
                )

                # Save normal visualization (map [-1,1] to [0,255])
                normal_viz = ((normal_map + 1) / 2 * 255).astype(np.uint8)
                normal_viz = cv2.cvtColor(normal_viz, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(moge_outputs_dir / f"normal_{img_id}.png"), normal_viz
                )
            else:
                plane_mask, normal_map = result
            write_group_mask(seg_labels_dir, img_id, plane_mask)
            # Save normal map if available
            if normal_map is not None:
                limapio.save_npy(normal_path, normal_map)

        if options.visualize:
            viz_path = viz_dir / f"plane_mask_{img_id}.png"
            if not viz_path.exists():
                bg_image = cv2.imread(str(image_name))
                viz_img = visualize_top_components(
                    plane_mask, background_image=bg_image
                )
                cv2.imwrite(str(viz_path), viz_img)

        yield img_id, plane_mask, normal_map


@typechecked
def group_description(
    image_names: dict[int, Path],
    output_dir: Path,
    db_path: Path,
    structure_db_path: Path,
    options: GroupDescriptionOptions,
    recon=None,
) -> dict[str, dict[int, int]]:
    """
    Detect groups (vanishing points, planes) and import into the structure DB.

    Points and lines are read from the databases per-image to avoid holding
    all keypoints and line detections in memory simultaneously.

    Args:
        recon: Optional pycolmap.Reconstruction. When provided, keypoints
            are read from recon.images[img_id].points2D instead of the
            COLMAP database. Use this when point detection was skipped
            and the database has no imported keypoints.

    Returns:
        start ids (for each image) corresponding to each group type
    """
    start_ids = {}
    if options.detect_vp:
        logging.info(f"Perform VP detection (n_images = {len(image_names)})")
        # VP detection needs all lines at once; read from structure DB
        with limap.scene.StructureDatabase.open(
            structure_db_path
        ) as structure_db:
            all_2d_lines = {
                img_id: list(structure_db.read_structure2d(img_id).lines)
                for img_id in image_names
            }
        vpresults = vp_detection(
            image_names,
            all_2d_lines,
            output_dir / "vp_detections",
            options.vp_detection,
        )
        del all_2d_lines
        all_groups2d_vp = convert_vpresults_to_groups2d(vpresults)
        with limap.scene.StructureDatabase.open(
            structure_db_path
        ) as structure_db:
            start_ids["VP"] = {
                img_id: structure_db.read_structure2d(img_id).num_groups()
                for img_id in image_names
            }
            limap.scene.import_group_descriptions(structure_db, all_groups2d_vp)

    if options.detect_plane:
        plane_groups_path = output_dir / "plane_groups.npy"
        if options.plane_detection.skip_exists and plane_groups_path.exists():
            all_group2ds_plane = limapio.read_npy(plane_groups_path).item()
        else:
            # Stream plane detection: detect one image at a time and
            # convert to groups2d immediately. This avoids accumulating
            # all plane masks (~8 GB) and normal maps (~26 GB) in memory
            # for large datasets. Points and lines are read per-image
            # from the databases.
            import pycolmap

            all_group2ds_plane = {}
            with (
                pycolmap.Database.open(db_path) as db,
                limap.scene.StructureDatabase.open(
                    structure_db_path
                ) as structure_db,
            ):
                for img_id, mask, normal_map in plane_detection(
                    image_names,
                    output_dir / "plane_detections",
                    options.plane_detection,
                ):
                    if recon is not None:
                        points = np.array(
                            [p.xy for p in recon.images[img_id].points2D]
                        )
                    else:
                        points = db.read_keypoints(img_id)
                    lines = list(structure_db.read_structure2d(img_id).lines)
                    all_group2ds_plane[img_id] = convert_plane_mask_to_groups2d(
                        mask,
                        points,
                        lines,
                        min_line_overlap_length=options.plane_min_line_overlap_length,
                        dilation_radius=options.plane_dilation_radius,
                        normal_map=normal_map,
                    )
            limapio.save_npy(plane_groups_path, all_group2ds_plane)

        with limap.scene.StructureDatabase.open(
            structure_db_path
        ) as structure_db:
            start_ids["PLANE"] = {
                img_id: structure_db.read_structure2d(img_id).num_groups()
                for img_id in image_names
            }
            limap.scene.import_group_descriptions(
                structure_db, all_group2ds_plane
            )
    limapio.save_npy(output_dir / "start_ids.npy", start_ids)
    return start_ids


@typechecked
def sam3_group_description(
    image_names: dict[int, Path],
    sam3_base_dir: Path,
    category_to_type: dict[str, limap.geometry.GroupType],
    output_dir: Path,
    db_path: Path,
    structure_db_path: Path,
    existing_start_ids: dict | None = None,
    min_line_overlap_length: float = 40.0,
    dilation_radius: float = 2.0,
    recon=None,
) -> dict[str, dict[int, int]]:
    """
    Load SAM3 segmentation results and import as Group2d into structure DB.

    Creates groups for each GroupType found in category_to_type, associates
    keypoints and lines, saves label masks for dense matching, and imports
    the groups into the structure database.

    Args:
        image_names: Mapping from COLMAP image_id to image path
        sam3_base_dir: Base directory of SAM3 output
        category_to_type: Mapping from SAM3 category name to GroupType,
            e.g. {"can": GroupType.CYLINDER, "football": GroupType.SPHERE}
        output_dir: Workspace directory for saving label masks.
            Masks go to output_dir/{TYPE}_detections/seg_labels/
        db_path: Path to COLMAP database (for reading keypoints)
        structure_db_path: Path to structure database
        existing_start_ids: Existing start_ids dict to merge with
            (e.g. from VP/plane detection)
        min_line_overlap_length: Minimum overlap length for line association
        dilation_radius: Dilation radius for point/line association
        recon: Optional pycolmap.Reconstruction for reading keypoints

    Returns:
        Merged start_ids dict (includes existing + new SAM3 group types)
    """
    import pycolmap

    # Build reverse mapping: image_name -> colmap image_id
    name_to_id = {}
    for img_id, img_path in image_names.items():
        name_to_id[img_path.name] = img_id

    # Collect unique group types from category mapping
    type_to_subdir = {
        limap.geometry.GroupType.PLANE: "plane_detections",
        limap.geometry.GroupType.CYLINDER: "cylinder_detections",
        limap.geometry.GroupType.SPHERE: "sphere_detections",
    }

    # Group categories by type
    type_to_categories: dict[limap.geometry.GroupType, list[str]] = {}
    for cat_name, gtype in category_to_type.items():
        type_to_categories.setdefault(gtype, []).append(cat_name)

    start_ids = dict(existing_start_ids) if existing_start_ids else {}

    # Process each group type
    for gtype, cat_names in type_to_categories.items():
        type_name = gtype.name  # e.g. "CYLINDER", "SPHERE", "PLANE"
        subdir = type_to_subdir.get(gtype)
        if subdir is None:
            logging.warning(f"Skipping unknown group type: {type_name}")
            continue

        seg_labels_dir = output_dir / subdir / "seg_labels"
        seg_labels_dir.mkdir(parents=True, exist_ok=True)

        # Build per-type category subset
        type_categories = {c: gtype for c in cat_names}

        # Record start IDs and collect groups
        all_groups2d: dict[int, list[limap.scene.Group2d]] = {}

        with (
            pycolmap.Database.open(db_path) as db,
            limap.scene.StructureDatabase.open(
                structure_db_path
            ) as structure_db,
        ):
            # Get working image size from cameras in the database.
            # This handles --max_image_dim resizing correctly.
            db_cameras = {c.camera_id: c for c in db.read_all_cameras()}
            db_images = {img.image_id: img for img in db.read_all_images()}

            # Record start IDs: current group count per image
            start_ids[type_name] = {
                img_id: structure_db.read_structure2d(img_id).num_groups()
                for img_id in image_names
            }

            for img_id, img_path in tqdm(
                image_names.items(),
                desc=f"SAM3 {type_name} groups",
            ):
                image_name = img_path.name
                sam3_data = load_sam3_data_for_image(
                    sam3_base_dir, type_categories, image_name
                )

                # Get working image dimensions from DB camera
                db_img = db_images[img_id]
                cam = db_cameras[db_img.camera_id]
                target_size = (cam.height, cam.width)

                if gtype not in sam3_data:
                    # No detections for this type in this image
                    empty_mask = np.zeros(target_size, dtype=np.int32)
                    write_group_mask(seg_labels_dir, img_id, empty_mask)
                    all_groups2d[img_id] = []
                    continue

                binary_masks, scores = sam3_data[gtype]

                # Read keypoints and lines
                if recon is not None:
                    points = np.array(
                        [p.xy for p in recon.images[img_id].points2D]
                    )
                else:
                    points = db.read_keypoints(img_id)
                lines = list(structure_db.read_structure2d(img_id).lines)

                groups, label_mask = convert_sam3_masks_to_groups2d(
                    binary_masks,
                    gtype,
                    points,
                    lines,
                    scores=scores,
                    min_line_overlap_length=min_line_overlap_length,
                    dilation_radius=dilation_radius,
                    target_size=target_size,
                )

                write_group_mask(seg_labels_dir, img_id, label_mask)
                all_groups2d[img_id] = groups

            # Import groups into structure DB
            limap.scene.import_group_descriptions(structure_db, all_groups2d)

        logging.info(
            f"SAM3 {type_name}: imported groups for {len(image_names)} images"
        )

    # Merge and save start_ids
    limapio.save_npy(output_dir / "start_ids.npy", start_ids)
    return start_ids
