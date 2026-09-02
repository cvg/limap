import json
from pathlib import Path

import numpy as np
import open3d as o3d
import seaborn as sns

from limap.geometry import GroupType
from limap.scene import HolisticReconstruction
from limap.util.types import Color, Ranges
import limap.util.io as limapio
import limap.visualize as limapvis
from limap.visualize.texture import (
    create_textured_plane_mesh,
    create_textured_cylinder_mesh,
    create_textured_sphere_mesh,
)
from limap.visualize.viz_utils import (
    test_point_inside_ranges,
    test_line_inside_ranges,
)


def load_viewpoint(path: str) -> dict | None:
    """Load viewpoint parameters from JSON file."""
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def get_viewpoint_from_image(
    hrecon: HolisticReconstruction, image_name: str
) -> dict | None:
    """Get viewpoint parameters from an image's camera pose."""
    point_recon = hrecon.point_recon

    # Find image by name
    target_image = None
    for img in point_recon.images.values():
        if img.name == image_name or Path(img.name).stem == image_name:
            target_image = img
            break

    if target_image is None:
        print(f"Warning: image '{image_name}' not found in reconstruction")
        return None

    # Get camera pose
    cam_from_world = target_image.cam_from_world()
    R = np.array(cam_from_world.rotation.matrix())
    t = np.array(cam_from_world.translation)

    # Camera position in world coords
    eye = -R.T @ t

    # Camera looks along +Z in camera space, so forward is R.T @ [0,0,1]
    forward = R.T @ np.array([0, 0, 1])
    lookat = eye + forward

    # Up is -Y in camera space
    up = -R.T @ np.array([0, 1, 0])

    # Get FOV from camera intrinsics
    camera = point_recon.cameras[target_image.camera_id]
    fov = 60.0  # default
    if hasattr(camera, "focal_length_x"):
        # Approximate FOV from focal length
        fov = 2 * np.arctan(camera.height / (2 * camera.focal_length_x))
        fov = np.degrees(fov)

    return {
        "eye": eye.tolist(),
        "lookat": lookat.tolist(),
        "up": up.tolist(),
        "field_of_view": fov,
    }


def save_viewpoint(path: str, eye, lookat, up, fov: float):
    """Save viewpoint parameters to JSON file."""
    data = {
        "eye": [float(x) for x in eye],
        "lookat": [float(x) for x in lookat],
        "up": [float(x) for x in up],
        "field_of_view": float(fov),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Viewpoint saved to {path}")


def get_group_colors(n_groups: int, soft: bool = False) -> list[Color]:
    """Generate distinct colors for groups using seaborn."""
    if n_groups == 0:
        return []
    palette_name = "tab10" if soft else "husl"
    palette = sns.color_palette(palette_name, n_groups)
    return [tuple(c) for c in palette]


def parse_args():
    import argparse

    arg_parser = argparse.ArgumentParser(
        description=(
            "visualize points and lines in holistic reconstruction "
            "using Open3D backend"
        )
    )
    arg_parser.add_argument(
        "-i", "--input_dir", type=str, required=True, help="input colmap folder"
    )
    arg_parser.add_argument(
        "--only_show_stats",
        action="store_true",
        help="Print reconstruction statistics and exit without visualization",
    )
    arg_parser.add_argument(
        "--min_track_length",
        type=int,
        default=2,
        help="Minimum track length (observations) for points and lines.",
    )
    arg_parser.add_argument(
        "--disable_robust_ranges",
        action="store_true",
        help="whether to use computed robust ranges",
    )
    arg_parser.add_argument(
        "--point_size",
        type=float,
        default=2.0,
        help="Point size",
    )
    arg_parser.add_argument(
        "--line_width",
        type=float,
        default=1.0,
        help="Line width",
    )
    arg_parser.add_argument(
        "--cam_scale",
        type=float,
        default=0.1,
        help="scale of the camera geometry",
    )
    arg_parser.add_argument(
        "--reproj_error_thresh",
        type=float,
        default=2.0,
        help="reprojection error threshold",
    )
    arg_parser.add_argument(
        "--color_points_by_recon",
        action="store_true",
        help="Use per-point RGB colors from COLMAP reconstruction",
    )
    arg_parser.add_argument(
        "--show_planes",
        action="store_true",
        help="whether to visualize plane groups",
    )
    arg_parser.add_argument(
        "--plane_alpha",
        type=float,
        default=0.5,
        help="plane transparency (0=transparent, 1=opaque)",
    )
    arg_parser.add_argument(
        "--plane_padding",
        type=float,
        default=0.4,
        help="relative padding around plane bounding box",
    )
    arg_parser.add_argument(
        "--min_associated_features",
        type=int,
        default=20,
        help="minimum number of associated features (points + lines) for plane",
    )
    arg_parser.add_argument(
        "--manhattan_planes",
        action="store_true",
        help="align plane bounding boxes with Manhattan world directions (VPs)",
    )
    arg_parser.add_argument(
        "--atlanta_planes",
        action="store_true",
        help="align vertical plane bboxes with gravity estimated from VPs",
    )
    arg_parser.add_argument(
        "--hide_lines",
        action="store_true",
        help="hide 3D lines from visualization",
    )
    arg_parser.add_argument(
        "--color_lines_by_vps",
        action="store_true",
        help="color lines by their vanishing point group",
    )
    arg_parser.add_argument(
        "--show_wireframes",
        action="store_true",
        help="show wireframe edges connecting points to lines (dashed)",
    )
    arg_parser.add_argument(
        "--show_spheres",
        action="store_true",
        help="whether to visualize sphere groups",
    )
    arg_parser.add_argument(
        "--sphere_alpha",
        type=float,
        default=0.5,
        help="sphere transparency (0=transparent, 1=opaque)",
    )
    arg_parser.add_argument(
        "--show_cylinders",
        action="store_true",
        help="whether to visualize cylinder groups",
    )
    arg_parser.add_argument(
        "--cylinder_alpha",
        type=float,
        default=0.5,
        help="cylinder transparency (0=transparent, 1=opaque)",
    )
    # Textured mesh options
    arg_parser.add_argument(
        "--textured",
        action="store_true",
        help="Use image-backed vertex-colored meshes instead of uniform colors",
    )
    arg_parser.add_argument(
        "--texture_workspace",
        type=str,
        default=None,
        help="Group workspace dir containing start_ids.npy and "
        "*_detections/seg_labels/. Required when --textured is used.",
    )
    arg_parser.add_argument(
        "--texture_image_dir",
        type=str,
        default=None,
        help="Undistorted images directory. Required when --textured is used.",
    )
    arg_parser.add_argument(
        "--texture_mesh_resolution",
        type=int,
        default=50,
        help="Grid density for planes / angular resolution for "
        "cylinders and spheres (textured mode)",
    )
    arg_parser.add_argument(
        "--texture_export_dir",
        type=str,
        default=None,
        help="Export textured meshes as PLY files to this directory",
    )
    arg_parser.add_argument(
        "--texture_use_all_images",
        action="store_true",
        help="Scan all images with masks (not just track images)",
    )
    arg_parser.add_argument(
        "--load_viewpoint",
        type=str,
        default=None,
        help="Load viewpoint from this JSON file.",
    )
    arg_parser.add_argument(
        "--viewpoint_image",
        type=str,
        default=None,
        help="Set viewpoint to match this image's camera pose (by name).",
    )
    arg_parser.add_argument(
        "--save_viewpoint",
        type=str,
        default=None,
        help="Save viewpoint to this JSON file (via Actions menu).",
    )
    arg_parser.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="Save screenshot to this path (PNG). Use action menu to save.",
    )
    arg_parser.add_argument(
        "--new_renderer",
        action="store_true",
        help="Use new Filament renderer (supports transparency, darker look).",
    )
    arg_parser.add_argument(
        "--line_max_pixel_uncertainty",
        type=float,
        default=30,
        help="Maximum pixel variance for line filtering (-1 to disable). "
        "Uses backprojected endpoints for per-view variance estimation.",
    )
    args = arg_parser.parse_args()
    return args


def get_line_sets_by_vp(
    hrecon: HolisticReconstruction,
    min_track_length: int = 2,
    ranges: Ranges | None = None,
    default_color: Color = (0.5, 0.5, 0.5),
    min_associated_features: int = 3,
    pixel_uncertainties: dict | None = None,
    max_pixel_uncertainty: float = -1,
) -> list[tuple[o3d.geometry.LineSet, str]]:
    """
    Get line sets grouped by vanishing point.

    Returns list of (LineSet, name) tuples for visualization.
    """
    structure_recon = hrecon.structure_recon

    # Get all VP groups that meet min_associated_features threshold
    vp_groups = [
        (gid, g)
        for gid, g in structure_recon.groups3D.items()
        if g.type == GroupType.VP
        and (len(g.points) + len(g.lines)) >= min_associated_features
    ]
    vp_colors = get_group_colors(len(vp_groups))

    # Build line_id -> vp_idx mapping
    line_to_vp: dict[int, int] = {}
    for vp_idx, (_gid, group) in enumerate(vp_groups):
        for assoc_ln in group.lines:
            line_to_vp[assoc_ln.idx] = vp_idx

    # Group lines by VP
    lines_by_vp: dict[int, list] = {i: [] for i in range(len(vp_groups))}
    lines_no_vp: list = []

    for line_id, line in structure_recon.lines3D.items():
        if line.track.length() < min_track_length:
            continue
        # Apply pixel uncertainty filter if enabled
        if (
            pixel_uncertainties is not None
            and max_pixel_uncertainty > 0
            and pixel_uncertainties.get(line_id, float("inf"))
            > max_pixel_uncertainty
        ):
            continue
        if line_id in line_to_vp:
            lines_by_vp[line_to_vp[line_id]].append(line)
        else:
            lines_no_vp.append(line)

    # Create line sets
    result = []

    # Lines without VP (gray)
    if lines_no_vp:
        line_set = limapvis.open3d_get_3d_lines(
            lines_no_vp, ranges=ranges, color=default_color
        )
        result.append((line_set, "lines_no_vp"))

    # Lines per VP with distinct colors
    for vp_idx, lines in lines_by_vp.items():
        if lines:
            line_set = limapvis.open3d_get_3d_lines(
                lines, ranges=ranges, color=vp_colors[vp_idx]
            )
            result.append((line_set, f"lines_vp_{vp_idx}"))

    n_with_vp = sum(len(lines) for lines in lines_by_vp.values())
    print(
        f"Lines by VP: {n_with_vp} assigned to {len(vp_groups)} VPs, "
        f"{len(lines_no_vp)} unassigned"
    )

    return result


def _closest_point_on_segment(
    point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray
) -> np.ndarray:
    """Find the closest point on a line segment to a given point."""
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)
    if seg_len_sq < 1e-12:
        return seg_start
    t = np.clip(np.dot(point - seg_start, seg_vec) / seg_len_sq, 0.0, 1.0)
    return seg_start + t * seg_vec


def _generate_dashed_segments(
    start: np.ndarray,
    end: np.ndarray,
    num_dashes: int = 10,
    dash_ratio: float = 0.6,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate dashed line segments between start and end points.

    Divides the line into num_dashes dashes with gaps between them.
    Pattern: dash - gap - dash - gap - ... - dash

    Args:
        start: Start point of the line
        end: End point of the line
        num_dashes: Number of dash segments
        dash_ratio: Fraction of each slot that is dash vs gap (0.6 = 60% dash)

    Returns list of (segment_start, segment_end) tuples.
    """
    direction = end - start
    total_length = np.linalg.norm(direction)
    if total_length < 1e-9:
        return []

    direction = direction / total_length

    # Pattern has num_dashes dashes and (num_dashes - 1) gaps
    num_slots = num_dashes + (num_dashes - 1) * (1 - dash_ratio) / dash_ratio
    slot_length = total_length / num_slots
    dash_length = slot_length
    gap_length = slot_length * (1 - dash_ratio) / dash_ratio

    segments = []
    pos = 0.0

    for _ in range(num_dashes):
        seg_start = start + direction * pos
        seg_end_pos = min(pos + dash_length, total_length)
        seg_end = start + direction * seg_end_pos
        segments.append((seg_start, seg_end))
        pos = seg_end_pos + gap_length

    return segments


def get_wireframe_lineset(
    hrecon: HolisticReconstruction,
    ranges: Ranges | None = None,
    color: Color = (0.6, 0.6, 0.6),
    num_dashes: int = 10,
) -> o3d.geometry.LineSet | None:
    """
    Create a dashed LineSet visualizing wireframe edges.

    Args:
        hrecon: Holistic reconstruction containing wireframe data
        ranges: Optional range filter for points
        color: RGB color for wireframe edges (default: soft gray)
        num_dashes: Number of dash segments per edge (default: 10)

    Returns:
        Open3D LineSet with dashed wireframe edges, or None if no edges
    """
    structure_recon = hrecon.structure_recon
    point_recon = hrecon.point_recon
    wireframe = structure_recon.wireframe

    if wireframe is None or wireframe.count_edges() == 0:
        print("No wireframe edges to visualize")
        return None

    all_points = []
    all_lines = []

    edges = wireframe.get_all_edges()
    for edge in edges:
        pt_id = edge.point_idx
        ln_id = edge.line_idx

        # Get 3D point coordinates
        if pt_id not in point_recon.points3D:
            continue
        pt_xyz = np.array(point_recon.points3D[pt_id].xyz)

        # Get 3D line
        if ln_id not in structure_recon.lines3D:
            continue
        line = structure_recon.lines3D[ln_id]
        line_start = np.array(line.start)
        line_end = np.array(line.end)

        # Find closest point on line to the 3D point
        closest_on_line = _closest_point_on_segment(
            pt_xyz, line_start, line_end
        )

        # Apply range filter if provided
        # ranges is (min_array, max_array) where each array has [x, y, z]
        if ranges is not None:
            mins, maxs = ranges
            pt_out = not all(mins[i] <= pt_xyz[i] <= maxs[i] for i in range(3))
            closest_out = not all(
                mins[i] <= closest_on_line[i] <= maxs[i] for i in range(3)
            )
            if pt_out or closest_out:
                continue

        # Generate dashed segments
        dashed_segs = _generate_dashed_segments(
            pt_xyz, closest_on_line, num_dashes
        )

        for seg_start, seg_end in dashed_segs:
            idx_start = len(all_points)
            all_points.append(seg_start)
            all_points.append(seg_end)
            all_lines.append([idx_start, idx_start + 1])

    if not all_points:
        print("No wireframe edges to visualize after filtering")
        return None

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(all_points))
    line_set.lines = o3d.utility.Vector2iVector(
        np.array(all_lines, dtype=np.int32)
    )
    line_set.colors = o3d.utility.Vector3dVector(
        np.array([color] * len(all_lines), dtype=np.float64)
    )

    print(
        f"Wireframe: {wireframe.count_edges()} edges -> "
        f"{len(all_lines)} dashed segments"
    )
    return line_set


def estimate_manhattan_frame(
    hrecon: HolisticReconstruction,
) -> np.ndarray | None:
    """Estimate 3 orthogonal Manhattan world directions from VP groups.

    Strategy:
    1. Rank VPs by number of associated lines (most supported first).
    2. Take the strongest VP as axis 1.
    3. Find the VP most orthogonal to axis 1 as axis 2, then orthogonalize.
    4. Axis 3 = cross product of axes 1 and 2.

    Returns:
        3x3 array of unit directions, or None if fewer than 2 VPs.
    """
    structure_recon = hrecon.structure_recon
    vp_info = []
    for g in structure_recon.groups3D.values():
        if g.type != GroupType.VP:
            continue
        n_lines = len(g.lines)
        direction = np.array(g.params[:3])
        vp_info.append((n_lines, direction))

    if len(vp_info) < 2:
        return None

    # Sort by support (most associated lines first)
    vp_info.sort(key=lambda x: x[0], reverse=True)

    # Axis 1: strongest VP
    axis1 = vp_info[0][1]
    axis1 = axis1 / np.linalg.norm(axis1)

    # Axis 2: VP most orthogonal to axis1 (smallest |dot product|)
    best_idx = 1
    best_orth = abs(np.dot(vp_info[1][1], axis1))
    for i in range(2, len(vp_info)):
        orth = abs(np.dot(vp_info[i][1], axis1))
        if orth < best_orth:
            best_orth = orth
            best_idx = i

    axis2 = vp_info[best_idx][1]
    # Orthogonalize axis2 w.r.t. axis1
    axis2 = axis2 - np.dot(axis2, axis1) * axis1
    if np.linalg.norm(axis2) < 1e-8:
        return None
    axis2 = axis2 / np.linalg.norm(axis2)

    # Axis 3: cross product
    axis3 = np.cross(axis1, axis2)
    axis3 = axis3 / np.linalg.norm(axis3)

    return np.array([axis1, axis2, axis3])


def estimate_atlanta_gravity(
    hrecon: HolisticReconstruction,
) -> np.ndarray | None:
    """Estimate gravity direction from VP groups for an Atlanta world model.

    Gravity is identified as the most-supported VP whose direction is
    "roughly vertical" (largest absolute component along some axis).
    Among VPs with at least 3 associated lines, we pick the one whose
    direction has the largest absolute Z component (assuming Z-up or
    Z-down convention).  If no clear vertical VP is found, we fall back
    to the most-supported VP overall.

    Returns:
        Unit gravity vector (pointing in the dominant vertical direction),
        or None if fewer than 1 VP exists.
    """
    structure_recon = hrecon.structure_recon
    vp_info = []
    for g in structure_recon.groups3D.values():
        if g.type != GroupType.VP:
            continue
        n_lines = len(g.lines)
        direction = np.array(g.params[:3])
        direction = direction / np.linalg.norm(direction)
        vp_info.append((n_lines, direction))

    if len(vp_info) == 0:
        return None

    # Sort by support (most associated lines first)
    vp_info.sort(key=lambda x: x[0], reverse=True)

    # Among well-supported VPs, find the most vertical one
    # "Vertical" = largest absolute component along any single axis,
    # then we pick that axis direction.  This is robust to Z-up / Y-up.
    best_gravity = None
    best_score = -1.0
    for n_lines, direction in vp_info:
        # Score: how "vertical" is this VP (max absolute component)
        verticality = np.max(np.abs(direction))
        # Weight by support count (log scale to avoid domination)
        score = verticality * np.log1p(n_lines)
        if score > best_score:
            best_score = score
            best_gravity = direction

    if best_gravity is None:
        return None

    # Ensure consistent sign: point toward positive dominant axis
    dominant_axis = np.argmax(np.abs(best_gravity))
    if best_gravity[dominant_axis] < 0:
        best_gravity = -best_gravity

    return best_gravity


def get_plane_meshes(
    hrecon: HolisticReconstruction,
    ranges: Ranges | None = None,
    min_associated_features: int = 3,
    alpha: float = 0.5,
    padding: float = 0.2,
    soft_colors: bool = False,
    manhattan_directions: np.ndarray | None = None,
    atlanta_gravity: np.ndarray | None = None,
) -> list[limapvis.PlaneMesh]:
    """
    Get plane meshes for all PLANE groups in the reconstruction.
    """
    structure_recon = hrecon.structure_recon
    point_recon = hrecon.point_recon

    # Get all plane groups
    plane_groups = [
        (gid, g)
        for gid, g in structure_recon.groups3D.items()
        if g.type == GroupType.PLANE
    ]
    colors = get_group_colors(len(plane_groups), soft=soft_colors)

    meshes = []
    for plane_idx, (group_id, group) in enumerate(plane_groups):
        # Collect associated points
        pts_list = []
        for assoc_pt in group.points:
            pt_id = assoc_pt.idx
            if pt_id in point_recon.points3D:
                pt = point_recon.points3D[pt_id].xyz
                if ranges is not None and not test_point_inside_ranges(
                    pt, ranges
                ):
                    continue
                pts_list.append(pt)

        # Collect associated line endpoints
        for assoc_ln in group.lines:
            ln_id = assoc_ln.idx
            if ln_id in structure_recon.lines3D:
                line = structure_recon.lines3D[ln_id]
                if ranges is not None and not test_line_inside_ranges(
                    line, ranges
                ):
                    continue
                pts_list.append(np.array(line.start))
                pts_list.append(np.array(line.end))

        num_associated_features = len(group.points) + len(group.lines)
        if num_associated_features < min_associated_features:
            continue
        if len(pts_list) < 3:
            continue

        pts_array = np.array(pts_list)
        plane_mesh = limapvis.open3d_get_plane_mesh(
            group.params,
            pts_array,
            color=colors[plane_idx],
            padding=padding,
            alpha=alpha,
            manhattan_directions=manhattan_directions,
            atlanta_gravity=atlanta_gravity,
        )
        if plane_mesh is not None:
            meshes.append(plane_mesh)
            p = group.params
            print(
                f"Plane {group_id}: {len(group.points)} pts, "
                f"{len(group.lines)} lns, "
                f"normal=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}], d={p[3]:.3f}"
            )

    return meshes


def _collect_group_3d_points(
    hrecon: HolisticReconstruction,
    group,
    ranges: Ranges | None = None,
) -> np.ndarray:
    """Collect 3D points and line endpoints associated with a group."""
    structure_recon = hrecon.structure_recon
    point_recon = hrecon.point_recon
    pts_list = []

    for assoc_pt in group.points:
        pt_id = assoc_pt.idx
        if pt_id in point_recon.points3D:
            pt = point_recon.points3D[pt_id].xyz
            if ranges is not None and not test_point_inside_ranges(pt, ranges):
                continue
            pts_list.append(pt)

    for assoc_ln in group.lines:
        ln_id = assoc_ln.idx
        if ln_id in structure_recon.lines3D:
            line = structure_recon.lines3D[ln_id]
            if ranges is not None and not test_line_inside_ranges(line, ranges):
                continue
            pts_list.append(np.array(line.start))
            pts_list.append(np.array(line.end))

    if len(pts_list) == 0:
        return np.empty((0, 3))
    return np.array(pts_list)


def get_sphere_meshes(
    hrecon: HolisticReconstruction,
    ranges: Ranges | None = None,
    min_associated_features: int = 3,
    alpha: float = 0.5,
    soft_colors: bool = False,
) -> list[limapvis.SphereMesh]:
    """Get sphere meshes for all SPHERE groups in the reconstruction."""
    structure_recon = hrecon.structure_recon

    sphere_groups = [
        (gid, g)
        for gid, g in structure_recon.groups3D.items()
        if g.type == GroupType.SPHERE
    ]
    colors = get_group_colors(len(sphere_groups), soft=soft_colors)

    meshes = []
    for idx, (group_id, group) in enumerate(sphere_groups):
        num_associated = len(group.points) + len(group.lines)
        if num_associated < min_associated_features:
            continue

        mesh = limapvis.open3d_get_sphere_mesh(
            group.params,
            color=colors[idx],
            alpha=alpha,
        )
        if mesh is not None:
            meshes.append(mesh)
            p = group.params
            r = np.exp(p[3])
            print(
                f"Sphere {group_id}: {len(group.points)} pts, "
                f"{len(group.lines)} lns, "
                f"center=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}], r={r:.3f}"
            )

    return meshes


def get_cylinder_meshes(
    hrecon: HolisticReconstruction,
    ranges: Ranges | None = None,
    min_associated_features: int = 3,
    alpha: float = 0.5,
    soft_colors: bool = False,
) -> list[limapvis.CylinderMesh]:
    """Get cylinder meshes for all CYLINDER groups in the reconstruction."""
    structure_recon = hrecon.structure_recon

    cylinder_groups = [
        (gid, g)
        for gid, g in structure_recon.groups3D.items()
        if g.type == GroupType.CYLINDER
    ]
    colors = get_group_colors(len(cylinder_groups), soft=soft_colors)

    meshes = []
    for idx, (group_id, group) in enumerate(cylinder_groups):
        num_associated = len(group.points) + len(group.lines)
        if num_associated < min_associated_features:
            continue

        pts_array = _collect_group_3d_points(hrecon, group, ranges=ranges)
        if pts_array.shape[0] < 2:
            continue

        mesh = limapvis.open3d_get_cylinder_mesh(
            group.params,
            pts_array,
            color=colors[idx],
            alpha=alpha,
        )
        if mesh is not None:
            meshes.append(mesh)
            r = np.exp(group.params[6])
            print(
                f"Cylinder {group_id}: {len(group.points)} pts, "
                f"{len(group.lines)} lns, r={r:.3f}"
            )

    return meshes


def get_textured_group_meshes(
    hrecon: HolisticReconstruction,
    group_workspace: Path,
    image_dir: Path,
    show_planes: bool = False,
    show_cylinders: bool = False,
    show_spheres: bool = False,
    min_associated_features: int = 3,
    mesh_resolution: int = 50,
    padding: float = 0.2,
    use_all_masks: bool = False,
) -> list[tuple[o3d.geometry.TriangleMesh, str]]:
    """Generate textured meshes for requested group types.

    Returns list of (mesh, name) tuples for visualization.
    """
    structure_recon = hrecon.structure_recon

    # Load start_ids
    start_ids_path = group_workspace / "start_ids.npy"
    if not start_ids_path.exists():
        print(f"Warning: start_ids.npy not found at {start_ids_path}")
        return []
    start_ids = limapio.read_npy(start_ids_path).item()

    requested_types = set()
    if show_planes:
        requested_types.add(GroupType.PLANE)
    if show_cylinders:
        requested_types.add(GroupType.CYLINDER)
    if show_spheres:
        requested_types.add(GroupType.SPHERE)

    results = []
    for group_id, group in structure_recon.groups3D.items():
        if group.type not in requested_types:
            continue
        num_assoc = len(group.points) + len(group.lines)
        if num_assoc < min_associated_features:
            continue

        type_name = group.type.name.lower()
        mesh = None

        if group.type == GroupType.PLANE:
            pts = _collect_group_3d_points(hrecon, group)
            if pts.shape[0] < 3:
                continue
            mesh = create_textured_plane_mesh(
                group3d_id=group_id,
                group=group,
                associated_points=pts,
                hrecon=hrecon,
                start_ids=start_ids,
                group_workspace=group_workspace,
                image_dir=image_dir,
                mesh_resolution=mesh_resolution,
                padding=padding,
                use_all_masks=use_all_masks,
            )
        elif group.type == GroupType.CYLINDER:
            pts = _collect_group_3d_points(hrecon, group)
            if pts.shape[0] < 2:
                continue
            mesh = create_textured_cylinder_mesh(
                group3d_id=group_id,
                group=group,
                associated_points=pts,
                hrecon=hrecon,
                start_ids=start_ids,
                group_workspace=group_workspace,
                image_dir=image_dir,
                mesh_resolution=mesh_resolution,
                use_all_masks=use_all_masks,
            )
        elif group.type == GroupType.SPHERE:
            mesh = create_textured_sphere_mesh(
                group3d_id=group_id,
                group=group,
                hrecon=hrecon,
                start_ids=start_ids,
                group_workspace=group_workspace,
                image_dir=image_dir,
                mesh_resolution=mesh_resolution,
                use_all_masks=use_all_masks,
            )

        if mesh is not None:
            n_v = len(np.asarray(mesh.vertices))
            n_t = len(np.asarray(mesh.triangles))
            print(f"  Textured {type_name} {group_id}: {n_v} verts, {n_t} tris")
            results.append((mesh, f"textured_{type_name}_{group_id}"))

    print(f"Generated {len(results)} textured meshes")
    return results


def show_stats(
    hrecon: HolisticReconstruction, min_associated_features: int = 3
):
    """Print reconstruction statistics and exit."""
    point_recon = hrecon.point_recon
    structure_recon = hrecon.structure_recon

    print("=" * 60)
    print("Reconstruction Statistics")
    print("=" * 60)
    print(f"  Cameras: {len(point_recon.cameras)}")
    print(f"  Images:  {len(point_recon.images)}")
    print(f"  Points:  {len(point_recon.points3D)}")
    print(f"  Lines:   {len(structure_recon.lines3D)}")

    # Groups by type, filtered by min_associated_features
    type_counts: dict[str, tuple[int, int]] = {}  # name -> (passing, total)
    for group in structure_recon.groups3D.values():
        name = group.type.name
        num_assoc = len(group.points) + len(group.lines)
        total_prev, passing_prev = type_counts.get(name, (0, 0))
        passes = 1 if num_assoc >= min_associated_features else 0
        type_counts[name] = (total_prev + 1, passing_prev + passes)

    total_groups = sum(t for t, _ in type_counts.values())
    total_passing = sum(p for _, p in type_counts.values())
    print(
        f"  Groups:  {total_passing} / {total_groups} total "
        f"(min_associated_features={min_associated_features})"
    )
    for name, (total, passing) in sorted(type_counts.items()):
        print(f"    {name}: {passing} / {total}")

    # Wireframe
    wf = structure_recon.wireframe
    if wf is not None and wf.count_edges() > 0:
        print(f"  Wireframe edges: {wf.count_edges()}")

    print("=" * 60)


def vis_holistic_reconstruction(
    hrecon: HolisticReconstruction,
    ranges: Ranges | None = None,
):
    # Filter points by track length
    points3D = [
        p
        for p in hrecon.point_recon.points3D.values()
        if p.track.length() >= args.min_track_length
    ]
    pts = np.array([p.xyz for p in points3D], dtype=np.float32)
    if args.color_points_by_recon:
        colors = np.array([p.color for p in points3D], dtype=np.float32) / 255.0
    if args.reproj_error_thresh > 0.0:
        errs = np.array([p.error for p in points3D], dtype=np.float32)
        mask = errs <= args.reproj_error_thresh
        pts = pts[mask]
        if args.color_points_by_recon:
            colors = colors[mask]
    # Apply range filtering
    if ranges is not None and pts.shape[0] > 0:
        mask = np.all((pts >= ranges[0]) & (pts <= ranges[1]), axis=1)
        pts = pts[mask]
        if args.color_points_by_recon:
            colors = colors[mask]
    print(f"Number of valid points for visualization: {pts.shape[0]}")
    pcd = o3d.geometry.PointCloud()
    if pts.shape[0] > 0:
        pcd.points = o3d.utility.Vector3dVector(pts)
        if args.color_points_by_recon:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.paint_uniform_color([0.0, 0.0, 0.0])
    camera_set = limapvis.open3d_get_camera_frustums(
        hrecon.point_recon,
        ranges=ranges,
        scale_cam_geometry=args.cam_scale,
    )

    # Compute pixel variances if filtering is requested
    pixel_uncertainties = None
    if args.line_max_pixel_uncertainty > 0:
        print("Computing pixel variances with backprojected endpoints...")
        pixel_uncertainties = (
            hrecon.structure_recon.compute_line_pixel_uncertainties()
        )
        # Count lines passing filter
        n_passing = sum(
            1
            for line_id in hrecon.structure_recon.lines3D
            if pixel_uncertainties.get(line_id, float("inf"))
            <= args.line_max_pixel_uncertainty
        )
        print(
            f"Lines passing pixel variance filter "
            f"(<= {args.line_max_pixel_uncertainty}): {n_passing}"
        )

    # Get line sets (either by VP or single set)
    if args.hide_lines:
        line_sets = []
    elif args.color_lines_by_vps:
        line_sets = get_line_sets_by_vp(
            hrecon,
            min_track_length=args.min_track_length,
            ranges=ranges,
            min_associated_features=args.min_associated_features,
            pixel_uncertainties=pixel_uncertainties,
            max_pixel_uncertainty=args.line_max_pixel_uncertainty,
        )
    else:
        lines = [
            line
            for line_id, line in hrecon.structure_recon.lines3D.items()
            if line.track.length() >= args.min_track_length
            and (
                pixel_uncertainties is None
                or pixel_uncertainties.get(line_id, float("inf"))
                <= args.line_max_pixel_uncertainty
            )
        ]
        print(f"Number of valid lines for visualization: {len(lines)}")
        line_set = limapvis.open3d_get_3d_lines(
            lines, ranges=ranges, color=(1, 0.5, 0)
        )
        line_sets = [(line_set, "lines")]

    # Estimate Manhattan world frame from VPs if requested
    manhattan_directions = None
    if args.manhattan_planes:
        manhattan_directions = estimate_manhattan_frame(hrecon)
        if manhattan_directions is not None:
            print("Manhattan frame estimated:")
            for i, d in enumerate(manhattan_directions):
                print(f"  axis {i}: [{d[0]:.4f}, {d[1]:.4f}, {d[2]:.4f}]")
        else:
            print("Warning: could not estimate Manhattan frame from VPs")

    # Estimate Atlanta gravity from VPs if requested
    atlanta_gravity = None
    if args.atlanta_planes:
        atlanta_gravity = estimate_atlanta_gravity(hrecon)
        if atlanta_gravity is not None:
            g = atlanta_gravity
            print(
                "Atlanta gravity estimated: "
                f"[{g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f}]"
            )
        else:
            print("Warning: could not estimate Atlanta gravity from VPs")

    show_meshes = (
        args.show_planes or args.show_spheres or args.show_cylinders
    ) and hrecon.structure_recon.num_groups3D() > 0

    # Textured mode: generate textured meshes and optionally export
    textured_meshes = []
    if args.textured and show_meshes:
        if not args.texture_workspace or not args.texture_image_dir:
            print(
                "Error: --texture_workspace and --texture_image_dir are "
                "required when --textured is used"
            )
            return

        group_workspace = Path(args.texture_workspace)
        image_dir = Path(args.texture_image_dir)
        assert group_workspace.exists(), (
            f"texture_workspace not found at {group_workspace}"
        )
        assert image_dir.exists(), f"Image directory not found at {image_dir}"
        textured_meshes = get_textured_group_meshes(
            hrecon,
            group_workspace=group_workspace,
            image_dir=image_dir,
            show_planes=args.show_planes,
            show_cylinders=args.show_cylinders,
            show_spheres=args.show_spheres,
            min_associated_features=args.min_associated_features,
            mesh_resolution=args.texture_mesh_resolution,
            padding=args.plane_padding,
            use_all_masks=args.texture_use_all_images,
        )

        # Export PLY files if requested
        if args.texture_export_dir and textured_meshes:
            export_dir = Path(args.texture_export_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
            for mesh, name in textured_meshes:
                out_path = export_dir / f"{name}.ply"
                o3d.io.write_triangle_mesh(str(out_path), mesh)
                print(f"  Exported {out_path.name}")
            # Combined scene
            combined = o3d.geometry.TriangleMesh()
            for mesh, _ in textured_meshes:
                combined += mesh
            scene_path = export_dir / "scene.ply"
            o3d.io.write_triangle_mesh(str(scene_path), combined)
            print(f"  Exported {scene_path.name}")

    use_new_renderer = args.new_renderer
    if use_new_renderer:
        # Use o3d.visualization.draw() API when showing meshes.
        # This API supports transparency via MaterialRecord with
        # "defaultLitTransparency" shader, which is required for proper
        # alpha blending. The trade-off is that this renderer
        # (Filament-based) has different visual characteristics than
        # the legacy Visualizer.
        draw_geometries = []

        # Add point cloud (if non-empty)
        if pcd.has_points():
            point_mat = o3d.visualization.rendering.MaterialRecord()
            point_mat.point_size = args.point_size
            draw_geometries.append(
                {
                    "geometry": pcd,
                    "name": "points",
                    "material": point_mat,
                }
            )

        # Add camera frustums
        cam_mat = o3d.visualization.rendering.MaterialRecord()
        cam_mat.shader = "unlitLine"
        cam_mat.line_width = args.line_width
        draw_geometries.append(
            {
                "geometry": camera_set,
                "name": "cameras",
                "material": cam_mat,
            }
        )

        # Add lines (if non-empty)
        line_mat = o3d.visualization.rendering.MaterialRecord()
        line_mat.shader = "unlitLine"
        line_mat.line_width = args.line_width
        for ls, name in line_sets:
            if ls.has_lines():
                draw_geometries.append(
                    {
                        "geometry": ls,
                        "name": name,
                        "material": line_mat,
                    }
                )

        # Add wireframe (dashed lines connecting points to lines)
        if args.show_wireframes:
            wireframe_ls = get_wireframe_lineset(hrecon, ranges=ranges)
            if wireframe_ls is not None:
                wf_mat = o3d.visualization.rendering.MaterialRecord()
                wf_mat.shader = "unlitLine"
                wf_mat.line_width = args.line_width
                draw_geometries.append(
                    {
                        "geometry": wireframe_ls,
                        "name": "wireframe",
                        "material": wf_mat,
                    }
                )

        if textured_meshes:
            # Textured meshes: vertex-colored, use defaultLit shader
            for mesh, name in textured_meshes:
                mat = o3d.visualization.rendering.MaterialRecord()
                mat.shader = "defaultLit"
                draw_geometries.append(
                    {
                        "geometry": mesh,
                        "name": name,
                        "material": mat,
                    }
                )
        else:
            # Uniform-colored transparent meshes (original behavior)
            def _add_transparent_meshes(meshes, prefix):
                for i, m in enumerate(meshes):
                    mat = o3d.visualization.rendering.MaterialRecord()
                    mat.shader = "defaultLitTransparency"
                    mat.base_color = [
                        m.color[0],
                        m.color[1],
                        m.color[2],
                        m.alpha,
                    ]
                    draw_geometries.append(
                        {
                            "geometry": m.mesh,
                            "name": f"{prefix}_{i}",
                            "material": mat,
                        }
                    )

            if args.show_planes:
                plane_meshes = get_plane_meshes(
                    hrecon,
                    ranges=ranges,
                    min_associated_features=args.min_associated_features,
                    alpha=args.plane_alpha,
                    padding=args.plane_padding,
                    manhattan_directions=manhattan_directions,
                    atlanta_gravity=atlanta_gravity,
                )
                print(f"Number of plane meshes: {len(plane_meshes)}")
                _add_transparent_meshes(plane_meshes, "plane")

            if args.show_spheres:
                sphere_meshes = get_sphere_meshes(
                    hrecon,
                    ranges=ranges,
                    min_associated_features=args.min_associated_features,
                    alpha=args.sphere_alpha,
                )
                print(f"Number of sphere meshes: {len(sphere_meshes)}")
                _add_transparent_meshes(sphere_meshes, "sphere")

            if args.show_cylinders:
                cylinder_meshes = get_cylinder_meshes(
                    hrecon,
                    ranges=ranges,
                    min_associated_features=args.min_associated_features,
                    alpha=args.cylinder_alpha,
                )
                print(f"Number of cylinder meshes: {len(cylinder_meshes)}")
                _add_transparent_meshes(cylinder_meshes, "cylinder")

        # Prepare viewpoint parameters
        viewpoint_kwargs = {}
        if args.viewpoint_image:
            viewpoint = get_viewpoint_from_image(hrecon, args.viewpoint_image)
            if viewpoint:
                viewpoint_kwargs = {
                    "eye": viewpoint["eye"],
                    "lookat": viewpoint["lookat"],
                    "up": viewpoint["up"],
                    "field_of_view": viewpoint.get("field_of_view", 60.0),
                }
                print(f"Loaded viewpoint from image {args.viewpoint_image}")
        elif args.load_viewpoint:
            viewpoint = load_viewpoint(args.load_viewpoint)
            if viewpoint:
                viewpoint_kwargs = {
                    "eye": viewpoint["eye"],
                    "lookat": viewpoint["lookat"],
                    "up": viewpoint["up"],
                    "field_of_view": viewpoint.get("field_of_view", 60.0),
                }
                print(f"Loaded viewpoint from {args.load_viewpoint}")

        # Action to save viewpoint
        def save_viewpoint_action(vis):
            if args.save_viewpoint:
                scene = vis.scene
                cam = scene.camera
                # Extract camera params from view matrix
                view = np.asarray(cam.get_view_matrix())
                R = view[:3, :3]
                t = view[:3, 3]
                eye = -R.T @ t
                # Camera looks along -Z in camera space
                front = -R[2, :]
                lookat = eye + front
                up = R[1, :]
                save_viewpoint(
                    args.save_viewpoint,
                    eye=eye,
                    lookat=lookat,
                    up=up,
                    fov=cam.get_field_of_view(),
                )

        # Action to save screenshot
        def save_screenshot_action(vis):
            if args.screenshot:
                vis.export_current_image(args.screenshot)
                print(f"Screenshot saved to {args.screenshot}")

        actions = []
        if args.save_viewpoint:
            actions.append(("Save Viewpoint [S]", save_viewpoint_action))
        if args.screenshot:
            actions.append(("Save Screenshot [P]", save_screenshot_action))
        if not actions:
            actions = None

        o3d.visualization.draw(
            draw_geometries,
            show_ui=True,
            width=1920,
            height=1080,
            point_size=int(args.point_size),
            line_width=int(args.line_width),
            bg_color=(1.0, 1.0, 1.0, 1.0),
            show_skybox=False,
            ibl_intensity=60000,
            actions=actions,
            **viewpoint_kwargs,
        )
    else:
        # Use legacy o3d.visualization.Visualizer.
        # This provides better visual quality for points and lines with
        # more predictable point_size and line_width behavior compared
        # to the newer Filament-based draw() API.
        # Note: mesh transparency is not supported in legacy mode.
        # Use VisualizerWithKeyCallback if screenshot is needed.
        if args.screenshot:
            vis = o3d.visualization.VisualizerWithKeyCallback()
        else:
            vis = o3d.visualization.Visualizer()
        vis.create_window(height=1080, width=1920)
        if pcd.has_points():
            vis.add_geometry(pcd)
        vis.add_geometry(camera_set)
        for ls, _ in line_sets:
            if ls.has_lines():
                vis.add_geometry(ls)

        # Add wireframe (dashed lines connecting points to lines)
        if args.show_wireframes:
            wireframe_ls = get_wireframe_lineset(hrecon, ranges=ranges)
            if wireframe_ls is not None and wireframe_ls.has_lines():
                vis.add_geometry(wireframe_ls)

        # Add meshes if showing (legacy mode - no transparency)
        if show_meshes:
            if textured_meshes:
                print(f"Adding {len(textured_meshes)} textured meshes (legacy)")
                for mesh, name in textured_meshes:
                    print(f"  Adding {name}")
                    vis.add_geometry(mesh)
            else:
                if args.show_planes:
                    plane_meshes = get_plane_meshes(
                        hrecon,
                        ranges=ranges,
                        min_associated_features=args.min_associated_features,
                        alpha=args.plane_alpha,
                        padding=args.plane_padding,
                        soft_colors=True,
                        manhattan_directions=manhattan_directions,
                        atlanta_gravity=atlanta_gravity,
                    )
                    print(f"Adding {len(plane_meshes)} planes (legacy)")
                    for pm in plane_meshes:
                        vis.add_geometry(pm.mesh)
                if args.show_spheres:
                    for sm in get_sphere_meshes(
                        hrecon,
                        ranges=ranges,
                        min_associated_features=args.min_associated_features,
                        alpha=args.sphere_alpha,
                        soft_colors=True,
                    ):
                        vis.add_geometry(sm.mesh)
                if args.show_cylinders:
                    for cm in get_cylinder_meshes(
                        hrecon,
                        ranges=ranges,
                        min_associated_features=args.min_associated_features,
                        alpha=args.cylinder_alpha,
                        soft_colors=True,
                    ):
                        vis.add_geometry(cm.mesh)

        vis.poll_events()
        vis.update_renderer()
        opt = vis.get_render_option()
        if opt is not None:
            opt.point_size = args.point_size
            opt.line_width = args.line_width
            opt.mesh_show_back_face = True

        # Load viewpoint if exists
        viewpoint = None
        if args.viewpoint_image:
            viewpoint = get_viewpoint_from_image(hrecon, args.viewpoint_image)
            if viewpoint:
                print(f"Loaded viewpoint from image {args.viewpoint_image}")
        elif args.load_viewpoint:
            viewpoint = load_viewpoint(args.load_viewpoint)
            if viewpoint:
                print(f"Loaded viewpoint from {args.load_viewpoint}")

        if viewpoint:
            vc = vis.get_view_control()
            vc.set_lookat(viewpoint["lookat"])
            vc.set_front(
                np.array(viewpoint["eye"]) - np.array(viewpoint["lookat"])
            )
            vc.set_up(viewpoint["up"])
            vc.set_zoom(viewpoint.get("zoom", 0.5))

        # Register key callback for screenshot (press 'S')
        if args.screenshot:

            def save_screenshot_callback(vis):
                vis.capture_screen_image(args.screenshot)
                print(f"Screenshot saved to {args.screenshot}")
                return False  # Don't stop the visualizer

            vis.register_key_callback(ord("S"), save_screenshot_callback)
            print(f"Press 'S' to save screenshot to {args.screenshot}")

        vis.run()

        vis.destroy_window()


def main(args):
    hrecon = HolisticReconstruction(args.input_dir)

    if args.only_show_stats:
        show_stats(hrecon, min_associated_features=args.min_associated_features)
        return

    ranges = None
    if not args.disable_robust_ranges:
        points = np.array(
            [point.xyz for _, point in hrecon.point_recon.points3D.items()]
        )
        if points.shape[0] > 0:
            ranges = limapvis.compute_robust_range_points(points)
        else:
            # Use lines if no points
            lines = list(hrecon.structure_recon.lines3D.values())
            if len(lines) > 0:
                ranges = limapvis.compute_robust_range_lines(lines)
    vis_holistic_reconstruction(hrecon, ranges=ranges)


if __name__ == "__main__":
    args = parse_args()
    main(args)
