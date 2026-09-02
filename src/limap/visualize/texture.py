"""Textured mesh generation for group primitives (planes, cylinders, spheres).

Projects images through per-image segmentation masks to produce vertex-colored
meshes that show real surface appearance.
"""

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap

from limap.geometry import GroupType
from limap.image.groups.group_io import read_group_mask
from limap.scene import HolisticReconstruction

# Mapping from GroupType enum to on-disk subdirectory name
_GROUP_TYPE_TO_SUBDIR = {
    GroupType.PLANE: "plane_detections",
    GroupType.CYLINDER: "cylinder_detections",
    GroupType.SPHERE: "sphere_detections",
}

# String-keyed version (matches start_ids dict keys)
_GROUP_NAME_TO_SUBDIR = {
    "PLANE": "plane_detections",
    "CYLINDER": "cylinder_detections",
    "SPHERE": "sphere_detections",
}


def get_group_binary_mask(
    group3d_id: int,
    image_id: int,
    structure_recon,
    start_ids: dict,
    group_workspace: Path,
) -> np.ndarray | None:
    """Get binary mask for a group3D in a specific image.

    Resolves group3D -> track -> group2D_idx -> mask_label -> binary.

    Args:
        group3d_id: 3D group identifier
        image_id: Image to look up the mask in
        structure_recon: StructureReconstruction instance
        start_ids: Dict mapping {TYPE_NAME: {image_id: offset}}
        group_workspace: Path to group_description/ directory

    Returns:
        (H, W) bool array, or None if the group is not visible in image
    """
    group = structure_recon.groups3D.get(group3d_id)
    if group is None:
        return None

    # Check group is observed in this image
    found = False
    group2d_idx = -1
    for el in group.track.elements:
        if el.image_id == image_id:
            group2d_idx = el.point2D_idx
            found = True
            break
    if not found:
        return None

    # Resolve type name and mask directory
    type_name = group.type.name  # "PLANE", "CYLINDER", "SPHERE"
    if type_name not in _GROUP_NAME_TO_SUBDIR:
        return None
    if type_name not in start_ids:
        return None
    if image_id not in start_ids[type_name]:
        return None

    # Convert global group2D index to per-type mask label
    # group2D_idx = start_ids[TYPE][img_id] + mask_label - 1
    # => mask_label = group2D_idx - start_ids[TYPE][img_id] + 1
    offset = start_ids[type_name][image_id]
    mask_label = group2d_idx - offset + 1

    if mask_label < 1:
        return None

    # Load label mask and extract binary
    mask_dir = group_workspace / _GROUP_NAME_TO_SUBDIR[type_name] / "seg_labels"
    try:
        label_mask = read_group_mask(mask_dir, image_id).astype(int)
    except FileNotFoundError:
        return None

    binary = label_mask == mask_label
    if not np.any(binary):
        return None
    return binary


def project_points_to_image(
    points_3d: np.ndarray,
    image: pycolmap.Image,
    camera: pycolmap.Camera,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D points to 2D pixel coordinates.

    Args:
        points_3d: (N, 3) array of 3D world points
        image: pycolmap.Image with pose
        camera: pycolmap.Camera with intrinsics

    Returns:
        uv: (N, 2) pixel coordinates
        valid: (N,) bool mask (in front of camera and within image bounds)
    """
    cam_from_world = image.cam_from_world()

    # Transform to camera frame
    pts_cam = np.array([cam_from_world * p for p in points_3d])  # (N, 3)

    # Check in front of camera
    valid_depth = pts_cam[:, 2] > 1e-4

    # Project to pixel coordinates (img_from_cam handles projection)
    uv = camera.img_from_cam(pts_cam)  # (N, 2) pixel coords

    # Check within image bounds (1px padding)
    pad = 1
    in_bounds = (
        (uv[:, 0] >= pad)
        & (uv[:, 0] <= camera.width - pad - 1)
        & (uv[:, 1] >= pad)
        & (uv[:, 1] <= camera.height - pad - 1)
    )

    valid = valid_depth & in_bounds
    return uv, valid


def _bilinear_sample(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sampling of image at sub-pixel locations.

    Args:
        image: (H, W, 3) uint8 or float image
        uv: (N, 2) pixel coordinates (x=col, y=row)

    Returns:
        (N, 3) sampled colors as float in [0, 1]
    """
    H, W = image.shape[:2]
    x = uv[:, 0]
    y = uv[:, 1]

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # Clamp to valid range
    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)

    # Fractional parts
    fx = (x - x0.astype(float)).reshape(-1, 1)
    fy = (y - y0.astype(float)).reshape(-1, 1)

    img = (
        image.astype(np.float64) / 255.0
        if image.dtype == np.uint8
        else image.astype(np.float64)
    )

    # Bilinear interpolation
    v00 = img[y0, x0]  # (N, 3)
    v01 = img[y0, x1]
    v10 = img[y1, x0]
    v11 = img[y1, x1]

    result = (
        v00 * (1 - fx) * (1 - fy)
        + v01 * fx * (1 - fy)
        + v10 * (1 - fx) * fy
        + v11 * fx * fy
    )
    return result


def _get_images_with_masks(
    group_type: GroupType,
    group_workspace: Path,
    point_recon,
) -> list[int]:
    """Get all image IDs that have mask files for a given group type."""
    type_name = group_type.name
    if type_name not in _GROUP_NAME_TO_SUBDIR:
        return []

    mask_dir = group_workspace / _GROUP_NAME_TO_SUBDIR[type_name] / "seg_labels"
    if not mask_dir.exists():
        return []

    # Find all mask files and extract image IDs (format: mask_{image_id}.npy)
    image_ids = []
    for mask_file in mask_dir.glob("mask_*.npy"):
        stem = mask_file.stem  # e.g. "mask_12"
        try:
            img_id = int(stem[len("mask_") :])
            if img_id in point_recon.images:
                image_ids.append(img_id)
        except ValueError:
            continue

    return image_ids


def _find_mask_for_projected_vertices(
    vertices_3d: np.ndarray,
    image: pycolmap.Image,
    camera: pycolmap.Camera,
    label_mask: np.ndarray,
) -> np.ndarray | None:
    """Find the best mask label that covers projected vertices.

    Projects 3D vertices into the image and finds which mask label
    (if any) covers the majority of valid projections.

    Args:
        vertices_3d: (V, 3) vertex positions
        image: pycolmap.Image with pose
        camera: pycolmap.Camera with intrinsics
        label_mask: (H, W) integer label mask

    Returns:
        (H, W) bool mask for the best label, or None if no good match
    """
    uv, valid = project_points_to_image(vertices_3d, image, camera)

    if not np.any(valid):
        return None

    # Sample labels at valid projected points
    uv_int = np.round(uv[valid]).astype(int)
    xi = np.clip(uv_int[:, 0], 0, label_mask.shape[1] - 1)
    yi = np.clip(uv_int[:, 1], 0, label_mask.shape[0] - 1)
    sampled_labels = label_mask[yi, xi]

    # Find most common non-zero label
    nonzero_labels = sampled_labels[sampled_labels > 0]
    if len(nonzero_labels) == 0:
        return None

    # Count occurrences and pick the most frequent
    unique, counts = np.unique(nonzero_labels, return_counts=True)
    best_label = unique[np.argmax(counts)]
    best_count = counts[np.argmax(counts)]

    # Require at least some vertices to be covered (relaxed threshold)
    # At least 5 vertices or 1% of valid projections
    min_required = max(5, int(0.01 * np.sum(valid)))
    if best_count < min_required:
        return None

    return label_mask == best_label


def sample_vertex_colors(
    vertices_3d: np.ndarray,
    vertex_normals: np.ndarray,
    group3d_id: int,
    hrecon: HolisticReconstruction,
    start_ids: dict,
    group_workspace: Path,
    image_dir: Path,
    use_all_masks: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample vertex colors by picking the best single view per vertex.

    For each vertex, selects the view with the most frontal viewing angle
    (highest cos between surface normal and view direction) among those
    where the vertex projects inside the segmentation mask.

    Args:
        vertices_3d: (V, 3) vertex positions
        vertex_normals: (V, 3) vertex normals
        group3d_id: 3D group identifier
        hrecon: HolisticReconstruction instance
        start_ids: Mask offset dict
        group_workspace: Path to group_description/ directory
        image_dir: Directory containing undistorted images
        use_all_masks: If True, scan all images with masks instead of just
            track images. Projects vertices to find overlapping masks.

    Returns:
        colors: (V, 3) RGB in [0, 1]
        confidence: (V,) sum of blending weights (0 = no valid samples)
    """
    V = vertices_3d.shape[0]
    colors = np.full((V, 3), 0.5)  # neutral gray fallback
    confidence = np.zeros(V)

    structure_recon = hrecon.structure_recon
    point_recon = hrecon.point_recon
    group = structure_recon.groups3D.get(group3d_id)
    if group is None:
        return colors, confidence

    # Collect candidate images
    if use_all_masks:
        # Scan all images that have mask files for this group type
        candidate_images = _get_images_with_masks(
            group.type, group_workspace, point_recon
        )
        print(
            f"    Group {group3d_id}: "
            f"found {len(candidate_images)} images with masks"
        )
    else:
        # Only use images from the group's track
        candidate_images = []
        for el in group.track.elements:
            if el.image_id in point_recon.images:
                candidate_images.append(el.image_id)

    if not candidate_images:
        return colors, confidence

    # Precompute camera positions for each candidate image
    cam_positions = {}
    for img_id in candidate_images:
        image = point_recon.images[img_id]
        pose = image.cam_from_world()
        cam_pos = pose.inverse().translation
        cam_positions[img_id] = cam_pos

    # Per-vertex best view: track the highest cos_angle seen so far
    best_cos = np.full(V, -1.0)

    # Cache loaded images and masks
    _image_cache = {}
    _mask_cache = {}
    _label_mask_cache = {}

    # Get mask directory for use_all_masks mode
    type_name = group.type.name
    mask_dir = (
        group_workspace
        / _GROUP_NAME_TO_SUBDIR.get(type_name, "")
        / "seg_labels"
    )

    for img_id in candidate_images:
        image = point_recon.images[img_id]
        camera = point_recon.cameras[image.camera_id]

        # Project all vertices
        uv, valid_proj = project_points_to_image(vertices_3d, image, camera)

        if not np.any(valid_proj):
            continue

        # Compute viewing angle: cos(angle) between normal and view dir
        cam_pos = cam_positions[img_id]
        view_dirs = cam_pos - vertices_3d  # (V, 3)
        view_dists = np.linalg.norm(view_dirs, axis=1, keepdims=True)
        view_dists = np.maximum(view_dists, 1e-8)
        view_dirs_normalized = view_dirs / view_dists

        cos_angle = np.sum(vertex_normals * view_dirs_normalized, axis=1)
        valid = valid_proj

        # Get binary mask - method depends on use_all_masks
        if img_id not in _mask_cache:
            if use_all_masks:
                # Load label mask and find best matching label by projection
                if img_id not in _label_mask_cache:
                    try:
                        _label_mask_cache[img_id] = read_group_mask(
                            mask_dir, img_id
                        ).astype(int)
                    except FileNotFoundError:
                        _label_mask_cache[img_id] = None

                label_mask = _label_mask_cache[img_id]
                if label_mask is not None:
                    _mask_cache[img_id] = _find_mask_for_projected_vertices(
                        vertices_3d, image, camera, label_mask
                    )
                else:
                    _mask_cache[img_id] = None
            else:
                # Use track association to find the mask
                _mask_cache[img_id] = get_group_binary_mask(
                    group3d_id,
                    img_id,
                    structure_recon,
                    start_ids,
                    group_workspace,
                )

        binary_mask = _mask_cache[img_id]

        if binary_mask is None:
            continue

        # Vectorized mask check for valid projected vertices
        in_mask = np.zeros(V, dtype=bool)
        valid_indices = np.where(valid)[0]
        if len(valid_indices) > 0:
            uv_valid = np.round(uv[valid_indices]).astype(int)
            xi = np.clip(uv_valid[:, 0], 0, binary_mask.shape[1] - 1)
            yi = np.clip(uv_valid[:, 1], 0, binary_mask.shape[0] - 1)
            in_mask[valid_indices] = binary_mask[yi, xi]

        valid_final = valid & in_mask
        if not np.any(valid_final):
            continue

        # Only update vertices where this view has a better angle
        final_indices = np.where(valid_final)[0]
        better = cos_angle[final_indices] > best_cos[final_indices]
        update_indices = final_indices[better]

        if len(update_indices) == 0:
            continue

        # Load image (lazily)
        if img_id not in _image_cache:
            img_name = point_recon.images[img_id].name
            img_path = image_dir / img_name
            if not img_path.exists():
                continue
            loaded = cv2.imread(str(img_path))
            if loaded is None:
                continue
            _image_cache[img_id] = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)

        img_data = _image_cache.get(img_id)
        if img_data is None:
            continue

        # Sample and assign colors for vertices where this is the best view
        sampled = _bilinear_sample(img_data, uv[update_indices])  # (K, 3)
        colors[update_indices] = sampled
        best_cos[update_indices] = cos_angle[update_indices]

    # A vertex was textured if best_cos was updated from its initial -1.0
    confidence = np.where(best_cos > -1.0, 1.0, 0.0)

    return colors, confidence


def _trim_mesh_by_confidence(
    mesh: o3d.geometry.TriangleMesh,
    confidence: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    """Remove triangles where all 3 vertices have zero confidence.

    Args:
        mesh: Input mesh with vertex colors already set
        confidence: (V,) per-vertex confidence (0 = untextured)

    Returns:
        Trimmed mesh with untextured triangles removed
    """
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0:
        return mesh

    # Keep triangle only if all 3 vertices have nonzero confidence
    tri_conf = confidence[triangles]  # (T, 3)
    keep = np.all(tri_conf > 0, axis=1)

    if np.all(keep):
        return mesh

    kept_triangles = triangles[keep]
    if len(kept_triangles) == 0:
        return o3d.geometry.TriangleMesh()

    # Remap vertices: only keep those referenced by kept triangles
    used_verts = np.unique(kept_triangles)
    old_to_new = np.full(len(np.asarray(mesh.vertices)), -1, dtype=int)
    old_to_new[used_verts] = np.arange(len(used_verts))

    new_vertices = np.asarray(mesh.vertices)[used_verts]
    new_triangles = old_to_new[kept_triangles]

    trimmed = o3d.geometry.TriangleMesh()
    trimmed.vertices = o3d.utility.Vector3dVector(new_vertices)
    trimmed.triangles = o3d.utility.Vector3iVector(new_triangles)

    if mesh.has_vertex_colors():
        new_colors = np.asarray(mesh.vertex_colors)[used_verts]
        trimmed.vertex_colors = o3d.utility.Vector3dVector(new_colors)

    if mesh.has_vertex_normals():
        new_normals = np.asarray(mesh.vertex_normals)[used_verts]
        trimmed.vertex_normals = o3d.utility.Vector3dVector(new_normals)

    return trimmed


def create_textured_plane_mesh(
    group3d_id: int,
    group,
    associated_points: np.ndarray,
    hrecon: HolisticReconstruction,
    start_ids: dict,
    group_workspace: Path,
    image_dir: Path,
    mesh_resolution: int = 50,
    padding: float = 0.2,
    use_all_masks: bool = False,
) -> o3d.geometry.TriangleMesh | None:
    """Create a textured plane mesh with vertex colors from images.

    Reuses the local (u, v) frame from open3d_get_plane_mesh but creates
    a dense grid mesh instead of 2 triangles.

    Args:
        group3d_id: 3D group identifier
        group: Group3d object with plane params
        associated_points: (N, 3) 3D points on the plane
        hrecon: HolisticReconstruction
        start_ids: Mask offset dict
        group_workspace: Path to group_description/ directory
        image_dir: Undistorted images directory
        mesh_resolution: Grid density (mesh_resolution x mesh_resolution)
        padding: Relative padding around bounding box
        use_all_masks: If True, scan all images with masks instead of track

    Returns:
        Vertex-colored TriangleMesh, or None if insufficient data
    """
    plane_params = group.params
    if plane_params is None or len(plane_params) != 4:
        return None
    if associated_points.shape[0] < 3:
        return None

    # Plane equation: ax + by + cz + d = 0
    normal = np.array(plane_params[:3])
    d = plane_params[3]

    # Project points onto the plane
    distances = associated_points @ normal + d
    projected = associated_points - np.outer(distances, normal)
    center = np.mean(projected, axis=0)

    # Local coordinate frame
    ref = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u_axis = np.cross(normal, ref)
    u_axis = u_axis / np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)

    # Project to local 2D
    diffs = projected - center
    local_coords = np.column_stack([diffs @ u_axis, diffs @ v_axis])

    # Minimum-area bounding rectangle
    local_f32 = local_coords.astype(np.float32)
    rect = cv2.minAreaRect(local_f32)
    box_2d = cv2.boxPoints(rect)

    # Apply padding
    rect_center = box_2d.mean(axis=0)
    box_2d = rect_center + (1.0 + padding) * (box_2d - rect_center)

    # Compute oriented bounding box extents in local frame
    # Use edges of the oriented box to define grid axes
    edge0 = box_2d[1] - box_2d[0]
    edge1 = box_2d[3] - box_2d[0]

    # Create dense grid in [0, 1] x [0, 1] mapped to the box
    res = mesh_resolution
    t0 = np.linspace(0, 1, res)
    t1 = np.linspace(0, 1, res)
    tt0, tt1 = np.meshgrid(t0, t1)
    tt0 = tt0.ravel()
    tt1 = tt1.ravel()

    # Grid points in local 2D
    grid_local = box_2d[0] + np.outer(tt0, edge0) + np.outer(tt1, edge1)

    # Map back to 3D
    vertices = (
        center + grid_local[:, 0:1] * u_axis + grid_local[:, 1:2] * v_axis
    )

    # Create triangle mesh from grid
    triangles = []
    for j in range(res - 1):
        for i in range(res - 1):
            idx = j * res + i
            # Two triangles per grid cell
            triangles.append([idx, idx + 1, idx + res])
            triangles.append([idx + 1, idx + res + 1, idx + res])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.compute_vertex_normals()

    # Ensure normals point consistently (towards the normal direction)
    vnormals = np.asarray(mesh.vertex_normals)
    if vnormals.shape[0] > 0 and np.mean(vnormals @ normal) < 0:
        # Flip normals
        vnormals *= -1
        mesh.vertex_normals = o3d.utility.Vector3dVector(vnormals)

    # Sample vertex colors
    colors, confidence = sample_vertex_colors(
        np.asarray(mesh.vertices),
        np.asarray(mesh.vertex_normals),
        group3d_id,
        hrecon,
        start_ids,
        group_workspace,
        image_dir,
        use_all_masks=use_all_masks,
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Trim untextured regions
    mesh = _trim_mesh_by_confidence(mesh, confidence)
    if len(np.asarray(mesh.triangles)) == 0:
        return None

    mesh.compute_vertex_normals()
    return mesh


def create_textured_cylinder_mesh(
    group3d_id: int,
    group,
    associated_points: np.ndarray,
    hrecon: HolisticReconstruction,
    start_ids: dict,
    group_workspace: Path,
    image_dir: Path,
    mesh_resolution: int = 50,
    use_all_masks: bool = False,
) -> o3d.geometry.TriangleMesh | None:
    """Create a textured cylinder mesh with vertex colors from images.

    Reuses axis recovery from open3d_get_cylinder_mesh.

    Args:
        group3d_id: 3D group identifier
        group: Group3d object with cylinder params
        associated_points: (N, 3) 3D points for height estimation
        hrecon: HolisticReconstruction
        start_ids: Mask offset dict
        group_workspace: Path to group_description/ directory
        image_dir: Undistorted images directory
        mesh_resolution: Cylinder angular resolution
        use_all_masks: If True, scan all images with masks instead of track

    Returns:
        Vertex-colored TriangleMesh, or None if insufficient data
    """
    from scipy.spatial.transform import Rotation

    cylinder_params = group.params
    if cylinder_params is None or len(cylinder_params) != 7:
        return None
    if associated_points.shape[0] < 2:
        return None

    # Extract parameters (same as open3d_get_cylinder_mesh)
    qx, qy, qz, qw = cylinder_params[:4]
    wv0, wv1 = cylinder_params[4:6]
    log_r = cylinder_params[6]
    radius = np.exp(log_r)

    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    direction = R[:, 0]
    moment = (wv1 / wv0) * R[:, 1]
    point_on_axis = np.cross(direction, moment)

    # Height from associated points
    diffs = associated_points - point_on_axis
    projections = diffs @ direction
    t_min = np.min(projections)
    t_max = np.max(projections)
    height = t_max - t_min

    if height < 1e-6:
        return None

    # Add 10% padding
    pad = 0.1 * height
    t_min -= pad
    t_max += pad
    height = t_max - t_min
    midpoint = point_on_axis + ((t_min + t_max) / 2.0) * direction

    # Create high-resolution cylinder mesh
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=height, resolution=mesh_resolution
    )

    # Rotate from Z axis to cylinder direction
    z_axis = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(z_axis, direction)) < 1.0 - 1e-6:
        rot_axis = np.cross(z_axis, direction)
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        rot = Rotation.from_rotvec(rot_axis * angle).as_matrix()
    else:
        if np.dot(z_axis, direction) > 0:
            rot = np.eye(3)
        else:
            rot = np.diag([1.0, -1.0, -1.0])

    mesh.rotate(rot, center=[0, 0, 0])
    mesh.translate(midpoint)
    mesh.compute_vertex_normals()

    # Sample vertex colors
    colors, confidence = sample_vertex_colors(
        np.asarray(mesh.vertices),
        np.asarray(mesh.vertex_normals),
        group3d_id,
        hrecon,
        start_ids,
        group_workspace,
        image_dir,
        use_all_masks=use_all_masks,
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Trim untextured regions
    mesh = _trim_mesh_by_confidence(mesh, confidence)
    if len(np.asarray(mesh.triangles)) == 0:
        return None

    mesh.compute_vertex_normals()
    return mesh


def create_textured_sphere_mesh(
    group3d_id: int,
    group,
    hrecon: HolisticReconstruction,
    start_ids: dict,
    group_workspace: Path,
    image_dir: Path,
    mesh_resolution: int = 50,
    use_all_masks: bool = False,
) -> o3d.geometry.TriangleMesh | None:
    """Create a textured sphere mesh with vertex colors from images.

    Args:
        group3d_id: 3D group identifier
        group: Group3d object with sphere params
        hrecon: HolisticReconstruction
        start_ids: Mask offset dict
        group_workspace: Path to group_description/ directory
        image_dir: Undistorted images directory
        mesh_resolution: Sphere subdivision resolution
        use_all_masks: If True, scan all images with masks instead of track

    Returns:
        Vertex-colored TriangleMesh, or None if insufficient data
    """
    sphere_params = group.params
    if sphere_params is None or len(sphere_params) != 4:
        return None

    cx, cy, cz = sphere_params[0], sphere_params[1], sphere_params[2]
    radius = np.exp(sphere_params[3])

    mesh = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius, resolution=mesh_resolution
    )
    mesh.translate([cx, cy, cz])
    mesh.compute_vertex_normals()

    # Sample vertex colors
    colors, confidence = sample_vertex_colors(
        np.asarray(mesh.vertices),
        np.asarray(mesh.vertex_normals),
        group3d_id,
        hrecon,
        start_ids,
        group_workspace,
        image_dir,
        use_all_masks=use_all_masks,
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Trim untextured regions
    mesh = _trim_mesh_by_confidence(mesh, confidence)
    if len(np.asarray(mesh.triangles)) == 0:
        return None

    mesh.compute_vertex_normals()
    return mesh
