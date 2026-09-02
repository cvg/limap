import copy
from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
import pycolmap
from typeguard import typechecked

import limap.geometry
from limap.util.types import Color, Ranges

from .viz_utils import test_line_inside_ranges, test_point_inside_ranges


@dataclass
class PlaneMesh:
    """Wrapper for plane mesh with transparency info."""

    mesh: o3d.geometry.TriangleMesh
    color: Color
    alpha: float


@typechecked
def open3d_get_3d_points(
    points: np.ndarray,
    color: Color | None = None,
    ranges: Ranges | None = None,
) -> o3d.geometry.PointCloud:
    if color is None:
        color = (0.0, 0.0, 0.0)
    if np.array(points).shape[0] == 0:
        return o3d.geometry.PointCloud()  # Return empty point cloud
    o3d_points, o3d_colors = [], []
    for idx in range(np.array(points).shape[0]):
        if (ranges is not None) and (
            not test_point_inside_ranges(points[idx], ranges)
        ):
            continue
        o3d_points.append(points[idx])
        o3d_colors.append(color)
    pcd = o3d.geometry.PointCloud()
    if len(o3d_points) > 0:
        pcd.points = o3d.utility.Vector3dVector(np.stack(o3d_points))
        pcd.colors = o3d.utility.Vector3dVector(np.stack(o3d_colors))
    return pcd


@typechecked
def open3d_get_3d_lines(
    lines: list[limap.geometry.Line3d],
    color: Color | None = None,
    ranges: Ranges | None = None,
) -> o3d.geometry.LineSet:
    if color is None:
        color = (0.0, 0.0, 0.0)
    o3d_points, o3d_lines, o3d_colors = [], [], []
    counter = 0
    for line in lines:
        if (ranges is not None) and (not test_line_inside_ranges(line, ranges)):
            continue
        o3d_points.append(line.start)
        o3d_points.append(line.end)
        o3d_lines.append([2 * counter, 2 * counter + 1])
        counter += 1
        o3d_colors.append(color)
    line_set = o3d.geometry.LineSet()
    if len(o3d_points) > 0:
        line_set.points = o3d.utility.Vector3dVector(np.array(o3d_points))
        line_set.lines = o3d.utility.Vector2iVector(
            np.array(o3d_lines).astype(np.int32)
        )
        line_set.colors = o3d.utility.Vector3dVector(np.array(o3d_colors))
    return line_set


@typechecked
def open3d_get_camera_frustums(
    recon: pycolmap.Reconstruction,
    color: Color | None = None,
    ranges: Ranges | None = None,
    scale_cam_geometry: float = 1.0,
) -> o3d.geometry.LineSet:
    if color is None:
        color = (1.0, 0.0, 0.0)
    cameras = o3d.geometry.LineSet()

    camera_lines = {}
    for _image_id, image in recon.images.items():
        cam = image.camera
        camera_lines[cam.camera_id] = (
            o3d.geometry.LineSet.create_camera_visualization(
                cam.width,
                cam.height,
                cam.calibration_matrix(),
                np.eye(4),
                scale=scale_cam_geometry,
            )
        )
    for _image_id, image in recon.images.items():
        pose = image.cam_from_world()
        if (ranges is not None) and (
            not test_point_inside_ranges(pose.inverse().translation, ranges)
        ):
            continue
        T = np.eye(4)
        T[:3, :3] = pose.rotation.matrix()
        T[:3, 3] = pose.translation
        T = np.linalg.inv(T)
        cam = copy.deepcopy(camera_lines[image.camera_id]).transform(T)
        cam.paint_uniform_color(color)
        cameras += cam
    return cameras


@typechecked
def open3d_get_plane_mesh(
    plane_params: list[float],
    points: np.ndarray,
    color: Color | None = None,
    padding: float = 0.4,
    alpha: float = 0.5,
    manhattan_directions: np.ndarray | None = None,
    atlanta_gravity: np.ndarray | None = None,
) -> PlaneMesh | None:
    """
    Create a plane rectangle mesh from plane parameters and associated points.

    Args:
        plane_params: [a, b, c, d] where ax + by + cz + d = 0
        points: Nx3 array of 3D points associated with the plane
        color: RGB color tuple
        padding: Relative padding around the bounding box (0.5 = 50% padding)
        alpha: Transparency value (0.0 = fully transparent, 1.0 = opaque).
            Used with o3d.visualization.draw() API which supports transparency.
        manhattan_directions: Optional Kx3 array of Manhattan world directions
            (e.g. VP directions). When provided, the plane bbox edges align with
            the two directions most orthogonal to the plane normal, and an
            axis-aligned bbox is used instead of an oriented bbox.
        atlanta_gravity: Optional 3-vector for Atlanta world gravity direction.
            When provided, vertical planes (normal orthogonal to gravity) get
            their bbox aligned with gravity as one axis.

    Returns:
        PlaneMesh containing the mesh, color, and alpha, or None if insufficient
        points
    """
    if color is None:
        color = (0.2, 0.6, 0.8)
    if plane_params is None or len(plane_params) != 4:
        return None
    if points.shape[0] < 3:
        return None

    # Plane params: [a, b, c, d] with ||(a,b,c)|| = 1, d is signed distance
    # Guaranteed by ProductManifold<SphereManifold<3>, EuclideanManifold<1>>
    normal = np.array(plane_params[:3])
    d = plane_params[3]

    # Project all points onto the plane
    # For point p, projection is: p_proj = p - ((n.p + d) * n)
    distances = points @ normal + d
    projected = points - np.outer(distances, normal)

    # Compute plane center
    center = np.mean(projected, axis=0)

    # Create local coordinate frame on the plane
    is_vertical_atlanta = False
    if atlanta_gravity is not None:
        # Atlanta world: vertical planes get gravity-aligned bbox
        gravity = atlanta_gravity / np.linalg.norm(atlanta_gravity)
        cos_angle = abs(np.dot(normal, gravity))
        # Vertical plane = normal roughly orthogonal to gravity
        if cos_angle < 0.5:
            is_vertical_atlanta = True
            # One bbox axis = gravity projected onto the plane
            u = gravity - np.dot(gravity, normal) * normal
            u = u / np.linalg.norm(u)
            # Other axis = normal x gravity (horizontal along the wall)
            v = np.cross(normal, u)
            v = v / np.linalg.norm(v)

    if (
        not is_vertical_atlanta
        and manhattan_directions is not None
        and len(manhattan_directions) >= 2
    ):
        # Pick the two Manhattan directions most orthogonal to the plane normal
        # (i.e. smallest |cos(angle)|)
        dots = np.abs(manhattan_directions @ normal)
        order = np.argsort(dots)
        u = manhattan_directions[order[0]].copy()
        v = manhattan_directions[order[1]].copy()
        # Project u,v onto the plane to ensure they lie in it
        u = u - np.dot(u, normal) * normal
        u = u / np.linalg.norm(u)
        v = v - np.dot(v, normal) * normal
        v = v / np.linalg.norm(v)
    elif not is_vertical_atlanta:
        # Fallback: arbitrary orthonormal frame
        ref = (
            np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
        )
        u = np.cross(normal, ref)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)

    # Project points to 2D local coordinates
    diffs = projected - center
    local_coords = np.column_stack([diffs @ u, diffs @ v])

    use_axis_aligned_bbox = is_vertical_atlanta or (
        manhattan_directions is not None and len(manhattan_directions) >= 2
    )
    if use_axis_aligned_bbox:
        # Axis-aligned bbox in the gravity/Manhattan-aligned local frame
        umin, vmin = local_coords.min(axis=0)
        umax, vmax = local_coords.max(axis=0)
        u_pad = (umax - umin) * padding * 0.5
        v_pad = (vmax - vmin) * padding * 0.5
        box_2d = np.array(
            [
                [umin - u_pad, vmin - v_pad],
                [umax + u_pad, vmin - v_pad],
                [umax + u_pad, vmax + v_pad],
                [umin - u_pad, vmax + v_pad],
            ]
        )
    else:
        # Compute minimum-area oriented bounding box via OpenCV
        local_f32 = local_coords.astype(np.float32)
        rect = cv2.minAreaRect(local_f32)
        box_2d = cv2.boxPoints(rect)  # 4x2 corners in local (u, v)
        # Apply padding: scale outward from the rectangle center
        rect_center = box_2d.mean(axis=0)
        box_2d = rect_center + (1.0 + padding) * (box_2d - rect_center)

    # Map 2D corners back to 3D
    corners = np.array([center + cu * u + cv * v for cu, cv in box_2d])

    # Create mesh with two triangles
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector(
        np.array([[0, 1, 2], [0, 2, 3]])
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return PlaneMesh(mesh=mesh, color=color, alpha=alpha)


@dataclass
class SphereMesh:
    """Wrapper for sphere mesh with transparency info."""

    mesh: o3d.geometry.TriangleMesh
    color: Color
    alpha: float


@dataclass
class CylinderMesh:
    """Wrapper for cylinder mesh with transparency info."""

    mesh: o3d.geometry.TriangleMesh
    color: Color
    alpha: float


@typechecked
def open3d_get_sphere_mesh(
    sphere_params: list[float],
    color: Color | None = None,
    alpha: float = 0.5,
    resolution: int = 20,
) -> SphereMesh | None:
    """
    Create a sphere mesh from sphere parameters.

    Args:
        sphere_params: [cx, cy, cz, log_r] where (cx, cy, cz) is the center
            and r = exp(log_r) is the radius
        color: RGB color tuple
        alpha: Transparency value (0.0 = fully transparent, 1.0 = opaque)
        resolution: Number of subdivisions for the sphere

    Returns:
        SphereMesh or None if invalid parameters
    """
    if color is None:
        color = (0.8, 0.2, 0.2)
    if sphere_params is None or len(sphere_params) != 4:
        return None

    cx, cy, cz = sphere_params[0], sphere_params[1], sphere_params[2]
    radius = np.exp(sphere_params[3])

    mesh = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius, resolution=resolution
    )
    mesh.translate([cx, cy, cz])
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return SphereMesh(mesh=mesh, color=color, alpha=alpha)


@typechecked
def open3d_get_cylinder_mesh(
    cylinder_params: list[float],
    associated_points: np.ndarray,
    color: Color | None = None,
    alpha: float = 0.5,
    resolution: int = 20,
) -> CylinderMesh | None:
    """
    Create a cylinder mesh from cylinder parameters and associated points.

    The cylinder is parameterized using MinimalInfiniteLine3d + log-radius:
    [qx, qy, qz, qw, wvec0, wvec1, log_r] where the quaternion is in Eigen
    storage order (x, y, z, w).

    The axis is recovered as a Plucker line (d, m) where:
    - d = R[:, 0] (direction = first column of rotation matrix)
    - m = (wvec1 / wvec0) * R[:, 1] (moment vector)

    Args:
        cylinder_params: [qx, qy, qz, qw, wvec0, wvec1, log_r]
        associated_points: Nx3 array of 3D points for height estimation
        color: RGB color tuple
        alpha: Transparency value (0.0 = fully transparent, 1.0 = opaque)
        resolution: Number of subdivisions for the cylinder

    Returns:
        CylinderMesh or None if invalid parameters or insufficient points
    """
    from scipy.spatial.transform import Rotation

    if color is None:
        color = (0.2, 0.8, 0.2)
    if cylinder_params is None or len(cylinder_params) != 7:
        return None
    if associated_points.shape[0] < 2:
        return None

    # Extract parameters — quaternion in Eigen order (x, y, z, w)
    qx, qy, qz, qw = cylinder_params[:4]
    wv0, wv1 = cylinder_params[4:6]
    log_r = cylinder_params[6]
    radius = np.exp(log_r)

    # Quaternion to rotation matrix
    # scipy.Rotation.from_quat expects [x, y, z, w] which matches Eigen order
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

    # Recover Plucker line (matches MinimalInfiniteLine3d::GetInfiniteLine)
    direction = R[:, 0]  # d = Q.col(0)
    moment = (wv1 / wv0) * R[:, 1]  # m = (wvec[1]/wvec[0]) * Q.col(1)

    # Closest point on axis to origin: p = d x m (Plucker identity)
    point_on_axis = np.cross(direction, moment)

    # Project associated points onto the axis to determine height
    diffs = associated_points - point_on_axis
    projections = diffs @ direction
    t_min = np.min(projections)
    t_max = np.max(projections)
    height = t_max - t_min

    if height < 1e-6:
        return None

    # Add 10% padding
    padding = 0.1 * height
    t_min -= padding
    t_max += padding
    height = t_max - t_min

    # Midpoint of the cylinder along the axis
    midpoint = point_on_axis + ((t_min + t_max) / 2.0) * direction

    # Create default cylinder (along Z axis, centered at origin)
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=height, resolution=resolution
    )

    # Rotate from Z axis to the cylinder's direction
    z_axis = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(z_axis, direction)) < 1.0 - 1e-6:
        rot_axis = np.cross(z_axis, direction)
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        rot = Rotation.from_rotvec(rot_axis * angle).as_matrix()
    else:
        # Nearly parallel or anti-parallel
        if np.dot(z_axis, direction) > 0:
            rot = np.eye(3)
        else:
            rot = np.diag([1.0, -1.0, -1.0])

    mesh.rotate(rot, center=[0, 0, 0])
    mesh.translate(midpoint)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return CylinderMesh(mesh=mesh, color=color, alpha=alpha)


@typechecked
def open3d_visualize_3d_lines(
    lines: list[limap.geometry.Line3d],
    ranges: Ranges | None = None,
) -> None:
    """
    Visualize a 3D line map with `Open3D <http://www.open3d.org/>`_

    Args:
        lines (list[:class:`limap.geometry.Line3d`]): The 3D line map
        width (float, optional): width of the line
    """
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window(height=1080, width=1920)
    line_set = open3d_get_3d_lines(lines, ranges=ranges)
    vis.add_geometry(line_set)
    vis.run()
    vis.destroy_window()
