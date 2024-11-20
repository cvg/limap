import copy

import numpy as np
import open3d as o3d

from .vis_utils import test_line_inside_ranges, test_point_inside_ranges


def pyvista_vis_3d_lines(
    lines, img_hw=(600, 800), width=2, ranges=None, scale=1.0
):
    """
    Visualize a 3D line map with `PyVista <https://docs.pyvista.org/version/stable/>`_

    Args:
        lines (list[:class:`limap.base.Line3d`]): The 3D line map
        width (float, optional): width of the line
    """
    import pyvista as pv

    plotter = pv.Plotter(window_size=[img_hw[1], img_hw[0]])
    for line in lines:
        if (ranges is not None) and (not test_line_inside_ranges(line, ranges)):
            continue
        plotter.add_lines(line.as_array() * scale, "#ff0000", width=width)
    plotter.show()


def open3d_get_points(
    points,
    color=None,
    ranges=None,
    scale=1.0,
):
    if color is None:
        color = [0.0, 0.0, 0.0]
    if np.array(points).shape[0] == 0:
        return None
    o3d_points, o3d_colors = [], []
    for idx in range(np.array(points).shape[0]):
        if (ranges is not None) and (
            not test_point_inside_ranges(points[idx], ranges)
        ):
            continue
        o3d_points.append(points[idx] * scale)
        o3d_colors.append(color)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack(o3d_points))
    pcd.colors = o3d.utility.Vector3dVector(np.stack(o3d_colors))
    return pcd


def open3d_add_points(
    w,
    points,
    color=None,
    ranges=None,
    scale=1.0,
    name="pcd",
    psize=1.0,
):
    if color is None:
        color = [0.0, 0.0, 0.0]
    pcd = open3d_get_points(points, color=color, ranges=ranges, scale=scale)
    if pcd is None:
        return w
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = psize
    w.add_geometry(name, pcd, mat)
    return w


def open3d_get_line_set(lines, color=None, ranges=None, scale=1.0):
    if color is None:
        color = [0.0, 0.0, 0.0]
    o3d_points, o3d_lines, o3d_colors = [], [], []
    counter = 0
    for line in lines:
        if (ranges is not None) and (not test_line_inside_ranges(line, ranges)):
            continue
        o3d_points.append(line.start * scale)
        o3d_points.append(line.end * scale)
        o3d_lines.append([2 * counter, 2 * counter + 1])
        counter += 1
        o3d_colors.append(color)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(o3d_points)
    line_set.lines = o3d.utility.Vector2iVector(o3d_lines)
    line_set.colors = o3d.utility.Vector3dVector(o3d_colors)
    return line_set


def open3d_add_line_set(
    w,
    lines,
    color=None,
    ranges=None,
    scale=1.0,
    name="lineset",
    width=2,
):
    if color is None:
        color = [0.0, 0.0, 0.0]
    import open3d as o3d

    line_set = open3d_get_line_set(
        lines, color=color, ranges=ranges, scale=scale
    )
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.line_width = width
    w.add_geometry(name, line_set, mat)
    return w


def open3d_get_cameras(
    imagecols,
    color=None,
    ranges=None,
    scale_cam_geometry=1.0,
    scale=1.0,
):
    if color is None:
        color = [1.0, 0.0, 0.0]
    cameras = o3d.geometry.LineSet()

    camera_lines = {}
    for cam_id in imagecols.get_cam_ids():
        cam = imagecols.cam(cam_id)
        camera_lines[cam_id] = o3d.geometry.LineSet.create_camera_visualization(
            cam.w(),
            cam.h(),
            cam.K(),
            np.eye(4),
            scale=0.005 * scale_cam_geometry * scale,
        )
    for img_id in imagecols.get_img_ids():
        camimage = imagecols.camimage(img_id)
        if (ranges is not None) and (
            not test_point_inside_ranges(camimage.pose.center(), ranges)
        ):
            continue
        T = np.eye(4)
        T[:3, :3] = camimage.R()
        T[:3, 3] = camimage.T() * scale
        T = np.linalg.inv(T)
        cam = copy.deepcopy(camera_lines[camimage.cam_id]).transform(T)
        cam.paint_uniform_color(color)
        cameras += cam
    return cameras


def open3d_add_cameras(
    w,
    imagecols,
    color=None,
    ranges=None,
    scale_cam_geometry=1.0,
    scale=1.0,
):
    if color is None:
        color = [1.0, 0.0, 0.0]
    camera_lines = {}
    for cam_id in imagecols.get_cam_ids():
        cam = imagecols.cam(cam_id)
        camera_lines[cam_id] = o3d.geometry.LineSet.create_camera_visualization(
            cam.w(),
            cam.h(),
            cam.K(),
            np.eye(4),
            scale=0.005 * scale_cam_geometry * scale,
        )
    for img_id in imagecols.get_img_ids():
        camimage = imagecols.camimage(img_id)
        if (ranges is not None) and (
            not test_point_inside_ranges(camimage.pose.center(), ranges)
        ):
            continue
        T = np.eye(4)
        T[:3, :3] = camimage.R()
        T[:3, 3] = camimage.T() * scale
        T = np.linalg.inv(T)
        cam = copy.deepcopy(camera_lines[camimage.cam_id]).transform(T)
        cam.paint_uniform_color(color)
        w.add_geometry(camimage.image_name(), cam)
    return w


def open3d_vis_3d_lines(lines, ranges=None, scale=1.0):
    """
    Visualize a 3D line map with `Open3D <http://www.open3d.org/>`_

    Args:
        lines (list[:class:`limap.base.Line3d`]): The 3D line map
        width (float, optional): width of the line
    """
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window(height=1080, width=1920)
    line_set = open3d_get_line_set(lines, ranges=ranges, scale=scale)
    vis.add_geometry(line_set)
    vis.run()
    vis.destroy_window()
