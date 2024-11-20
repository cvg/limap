import copy

import _limap._base as _base
import numpy as np

from .vis_lines import (
    open3d_add_cameras,
    open3d_add_line_set,
    open3d_add_points,
)
from .vis_utils import (
    draw_points,
    draw_segments,
    test_line_inside_ranges,
    test_point_inside_ranges,
)


def draw_bipartite2d(image, bpt2d):
    image = copy.deepcopy(image)
    lines = bpt2d.get_all_lines()
    image = draw_segments(
        image, [line.as_array().reshape(-1) for line in lines], (0, 255, 0)
    )
    junctions = bpt2d.get_all_junctions()
    image = draw_points(
        image,
        [junc.p.p for junc in junctions if junc.degree() == 1],
        (255, 0, 0),
        1,
    )
    image = draw_points(
        image,
        [junc.p.p for junc in junctions if junc.degree() == 2],
        (0, 0, 255),
        1,
    )
    image = draw_points(
        image,
        [junc.p.p for junc in junctions if junc.degree() > 2],
        (0, 0, 0),
        1,
    )
    return image


def open3d_draw_bipartite3d_pointline(
    bpt3d, ranges=None, draw_edges=True, imagecols=None, draw_planes=False
):
    """
    Visualize point-line bipartite on 3D with Open3D
    """
    points, degrees = [], []
    for idx, ptrack in bpt3d.get_dict_points().items():
        p = ptrack.p
        deg = bpt3d.pdegree(idx)
        if (ranges is not None) and (not test_point_inside_ranges(p, ranges)):
            continue
        points.append(p)
        degrees.append(deg)
    # points_deg0 = [p for p, deg in zip(points, degrees) if deg == 0]
    points_deg1 = [p for p, deg in zip(points, degrees) if deg == 1]
    points_deg2 = [p for p, deg in zip(points, degrees) if deg == 2]
    points_deg3p = [p for p, deg in zip(points, degrees) if deg >= 3]
    lines = bpt3d.get_line_cloud()
    if ranges is not None:
        lines = [
            line for line in lines if test_line_inside_ranges(line, ranges)
        ]

    # optionally draw edges
    edges = None
    if draw_edges:
        edges = []
        for p_id, ptrack in bpt3d.get_dict_points().items():
            if bpt3d.pdegree(p_id) == 0:
                continue
            p = ptrack.p
            if (ranges is not None) and (
                not test_point_inside_ranges(p, ranges)
            ):
                continue
            for line_id in bpt3d.neighbor_lines(p_id):
                line = bpt3d.line(line_id).line
                p_proj = line.point_projection(p)
                edges.append(_base.Line3d(p, p_proj))

    # optiionally draw planes
    planes = None
    if draw_planes:
        planes = []
        scale = 0.2  # TODO: port properties
        for point_id in bpt3d.get_point_ids():
            # TODO: now we only consider degree-2 points
            if bpt3d.pdegree(point_id) != 2:
                continue

            point = bpt3d.point(point_id).p
            line_ids = bpt3d.neighbor_lines(point_id)
            direc1 = bpt3d.line(line_ids[0]).line.direction()
            direc2 = bpt3d.line(line_ids[1]).line.direction()
            normal = np.cross(direc1, direc2)
            # test if it is a psuedo junction
            if np.linalg.norm(normal) < 0.80:
                continue
            normal /= np.linalg.norm(normal)

            # draw plane
            base1 = direc1
            base2 = np.cross(normal, base1)

            vert1 = point + base1 * scale * 0.5
            vert2 = point + base2 * scale * 0.5
            vert3 = point + base1 * scale * (-0.5)
            vert4 = point + base2 * scale * (-0.5)
            planes.append([vert1, vert2, vert3, vert4])

    import open3d as o3d

    app = o3d.visualization.gui.Application.instance
    app.initialize()
    w = o3d.visualization.O3DVisualizer(height=1080, width=1920)
    w.show_skybox(False)
    w = open3d_add_points(
        w, points_deg1, color=(0.0, 0.0, 1.0), psize=1, name="pcd_deg1"
    )
    w = open3d_add_points(
        w, points_deg2, color=(1.0, 0.0, 0.0), psize=3, name="pcd_deg2"
    )
    w = open3d_add_points(
        w, points_deg3p, color=(1.0, 0.0, 0.0), psize=5, name="pcd_deg3p"
    )
    w = open3d_add_line_set(
        w, lines, color=(0.0, 1.0, 0.0), width=-1, name="line_set"
    )
    if edges is not None:
        w = open3d_add_line_set(
            w,
            edges,
            color=(0.0, 0.0, 0.0),
            width=-1,
            name="line_set_constraints",
        )

    if planes is not None:
        for plane_id, plane in enumerate(planes):
            # plane_c = (0.5, 0.4, 0.6)
            mesh = o3d.geometry.TriangleMesh()
            np_vertices = np.array(plane)
            np_triangles = np.array([[0, 1, 2], [0, 2, 3]]).astype(np.int32)
            mesh.vertices = o3d.utility.Vector3dVector(np_vertices)
            mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
            w.add_geometry(f"plane_{plane_id}", mesh)

    # optionally draw cameras
    if imagecols is not None:
        w = open3d_add_cameras(w, imagecols)
    w.reset_camera_to_default()
    w.scene_shader = w.UNLIT
    w.enable_raw_mode(True)
    app.add_window(w)
    app.run()


def open3d_draw_bipartite3d_vpline(bpt3d, ranges=None):
    """
    Visualize point-line bipartite on 3D with Open3D
    """
    import seaborn as sns

    n_vps = bpt3d.count_points()
    colors = sns.color_palette("husl", n_colors=n_vps)

    vp_ids = bpt3d.get_point_ids()
    vp_id_to_color = {vp_id: colors[idx] for idx, vp_id in enumerate(vp_ids)}
    vp_line_sets = {vp_id: [] for vp_id in vp_ids}
    nonvp_line_set = []
    for line_id, ltrack in bpt3d.get_dict_lines().items():
        if (ranges is not None) and (
            not test_line_inside_ranges(ltrack.line, ranges)
        ):
            continue
        labels = bpt3d.neighbor_points(line_id)
        if len(labels) == 0:
            nonvp_line_set.append(ltrack.line)
            continue
        assert len(labels) == 1
        label = labels[0]
        vp_line_sets[label].append(ltrack.line)

    # open3d
    import open3d as o3d

    app = o3d.visualization.gui.Application.instance
    app.initialize()
    w = o3d.visualization.O3DVisualizer(height=1080, width=1920)
    w.show_skybox(False)
    for vp_id in vp_ids:
        if len(vp_line_sets[vp_id]) == 0:
            continue
        w = open3d_add_line_set(
            w,
            vp_line_sets[vp_id],
            color=vp_id_to_color[vp_id],
            width=2,
            name=f"lineset_vp_{vp_id}",
        )
    w.reset_camera_to_default()
    w.scene_shader = w.UNLIT
    w.enable_raw_mode(True)
    app.add_window(w)
    app.run()
