import numpy as np

def pyvista_vis_3d_lines(lines, img_hw=(600, 800), width=2):
    '''
    Input:
    - lines: list of _base.Line3d
    '''
    import pyvista as pv
    plotter = pv.Plotter(window_size=[img_hw[1], img_hw[0]])
    for line in lines:
        plotter.add_lines(line.as_array(), '#ff0000', width=width)
    plotter.show()

def open3d_add_points(w, points, color=[0.0, 0.0, 0.0], psize=1.0, name="pcd"):
    if np.array(points).shape[0] == 0:
        return w
    import open3d as o3d
    o3d_points, o3d_colors = [], []
    for idx in range(np.array(points).shape[0]):
        o3d_points.append(points[idx])
        o3d_colors.append(color)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack(o3d_points))
    pcd.colors = o3d.utility.Vector3dVector(np.stack(o3d_colors))
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = psize
    w.add_geometry(name, pcd, mat)
    return w

def open3d_add_line_set(w, lines, color=[0.0, 0.0, 0.0], width=2, name="lineset"):
    import open3d as o3d
    o3d_points, o3d_lines, o3d_colors = [], [], []
    for idx, line in enumerate(lines):
        o3d_points.append(line.start)
        o3d_points.append(line.end)
        o3d_lines.append([2*idx, 2*idx+1])
        o3d_colors.append(color)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(o3d_points)
    line_set.lines = o3d.utility.Vector2iVector(o3d_lines)
    line_set.colors = o3d.utility.Vector3dVector(o3d_colors)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.line_width = width
    w.add_geometry(name, line_set, mat)
    return w

def open3d_add_cameras(w, imagecols, color=[1.0, 0.0, 0.0]):
    import open3d as o3d
    import copy
    camera_lines = {}
    for cam_id in imagecols.get_cam_ids():
        cam = imagecols.cam(cam_id)
        camera_lines[cam_id] = o3d.geometry.LineSet.create_camera_visualization(cam.w(), cam.h(), cam.K(), np.eye(4), scale=0.1)
    for img_id in imagecols.get_img_ids():
        camimage = imagecols.camimage(img_id)
        T = np.eye(4)
        T[:3, :3] = camimage.R()
        T[:3, 3] = camimage.T()
        T = np.linalg.inv(T)
        cam = copy.deepcopy(camera_lines[camimage.cam_id]).transform(T)
        cam.paint_uniform_color(color)
        w.add_geometry(camimage.image_name(), cam)
    return w

def open3d_vis_3d_lines(lines, width=2):
    import open3d as o3d
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    # initiate a window
    w = o3d.visualization.O3DVisualizer(width=2048)
    w.show_ground = False
    w.show_axes = False

    w = open3d_add_line_set(w, lines, width=width)

    w.reset_camera_to_default()
    w.scene_shader = w.UNLIT
    w.enable_raw_mode(True)

    # initiate app
    app.add_window(w)
    app.run()

