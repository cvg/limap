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

def open3d_vis_3d_lines(lines):
    import open3d as o3d
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    # generate lineset
    o3d_points, o3d_lines, o3d_colors = [], [], []
    for idx, line in enumerate(lines):
        o3d_points.append(line.start)
        o3d_points.append(line.end)
        o3d_lines.append([2*idx, 2*idx+1])
        o3d_colors.append([1, 0, 0])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(o3d_points)
    line_set.lines = o3d.utility.Vector2iVector(o3d_lines)
    line_set.colors = o3d.utility.Vector3dVector(o3d_colors)

    # initiate a window
    w = o3d.visualization.O3DVisualizer(width=2048)
    w.show_ground = False
    w.show_axes = False
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnit"
    w.add_geometry("lineset", line_set, mat)
    w.reset_camera_to_default()
    w.scene_shader = w.UNLIT
    w.enable_raw_mode(True)

    # initiate app
    app.add_window(w)
    app.run()

