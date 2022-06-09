from .base import BaseTrackVisualizer
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vis_utils import *

import open3d as o3d

class Open3DTrackVisualizer(BaseTrackVisualizer):
    def __init__(self, tracks):
        super(Open3DTrackVisualizer, self).__init__(tracks)

    def reset(self):
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        return app

    def get_line_set(self, n_visible_views=4):
        # generate lineset
        o3d_points, o3d_lines, o3d_colors = [], [], []
        counter = 0
        for track in self.tracks:
            if track.count_images() < n_visible_views:
                continue
            counter += 1
            line = track.line
            o3d_points.append(line.start)
            o3d_points.append(line.end)
            o3d_lines.append([2*counter, 2*counter+1])
            o3d_colors.append([1, 0, 0])
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(o3d_points)
        line_set.lines = o3d.utility.Vector2iVector(o3d_lines)
        line_set.colors = o3d.utility.Vector3dVector(o3d_colors)
        return line_set

    def vis_all_lines(self, n_visible_views=4, width=2):
        app = self.reset()

        line_set = self.get_line_set(n_visible_views=n_visible_views)

        w = o3d.visualization.O3DVisualizer(width=2048)
        w.show_ground = False
        w.show_axes = False
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnit"
        w.add_geometry("lineset", line_set, mat)
        w.reset_camera_to_default()
        w.scene_shader = w.UNLIT
        w.enable_raw_mode(True)
        app = o3d.visualization.gui.Application.instance
        app.add_window(w)
        app.run()

    def vis_reconstruction(self, imagecols):
        # TODO: raise NotImplementedError
        app = self.reset()

        line_set = self.get_line_set(n_visible_views=n_visible_views)

        w = o3d.visualization.O3DVisualizer(width=2048)
        w.show_ground = False
        w.show_axes = False
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnit"
        w.add_geometry("lineset", line_set, mat)
        w.reset_camera_to_default()
        w.scene_shader = w.UNLIT
        w.enable_raw_mode(True)
        app = o3d.visualization.gui.Application.instance
        app.add_window(w)
        app.run()
        pass


