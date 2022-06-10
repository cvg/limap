from .base import BaseTrackVisualizer
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vis_lines import open3d_add_line_set, open3d_add_cameras

import open3d as o3d

class Open3DTrackVisualizer(BaseTrackVisualizer):
    def __init__(self, tracks):
        super(Open3DTrackVisualizer, self).__init__(tracks)

    def reset(self):
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        return app

    def vis_all_lines(self, n_visible_views=4, width=2):
        app = self.reset()
        w = o3d.visualization.O3DVisualizer(width=1600)
        w.show_ground = False
        w.show_axes = False

        lines = self.get_lines_n_visible_views(n_visible_views)
        open3d_add_line_set(w, lines, width=width)

        w.reset_camera_to_default()
        w.scene_shader = w.UNLIT
        w.enable_raw_mode(True)
        app.add_window(w)
        app.run()

    def vis_reconstruction(self, imagecols, n_visible_views=4, width=2):
        app = self.reset()
        w = o3d.visualization.O3DVisualizer(width=1600)
        w.show_ground = False
        w.show_axes = False

        lines = self.get_lines_n_visible_views(n_visible_views)
        open3d_add_line_set(w, lines, width=width)
        open3d_add_cameras(w, imagecols)

        w.reset_camera_to_default()
        w.scene_shader = w.UNLIT
        w.enable_raw_mode(True)
        app.add_window(w)
        app.run()
        pass


