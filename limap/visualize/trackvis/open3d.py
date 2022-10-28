from .base import BaseTrackVisualizer
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vis_utils import compute_robust_range_lines
from vis_lines import open3d_add_line_set, open3d_add_cameras

import open3d as o3d

class Open3DTrackVisualizer(BaseTrackVisualizer):
    def __init__(self, tracks):
        super(Open3DTrackVisualizer, self).__init__(tracks)

    def reset(self):
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        return app

    def vis_all_lines(self, n_visible_views=4, width=2, scale=1.0):
        app = self.reset()
        w = o3d.visualization.O3DVisualizer(width=1600)
        w.show_ground = False
        w.show_axes = False

        lines = self.get_lines_n_visible_views(n_visible_views)
        open3d_add_line_set(w, lines, width=width, scale=scale)

        w.reset_camera_to_default()
        w.scene_shader = w.UNLIT
        w.enable_raw_mode(True)
        app.add_window(w)
        app.run()

    def vis_reconstruction(self, imagecols, n_visible_views=4, width=2, ranges=None, scale=1.0, cam_scale=1.0):
        app = self.reset()
        w = o3d.visualization.O3DVisualizer(width=1600)
        w.show_ground = False
        w.show_axes = False

        lines = self.get_lines_n_visible_views(n_visible_views)
        open3d_add_line_set(w, lines, width=width, ranges=ranges, scale=scale)
        ranges = compute_robust_range_lines(lines)
        scale_cam_geometry = (ranges[1, :] - ranges[0, :]).max()
        open3d_add_cameras(w, imagecols, ranges=ranges, scale_cam_geometry=scale_cam_geometry * cam_scale, scale=scale)

        w.reset_camera_to_default()
        w.scene_shader = w.UNLIT
        w.enable_raw_mode(True)
        app.add_window(w)
        app.run()
        pass


