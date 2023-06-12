import open3d as o3d
from .base import BaseTrackVisualizer
from ..vis_utils import compute_robust_range_lines
from ..vis_lines import open3d_get_line_set, open3d_get_cameras

class Open3DTrackVisualizer(BaseTrackVisualizer):
    def __init__(self, tracks):
        super(Open3DTrackVisualizer, self).__init__(tracks)

    def reset(self):
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        return app

    def vis_all_lines(self, n_visible_views=4, width=2, scale=1.0):
        lines = self.get_lines_n_visible_views(n_visible_views)
        vis = o3d.visualization.Visualizer()
        vis.create_window(height=1080, width=1920)
        line_set = open3d_get_line_set(lines, width=width, ranges=ranges, scale=scale)
        vis.add_geometry(line_set)
        vis.run()
        vis.destroy_window()

    def vis_reconstruction(self, imagecols, n_visible_views=4, width=2, ranges=None, scale=1.0, cam_scale=1.0):
        lines = self.get_lines_n_visible_views(n_visible_views)
        lranges = compute_robust_range_lines(lines)
        scale_cam_geometry = abs(lranges[1, :] - lranges[0, :]).max()

        vis = o3d.visualization.Visualizer()
        vis.create_window(height=1080, width=1920)
        line_set = open3d_get_line_set(lines, width=width, ranges=ranges, scale=scale)
        vis.add_geometry(line_set)
        camera_set = open3d_get_cameras(imagecols, ranges=ranges, scale_cam_geometry=scale_cam_geometry * cam_scale, scale=scale)
        vis.add_geometry(camera_set)
        vis.run()
        vis.destroy_window()

