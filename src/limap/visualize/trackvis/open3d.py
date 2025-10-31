import open3d as o3d

from ..vis_lines import open3d_get_cameras, open3d_get_line_set
from ..vis_utils import compute_robust_range_lines
from .base import BaseTrackVisualizer


class Open3DTrackVisualizer(BaseTrackVisualizer):
    def __init__(self, tracks):
        super().__init__(tracks)

    def reset(self):
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        return app

    def vis_all_lines(self, n_visible_views=4, width=2, ranges=None, scale=1.0):
        # TODO: support width
        lines = self.get_lines_n_visible_views(n_visible_views)
        vis = o3d.visualization.Visualizer()
        vis.create_window(height=1080, width=1920)
        line_set = open3d_get_line_set(lines, ranges=ranges, scale=scale)
        vis.add_geometry(line_set)
        vis.run()
        vis.destroy_window()

    def vis_reconstruction(
        self,
        imagecols,
        n_visible_views=4,
        ranges=None,
        scale=1.0,
        cam_scale=1.0,
    ):
        lines = self.get_lines_n_visible_views(n_visible_views)
        lranges = compute_robust_range_lines(lines)
        scale_cam_geometry = abs(lranges[1, :] - lranges[0, :]).max()

        vis = o3d.visualization.Visualizer()
        vis.create_window(height=1080, width=1920)
        line_set = open3d_get_line_set(lines, ranges=ranges, scale=scale)
        vis.add_geometry(line_set)
        camera_set = open3d_get_cameras(
            imagecols,
            ranges=ranges,
            scale_cam_geometry=scale_cam_geometry * cam_scale,
            scale=scale,
        )
        vis.add_geometry(camera_set)
        vis.run()
        vis.destroy_window()
