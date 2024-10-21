from .base import BaseTrackVisualizer


class PyVistaTrackVisualizer(BaseTrackVisualizer):
    def __init__(self, tracks):
        super().__init__(tracks)
        self.reset()

    def reset(self, img_hw=(600, 800)):
        import pyvista as pv
        from pyvista import themes

        my_theme = themes.DefaultTheme()
        my_theme.lighting = True
        my_theme.show_edges = True
        my_theme.edge_color = "white"
        my_theme.background = "white"
        self.plotter = pv.Plotter(
            window_size=[img_hw[1], img_hw[0]], theme=my_theme
        )

    def vis_all_lines(self, n_visible_views=4, width=2, scale=1.0):
        lines = self.get_lines_n_visible_views(n_visible_views)
        color = "#ff0000"
        for line in lines:
            self.plotter.add_lines(line.as_array() * scale, color, width=width)
        self.plotter.show()

    def vis_all_lines_image(
        self, img_id, img_hw=(600, 800), n_visible_views=4, width=2
    ):
        flags = [track.HasImage(img_id) for track in self.tracks]
        for track_id, line in enumerate(self.lines):
            if self.counts[track_id] < n_visible_views:
                continue
            color = "#ff0000"
            if flags[track_id]:
                color = "#00ff00"
            self.plotter.add_lines(line.as_array(), color, width=width)
        self.plotter.show()

    def vis_additional_lines(self, lines, img_hw=(600, 800), width=2):
        for line in self.lines:
            color = "#ff0000"
            self.plotter.add_lines(line.as_array(), color, width=width)
        for line in lines:
            color = "#00ff00"
            self.plotter.add_lines(line.as_array(), color, width=width)
        self.plotter.show()
