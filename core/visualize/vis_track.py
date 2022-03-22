from .vis_utils import *
import pyvista as pv
from pyvista import themes

class TrackVisualizer(object):
    def __init__(self, tracks, visualize=False):
        self.tracks = tracks
        self.counts = [track.count_images() for track in tracks]
        self.counts_lines = [track.count_lines() for track in tracks]
        self.lines = [track.line for track in tracks]
        if visualize:
            self.reset_plotter()

    def reset_plotter(self, img_hw=(600, 800)):
        my_theme = themes.DefaultTheme()
        my_theme.lighting = True
        my_theme.show_edges = True
        my_theme.edge_color = 'white'
        my_theme.background = 'white'
        self.plotter = pv.Plotter(window_size=[img_hw[1], img_hw[0]], theme=my_theme)

    def report(self):
        counts = np.array(self.counts)
        counts_lines = np.array(self.counts_lines)
        print('[Track Report] (N2, N4, N6, N8, N10, N20, N50) = ({0}, {1}, {2}, {3}, {4}, {5}, {6})'
            .format(counts[counts >= 2].shape[0],
                    counts[counts >= 4].shape[0],
                    counts[counts >= 6].shape[0],
                    counts[counts >= 8].shape[0],
                    counts[counts >= 10].shape[0],
                    counts[counts >= 20].shape[0],
                    counts[counts >= 50].shape[0]))
        arr = counts[counts >= 3]
        arr_lines = counts_lines[counts >= 3]
        print("average supporting images (>= 3): {0} / {1} = {2:.2f}".format(arr.sum(), arr.shape[0], arr.mean()))
        print("average supporting lines (>= 3): {0} / {1} = {2:.2f}".format(arr_lines.sum(), arr_lines.shape[0], arr_lines.mean()))
        arr = counts[counts >= 4]
        arr_lines = counts_lines[counts >= 4]
        print("average supporting images (>= 4): {0} / {1} = {2:.2f}".format(arr.sum(), arr.shape[0], arr.mean()))
        print("average supporting lines (>= 4): {0} / {1} = {2:.2f}".format(arr_lines.sum(), arr_lines.shape[0], arr_lines.mean()))

    def get_counts_np(self):
        return np.array(self.counts)

    def get_lines_np(self):
        lines_np = np.array([line.as_array() for line in self.lines])
        return lines_np

    def vis_all_lines(self, img_hw=(600, 800), n_visible_views=4, width=2):
        self.report()
        self.reset_plotter(img_hw)
        for track_id, line in enumerate(self.lines):
            if self.counts[track_id] < n_visible_views:
                continue
            color = '000000'
            self.plotter.add_lines(line.as_array(), color, width=width)
        self.plotter.show()

    def vis_all_lines_image(self, img_id, img_hw=(600, 800), n_visible_views=4, width=2):
        flags = [track.HasImage(img_id) for track in self.tracks]
        self.reset_plotter(img_hw)
        for track_id, line in enumerate(self.lines):
            if self.counts[track_id] < n_visible_views:
                continue
            color = "#ff0000"
            if flags[track_id]:
                color = "#00ff00"
            self.plotter.add_lines(line.as_array(), color, width=width)
        self.plotter.show()

    def vis_additional_lines(self, lines, img_hw=(600, 800), width=2):
        self.reset_plotter(img_hw)
        for track_id, line in enumerate(self.lines):
            color = "#ff0000"
            self.plotter.add_lines(line.as_array(), color, width=width)
        for line in lines:
            color = "#00ff00"
            self.plotter.add_lines(line.as_array(), color, width=width)
        self.plotter.show()

    def get_lines_for_images(self, image_list):
        lines, counts = [], []
        for track_id, line in enumerate(self.lines):
            flag = False
            for img_id in image_list:
                if self.tracks[track_id].HasImage(img_id):
                    flag = True
                    break
            if flag:
                lines.append(line)
                counts.append(self.counts[track_id])
        lines_np = np.array([line.as_array() for line in lines])
        counts_np = np.array(counts)
        return lines_np, counts_np

    def get_lines_within_ranges(self, ranges):
        lines, counts = [], []
        for track_id, line in enumerate(self.lines):
            if test_line_inside_ranges(line.as_array(), ranges):
                lines.append(line)
                counts.append(self.counts[track_id])
        lines_np = np.array([line.as_array() for line in lines])
        counts_np = np.array(counts)
        return lines_np, counts_np

