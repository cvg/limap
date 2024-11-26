import numpy as np
from pycolmap import logging

from ..vis_utils import test_line_inside_ranges


class BaseTrackVisualizer:
    def __init__(self, tracks):
        self.tracks = tracks
        self.counts = [track.count_images() for track in tracks]
        self.counts_lines = [track.count_lines() for track in tracks]
        self.lines = [track.line for track in tracks]

    def vis_all_lines(self, n_visible_views=4, width=2):
        raise NotImplementedError

    def vis_reconstruction(self, imagecols):
        raise NotImplementedError

    def report(self):
        self.report_stats()
        self.report_avg_supports(n_visible_views=3)
        self.report_avg_supports(n_visible_views=4)

    def report_stats(self):
        counts = np.array(self.counts)
        logging.info(
            f"[Track Report] (N2, N4, N6, N8, N10, N20, N50) ="
            f" ({counts[counts >= 2].shape[0]},"
            f" {counts[counts >= 4].shape[0]},"
            f" {counts[counts >= 6].shape[0]},"
            f" {counts[counts >= 8].shape[0]},"
            f" {counts[counts >= 10].shape[0]},"
            f" {counts[counts >= 20].shape[0]},"
            f" {counts[counts >= 50].shape[0]})"
        )

    def report_avg_supports(self, n_visible_views=4):
        counts = np.array(self.counts)
        counts_lines = np.array(self.counts_lines)
        arr = counts[counts >= n_visible_views]
        arr_lines = counts_lines[counts >= n_visible_views]
        logging.info(
            f"average supporting images (>= {n_visible_views}):"
            f" {arr.sum()} / {arr.shape[0]} = {arr.mean():.2f}"
        )
        logging.info(
            f"average supporting lines (>= {n_visible_views}): "
            f"{arr_lines.sum()} / {arr_lines.shape[0]} = {arr_lines.mean():.2f}"
        )

    def get_counts_np(self):
        return np.array(self.counts)

    def get_lines_np(self, n_visible_views=0):
        lines_np_list = []
        for idx, line in enumerate(self.lines):
            if self.counts[idx] < n_visible_views:
                continue
            lines_np_list.append(line.as_array())
        lines_np = np.array(lines_np_list)
        return lines_np

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

    def get_lines_n_visible_views(self, n_visible_views):
        lines = []
        for track in self.tracks:
            if track.count_images() < n_visible_views:
                continue
            lines.append(track.line)
        return lines
