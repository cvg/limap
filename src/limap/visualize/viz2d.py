import copy

import cv2
import numpy as np
import seaborn as sns
from typeguard import typechecked

import limap.geometry
from limap.image.groups.vplib import VPResult
from limap.util.types import Color

from .viz_utils import random_color


@typechecked
def draw_2d_points(
    image: np.ndarray,
    points: np.ndarray,
    color: Color | None = None,
    thickness=1,
) -> np.ndarray:
    image = copy.deepcopy(image)
    for p in points:
        c = random_color() if color is None else color
        pos_x, pos_y = int(round(p[0])), int(round(p[1]))
        cv2.circle(image, (pos_x, pos_y), thickness * 2, c, -1)
    return image


@typechecked
def draw_2d_lines(
    image: np.ndarray,
    lines: list[np.ndarray] | list[limap.geometry.Line2d],
    color: Color | None = None,
    thickness: int = 1,
    endpoints: bool = True,
) -> np.ndarray:
    if isinstance(lines[0], np.ndarray):
        lines = [limap.geometry.Line2d(line_array) for line_array in lines]
    image = copy.deepcopy(image)
    for line in lines:
        c = random_color() if color is None else color
        p1 = (int(line.start[0]), int(line.start[1]))
        p2 = (int(line.end[0]), int(line.end[1]))
        cv2.line(image, p1, p2, c, thickness)
        if endpoints:
            cv2.circle(image, p1, thickness * 2, c, -1)
            cv2.circle(image, p2, thickness * 2, c, -1)
    return image


@typechecked
def draw_2d_vpresult(
    img: np.ndarray,
    lines: list[limap.geometry.Line2d],
    vpres: VPResult,
    vp_id: float = -1,
    show_original: bool = False,
    endpoints: bool = False,
) -> np.ndarray:
    n_vps = vpres.count_vps()
    colors = sns.color_palette("husl", n_colors=n_vps)
    colors = (np.array(colors) * 255).astype(np.uint8).tolist()
    if n_vps == 1:
        colors = [[255, 0, 0]]
    for line_id, line in enumerate(lines):
        c = [255, 255, 255]  # default color: white
        if (
            not vpres.has_vp(line_id)
            or vp_id >= 0
            and vpres.labels[line_id] != vp_id
        ):
            if not show_original:
                continue
        else:
            c = colors[vpres.labels[line_id]]
        cv2.line(
            img,
            (int(line.start[0]), int(line.start[1])),
            (int(line.end[0]), int(line.end[1])),
            c,
            2,
        )
        if endpoints:
            cv2.circle(
                img, (int(line.start[0]), int(line.start[1])), 3, [0, 0, 0], -1
            )
            cv2.circle(
                img, (int(line.end[0]), int(line.end[1])), 3, [0, 0, 0], -1
            )
    return img
