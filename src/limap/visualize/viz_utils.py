import numpy as np
from typeguard import typechecked

import limap.geometry
from limap.util.types import Color, Ranges


@typechecked
def random_color() -> Color:
    r = int(255 * np.random.rand())
    g = int(255 * np.random.rand())
    b = 255 - r
    return b, g, r


@typechecked
def _cat_to_big_image(imgs: list[list[np.ndarray]], pad=20) -> np.ndarray:
    img = imgs[0][0]
    img_h, img_w = img.shape[0], img.shape[1]
    channel = img.shape[2]
    n_rows, n_cols = len(imgs), len(imgs[0])

    final_img_h = img_w * n_cols + pad * (n_cols - 1)
    final_img_w = img_h * n_rows + pad * (n_rows - 1)
    bigimg = (np.ones((final_img_h, final_img_w, channel)) * 255).astype(
        np.uint8
    )
    for col in range(n_cols):
        for row in range(n_rows):
            img = imgs[col][row]
            assert img.shape[0] == img_h
            assert img.shape[1] == img_w
            row_start = (img_h + pad) * row
            row_end = row_start + img_h
            col_start = (img_w + pad) * col
            col_end = col_start + img_w
            bigimg[row_start:row_end, col_start:col_end, :] = img
    return bigimg


@typechecked
def make_big_image(imgs: list[np.ndarray], pad=20) -> np.ndarray:
    """
    make a big image with 2d image collections
    all images should have the same size
    """
    n_images = len(imgs)
    if n_images <= 5:
        return _cat_to_big_image([[imgs]], pad=pad)

    n_cols = int(np.ceil(np.sqrt(n_images)))
    int(np.ceil(n_images / n_cols))
    imgs_collections = []
    imgs_now = []
    for image in imgs:
        imgs_now.append(image)
        if len(imgs_now) == n_cols:
            imgs_collections.append(imgs_now)
            imgs_now = []
    blank_image = (
        np.ones((imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2])) * 255
    ).astype(np.uint8)
    while len(imgs_now) < n_cols:
        imgs_now.append(blank_image)
    imgs_collections.append(imgs_now)
    return _cat_to_big_image(imgs_collections, pad=pad)


@typechecked
def test_point_inside_ranges(point: np.ndarray, ranges: Ranges) -> bool:
    point = np.array(point)
    return bool(np.all(point > ranges[0]) and np.all(point < ranges[1]))


@typechecked
def test_line_inside_ranges(
    line: limap.geometry.Line3d, ranges: Ranges
) -> bool:
    if not test_point_inside_ranges(line.start, ranges):
        return False
    return test_point_inside_ranges(line.end, ranges)


@typechecked
def compute_robust_range(
    arr: np.ndarray,
    range_robust: tuple[float, float] | None = None,
    k_stretch: float = 2.0,
) -> tuple[float, float]:
    if range_robust is None:
        range_robust = (0.05, 0.95)
    N = arr.shape[0]
    if N == 0:
        return (0.0, 1.0)  # Default range for empty arrays
    start_idx = int(round((N - 1) * range_robust[0]))
    end_idx = int(round((N - 1) * range_robust[1]))
    arr_sorted = np.sort(arr)
    start = arr_sorted[start_idx]
    end = arr_sorted[end_idx]
    start_stretched = (start + end) / 2.0 - k_stretch * (end - start) / 2.0
    end_stretched = (start + end) / 2.0 + k_stretch * (end - start) / 2.0
    return start_stretched, end_stretched


@typechecked
def compute_robust_range_points(
    points: np.ndarray,
    range_robust: tuple[float, float] | None = None,
    k_stretch: float = 2.0,
) -> Ranges:
    if range_robust is None:
        range_robust = (0.05, 0.95)
    points_array = np.array(points)
    x_array = points_array.reshape(-1, 3)[:, 0]
    y_array = points_array.reshape(-1, 3)[:, 1]
    z_array = points_array.reshape(-1, 3)[:, 2]

    x_start, x_end = compute_robust_range(
        x_array, range_robust=range_robust, k_stretch=k_stretch
    )
    y_start, y_end = compute_robust_range(
        y_array, range_robust=range_robust, k_stretch=k_stretch
    )
    z_start, z_end = compute_robust_range(
        z_array, range_robust=range_robust, k_stretch=k_stretch
    )
    ranges = (
        np.array([x_start, y_start, z_start]),
        np.array([x_end, y_end, z_end]),
    )
    return ranges


@typechecked
def compute_robust_range_lines(
    lines: list[limap.geometry.Line3d],
    range_robust: tuple[float, float] | None = None,
    k_stretch: float = 2.0,
) -> Ranges:
    if range_robust is None:
        range_robust = (0.05, 0.95)
    lines_array = np.array([line.as_array() for line in lines])
    x_array = lines_array.reshape(-1, 3)[:, 0]
    y_array = lines_array.reshape(-1, 3)[:, 1]
    z_array = lines_array.reshape(-1, 3)[:, 2]

    x_start, x_end = compute_robust_range(
        x_array, range_robust=range_robust, k_stretch=k_stretch
    )
    y_start, y_end = compute_robust_range(
        y_array, range_robust=range_robust, k_stretch=k_stretch
    )
    z_start, z_end = compute_robust_range(
        z_array, range_robust=range_robust, k_stretch=k_stretch
    )
    ranges = (
        np.array([x_start, y_start, z_start]),
        np.array([x_end, y_end, z_end]),
    )
    return ranges
