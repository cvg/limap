from limap._limap import _geometry
import numpy as np


def get_all_lines_2d(
    all_2d_segs: dict[int, np.ndarray],
) -> dict[int, list[_geometry.Line2d]]:
    """
    Convert :class:`np.array` representations of 2D line segments \
        to dict of :class:`~limap.geometry.Line2d`.

    Args:
        all_2d_segs (dict[int -> :class:`np.array`]): \
            Map image IDs to :class:`np.array` of shape (N, 4), \
            each row (4 numbers) is concatenated by the start and end \
            of a 2D line segment.

    Returns:
        dict[int -> list[:class:`~limap.geometry.Line2d`]]: \
            Map image IDs to list of :class:`~limap.geometry.Line2d`.
    """
    all_lines_2d = {}
    for img_id in all_2d_segs:
        all_lines_2d[img_id] = _geometry.get_line2d_vector_from_array(
            all_2d_segs[img_id]
        )
    return all_lines_2d


def get_all_lines_3d(
    all_3d_segs: dict[int, np.ndarray],
) -> dict[int, list[_geometry.Line3d]]:
    """
    Convert :class:`np.array` representations of 3D line segments \
        to dict of :class:`~limap.geometry.Line3d`.

    Args:
        all_3d_segs (dict[int -> :class:`np.array`]): \
            Map image IDs to :class:`np.array` of shape (N, 2, 3), \
            each 2*3 matrix is stacked from the two endpoints \
            of a 3D line segment.

    Returns:
        dict[int -> list[:class:`~limap.geometry.Line3d`]]: \
            Map image IDs to list of :class:`~limap.geometry.Line3d`.
    """
    all_lines_3d = {}
    for img_id, segs3d in all_3d_segs.items():
        all_lines_3d[img_id] = _geometry.get_line3d_vector_from_array(segs3d)
    return all_lines_3d
