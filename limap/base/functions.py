import _limap._base as _base


def get_all_lines_2d(all_2d_segs):
    """
    Convert :class:`np.array` representations of 2D line segments \
        to dict of :class:`~limap.base.Line2d`.

    Args:
        all_2d_segs (dict[int -> :class:`np.array`]): \
            Map image IDs to :class:`np.array` of shape (N, 4), \
            each row (4 numbers) is concatenated by the start and end \
            of a 2D line segment.

    Returns:
        dict[int -> list[:class:`~limap.base.Line2d`]]: \
            Map image IDs to list of :class:`~limap.base.Line2d`.
    """
    all_lines_2d = {}
    for img_id in all_2d_segs:
        all_lines_2d[img_id] = _base._GetLine2dVectorFromArray(
            all_2d_segs[img_id]
        )
    return all_lines_2d


def get_all_lines_3d(all_3d_segs):
    """
    Convert :class:`np.array` representations of 3D line segments \
        to dict of :class:`~limap.base.Line3d`.

    Args:
        all_3d_segs (dict[int -> :class:`np.array`]): \
            Map image IDs to :class:`np.array` of shape (N, 2, 3), \
            each 2*3 matrix is stacked from the two endpoints \
            of a 3D line segment.

    Returns:
        dict[int -> list[:class:`~limap.base.Line3d`]]: \
            Map image IDs to list of :class:`~limap.base.Line3d`.
    """
    all_lines_3d = {}
    for img_id, segs3d in all_3d_segs.items():
        all_lines_3d[img_id] = _base._GetLine3dVectorFromArray(segs3d)
    return all_lines_3d


def get_invert_idmap_from_linetracks(all_lines_2d, linetracks):
    """
    Get the mapping from a 2D line segment (identified by an image and \
    its line ID) to the index of its associated linetrack.

    Args:
        all_lines_2d (dict[int -> list[:class:`~limap.base.Line2d`]]): \
            Map image IDs to the list of 2D line segments in each image.
        linetracks (list[:class:`~limap.base.LineTrack`]): All line tracks.

    Returns:
        dict[int -> list[int]]: Map image ID to list of the associated \
            line track indices for each 2D line, \
            -1 if not associated to any track.
    """
    map = {}
    for img_id in all_lines_2d:
        lines_2d = all_lines_2d[img_id]
        map[img_id] = [-1] * len(lines_2d)
    for track_id, track in enumerate(linetracks):
        for img_id, line_id in zip(track.image_id_list, track.line_id_list):
            map[img_id][line_id] = track_id
    return map
