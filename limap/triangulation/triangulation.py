from _limap import _triangulation as _tri


def get_normal_direction(line2d, view):
    return _tri.get_normal_direction(line2d, view)


def get_direction_from_VP(vp, view):
    """
    Get the 3d direction from a 2D vanishing point

    Args:
        vp (:class:`np.array` of shape (3,))
        view (:class:`limap.base.CameraView`)
    Returns:
        direction (:class:`np.array` of shape (3,))
    """
    return _tri.get_direction_from_VP(vp, view)


def compute_essential_matrix(view1, view2):
    """
    Get the essential matrix between two views

    Args:
        view1 (:class:`limap.base.CameraView`)
        view2 (:class:`limap.base.CameraView`)
    Returns:
        essential_matrix (:class:`np.array` of shape (3, 3))
    """
    return _tri.compute_essential_matrix(view1, view2)


def compute_fundamental_matrix(view1, view2):
    """
    Get the essential matrix between two views

    Args:
        view1 (:class:`limap.base.CameraView`)
        view2 (:class:`limap.base.CameraView`)
    Returns:
        fundamental_matrix (:class:`np.array` of shape (3, 3))
    """
    return _tri.compute_fundamental_matrix(view1, view2)


def compute_epipolar_IoU(l1, view1, l2, view2):
    """
    Get the IoU between two lines from different views by \
    intersecting the epipolar lines

    Args:
        l1 (:class:`limap.base.Line2d`)
        view1 (:class:`limap.base.CameraView`)
        l2 (:class:`limap.base.Line2d`)
        view2 (:class:`limap.base.CameraView`)
    Returns:
        IoU (float): The calculated epipolar IoU
    """
    return _tri.compute_epipolar_IoU(l1, view1, l2, view2)


def triangulate_point(p1, view1, p2, view2):
    """
    Two-view point triangulation (mid-point)

    Args:
        p1 (:class:`np.array` of shape (2,))
        view1 (:class:`limap.base.CameraView`)
        p2 (:class:`np.array` of shape (2,))
        view2 (:class:`limap.base.CameraView`)
    Returns:
        point3d (:class:`np.array` of shape (3,))
    """
    return _tri.triangulate_point(p1, view1, p2, view2)


def triangulate_line_by_endpoints(l1, view1, l2, view2):
    """
    Two-view triangulation of lines with point triangulation \
    on both endpoints (assuming correspondences)

    Args:
        l1 (:class:`limap.base.Line2d`)
        view1 (:class:`limap.base.CameraView`)
        l2 (:class:`limap.base.Line2d`)
        view2 (:class:`limap.base.CameraView`)
    Returns:
        line3d (:class:`limap.base.Line3d`)
    """
    return _tri.triangulate_line_by_endpoints(l1, view1, l2, view2)


def triangulate_line(l1, view1, l2, view2):
    """
    Two-view triangulation of lines by ray-plane intersection

    Args:
        l1 (:class:`limap.base.Line2d`)
        view1 (:class:`limap.base.CameraView`)
        l2 (:class:`limap.base.Line2d`)
        view2 (:class:`limap.base.CameraView`)
    Returns:
        line3d (:class:`limap.base.Line3d`)
    """
    return _tri.triangulate_line(l1, view1, l2, view2)


def triangulate_line_with_one_point(l1, view1, l2, view2, p):
    """
    Two-view triangulation of lines with a known 3D point on the line

    Args:
        l1 (:class:`limap.base.Line2d`)
        view1 (:class:`limap.base.CameraView`)
        l2 (:class:`limap.base.Line2d`)
        view2 (:class:`limap.base.CameraView`)
        point (:class:`np.array` of shape (3,))
    Returns:
        line3d (:class:`limap.base.Line3d`)
    """
    return _tri.triangulate_line_with_one_point(l1, view1, l2, view2, p)


def triangulate_line_with_direction(l1, view1, l2, view2, direc):
    """
    Two-view triangulation of lines with known 3D line direction

    Args:
        l1 (:class:`limap.base.Line2d`)
        view1 (:class:`limap.base.CameraView`)
        l2 (:class:`limap.base.Line2d`)
        view2 (:class:`limap.base.CameraView`)
        direction (:class:`np.array` of shape (3,))
    Returns:
        line3d (:class:`limap.base.Line3d`)
    """
    return _tri.triangulate_line_with_direction(l1, view1, l2, view2, direc)
