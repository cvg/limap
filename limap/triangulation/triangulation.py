from _limap import _triangulation as _tri
from _limap import _base
import numpy as np

def get_normal_direction(l, view):
    return _tri.get_normal_direction(l, view)

def get_direction_from_VP(vp, view):
    return _tri.get_direction_from_VP(vp, view)

def compute_essential_matrix(view1, view2):
    return _tri.compute_essential_matrix(view1, view2)

def compute_fundamental_matrix(view1, view2):
    return _tri.compute_fundamental_matrix(view1, view2)

def compute_epipolar_IoU(l1, view1, l2, view2):
    return _tri.compute_epipolar_IoU(l1, view1, l2, view2)

def point_triangulation(p1, view1, p2, view2):
    return _tri.point_triangulation(p1, view1, p2, view2)

def triangulate_endpoints(l1, view1, l2, view2):
    return _tri.triangulate_endpoints(l1, view1, l2, view2)

def triangulate(l1, view1, l2, view2):
    return _tri.triangulate(l1, view1, l2, view2)

def triangulate_with_one_point(l1, view1, l2, view2, p):
    return _tri.triangulate_with_one_point(l1, view1, l2, view2, p)

def triangulate_with_direction(l1, view1, l2, view2, direc):
    return _tri.triangulate_with_direction(l1, view1, l2, view2, direc)

