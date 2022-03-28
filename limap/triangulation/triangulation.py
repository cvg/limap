from _limap import _triangulation as _tri
from _limap import _base
import numpy as np

def get_normal_direction(l, view):
    return _tri.get_normal_direction(l, view)

def get_direction_from_VP(vp, view):
    return _tri.get_direction_from_VP(vp, view)

def compute_epipolar_IoU(l1, view1, l2, view2):
    return _tri.compute_epipolar_IoU(l1, view1, l2, view2)

def triangulate_endpoints(l1, view1, l2, view2):
    return _tri.triangulate_endpoints(l1, view1, l2, view2)

def triangulate(l1, view1, l2, view2):
    return _tri.triangulate(l1, view1, l2, view2)

def triangulate_with_direction(l1, view1, l2, view2, direc):
    return _tri.triangulate_with_direction(l1, view1, l2, view2, direc)

def GetAllLines2D(all_2d_segs):
    all_lines_2d = _base._GetAllLines2D(all_2d_segs)
    return all_lines_2d

def BuildInitialGraph(all_2d_segs, # list (N_images) of [num_segs_k, 5]
                      all_matches, # list ([N_images, n_neighbors]) of [num_matches_k, 2]
                      neighbors):  # np.ndarray [N_images, n_neighbors], correspond to index
    '''
    Returns:
    - all_lines_2d: all_2d_segs all casted as Line2d object
    - basegraph: directed line graph with edges connecting all initial matches
    '''
    all_lines_2d = _base._GetAllLines2D(all_2d_segs)

    basegraph = _base.DirectedGraph()
    _tri._BuildInitialGraph(basegraph, all_matches, neighbors)
    return all_lines_2d, basegraph

