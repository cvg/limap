import numpy as np
from scipy.spatial import distance_matrix
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from line_utils import start_endpoints, end_endpoints

def endpoints_distance(segs1, segs2):
    # calculate list of start and end points from both lists
    start_points1, end_points1 = start_endpoints(segs1), end_endpoints(segs1)
    start_points2, end_points2 = start_endpoints(segs2), end_endpoints(segs2)
    # calculate the distance matrix for start points 1 and 2, and end points 1 and 2
    dist_s1_s2 = distance_matrix(start_points1, start_points2)
    dist_s1_e2 = distance_matrix(start_points1, end_points2)
    dist_e1_s2 = distance_matrix(end_points1, start_points2)
    dist_e1_e2 = distance_matrix(end_points1, end_points2)
    # There is two possible configurations, switching the endpoints or not
    dist_matrix1 = (dist_s1_s2 + dist_e1_e2) / 2
    dist_matrix2 = (dist_s1_e2 + dist_e1_s2) / 2
    # We take the correct distance as the minimum of both
    dist_matrix = np.minimum(dist_matrix1, dist_matrix2)
    return dist_matrix

def individual_endpoints_distance(segs1, segs2):
    # calculate list of start and end points from both lists
    start_points1, end_points1 = start_endpoints(segs1), end_endpoints(segs1)
    start_points2, end_points2 = start_endpoints(segs2), end_endpoints(segs2)
    # calculate the distance matrix for start points 1 and 2, and end points 1 and 2
    dist_s1_s2 = np.linalg.norm(start_points1 - start_points2, axis=1)
    dist_s1_e2 = np.linalg.norm(start_points1 - end_points2, axis=1)
    dist_e1_s2 = np.linalg.norm(end_points1 - start_points2, axis=1)
    dist_e1_e2 = np.linalg.norm(end_points1 - end_points2, axis=1)
    # There is two possible configurations, switching the endpoints or not
    dist_matrix1 = (dist_s1_s2 + dist_e1_e2) / 2
    dist_matrix2 = (dist_s1_e2 + dist_e1_s2) / 2
    # We take the correct distance as the minimum of both
    dist_matrix = np.minimum(dist_matrix1, dist_matrix2)
    return dist_matrix


def individual_3d_endpoints_distance(segs1, segs2):
    assert segs1.ndim == 3 and segs1.shape[1] == 2 and segs1.shape[2] == 3
    assert segs2.ndim == 3 and segs2.shape[1] == 2 and segs2.shape[2] == 3

    # calculate list of start and end points from both lists
    start_points1, end_points1 = segs1[:, 0], segs1[:, 1]
    start_points2, end_points2 = segs2[:, 0], segs2[:, 1]
    # calculate the distance matrix for start points 1 and 2, and end points 1 and 2
    dist_s1_s2 = np.linalg.norm(start_points1 - start_points2, axis=1)
    dist_s1_e2 = np.linalg.norm(start_points1 - end_points2, axis=1)
    dist_e1_s2 = np.linalg.norm(end_points1 - start_points2, axis=1)
    dist_e1_e2 = np.linalg.norm(end_points1 - end_points2, axis=1)
    # There is two possible configurations, switching the endpoints or not
    dists1 = (dist_s1_s2 + dist_e1_e2) / 2
    dists2 = (dist_s1_e2 + dist_e1_s2) / 2
    # We take the correct distance as the minimum of both
    dist_matrix = np.minimum(dists1, dists2)
    return dist_matrix


def endpoints_distance_3d(segs1, segs2, structural=True):
    # calculate list of start and end points from both lists
    start_points1, end_points1 = segs1[:, 0], segs1[:, 1]
    start_points2, end_points2 = segs2[:, 0], segs2[:, 1]
    # calculate the distance matrix for start points 1 and 2, and end points 1 and 2
    dist_s1_s2 = distance_matrix(start_points1, start_points2)
    dist_s1_e2 = distance_matrix(start_points1, end_points2)
    dist_e1_s2 = distance_matrix(end_points1, start_points2)
    dist_e1_e2 = distance_matrix(end_points1, end_points2)
    # There is two possible configurations, switching the endpoints or not
    dist_matrix1 = (dist_s1_s2 + dist_e1_e2) / 2
    if not structural:
        return dist_matrix1

    dist_matrix2 = (dist_s1_e2 + dist_e1_s2) / 2
    # We take the correct distance as the minimum of both
    dist_matrix = np.minimum(dist_matrix1, dist_matrix2)
    return dist_matrix


