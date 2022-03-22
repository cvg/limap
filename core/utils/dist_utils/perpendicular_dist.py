import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from constants import ETH_EPS
from line_utils import start_endpoints, end_endpoints

def individual_perpendicular_distance(segs1, segs2):
    # calculate list of start, end and midpoints points from both lists
    start_points1, end_points1 = to_homogeneous(start_endpoints(segs1)), to_homogeneous(end_endpoints(segs1))
    start_points2, end_points2 = to_homogeneous(start_endpoints(segs2)), to_homogeneous(end_endpoints(segs2))

    # Compute the centers of each segment (in homogeneous coords)
    centers1 = (start_points1 + end_points1) / 2.0
    centers2 = (start_points2 + end_points2) / 2.0
    # Compute the line equations as ax + by + c = 0 , where x^2 + y^2 = 1
    lines1 = np.cross(start_points1, end_points1)
    lines1 = lines1 / (np.sqrt(lines1[:, 0] ** 2 + lines1[:, 1] ** 2)[:, np.newaxis] + ETH_EPS)
    lines2 = np.cross(start_points2, end_points2)
    lines2 = lines2 / (np.sqrt(lines2[:, 0] ** 2 + lines2[:, 1] ** 2)[:, np.newaxis] + ETH_EPS)

    # Compute the perpendicular distance
    dist1 = np.abs(np.einsum('ij,ij->i', lines1, centers2))
    dist2 = np.abs(np.einsum('ij,ij->i', lines2, centers1))
    dist_matrix = (dist1 + dist2) / 2.0
    return dist_matrix


def perpendicular_distance(segs1, segs2):
    # calculate list of start, end and midpoints points from both lists
    start_points1, end_points1 = to_homogeneous(start_endpoints(segs1)), to_homogeneous(end_endpoints(segs1))
    start_points2, end_points2 = to_homogeneous(start_endpoints(segs2)), to_homogeneous(end_endpoints(segs2))

    # Compute the centers of each segment (in homogeneous coords)
    centers1 = (start_points1 + end_points1) / 2.0
    centers2 = (start_points2 + end_points2) / 2.0
    # Compute the line equations as ax + by + c = 0 , where x^2 + y^2 = 1
    lines1 = np.cross(start_points1, end_points1)
    lines1 = lines1 / (np.sqrt(lines1[:, 0] ** 2 + lines1[:, 1] ** 2)[:, np.newaxis] + ETH_EPS)
    lines2 = np.cross(start_points2, end_points2)
    lines2 = lines2 / (np.sqrt(lines2[:, 0] ** 2 + lines2[:, 1] ** 2)[:, np.newaxis] + ETH_EPS)

    # Compute the perpendicular distance
    dist1 = np.abs(np.einsum('ij,kj->ik', lines1, centers2))
    dist2 = np.abs(np.einsum('ij,kj->ki', lines2, centers1))
    dist_matrix = (dist1 + dist2) / 2.0
    return dist_matrix

def individual_3d_perpendicular_distance(segs1, segs2):
    # Get a vector with the direction of each segment
    dirs1 = segs1[:, 1] - segs1[:, 0]
    dirs1 /= (ETH_EPS + np.linalg.norm(dirs1, axis=1)[:, np.newaxis])

    # Create a projection (rotation) matrix, with two vector orthogonal to dirs1
    r1 = np.array([np.linalg.svd(d)[2][1:] for d in dirs1[:, np.newaxis]])

    # Do the projection. Shape: (n_segs, endpoints, {dx,dy} )
    perp_distances_2d = np.einsum('kij, kmj->kmi', r1, segs2 - segs1[:, 0, np.newaxis])
    # Compute the norm in the perpendicular direction
    perp_distances = np.linalg.norm(perp_distances_2d, axis=2)
    # Return the mean distance of both endpoints
    return perp_distances.mean(axis=1)


def perpendicular_distance_3d(segs1, segs2):
    # Get a vector with the direction of each segment
    dirs1 = segs1[:, 1] - segs1[:, 0]
    dirs1 /= (ETH_EPS + np.linalg.norm(dirs1, axis=1)[:, np.newaxis])

    # Create a projection (rotation) matrix, with two vector orthogonal to dirs1
    r1 = np.array([np.linalg.svd(d)[2][1:] for d in dirs1[:, np.newaxis]])

    # Do the projection. Shape: (n_segs1,n_segs2, endpoints, {dx,dy} )
    perp_distances_2d = np.einsum('kij, knmj->knmi', r1, segs2 - segs1[:, np.newaxis, np.newaxis, 0])
    # Compute the norm in the perpendicular direction
    perp_distances = np.linalg.norm(perp_distances_2d, axis=-1)
    # Return the mean distance of both endpoints
    return perp_distances.mean(axis=-1)


