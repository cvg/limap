import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from constants import ETH_EPS
from line_utils import start_endpoints, end_endpoints

def individual_overlap_distance(segs1, segs2):
    start_points1, end_points1 = start_endpoints(segs1), end_endpoints(segs1)
    start_points2, end_points2 = start_endpoints(segs2), end_endpoints(segs2)
    # Compute the direction vector of the segments in segs2 (the GT)
    dirs2 = end_points2 - start_points2
    dirs2 /= (ETH_EPS + np.linalg.norm(dirs2, axis=1)[:, np.newaxis])

    # Project the segments in its direction vector
    projected_start_pt2 = np.einsum('ij,ij->i', dirs2, start_points2)
    projected_end_pt2 = np.einsum('ij,ij->i', dirs2, end_points2)

    # Project segs1 in each direction vector of segs2 generating a matrix
    projected_start_pts1 = np.einsum('ij,ij->i', start_points1, dirs2)
    projected_end_pts1 = np.einsum('ij,ij->i', end_points1, dirs2)

    # Get the right(max) and left(min) endpoints of each reference segment
    pts1_max = np.maximum(projected_start_pts1, projected_end_pts1)
    pts1_min = np.minimum(projected_start_pts1, projected_end_pts1)
    # Get the  right(max) and left(min) endpoints of the projected segments
    pt2_max = np.maximum(projected_start_pt2, projected_end_pt2)
    pt2_min = np.minimum(projected_start_pt2, projected_end_pt2)
    # Compute the intersection over the union
    intersection = np.minimum(pt2_max, pts1_max) - np.maximum(pt2_min, pts1_min)
    union = np.maximum(pt2_max, pts1_max) - np.minimum(pt2_min, pts1_min)
    iou = intersection / (ETH_EPS + union)
    return np.maximum(0, intersection), np.maximum(0, union), np.maximum(0, iou)

def overlap_distance(segs1, segs2):
    start_points1, end_points1 = start_endpoints(segs1), end_endpoints(segs1)
    start_points2, end_points2 = start_endpoints(segs2), end_endpoints(segs2)
    # Compute the direction vector of the segments in segs2 (the GT)
    dirs2 = end_points2 - start_points2
    dirs2 /= (ETH_EPS + np.linalg.norm(dirs2, axis=1)[:, np.newaxis])

    # Project the segments in its direction vector
    projected_start_pt2 = np.einsum('ij,ij->i', dirs2, start_points2)
    projected_end_pt2 = np.einsum('ij,ij->i', dirs2, end_points2)

    # Project segs1 in each direction vector of segs2 generating a matrix
    projected_start_pts1 = np.einsum('ij,kj->ik', start_points1, dirs2)
    projected_end_pts1 = np.einsum('ij,kj->ik', end_points1, dirs2)

    # Get the right(max) and left(min) endpoints of each reference segment
    pts1_max = np.maximum(projected_start_pts1, projected_end_pts1)
    pts1_min = np.minimum(projected_start_pts1, projected_end_pts1)
    # Get the  right(max) and left(min) endpoints of the projected segments
    pt2_max = np.maximum(projected_start_pt2, projected_end_pt2)
    pt2_min = np.minimum(projected_start_pt2, projected_end_pt2)
    # Compute the intersection over the union
    intersection = np.minimum(pt2_max, pts1_max) - np.maximum(pt2_min, pts1_min)
    union = np.maximum(pt2_max, pts1_max) - np.minimum(pt2_min, pts1_min)
    iou = intersection / (ETH_EPS + union)
    return np.maximum(0, intersection), np.maximum(0, union), np.maximum(0, iou)

def individual_3d_overlap_distance(segs1, segs2):
    # Get a vector with the direction of each segment
    dirs1 = segs1[:, 1] - segs1[:, 0]
    dirs1 /= (ETH_EPS + np.linalg.norm(dirs1, axis=1)[:, np.newaxis])

    # Do the projection. Shape: (n_segs, endpoints, {dx,dy} )
    projections2 = np.einsum('ij, ikj->ik', dirs1, segs2 - segs1[:, 0, np.newaxis])

    pts1_max = np.linalg.norm(segs1[:, 1] - segs1[:, 0], axis=1)
    pt2_min = projections2.min(axis=1)
    pt2_max = projections2.max(axis=1)

    intersection = np.minimum(pt2_max, pts1_max) - np.maximum(pt2_min, 0)
    union = np.maximum(pt2_max, pts1_max) - np.minimum(pt2_min, 0)
    iou = intersection / (ETH_EPS + union)
    return intersection, union, iou

def overlap_distance_3d(segs1, segs2):
    # Get a vector with the direction of each segment
    dirs1 = segs1[:, 1] - segs1[:, 0]
    dirs1 /= (ETH_EPS + np.linalg.norm(dirs1, axis=1)[:, np.newaxis])

    # Do the projection. Shape: (n_segs1, n_segs2, endpoints, {dx,dy} )
    projections2 = np.einsum('ij, ikmj->ikm', dirs1, segs2 - segs1[:, np.newaxis, np.newaxis, 0])

    pts1_max = np.linalg.norm(segs1[:, 1] - segs1[:, 0], axis=1)
    pts1_max = np.repeat(pts1_max[:, np.newaxis], len(segs2), axis=1)
    pt2_min = projections2.min(axis=-1)
    pt2_max = projections2.max(axis=-1)

    intersection = np.minimum(pt2_max, pts1_max) - np.maximum(pt2_min, 0)
    union = np.maximum(pt2_max, pts1_max) - np.minimum(pt2_min, 0)
    iou = intersection / (ETH_EPS + union)
    return intersection, union, iou


