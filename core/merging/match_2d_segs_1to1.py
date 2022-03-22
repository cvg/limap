import numpy as np
from scipy.optimize import linear_sum_assignment

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import angular_distance, perpendicular_distance, overlap_distance, endpoints_distance

def match_2d_segs_1to1(segs1, segs2, angular_th=(15 * np.pi / 180), perp_dist_th=2 * 1.41421356, iou_th=0.1):
    """Match segs1 and segs2 1-to-1, minimizing the endpoints.
     The overlap is calculated projecting segs1 over segs2."""
    HIGH_VALUE = 100000

    full_distance_matrix = endpoints_distance(segs1, segs2)

    # We require that they have a similar angle (min ang =15 degreed)
    angular_dist = angular_distance(segs1, segs2)
    full_distance_matrix[angular_dist > angular_th] = HIGH_VALUE

    # A small per-perpendicular distance (2 * math.sqrt(2) pixels)
    perp_dist = perpendicular_distance(segs1, segs2)
    full_distance_matrix[perp_dist > perp_dist_th] = HIGH_VALUE

    # Project the detected segments over the GT
    intersection_mat, union_mat, overlap_matrix = overlap_distance(segs1, segs2)
    # Set a small threshold in the segments overlap (0.1 in the paper)
    full_distance_matrix[overlap_matrix < iou_th] = HIGH_VALUE

    # Enforce the 1-to-1 assignation
    matched_det, matched_gt = linear_sum_assignment(full_distance_matrix)
    # Use 1-to-N assignation
    # matched_det, matched_gt = np.argwhere(full_distance_matrix < HIGH_VALUE).T
    correct_mask = full_distance_matrix[matched_det, matched_gt] < HIGH_VALUE
    matched_det, matched_gt = matched_det[correct_mask], matched_gt[correct_mask]
    sort_indices = np.argsort(matched_gt)
    matched_det, matched_gt = matched_det[sort_indices], matched_gt[sort_indices]
    return matched_det, matched_gt, intersection_mat, union_mat


