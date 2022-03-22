import numpy as np
import cv2
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constants import ETH_EPS
from geometry import *

def process_pyramid(img, detector, n_levels=5, level_scale=np.sqrt(2), presmooth=True):
    octave_img = img.copy()
    pre_sigma2 = 0
    cur_sigma2 = 1.0
    pyramid = []
    multiscale_segs = []
    for i in range(n_levels):
        increase_sigma = np.sqrt(cur_sigma2 - pre_sigma2)
        blurred = cv2.GaussianBlur(octave_img, (5, 5), increase_sigma, borderType=cv2.BORDER_REPLICATE)
        pyramid.append(blurred)

        if presmooth:
            multiscale_segs.append(detector(blurred))
        else:
            multiscale_segs.append(detector(octave_img))

        new_size = (int(octave_img.shape[1] / level_scale), int(octave_img.shape[0] / level_scale))
        octave_img = cv2.resize(blurred, new_size, 0, 0, interpolation=cv2.INTER_NEAREST)
        pre_sigma2 = cur_sigma2
        cur_sigma2 = cur_sigma2 * 2
    return multiscale_segs, pyramid

def start_endpoints(segs):
    return segs[:, 0:2]

def end_endpoints(segs):
    return segs[:, 2:4]

def seg_equation(segs):
    # calculate list of start, end and midpoints points from both lists
    start_points1, end_points1 = to_homogeneous(start_endpoints(segs)), to_homogeneous(end_endpoints(segs))
    # Compute the line equations as ax + by + c = 0 , where x^2 + y^2 = 1
    lines1 = np.cross(start_points1, end_points1)
    lines1 = lines1 / (np.sqrt(lines1[:, 0] ** 2 + lines1[:, 1] ** 2)[:, np.newaxis] + ETH_EPS)
    return lines1

def is_list_of_lists(the_list):
    """ Checks if the object is a empty list os a list containing lists."""
    if not isinstance(the_list, list):
        return False
    if len(the_list) == 0:
        return True
    return isinstance(the_list[0], list)

def are_segs_inside_img(segs, img_shape):
    s, e = start_endpoints(segs), end_endpoints(segs)
    is_s_inside = (s[:, 0] >= 0) & (s[:, 0] < img_shape[1]) & (s[:, 1] >= 0) & (s[:, 1] < img_shape[0])
    is_e_inside = (e[:, 0] >= 0) & (e[:, 0] < img_shape[1]) & (e[:, 1] >= 0) & (e[:, 1] < img_shape[0])
    return np.logical_or(is_s_inside, is_e_inside)

def transform_segments(segs, H):
    projected_start = to_cartesian(to_homogeneous(start_endpoints(segs)) @ H.T)
    projected_end = to_cartesian(to_homogeneous(end_endpoints(segs)) @ H.T)
    return np.hstack([projected_start, projected_end])

def shrink_to_ref_image(segs, ref_shape, ref_to_query=np.eye(3)):
    """Having the reference image shape ref_shape, the method shrink the endpoints of segs,
     detected in other image to fit inside the reference image. The image where segs were detected
     and the reference image are related by ref_to_query homography."""
    w, h = ref_shape[1], ref_shape[0]
    # Project the segments to the reference image
    segs_ref = transform_segments(segs, np.linalg.inv(ref_to_query))
    eqs = seg_equation(segs_ref)
    x0, y0 = np.array([1, 0, 0]), np.array([0, 1, 0])

    pt_x0s = np.cross(eqs, x0)
    pt_x0s = pt_x0s[:, :-1] / (ETH_EPS + pt_x0s[:, np.newaxis, -1])
    pt_y0s = np.cross(eqs, y0)
    pt_y0s = pt_y0s[:, :-1] / (ETH_EPS + pt_y0s[:, np.newaxis, -1])

    xW, yH = np.array([1, 0, -w]), np.array([0, 1, -h])
    pt_xWs = np.cross(eqs, xW)
    pt_xWs = pt_xWs[:, :-1] / (ETH_EPS + pt_xWs[:, np.newaxis, -1])
    pt_yHs = np.cross(eqs, yH)
    pt_yHs = pt_yHs[:, :-1] / (ETH_EPS + pt_yHs[:, np.newaxis, -1])

    # If the X coordinate of the first keypoint is out
    mask = segs_ref[:, 0] < 0
    segs_ref[mask, :2] = pt_x0s[mask]
    mask = segs_ref[:, 0] > (w - 1)
    segs_ref[mask, :2] = pt_xWs[mask]
    # If the X coordinate of the second keypoint is out
    mask = segs_ref[:, 2] < 0
    segs_ref[mask, 2:] = pt_x0s[mask]
    mask = segs_ref[:, 2] > (w - 1)
    segs_ref[mask, 2:] = pt_xWs[mask]
    # If the Y coordinate of the first keypoint is out
    mask = segs_ref[:, 1] < 0
    segs_ref[mask, :2] = pt_y0s[mask]
    mask = segs_ref[:, 1] > (h - 1)
    segs_ref[mask, :2] = pt_yHs[mask]
    # If the Y coordinate of the second keypoint is out
    mask = segs_ref[:, 3] < 0
    segs_ref[mask, 2:] = pt_y0s[mask]
    mask = segs_ref[:, 3] > (h - 1)
    segs_ref[mask, 2:] = pt_yHs[mask]
    # Back-project the segments to the query image
    reproj_seg = transform_segments(segs_ref, ref_to_query)
    return reproj_seg


