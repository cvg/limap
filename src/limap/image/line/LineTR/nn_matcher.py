import numpy as np


def nn_matcher_distmat(dist_mat, nn_thresh, is_mutual_NN=True):
    """Nearest Neighbor Matching using a distance matrix"""
    n0 = dist_mat.shape[1]
    n1 = dist_mat.shape[2]
    b = 1
    mat_nn_ = np.zeros((b, n0, n1))
    if n0 == 0 or n1 == 0:
        return mat_nn_

    for b_idx in np.arange(b):
        dmat_tmp = dist_mat[b_idx].clip(min=0)
        # Get NN indices and scores.
        idx = np.argmin(dmat_tmp, axis=1)
        scores = dmat_tmp[np.arange(dmat_tmp.shape[0]), idx]
        # Threshold the NN matches.
        keep = scores < nn_thresh
        if is_mutual_NN:
            # Check if nearest neighbor goes both directions and keep those.
            idx2 = np.argmin(dmat_tmp, axis=0)
            keep_bi = np.arange(len(idx)) == idx2[idx]
            keep = np.logical_and(keep, keep_bi)
        idx = idx[keep]
        scores = scores[keep]
        # Get the surviving point indices.
        m_idx1 = np.arange(n0)[keep]
        m_idx2 = idx
        mat_nn_[b_idx, m_idx1, m_idx2] = 1

    return mat_nn_


def nn_matcher(desc0, desc1, nn_thresh=0.8, is_mutual_NN=True):
    """Nearest Neighbor Matching using two descriptors"""
    d, num0 = desc0.shape
    desc0_, desc1_ = desc0.T, desc1.T

    dmat = desc0_ @ desc1_.T
    dist_mat = (2.0 - 2.0 * dmat).clip(min=0)[None]

    mat_nn = nn_matcher_distmat(dist_mat, nn_thresh, is_mutual_NN)
    return mat_nn, dist_mat
