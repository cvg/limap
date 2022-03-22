import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from constants import ETH_EPS
from line_utils import start_endpoints, end_endpoints

def angular_distance(segs1, segs2):
    # Compute direction vector of segs1
    dirs1 = np.array([segs1[:, 2] - segs1[:, 0], segs1[:, 3] - segs1[:, 1]])
    dirs1 /= (np.linalg.norm(dirs1, axis=0) + ETH_EPS)
    # Compute direction vector of segs2
    dirs2 = np.array([segs2[:, 2] - segs2[:, 0], segs2[:, 3] - segs2[:, 1]])
    dirs2 /= (np.linalg.norm(dirs2, axis=0) + ETH_EPS)
    # https://en.wikipedia.org/wiki/Cosine_similarity
    return np.arccos(np.minimum(1, np.abs(np.einsum('ij,ik->jk', dirs1, dirs2))))


def individual_angular_distance(segs1, segs2):
    # Compute direction vector of segs1
    dirs1 = np.array([segs1[:, 2] - segs1[:, 0], segs1[:, 3] - segs1[:, 1]])
    dirs1 /= (np.linalg.norm(dirs1, axis=0) + ETH_EPS)
    # Compute direction vector of segs2
    dirs2 = np.array([segs2[:, 2] - segs2[:, 0], segs2[:, 3] - segs2[:, 1]])
    dirs2 /= (np.linalg.norm(dirs2, axis=0) + ETH_EPS)
    return np.arccos(np.minimum(1, np.abs(np.einsum('ij,ij->j', dirs1, dirs2))))


def angular_distance_3d(segs1, segs2):
    # Compute direction vector of segs1
    dirs1 = segs1[:, 1] - segs1[:, 0]
    dirs1 /= (np.linalg.norm(dirs1, axis=1)[:, np.newaxis] + ETH_EPS)
    # Compute direction vector of segs2
    dirs2 = segs2[:, 1] - segs2[:, 0]
    dirs2 /= (np.linalg.norm(dirs2, axis=1)[:, np.newaxis] + ETH_EPS)
    return np.arccos(np.minimum(1, np.abs(np.einsum('ij,kj->ik', dirs1, dirs2))))


def individual_3d_angular_distance(segs1, segs2):
    # Compute direction vector of segs1
    dirs1 = segs1[:, 1] - segs1[:, 0]
    dirs1 /= (np.linalg.norm(dirs1, axis=1)[:, np.newaxis] + ETH_EPS)
    # Compute direction vector of segs2
    dirs2 = segs2[:, 1] - segs2[:, 0]
    dirs2 /= (np.linalg.norm(dirs2, axis=1)[:, np.newaxis] + ETH_EPS)
    return np.arccos(np.minimum(1, np.abs(np.einsum('ij,ij->i', dirs1, dirs2))))


