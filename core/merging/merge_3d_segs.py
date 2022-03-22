import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import angular_distance_3d, perpendicular_distance_3d, overlap_distance_3d, endpoints_distance_3d
import time

def merge_3d_segs(segs3d, visibility=None, plotter=None, length_th=0.03):
    # TODO: need to be optimized
    # Sort by length
    if not isinstance(segs3d, np.ndarray):
        segs3d = np.array(segs3d)
    if visibility is not None and not isinstance(visibility, np.ndarray):
        visibility = np.array(visibility)

    t0 = time.time()

    lengths = np.linalg.norm(segs3d[:, 0] - segs3d[:, 1], axis=-1)
    ord_indices = np.argsort(-lengths)
    segs3d, lengths = segs3d[ord_indices], lengths[ord_indices]
    segs3d = segs3d[lengths > length_th]
    if visibility is not None:
        visibility = visibility[ord_indices]
        visibility = visibility[lengths > length_th]

    # close_segs stores an adjacency matrix between segments
    close_segs = np.ones((len(segs3d), len(segs3d)), dtype=bool)

    # Remove segments with big angular distance
    angular_dist = angular_distance_3d(segs3d, segs3d)
    close_segs &= angular_dist < np.pi / 6

    # Remove segments with big perpendicular distance
    perp_dist = perpendicular_distance_3d(segs3d, segs3d)
    perp_dist = 0.5 * (perp_dist + perp_dist.T)
    close_segs = close_segs & (perp_dist < 0.02)  # in meters

    # Remove segments that do not overlap
    intersec, union, overlap = overlap_distance_3d(segs3d, segs3d)
    close_segs &= (overlap > 0.1)

    # Structural distance do not take into account the direction of the segment
    structural_dists = endpoints_distance_3d(segs3d, segs3d, structural=True)
    # Pure endpoints distance take it into account
    endpoint_dists = endpoints_distance_3d(segs3d, segs3d, structural=False)
    # If segments have both small structural_dists and endpoint_dists can be safely merged,
    # if have small structural distance but big endpoint distance it means that belong to
    # different structures (because in the image they have opposite gradients).
    close_segs = close_segs & ~((endpoint_dists > length_th) & (structural_dists < length_th))
    segs_to_merge = close_segs & (endpoint_dists < length_th)

    # If the overlap is bigger than the 50% Extend the endpoints taking the ones of the segment farr away
    extend_endpts = close_segs & (overlap > 0.5)
    # If the endpoints are very close compute its mean instead of extend them
    extend_endpts = extend_endpts & (endpoint_dists > length_th)

    t1 = time.time()
    print("[DEBUG] pairwise matrix computation time: {}s".format(t1 - t0))

    merged_3d_segs, merged_visibility = [], []
    while len(segs3d) > 0:
        # Refine the segment endpoints with the ones that have small endpoint distance
        selected_close_segs = np.argwhere(close_segs[0]).flatten()
        merge_indices = np.argwhere(segs_to_merge[0]).flatten()
        new_enpoints = segs3d[merge_indices].mean(axis=0)

        if extend_endpts[0].sum() > 1:
            middle_ptn = new_enpoints.mean(axis=0)
            dir_vec = new_enpoints[0] - middle_ptn
            dir_vec /= np.linalg.norm(dir_vec)
            extension_indices = np.argwhere(extend_endpts[0]).flatten()
            extension_indices = np.append(extension_indices, merge_indices)
            extension_segs = segs3d[extension_indices]
            projection = np.sum(dir_vec * (extension_segs - middle_ptn), axis=-1)
            new_enpoints = np.array([middle_ptn + dir_vec * projection.max(),
                                     middle_ptn + dir_vec * projection.min()])

        merged_3d_segs.append(new_enpoints)
        if visibility is not None:
            merged_visibility.append(np.unique(np.concatenate(visibility[selected_close_segs])))
            visibility = np.delete(visibility, selected_close_segs, axis=0)

        segs3d = np.delete(segs3d, selected_close_segs, axis=0)
        segs_to_merge = np.delete(np.delete(segs_to_merge, selected_close_segs, axis=0), selected_close_segs, axis=1)
        close_segs = np.delete(np.delete(close_segs, selected_close_segs, axis=0), selected_close_segs, axis=1)
        extend_endpts = np.delete(np.delete(extend_endpts, selected_close_segs, axis=0), selected_close_segs, axis=1)

    t2 = time.time()
    print("[DEBUG] sequential merging time: {}s".format(t2 - t1))

    if visibility is not None:
        return merged_3d_segs, merged_visibility
    return merged_3d_segs



