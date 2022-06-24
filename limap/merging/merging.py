from _limap import _merging as _mrg
from _limap import _base
import numpy as np

def get_all_lines_3d(seg3d_list):
    all_lines_3d = {}
    for img_id, segs3d in seg3d_list.items():
        all_lines_3d[img_id] = _mrg._GetLines(segs3d)
    return all_lines_3d

def merging(linker, all_2d_segs, imagecols, seg3d_list, neighbors, var2d=5.0):
    all_lines_2d = _base._GetAllLines2D(all_2d_segs)
    all_lines_3d = {}
    all_lines_3d_with_uncertainty = {}
    for img_id, segs3d in seg3d_list.items():
        all_lines_3d[img_id] = _mrg._GetLines(segs3d)
        all_lines_3d_with_uncertainty[img_id] = _mrg._SetUncertaintySegs3d(_mrg._GetLines(segs3d), imagecols.camview(img_id), var2d)
    graph = _base.Graph()
    linetracks = _mrg._MergeToLineTracks(graph, all_lines_2d, imagecols, all_lines_3d_with_uncertainty, neighbors, linker)
    return graph, linetracks

def remerge(linker3d, linetracks, num_outliers=2):
    if len(linetracks) == 0:
        return linetracks
    new_linetracks = linetracks
    num_tracks = len(new_linetracks)
    # iterative remerging
    while True:
        new_linetracks = _mrg._RemergeLineTracks(new_linetracks, linker3d, num_outliers=num_outliers)
        num_tracks_new = len(new_linetracks)
        if num_tracks == num_tracks_new:
            break
        num_tracks = num_tracks_new
    return new_linetracks

def checktrackbyreprojection(track, imagecols, th_angular2d, th_perp2d):
    results = _mrg._CheckReprojection(track, imagecols, th_angular2d, th_perp2d)
    return results

def filtertracksbyreprojection(linetracks, imagecols, th_angular2d, th_perp2d, num_outliers=2):
    new_linetracks = _mrg._FilterSupportLines(linetracks, imagecols, th_angular2d, th_perp2d, num_outliers=num_outliers)
    return new_linetracks

def checksensitivity(linetracks, imagecols, th_angular3d):
    results = _mrg._CheckSensitivity(linetracks, imagecols, th_angular3d)
    return results

def filtertracksbysensitivity(linetracks, imagecols, th_angular3d, min_num_supports):
    new_linetracks = _mrg._FilterTracksBySensitivity(linetracks, imagecols, th_angular3d, min_num_supports)
    return new_linetracks

def filtertracksbyoverlap(linetracks, imagecols, th_overlap, min_num_supports):
    new_linetracks = _mrg._FilterTracksByOverlap(linetracks, imagecols, th_overlap, min_num_supports)
    return new_linetracks

