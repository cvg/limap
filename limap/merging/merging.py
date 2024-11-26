from _limap import _base
from _limap import _merging as _mrg
from pycolmap import logging


def merging(linker, all_2d_segs, imagecols, seg3d_list, neighbors, var2d=5.0):
    all_lines_2d, all_lines_3d = {}, {}
    for img_id in imagecols.get_img_ids():
        all_lines_2d[img_id] = _base._GetLine2dVectorFromArray(
            all_2d_segs[img_id]
        )
        all_lines_3d[img_id] = _mrg._SetUncertaintySegs3d(
            _base._GetLine3dVectorFromArray(seg3d_list[img_id]),
            imagecols.camview(img_id),
            var2d,
        )
    graph = _base.Graph()
    linetracks = _mrg._MergeToLineTracks(
        graph, all_lines_2d, imagecols, all_lines_3d, neighbors, linker
    )
    return graph, linetracks


def remerge(linker3d, linetracks, num_outliers=2):
    if len(linetracks) == 0:
        return linetracks
    new_linetracks = linetracks
    num_tracks = len(new_linetracks)
    # iterative remerging
    while True:
        new_linetracks = _mrg._RemergeLineTracks(
            new_linetracks, linker3d, num_outliers=num_outliers
        )
        num_tracks_new = len(new_linetracks)
        if num_tracks == num_tracks_new:
            break
        num_tracks = num_tracks_new
    logging.info(
        f"[LOG] tracks after iterative remerging:"
        f" {len(new_linetracks)} / {len(linetracks)}"
    )
    return new_linetracks


def check_track_by_reprojection(track, imagecols, th_angular2d, th_perp2d):
    results = _mrg._CheckReprojection(track, imagecols, th_angular2d, th_perp2d)
    return results


def filter_tracks_by_reprojection(
    linetracks, imagecols, th_angular2d, th_perp2d, num_outliers=2
):
    new_linetracks = _mrg._FilterSupportLines(
        linetracks,
        imagecols,
        th_angular2d,
        th_perp2d,
        num_outliers=num_outliers,
    )
    return new_linetracks


def check_sensitivity(linetracks, imagecols, th_angular3d):
    results = _mrg._CheckSensitivity(linetracks, imagecols, th_angular3d)
    return results


def filter_tracks_by_sensitivity(
    linetracks, imagecols, th_angular3d, min_num_supports
):
    new_linetracks = _mrg._FilterTracksBySensitivity(
        linetracks, imagecols, th_angular3d, min_num_supports
    )
    return new_linetracks


def filter_tracks_by_overlap(
    linetracks, imagecols, th_overlap, min_num_supports
):
    new_linetracks = _mrg._FilterTracksByOverlap(
        linetracks, imagecols, th_overlap, min_num_supports
    )
    return new_linetracks
