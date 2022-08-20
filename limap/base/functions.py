import _limap._base as _base

def get_all_lines_2d(all_2d_segs):
    all_lines_2d = {}
    for img_id in all_2d_segs:
        all_lines_2d[img_id] = _base._GetLine2dVectorFromArray(all_2d_segs[img_id])
    return all_lines_2d

def get_all_lines_3d(seg3d_list):
    all_lines_3d = {}
    for img_id, segs3d in seg3d_list.items():
        all_lines_3d[img_id] = _base._GetLine3dVectorFromArray(segs3d)
    return all_lines_3d

def get_invert_idmap_from_linetracks(all_lines_2d, linetracks):
    map = {}
    for img_id in all_lines_2d:
        lines_2d = all_lines_2d[img_id]
        map[img_id] = [-1] * len(lines_2d)
    for track_id, track in enumerate(linetracks):
        for img_id, line_id in zip(track.image_id_list, track.line_id_list):
            map[img_id][line_id] = track_id
    return map

