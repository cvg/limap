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

