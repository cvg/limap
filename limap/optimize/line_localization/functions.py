import numpy as np

import limap.base as base


def match_line_2to2_epipolarIoU(
    ref_lines, tgt_lines, ref_cam, ref_pose, tgt_cam, tgt_pose, IoU_threshold
):
    from limap.triangulation import compute_epipolar_IoU

    pairs = []
    if not isinstance(ref_pose, base.CameraPose):
        ref_pose = base.CameraPose(ref_pose[0], ref_pose[1])
    if not isinstance(tgt_pose, base.CameraPose):
        tgt_pose = base.CameraPose(tgt_pose[0], tgt_pose[1])
    ref_view = base.CameraView(ref_cam, ref_pose)
    tgt_view = base.CameraView(tgt_cam, tgt_pose)
    for ref_line_id, ref_line in enumerate(ref_lines):
        for tgt_line_id, tgt_line in enumerate(tgt_lines):
            res = compute_epipolar_IoU(ref_line, ref_view, tgt_line, tgt_view)
            if res > IoU_threshold:
                pairs.append((ref_line_id, tgt_line_id))
    return pairs


def filter_line_2to2_epipolarIoU(
    pairs,
    ref_lines,
    tgt_lines,
    ref_cam,
    ref_pose,
    tgt_cam,
    tgt_pose,
    IoU_threshold,
):
    from limap.triangulation import compute_epipolar_IoU

    if not isinstance(ref_pose, base.CameraPose):
        ref_pose = base.CameraPose(ref_pose[0], ref_pose[1])
    if not isinstance(tgt_pose, base.CameraPose):
        tgt_pose = base.CameraPose(tgt_pose[0], tgt_pose[1])
    ref_view = base.CameraView(ref_cam, ref_pose)
    tgt_view = base.CameraView(tgt_cam, tgt_pose)
    filtered = []
    for ref_line_id, tgt_line_id in pairs:
        res = compute_epipolar_IoU(
            ref_lines[ref_line_id], ref_view, tgt_lines[tgt_line_id], tgt_view
        )
        if res > IoU_threshold:
            filtered.append((ref_line_id, tgt_line_id))
    return filtered


def match_line_2to3(pairs_2to2, line2track, tgt_img_id):
    track_ids = line2track[tgt_img_id]
    pairs_2to3 = []
    for pair in pairs_2to2:
        ref_line_id, tgt_line_id = pair[:2]
        ref_line_id = int(ref_line_id)
        tgt_line_id = int(tgt_line_id)
        if track_ids[tgt_line_id] != -1:
            pairs_2to3.append((ref_line_id, track_ids[tgt_line_id]))
    return pairs_2to3


def midpoint_dist(ref_line, tgt_line):
    return np.linalg.norm(ref_line.midpoint() - tgt_line.midpoint())


def midpoint_perpendicular_dist(ref_line, tgt_line):
    m = ref_line.midpoint()
    t, dir = tgt_line.start, tgt_line.direction()

    dist = np.abs(np.cross(dir, t - m)) / ref_line.length()
    return dist


def perpendicular_dist(ref_line, tgt_line):
    s, e = ref_line.start, ref_line.end
    t, dir = tgt_line.start, tgt_line.direction()

    dist = 0.5 * (np.abs(np.cross(dir, t - s)) + np.abs(np.cross(dir, t - e)))
    return dist


def get_reprojection_dist_func(func_name):
    if func_name == "Perpendicular":
        return perpendicular_dist
    if func_name == "Midpoint":
        return midpoint_dist
    if func_name == "Midpoint_Perpendicular":
        return midpoint_perpendicular_dist
    raise ValueError(f"[Error] Unknown dist function: {func_name}")


def reprojection_filter_matches_2to3(
    ref_lines,
    ref_camview,
    all_pairs_2to3,
    linetracks,
    dist_thres=10,
    sine_thres=0.4,
    angle_scale=1.0,
    dist_func=midpoint_dist,
):
    matches = []
    for ref_line_id in all_pairs_2to3:
        ref_line = ref_lines[ref_line_id]
        # mp_ref = ref_line.midpoint()
        dir_ref = ref_line.direction()
        track_ids = np.unique(all_pairs_2to3[ref_line_id])

        min_loss = np.inf
        best_id = None
        for id in track_ids:
            l3d = linetracks[id].line
            l2d_start, l2d_end = (
                ref_camview.projection(l3d.start),
                ref_camview.projection(l3d.end),
            )
            l2d = base.Line2d(l2d_start, l2d_end)

            dist = dist_func(ref_line, l2d)
            loss_val = dist
            if dist_func == midpoint_dist:
                dir, length = l2d.direction(), l2d.length()
                cosine = np.clip(np.dot(dir_ref, dir), -1.0, 1.0)
                sine = np.sqrt(1 - cosine * cosine)
                loss_val += angle_scale * length * sine
                if sine > sine_thres:
                    continue
            if dist > dist_thres:
                continue
            if loss_val < min_loss:
                min_loss = loss_val
                best_id = id

        if best_id is not None:
            matches.append((ref_line_id, best_id))
    return matches
