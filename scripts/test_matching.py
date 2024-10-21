import cv2

import limap.base
import limap.line2d
import limap.util.config

detector = limap.line2d.get_detector(
    {"method": "lsd", "skip_exists": False}
)  # get a line detector
extractor = limap.line2d.get_extractor(
    {"method": "dense_naive", "skip_exists": False}
)  # get a line extractor
matcher = limap.line2d.get_matcher(
    {"method": "dense_roma", "skip_exists": False, "n_jobs": 1, "topk": 0},
    extractor,
)  # initiate a line matcher

view1 = limap.base.CameraView(
    "/home/shaoliu/workspace/GlueStick/resources/img1.jpg"
)  # initiate an limap.base.CameraView instance for detection.
view2 = limap.base.CameraView(
    "/home/shaoliu/workspace/GlueStick/resources/img2.jpg"
)  # initiate an limap.base.CameraView instance for detection.

segs1 = detector.detect(view1)  # detection
desc1 = extractor.extract(view1, segs1)  # description
segs2 = detector.detect(view2)  # detection
desc2 = extractor.extract(view2, segs2)  # description
matches = matcher.match_pair(desc1, desc2)  # matching


def vis_detections(img, segs):
    import copy

    from limap.visualize.vis_utils import draw_segments

    img_draw = copy.deepcopy(img)
    img_draw = draw_segments(img_draw, segs, color=[0, 255, 0])
    return img_draw


def vis_matches(img1, img2, segs1, segs2, matches):
    import copy

    import cv2
    import numpy as np
    import seaborn as sns

    matched_seg1 = segs1[matches[:, 0]]
    matched_seg2 = segs2[matches[:, 1]]
    n_lines = matched_seg1.shape[0]
    colors = sns.color_palette("husl", n_colors=n_lines)
    img1_draw = copy.deepcopy(img1)
    img2_draw = copy.deepcopy(img2)
    for idx in range(n_lines):
        color = np.array(colors[idx]) * 255.0
        color = color.astype(int).tolist()
        cv2.line(
            img1_draw,
            (int(matched_seg1[idx, 0]), int(matched_seg1[idx, 1])),
            (int(matched_seg1[idx, 2]), int(matched_seg1[idx, 3])),
            color,
            4,
        )
        cv2.line(
            img2_draw,
            (int(matched_seg2[idx, 0]), int(matched_seg2[idx, 1])),
            (int(matched_seg2[idx, 2]), int(matched_seg2[idx, 3])),
            color,
            4,
        )
    return img1_draw, img2_draw


img1 = view1.read_image()
img2 = view2.read_image()
img1_det = vis_detections(img1, segs1)
cv2.imwrite("tmp/img1_det.png", img1_det)
img2_det = vis_detections(img2, segs2)
cv2.imwrite("tmp/img2_det.png", img2_det)
img1_draw, img2_draw = vis_matches(img1, img2, segs1, segs2, matches)
cv2.imwrite("tmp/img1_draw.png", img1_draw)
cv2.imwrite("tmp/img2_draw.png", img2_draw)
