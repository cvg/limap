import cv2

import limap.base
import limap.line2d
import limap.util.config
import limap.visualize

# detect and describe lines
view1 = limap.base.CameraView(
    "runners/tests/data/line2d/frame.0000.color.jpg"
)  # initiate an limap.base.CameraView instance for detection.
view2 = limap.base.CameraView(
    limap.base.Camera("SIMPLE_PINHOLE", hw=(800, 800)),
    "runners/tests/data/line2d/frame.0000.color.jpg",
)
# You can specify the height and width to resize into
# in the limap.base.Camera instance at initialization.
detector = limap.line2d.get_detector(
    {"method": "deeplsd", "skip_exists": False}
)  # get a line detector
extractor = limap.line2d.get_extractor(
    {"method": "sold2", "skip_exists": False}
)  # get a line extractor
segs1 = detector.detect(view1)  # detection
desc1 = extractor.extract(view1, segs1)  # description
segs2 = detector.detect(view2)  # detection
desc2 = extractor.extract(view2, segs2)  # description

# visualize
img1 = view1.read_image(set_gray=False)
img1 = limap.visualize.draw_segments(img1, segs1, (0, 255, 0))
cv2.imshow("detections", img1)
cv2.waitKey(0)
img2 = view2.read_image(set_gray=False)
img2 = limap.visualize.draw_segments(img2, segs2, (0, 255, 0))
cv2.imshow("detections", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# match lines
matcher = limap.line2d.get_matcher(
    {"method": "sold2", "skip_exists": False, "n_jobs": 1, "topk": 0}, extractor
)  # set topk to 0 for mutual nearest neighbor matching
matches = matcher.match_pair(desc1, desc2)
print(matches.shape)  # (303, 2)
