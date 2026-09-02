import pytlsd

import cv2
from pathlib import Path
from ..base_detector import (
    BaseDetector,
    DefaultDetectorOptions,
)


class LSDDetector(BaseDetector):
    def __init__(self, options=DefaultDetectorOptions):
        super().__init__(options)

    def get_module_name(self):
        return "lsd"

    def detect(self, image_path: Path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        segs = pytlsd.lsd(img)
        return segs
