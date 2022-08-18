import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_detector import BaseDetector, BaseDetectorOptions

import pytlsd
import numpy as np

class LSDDetector(BaseDetector):
    def __init__(self, options = BaseDetectorOptions()):
        super(LSDDetector, self).__init__(options)

    def get_module_name(self):
        return "lsd"

    def detect(self, camview):
        img = camview.read_image(set_gray=self.set_gray)
        segs = pytlsd.lsd(img)
        return segs

