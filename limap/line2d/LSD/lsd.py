import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_detector import BaseDetector

import pytlsd
import numpy as np

class LSDDetector(BaseDetector):
    def __init__(self, set_gray=True, max_num_2d_segs=3000):
        super(LSDDetector, self).__init__(set_gray=set_gray, max_num_2d_segs=max_num_2d_segs)

    def get_module_name(self):
        return "lsd"

    def detect(self, camview):
        img = camview.read_image(set_gray=self.set_gray)
        segs = pytlsd.lsd(img)
        return segs

