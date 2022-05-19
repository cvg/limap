import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_detector import BaseDetector

import pytlsd
import numpy as np

class LSDDetector(BaseDetector):
    def __init__(self, set_gray=True, max_num_2d_segs=3000, n_jobs=1):
        super(LSDDetector, self).__init__(set_gray=set_gray, max_num_2d_segs=max_num_2d_segs, n_jobs=n_jobs)

    def get_module_name(self):
        return "lsd"

    def detect(self, output_folder, idx, camview):
        img = camview.read_image(set_gray=self.set_gray)
        segs = pytlsd.lsd(img)
        segs = self.take_longest_k(segs, max_num_2d_segs=self.max_num_2d_segs)
        self.save_segs(output_folder, idx, segs)

