import numpy as np
import joblib
from tqdm import tqdm

from collections import namedtuple
BaseVPDetectorOptions = namedtuple("BaseVPDetectorOptions",
                                   ["n_jobs"],
                                   defaults = [1])

class BaseVPDetector():
    def __init__(self, options = BaseVPDetectorOptions()):
        self.n_jobs = options.n_jobs

    # Module name needs to be set
    def get_module_name(self):
        raise NotImplementedError
    # The functions below are required for VP detectors
    def detect_vp(self, lines, camview=None):
        '''
        Input:
        - lines     type: std::vector<limap.base.Line2d>
        Output:
        - vpresult  type: limap.vplib.VPResult
        '''
        raise NotImplementedError

    def detect_vp_all_images(self, all_lines, camviews=None):
        def process(self, lines):
            return self.detect_vp(lines)
        def process_camview(self, lines, camview):
            return self.detect_vp(lines, camview)
        if camviews is None:
            vpresults_vector = joblib.Parallel(self.n_jobs)(joblib.delayed(process)(self, lines) for (img_id, lines) in tqdm(all_lines.items()))
        else:
            vpresults_vector = joblib.Parallel(self.n_jobs)(joblib.delayed(process_camview)(self, lines, camviews[img_id]) for (img_id, lines) in tqdm(all_lines.items()))
        # map vector back to map
        vpresults = dict()
        for idx, img_id in enumerate(list(all_lines.keys())):
            vpresults[img_id] = vpresults_vector[idx]
        return vpresults

    def visualize(self, fname, img, lines, vpresult, show_original=False, endpoints=False):
        import cv2
        import limap.visualize as limapvis
        img = limapvis.vis_vpresult(img, lines, vpresult, show_original=show_original, endpoints=endpoints)
        cv2.imwrite(fname, img)

