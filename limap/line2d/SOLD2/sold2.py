import os, sys
from .sold2_wrapper import SOLD2LineDetector
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_detector import BaseDetector
from base_matcher import BaseMatcher

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import limap.util.io_utils as limapio

class SOLD2Detector(BaseDetector):
    def __init__(self, set_gray=True, max_num_2d_segs=3000, n_jobs=1):
        super(SOLD2Detector, self).__init__(set_gray=set_gray, max_num_2d_segs=max_num_2d_segs, n_jobs=n_jobs)
        self.detector = SOLD2LineDetector()

    def get_module_name(self):
        return "sold2"

    def save_descinfo(self, descinfo_folder, idx, descinfo):
        if not os.path.exists(descinfo_folder):
            os.makedirs(descinfo_folder)
        fname = os.path.join(descinfo_folder, "descinfo_{0}.npy".format(idx))
        # special handling of None and same dimension
        if len(descinfo) == 2:
            descinfo[1] = descinfo[1][None,:]
        limapio.save_npy(fname, descinfo)

    def read_descinfo(self, descinfo_folder, idx):
        fname = os.path.join(descinfo_folder, "descinfo_{0}.npy".format(idx))
        descinfo = limapio.read_npy(fname)
        descinfo[1] = descinfo[1][0]
        return descinfo

    def detect(self, output_folder, idx, camview):
        img = camview.read_image(set_gray=self.set_gray)
        segs, descriptor, heatmap, descinfo = self.detector.detect(img)
        segs = self.take_longest_k(segs, max_num_2d_segs=self.max_num_2d_segs)
        self.save_segs(output_folder, idx, segs)

    def extract(self, output_folder, idx, camview, segs):
        img = camview.read_image(set_gray=self.set_gray)
        _, descriptor, _, _ = self.detector.detect(img)
        descinfo = self.compute_descinfo(segs, descriptor)
        descinfo_folder = self.get_descinfo_folder(output_folder)
        self.save_descinfo(descinfo_folder, idx, descinfo)

    def detect_and_extract(self, output_folder, idx, camview):
        img = camview.read_image(set_gray=self.set_gray)
        segs, descriptor, heatmap, descinfo = self.detector.detect(img)
        segs = self.take_longest_k(segs, max_num_2d_segs=self.max_num_2d_segs)
        self.save_segs(output_folder, idx, segs)
        descinfo_folder = self.get_descinfo_folder(output_folder)
        self.save_descinfo(descinfo_folder, idx, descinfo)

class SOLD2Matcher(BaseMatcher):
    def __init__(self, extractor, n_neighbors=20, topk=10, n_jobs=1):
        super(SOLD2Matcher, self).__init__(extractor, n_neighbors=n_neighbors, topk=topk, n_jobs=n_jobs)
        self.detector = SOLD2LineDetector()

    def get_module_name(self):
        return "sold2_matcher"

    def match_pair(self, descinfo1, descinfo2):
        if self.topk == 0:
            return self.detector.match_segs_with_descinfo(descinfo1, descinfo2)
        else:
            return self.detector.match_segs_with_descinfo_topk(descinfo1, descinfo2, topk=self.topk)


