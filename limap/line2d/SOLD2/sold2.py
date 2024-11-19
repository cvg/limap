import copy
import os

import limap.util.io as limapio

from ..base_detector import (
    BaseDetector,
    DefaultDetectorOptions,
)
from ..base_matcher import (
    BaseMatcher,
    DefaultMatcherOptions,
)
from .sold2_wrapper import SOLD2LineDetector


class SOLD2Detector(BaseDetector):
    def __init__(self, options=DefaultDetectorOptions):
        super().__init__(options)
        self.detector = SOLD2LineDetector(weight_path=self.weight_path)

    def get_module_name(self):
        return "sold2"

    def get_descinfo_fname(self, descinfo_folder, img_id):
        fname = os.path.join(descinfo_folder, f"descinfo_{img_id}.npy")
        return fname

    def save_descinfo(self, descinfo_folder, img_id, descinfo):
        limapio.check_makedirs(descinfo_folder)
        fname = self.get_descinfo_fname(descinfo_folder, img_id)
        # special handling of None and same dimension
        if len(descinfo) == 2:
            descinfo[1] = descinfo[1][None, :]
        limapio.save_npy(fname, descinfo)

    def read_descinfo(self, descinfo_folder, img_id):
        fname = self.get_descinfo_fname(descinfo_folder, img_id)
        descinfo = limapio.read_npy(fname)
        # special handling of None and same dimension
        if len(descinfo) > 0:
            descinfo[1] = descinfo[1][0]
        return descinfo

    def detect(self, camview):
        img = camview.read_image(set_gray=self.set_gray)
        segs, descriptor, heatmap, descinfo = self.detector.detect(img)
        return segs

    def extract(self, camview, segs):
        img = camview.read_image(set_gray=self.set_gray)
        _, descriptor, _, _ = self.detector.detect(img)
        descinfo = self.detector.compute_descinfo(segs, descriptor)
        return descinfo

    def detect_and_extract(self, camview):
        img = camview.read_image(set_gray=self.set_gray)
        segs, descriptor, heatmap, descinfo = self.detector.detect(img)
        return segs, descinfo

    def sample_descinfo_by_indexes(self, descinfo, indexes):
        descinfo = copy.deepcopy(descinfo)
        descinfo[0] = (
            descinfo[0].reshape(128, -1, 5)[:, indexes, :].reshape(128, -1)
        )
        descinfo[1] = descinfo[1][indexes, :]
        return descinfo

    def get_heatmap_fname(self, folder, img_id):
        return os.path.join(folder, f"heatmap_{img_id}.npy")

    def extract_heatmap(self, camview):
        img = camview.read_image(set_gray=self.set_gray)
        segs, descriptor, heatmap, descinfo = self.detector.detect(img)
        return heatmap

    def extract_heatmaps_all_images(self, folder, imagecols, skip_exists=False):
        from tqdm import tqdm

        heatmap_folder = os.path.join(folder, "sold2_heatmaps")
        if not skip_exists:
            limapio.delete_folder(heatmap_folder)
        limapio.check_makedirs(heatmap_folder)
        if not (
            skip_exists
            and os.path.exists(os.path.join(heatmap_folder, "imagecols.npy"))
        ):
            limapio.save_npy(
                os.path.join(heatmap_folder, "imagecols.npy"),
                imagecols.as_dict(),
            )
        for img_id in tqdm(imagecols.get_img_ids()):
            heatmap_fname = self.get_heatmap_fname(heatmap_folder, img_id)
            if skip_exists and os.path.exists(heatmap_fname):
                continue
            heatmap = self.extract_heatmap(imagecols.camview(img_id))
            limapio.save_npy(heatmap_fname, heatmap)
        return heatmap_folder


class SOLD2Matcher(BaseMatcher):
    def __init__(self, extractor, options=DefaultMatcherOptions):
        super().__init__(extractor, options)
        assert self.extractor.get_module_name() == "sold2"
        self.detector = SOLD2LineDetector(weight_path=self.weight_path)

    def get_module_name(self):
        return "sold2"

    def match_pair(self, descinfo1, descinfo2):
        if self.topk == 0:
            return self.detector.match_segs_with_descinfo(descinfo1, descinfo2)
        else:
            return self.detector.match_segs_with_descinfo_topk(
                descinfo1, descinfo2, topk=self.topk
            )
