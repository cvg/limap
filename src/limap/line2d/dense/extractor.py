import os

import numpy as np

import limap.util.io as limapio

from ..base_detector import BaseDetector, DefaultDetectorOptions


class DenseNaiveExtractor(BaseDetector):
    def __init__(self, options=DefaultDetectorOptions, device=None):
        super().__init__(options)

    def get_module_name(self):
        return "dense_naive"

    def get_descinfo_fname(self, descinfo_folder, img_id):
        fname = os.path.join(descinfo_folder, f"descinfo_{img_id}.npz")
        return fname

    def save_descinfo(self, descinfo_folder, img_id, descinfo):
        limapio.check_makedirs(descinfo_folder)
        fname = self.get_descinfo_fname(descinfo_folder, img_id)
        limapio.save_npz(fname, descinfo)

    def read_descinfo(self, descinfo_folder, img_id):
        fname = self.get_descinfo_fname(descinfo_folder, img_id)
        descinfo = limapio.read_npz(fname)
        return descinfo

    def extract(self, camview, segs):
        img = camview.read_image()
        lines = segs[:, :4].reshape(-1, 2, 2)
        scores = segs[:, -1] * np.sqrt(
            np.linalg.norm(segs[:, :2] - segs[:, 2:4], axis=1)
        )
        descinfo = {
            "image": img,
            "image_shape": img.shape,
            "lines": lines,
            "scores": scores,
        }
        return descinfo
