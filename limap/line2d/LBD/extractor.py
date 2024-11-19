import os

import cv2
import numpy as np
import pytlbd
import pytlsd

import limap.util.io as limapio

from ..base_detector import (
    BaseDetector,
    DefaultDetectorOptions,
)


def process_pyramid(
    img, detector, n_levels=5, level_scale=None, presmooth=True
):
    if level_scale is None:
        level_scale = np.sqrt(2)
    octave_img = img.copy()
    pre_sigma2 = 0
    cur_sigma2 = 1.0
    pyramid = []
    multiscale_segs = []
    for _ in range(n_levels):
        increase_sigma = np.sqrt(cur_sigma2 - pre_sigma2)
        blurred = cv2.GaussianBlur(
            octave_img, (5, 5), increase_sigma, borderType=cv2.BORDER_REPLICATE
        )
        pyramid.append(blurred)

        if presmooth:
            multiscale_segs.append(detector(blurred))
        else:
            multiscale_segs.append(detector(octave_img))

        # cv2.imshow(f"Mine L{i}", blurred)
        # down sample the current octave image to get the next octave image
        new_size = (
            int(octave_img.shape[1] / level_scale),
            int(octave_img.shape[0] / level_scale),
        )
        octave_img = cv2.resize(
            blurred, new_size, 0, 0, interpolation=cv2.INTER_NEAREST
        )
        pre_sigma2 = cur_sigma2
        cur_sigma2 = cur_sigma2 * 2

    return multiscale_segs, pyramid


def to_multiscale_lines(lines):
    ms_lines = []
    for line in lines.reshape(-1, 4):
        ll = np.append(line, [0, np.linalg.norm(line[:2] - line[2:4])])
        ms_lines.append(
            [(0, ll)] + [(i, ll / (i * np.sqrt(2))) for i in range(1, 5)]
        )
    return ms_lines


class LBDExtractor(BaseDetector):
    def __init__(self, options=DefaultDetectorOptions):
        super().__init__(options)

    def get_module_name(self):
        return "lbd"

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
        img = camview.read_image(set_gray=self.set_gray)
        descinfo = self.compute_descinfo(img, segs)
        return descinfo

    def compute_descinfo(self, img, segs):
        """A desc_info is composed of the following tuple / np arrays:
        - the multiscale lines [N, 5] containing tuples of (scale, scaled_line)
        - the line descriptors [N, dim]
        """
        ms_lines = to_multiscale_lines(segs)
        _, pyramid = process_pyramid(img, pytlsd.lsd, presmooth=False)
        descriptors = pytlbd.lbd_multiscale_pyr(pyramid, ms_lines, 9, 7)

        return {"ms_lines": ms_lines, "line_descriptors": descriptors}
