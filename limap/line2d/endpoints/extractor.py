import os

import numpy as np
import torch

import limap.util.io as limapio
from limap.point2d.superpoint import SuperPoint

from ..base_detector import (
    BaseDetector,
    DefaultDetectorOptions,
)


class SuperPointEndpointsExtractor(BaseDetector):
    def __init__(self, options=DefaultDetectorOptions, device=None):
        super().__init__(options)
        self.device = "cuda" if device is None else device
        self.sp = (
            SuperPoint({"weight_path": self.weight_path}).eval().to(self.device)
        )

    def get_module_name(self):
        return "superpoint_endpoints"

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
        - the original image shape (h, w)
        - the 2D endpoints of the lines in shape [N*2, 2] (xy convention)
        - the line score of shape [N] (NFA * sqrt(line_length))
        - the descriptor of each endpoints of shape [256, N*2]
        """
        if len(segs) == 0:
            return {
                "image_shape": img.shape,
                "lines": np.array([]),
                "lines_score": np.zeros((0,)),
                "endpoints_desc": np.zeros((256, 0)),
            }
        lines = segs[:, :4].reshape(-1, 2)
        scores = segs[:, -1] * np.sqrt(
            np.linalg.norm(segs[:, :2] - segs[:, 2:4], axis=1)
        )
        scores /= np.amax(scores) + 1e-8
        torch_img = {
            "image": torch.tensor(
                img.astype(np.float32) / 255,
                dtype=torch.float,
                device=self.device,
            )[None, None]
        }
        torch_endpoints = torch.tensor(
            lines.reshape(1, -1, 2), dtype=torch.float, device=self.device
        )
        with torch.no_grad():
            endpoint_descs = (
                self.sp.sample_descriptors(torch_img, torch_endpoints)[
                    "descriptors"
                ][0]
                .cpu()
                .numpy()
            )
        return {
            "image_shape": img.shape,
            "lines": lines,
            "lines_score": scores,
            "endpoints_desc": endpoint_descs,
        }
