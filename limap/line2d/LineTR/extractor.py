import os

import cv2
import numpy as np
import torch

import limap.util.io as limapio
from limap.point2d.superpoint import SuperPoint

from ..base_detector import (
    BaseDetector,
    DefaultDetectorOptions,
)
from .line_transformer import LineTransformer


class LineTRExtractor(BaseDetector):
    def __init__(self, options=DefaultDetectorOptions, device=None):
        super().__init__(options)
        self.device = "cuda" if device is None else device
        self.sp = SuperPoint({}).eval().to(self.device)
        self.linetr = (
            LineTransformer({"weight_path": self.weight_path})
            .eval()
            .to(self.device)
        )

    def get_module_name(self):
        return "linetr"

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
        - the line descriptors with shape [num_lines, 256]
        - mat_klines2sublines: matrix assigning sublines to lines
        """
        if len(segs) == 0:
            return {
                "line_descriptors": np.empty((0, 256)),
                "mat_klines2sublines": np.empty((0, 0)),
            }

        self.linetr.config["min_length"] = max(16, max(img.shape) / 40)
        self.linetr.config["token_distance"] = max(8, max(img.shape) / 80)

        # Resize the image to a fixed size where LineTR works
        orig_h, orig_w = img.shape[:2]
        new_h, new_w = 480, 640
        s_h, s_w = new_h / orig_h, new_w / orig_w
        new_img = cv2.resize(img, (new_w, new_h))
        new_segs = segs.reshape(-1, 2, 2) * [s_w, s_h]

        # Run dense SuperPoint prediction
        torch_img = {
            "image": torch.tensor(
                new_img.astype(np.float32) / 255,
                dtype=torch.float,
                device=self.device,
            )[None, None]
        }
        with torch.no_grad():
            pred_sp = self.sp.compute_dense_descriptor_and_score(torch_img)

        # Preprocess the lines
        klines = self.linetr.preprocess(new_segs, (new_h, new_w), pred_sp)

        # Run the LineTR inference
        with torch.no_grad():
            line_desc = self.linetr(klines)["line_descriptors"]
            line_desc = line_desc.cpu().numpy()[0].T

        return {
            "line_descriptors": line_desc,
            "mat_klines2sublines": klines["mat_klines2sublines"][0]
            .cpu()
            .numpy(),
        }
