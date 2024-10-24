import os

import romatch
from PIL import Image

from .base import BaseDenseMatcher


class RoMa(BaseDenseMatcher):
    def __init__(self, mode="outdoor", device="cuda"):
        super(RoMa).__init__()
        self.output_res = 864
        self.mode = mode
        if mode == "outdoor":
            self.model = romatch.roma_outdoor(
                device=device, coarse_res=560, upsample_res=self.output_res
            )
        elif mode == "indoor":
            self.model = romatch.roma_indoor(
                device=device, coarse_res=560, upsample_res=self.output_res
            )
        elif mode == "tiny_outdoor":
            self.model = romatch.tiny_roma_v1_outdoor(device=device)

    def get_sample_thresh(self):
        return self.model.sample_thresh

    def get_warping_symmetric(self, img1, img2):
        warp, certainty = self.model.match(
            Image.fromarray(img1), Image.fromarray(img2), batched=False
        )
        if self.mode.startswith("tiny"):
            warp2_to_1, certainty2_to_1 = self.model.match(
                Image.fromarray(img2), Image.fromarray(img1), batched=False
            )
            return (
                warp[:, :, 2:],
                certainty,
                warp2_to_1[:, :, 2:],
                certainty2_to_1,
            )
        else:
            return (
                warp[:, : self.output_res, 2:],
                certainty[:, : self.output_res],
                warp[:, self.output_res :, :2],
                certainty[:, self.output_res :],
            )
