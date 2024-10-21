import os

import romatch
from PIL import Image

from .base import BaseDenseMatcher


class RoMa(BaseDenseMatcher):
    def __init__(self, mode="outdoor", device="cuda"):
        super(RoMa).__init__()
        if mode == "outdoor":
            self.model = romatch.roma_outdoor(device=device, coarse_res=560)
        elif mode == "indoor":
            self.model = romatch.roma_indoor(device=device, coarse_res=560)
        elif mode == "tiny_outdoor":
            self.model = romatch.tiny_roma_v1_outdoor(device=device)

    def get_sample_thresh(self):
        return self.model.sample_thresh

    def get_warpping_symmetric(self, img1, img2):
        warp, certainty = self.model.match(
            Image.fromarray(img1), Image.fromarray(img2)
        )
        N = 864
        return (
            warp[:, :N, 2:],
            certainty[:, :N],
            warp[:, N:, :2],
            certainty[:, N:],
        )
