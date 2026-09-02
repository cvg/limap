import romatch
from PIL import Image
from pycolmap import logging

from ..base_dense_matcher import (
    DenseMatchingResult,
    BiDenseMatchingResult,
    BaseDenseMatcher,
)


class RoMa(BaseDenseMatcher):
    def __init__(self, mode="outdoor", device="cuda"):
        super(RoMa).__init__()
        self.output_res = 864
        self.mode = mode
        self.device = device
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
        else:
            raise ValueError(f"Unknown mode for RoMa: {mode}")

    def get_warping(self, img1, img2) -> DenseMatchingResult:
        pil_img1 = Image.fromarray(img1)
        pil_img2 = Image.fromarray(img2)
        warp, certainty = self.model.match(pil_img1, pil_img2, batched=False)
        sample_thresh = self.model.sample_thresh
        if self.mode.startswith("tiny"):
            return DenseMatchingResult(
                device=self.device,
                source_shape=img1.shape[:2],
                target_shape=img2.shape[:2],
                warp=warp[:, :, 2:],
                certainty=certainty,
                sample_threshold=sample_thresh,
            )
        else:
            logging.warning(
                "Full RoMa model does not expose one-way matching. "
                "Please consider using tiny RoMa. "
            )
            return DenseMatchingResult(
                device=self.device,
                source_shape=img1.shape[:2],
                arget_shape=img2.shape[:2],
                warp=warp[:, : self.output_res, 2:],
                certainty=certainty[:, : self.output_res],
                sample_threshold=sample_thresh,
            )

    def get_warping_symmetric(self, img1, img2) -> BiDenseMatchingResult:
        pil_img1 = Image.fromarray(img1)
        pil_img2 = Image.fromarray(img2)
        warp, certainty = self.model.match(pil_img1, pil_img2, batched=False)
        sample_thresh = self.model.sample_thresh
        if self.mode.startswith("tiny"):
            warp2_to_1, certainty2_to_1 = self.model.match(
                pil_img2, pil_img1, batched=False
            )
            return BiDenseMatchingResult(
                match_1to2=DenseMatchingResult(
                    device=self.device,
                    source_shape=img1.shape[:2],
                    target_shape=img2.shape[:2],
                    warp=warp[:, :, 2:],
                    certainty=certainty,
                    sample_threshold=sample_thresh,
                ),
                match_2to1=DenseMatchingResult(
                    device=self.device,
                    source_shape=img2.shape[:2],
                    target_shape=img1.shape[:2],
                    warp=warp2_to_1[:, :, 2:],
                    certainty=certainty2_to_1,
                    sample_threshold=sample_thresh,
                ),
            )
        else:
            return BiDenseMatchingResult(
                match_1to2=DenseMatchingResult(
                    device=self.device,
                    source_shape=img1.shape[:2],
                    target_shape=img2.shape[:2],
                    warp=warp[:, : self.output_res, 2:],
                    certainty=certainty[:, : self.output_res],
                    sample_threshold=sample_thresh,
                ),
                match_2to1=DenseMatchingResult(
                    device=self.device,
                    source_shape=img2.shape[:2],
                    target_shape=img1.shape[:2],
                    warp=warp[:, self.output_res :, 2:],
                    certainty=certainty[:, self.output_res :],
                    sample_threshold=sample_thresh,
                ),
            )
