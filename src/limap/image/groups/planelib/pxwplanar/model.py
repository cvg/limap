"""
Plane detector on top of the MoGe 4-head planarity model, with inference
and region growing from pxwplanar (github.com/alpayozkan/PixelwisePlanarity).
"""

import cv2
import numpy as np
from pathlib import Path

from pxwplanar.inference.planarity.moge_inference import MoGePlanarityInference
from pxwplanar.shared.segmentation import (
    compute_planar_segments,
    remove_small_components,
)

from ..base_plane_detector import (
    BasePlaneDetector,
    BasePlaneDetectorOptions,
    DefaultBasePlaneDetectorOptions,
)
from ..register_plane_detector import PxwPlanarOptions


class PxwPlanar(BasePlaneDetector):
    def __init__(
        self,
        options: BasePlaneDetectorOptions = DefaultBasePlaneDetectorOptions,
        pxw_options: PxwPlanarOptions | None = None,
    ):
        super().__init__(options)
        self.pxw_options = pxw_options or PxwPlanarOptions()

        self.model = MoGePlanarityInference.from_pretrained(
            self.pxw_options.model_path, device=self.pxw_options.device
        )
        # Use PyTorch native SDPA for attention (matching pxwplanar)
        self.model.model.encoder.use_memory_efficient_attention = False
        self.model.model.encoder.enable_pytorch_native_sdpa()

    def get_module_name(self):
        return "pxwplanar"

    def _detect_plane_mask_impl(self, image_name: Path):
        """
        Internal implementation that returns plane mask and all MoGe outputs.

        Returns:
            filtered_segmentation: (H, W) plane mask with label 0 = background
            normal_map: (H, W, 3) normal vectors in camera frame
            depth: (H, W) metric depth map in meters
            planarity: (H, W) planarity probability in [0, 1]
        """
        opts = self.pxw_options

        image = cv2.imread(str(image_name))
        if image is None:
            raise OSError(f"Failed to read image: {image_name}")
        img_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = img_np.shape[:2]

        # metric depth in meters, at the model resolution
        res = self.model.predict_metric(
            img_np, num_tokens=opts.num_tokens, return_all_heads=True
        )

        depth = res["depth"]
        normal = res["normal"]  # (H, W, 3)
        planarity = res["planarity_probability"]

        depth = cv2.resize(
            depth.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR
        )
        normal = np.ascontiguousarray(
            cv2.resize(
                normal.astype(np.float32),
                (W, H),
                interpolation=cv2.INTER_LINEAR,
            )
        )
        planarity = cv2.resize(
            planarity.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR
        )
        planarity_mask = (planarity > opts.threshold_planarity).astype(np.int16)

        assert (
            img_np.shape[:2]
            == depth.shape[:2]
            == normal.shape[:2]
            == planarity.shape[:2]
        ), "All inputs must have the same resolution"

        labels, _ = compute_planar_segments(
            planarity_mask,
            normal,
            depth,
            np.deg2rad(opts.normal_threshold_deg),
            opts.depth_threshold,
            neighbor_match_count_thresh=opts.neighbor_match_count_thresh,
            device=opts.device,
        )
        filtered_segmentation = remove_small_components(
            labels, min_size=self.options.min_num_pixels
        )
        return filtered_segmentation, normal, depth, planarity

    def _detect_plane_mask(self, image_name: Path):
        filtered_segmentation, _, _, _ = self._detect_plane_mask_impl(
            image_name
        )
        return filtered_segmentation

    def detect_plane_mask_with_normals(self, image_name: Path):
        """
        Detect planar regions and return plane mask with all MoGe outputs.

        Returns:
            plane_mask: (H, W) int array with filtered plane labels
                (0 = background)
            normal_map: (H, W, 3) float32 array with per-pixel normal vectors
            depth_map: (H, W) float32 metric depth map in meters
            planarity_mask: (H, W) uint8 binary mask (0 or 255)
        """
        raw_mask, normal_map, depth_map, planarity = (
            self._detect_plane_mask_impl(image_name)
        )
        filtered_mask = self._filter_and_relabel(raw_mask)
        planarity_mask = (
            planarity > self.pxw_options.threshold_planarity
        ).astype(np.uint8) * 255
        return filtered_mask, normal_map, depth_map, planarity_mask
