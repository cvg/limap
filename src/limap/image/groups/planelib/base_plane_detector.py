import numpy as np
from pathlib import Path
from typeguard import typechecked
from dataclasses import dataclass


@dataclass
class BasePlaneDetectorOptions:
    min_num_pixels: int = 500
    weight_path: Path | None = None


DefaultBasePlaneDetectorOptions = BasePlaneDetectorOptions()


class BasePlaneDetector:
    def __init__(self, options: BasePlaneDetectorOptions):
        self.options = options

    # Module name needs to be set
    def get_module_name(self) -> str:
        """
        Virtual method (need to be implemented) - return the name of the module
        """
        raise NotImplementedError

    # The functions below are required for plane detectors
    @typechecked
    def _detect_plane_mask(self, image_name: Path) -> np.ndarray:
        """
        Virtual method (need to be implemented) - single-view plane segmentation
        """
        raise NotImplementedError

    @typechecked
    def detect_plane_mask(self, image_name: Path) -> np.ndarray:
        """
        Single-view plane segmentation. Filter and ensure continous labels.
        """
        plane_mask = self._detect_plane_mask(image_name)
        return self._filter_and_relabel(plane_mask)

    def _filter_and_relabel(
        self, plane_mask: np.ndarray, label_map: dict | None = None
    ) -> np.ndarray:
        """
        Filter small regions and ensure continuous labels.

        Args:
            plane_mask: (H, W) raw plane mask
            label_map: if provided, will be populated with
                old_label -> new_label

        Returns:
            Filtered plane mask with continuous labels starting from 1
        """
        # Unique labels (ignore 0 as background)
        labels = np.unique(plane_mask)
        labels = labels[labels > 0]

        region_sizes = []
        for lbl in labels:
            size = int(np.sum(plane_mask == lbl))
            region_sizes.append((lbl, size))
        region_sizes = [
            (lbl, size)
            for (lbl, size) in region_sizes
            if size >= self.options.min_num_pixels
        ]
        region_sizes.sort(key=lambda x: -x[1])

        # Fill in new mask with continuous labels
        new_plane_mask = np.zeros_like(plane_mask)
        next_label = 1  # start new plane labels from 1; background = 0

        for lbl, _ in region_sizes:
            region = plane_mask == lbl
            new_plane_mask[region] = next_label
            if label_map is not None:
                label_map[lbl] = next_label
            next_label += 1
        return new_plane_mask

    def detect_plane_mask_with_normals(
        self, image_name: Path
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Detect planes and optionally return normal map.

        Returns:
            plane_mask: (H, W) filtered plane mask with continuous labels
            normal_map: (H, W, 3) normal map or None if not supported
        """
        # Default implementation: no normal map support
        plane_mask = self.detect_plane_mask(image_name)
        return plane_mask, None
