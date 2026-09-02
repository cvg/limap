import numpy as np
import pyprogressivex
import imagesize

from limap._limap._image._groups import _vplib

from ..base_vp_detector import BaseVPDetector, DefaultVPDetectorOptions


class ProgressiveX(BaseVPDetector):
    def __init__(self, options=DefaultVPDetectorOptions):
        super().__init__(options)
        self.options = options

    def get_module_name(self):
        return "progressive-x"

    def detect_vp(self, lines, image_path=None):
        if image_path is None:
            raise NotImplementedError
        width, height = imagesize.get(image_path)

        # Initialize
        labels = (np.ones(len(lines)) * -1).astype(int)
        flags = [line.length() >= self.options.min_length for line in lines]

        # Progressive-X inference
        lines = [
            line for line in lines if line.length() >= self.options.min_length
        ]
        lines_array = np.array([line.as_array().reshape(-1) for line in lines])
        weights_array = np.array([line.length() for line in lines])

        vanishing_points, labeling = pyprogressivex.findVanishingPoints(
            np.ascontiguousarray(lines_array),
            np.ascontiguousarray(weights_array),
            width,
            height,
            threshold=self.options.inlier_threshold,
            conf=0.99,
            spatial_coherence_weight=0.0,
            neighborhood_ball_radius=1.0,
            maximum_tanimoto_similarity=1.0,
            max_iters=1000,
            minimum_point_number=self.options.min_num_supports,
            maximum_model_number=-1,
            sampler_id=0,
            scoring_exponent=1.0,
            do_logging=False,
        )

        # Output
        labels[flags] = labeling - 1
        vps = vanishing_points.tolist()
        vpres = _vplib.VPResult(labels, vps)
        return vpres
