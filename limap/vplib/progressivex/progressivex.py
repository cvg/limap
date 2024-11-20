from collections import namedtuple

import numpy as np
import pyprogressivex
from _limap import _vplib

from ..base_vp_detector import BaseVPDetector, DefaultVPDetectorOptions

ProgressiveXOptions = namedtuple(
    "ProgressiveXOptions",
    ["min_length", "inlier_threshold"],
    defaults=[20, 1.0],
)


class ProgressiveX(BaseVPDetector):
    def __init__(self, cfg, options=DefaultVPDetectorOptions):
        super().__init__(options)
        self.options = ProgressiveXOptions()
        for fld in self.options._fields:
            if fld in cfg:
                self.options = self.options._replace(fld=cfg[fld])

    def get_module_name(self):
        return "progressive-x"

    def detect_vp(self, lines, camview=None):
        if camview is None:
            raise NotImplementedError

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
            camview.w(),
            camview.h(),
            threshold=self.options.inlier_threshold,
            conf=0.99,
            spatial_coherence_weight=0.0,
            neighborhood_ball_radius=1.0,
            maximum_tanimoto_similarity=1.0,
            max_iters=1000,
            minimum_point_number=5,
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
