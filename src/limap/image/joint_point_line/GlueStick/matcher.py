import numpy as np
import torch

from ...line.base_matcher import (
    BaseMatcher,
    DefaultMatcherOptions,
)
from .common import build_inputs, load_gluestick


class GlueStickMatcher(BaseMatcher):
    """The line half of GlueStick.

    Use :class:`~limap.image.joint_point_line.GlueStick.GlueStickJointMatcher`
    instead to keep the point matches the same forward pass produces.
    """

    def __init__(self, extractor, options=DefaultMatcherOptions, device=None):
        super().__init__(extractor, options)
        self.device = "cuda" if device is None else device
        self.gs = load_gluestick(self.weight_path, self.device)

    def get_module_name(self):
        return "gluestick"

    def check_compatibility(self, extractor):
        return extractor.get_module_name() == "wireframe"

    def match_pair(self, descinfo1, descinfo2):
        if self.topk == 0:
            return self.match_segs_with_descinfo(descinfo1, descinfo2)
        else:
            return self.match_segs_with_descinfo_topk(
                descinfo1, descinfo2, topk=self.topk
            )

    def match_segs_with_descinfo(self, descinfo1, descinfo2):
        inputs = build_inputs(descinfo1, descinfo2, self.device)
        with torch.no_grad():
            # Run the point-line matching
            out = self.gs(inputs)
            matches = out["line_matches0"].cpu().numpy()[0]

        # Transform matches to [n_matches, 2]
        id_list_1 = np.arange(0, matches.shape[0])[matches != -1]
        id_list_2 = matches[matches != -1]
        matches_t = np.stack([id_list_1, id_list_2], 1)
        return matches_t

    def match_segs_with_descinfo_topk(self, descinfo1, descinfo2, topk=10):
        inputs = build_inputs(descinfo1, descinfo2, self.device)
        with torch.no_grad():
            # Run the point matching
            scores = self.gs(inputs)["raw_line_scores"][0]

            # For each line in img1, retrieve the topk matches in img2
            matches = torch.argsort(scores, dim=1)[:, -topk:]
            matches = torch.flip(matches, dims=(1,))
            matches = matches.cpu().numpy()

        # Transform matches to [n_matches, 2]
        n_lines = matches.shape[0]
        topk = matches.shape[1]
        matches_t = np.stack(
            [np.arange(n_lines).repeat(topk), matches.flatten()], axis=1
        )
        return matches_t
