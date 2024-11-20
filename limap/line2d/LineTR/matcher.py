import numpy as np

from ..base_matcher import (
    BaseMatcher,
    DefaultMatcherOptions,
)
from .line_process import get_dist_matrix
from .line_transformer import LineTransformer
from .nn_matcher import nn_matcher_distmat


class LineTRMatcher(BaseMatcher):
    def __init__(
        self, extractor, options=DefaultMatcherOptions, topk=0, device=None
    ):
        super().__init__(extractor, options)
        self.device = "cuda" if device is None else device
        self.linetr = (
            LineTransformer({"weight_path": self.weight_path})
            .eval()
            .to(self.device)
        )

    def get_module_name(self):
        return "linetr"

    def check_compatibility(self, extractor):
        return extractor.get_module_name() == "linetr"

    def match_pair(self, descinfo1, descinfo2):
        if self.topk == 0:
            return self.match_segs_with_descinfo(descinfo1, descinfo2)
        else:
            return self.match_segs_with_descinfo_topk(
                descinfo1, descinfo2, topk=self.topk
            )

    def match_segs_with_descinfo(self, descinfo1, descinfo2):
        line_desc1 = descinfo1["line_descriptors"].T[None]
        line_desc2 = descinfo2["line_descriptors"].T[None]
        distance_sublines = get_dist_matrix(line_desc1, line_desc2)
        distance_matrix = self.linetr.subline2keyline(
            distance_sublines,
            descinfo1["mat_klines2sublines"],
            descinfo2["mat_klines2sublines"],
        )[0]
        match_mat = nn_matcher_distmat(
            distance_matrix,
            self.linetr.config["nn_threshold"],
            is_mutual_NN=True,
        )[0]
        matches = np.stack(np.where(match_mat > 0), axis=-1)
        return matches

    def match_segs_with_descinfo_topk(self, descinfo1, descinfo2, topk=10):
        line_desc1 = descinfo1["line_descriptors"].T[None]
        line_desc2 = descinfo2["line_descriptors"].T[None]
        distance_sublines = get_dist_matrix(line_desc1, line_desc2)
        distance_matrix = self.linetr.subline2keyline(
            distance_sublines,
            descinfo1["mat_klines2sublines"],
            descinfo2["mat_klines2sublines"],
        )[0, 0]

        # For each line in img1, retrieve the topk matches in img2
        matches = np.argsort(distance_matrix, axis=1)[:, :topk]

        # Transform matches to [n_matches, 2]
        n_lines = matches.shape[0]
        matches_t = np.stack(
            [np.arange(n_lines).repeat(topk), matches.flatten()], axis=1
        )
        return matches_t
