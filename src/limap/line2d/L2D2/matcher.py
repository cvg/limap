import numpy as np

from ..base_matcher import (
    BaseMatcher,
    DefaultMatcherOptions,
)


class L2D2Matcher(BaseMatcher):
    def __init__(self, extractor, options=DefaultMatcherOptions):
        super().__init__(extractor, options)

    def get_module_name(self):
        return "l2d2"

    def check_compatibility(self, extractor):
        return extractor.get_module_name() == "l2d2"

    def match_pair(self, descinfo1, descinfo2):
        if self.topk == 0:
            return self.match_segs_with_descinfo(descinfo1, descinfo2)
        else:
            return self.match_segs_with_descinfo_topk(
                descinfo1, descinfo2, topk=self.topk
            )

    def match_segs_with_descinfo(self, descinfo1, descinfo2):
        desc1 = descinfo1["line_descriptors"]
        desc2 = descinfo2["line_descriptors"]

        # Default case when an image has no lines
        if len(desc1) == 0 or len(desc2) == 0:
            return np.empty((0, 2))

        # Mutual nearest neighbor matching
        score_mat = desc1 @ desc2.T
        nearest1 = np.argmax(score_mat, axis=1)
        nearest2 = np.argmax(score_mat, axis=0)
        mutual = nearest2[nearest1] == np.arange(len(desc1))
        nearest1[~mutual] = -1

        # Transform matches to [n_matches, 2]
        id_list_1 = np.arange(0, len(nearest1))[mutual]
        id_list_2 = nearest1[mutual]
        matches_t = np.stack([id_list_1, id_list_2], 1)
        return matches_t

    def match_segs_with_descinfo_topk(self, descinfo1, descinfo2, topk=10):
        desc1 = descinfo1["line_descriptors"]
        desc2 = descinfo2["line_descriptors"]

        # Default case when an image has no lines
        if len(desc1) == 0 or len(desc2) == 0:
            return np.empty((0, 2))

        # Top k nearest neighbor matching
        score_mat = desc1 @ desc2.T
        matches = np.argsort(score_mat, axis=1)[:, -topk:]
        matches = np.flip(matches, axis=1)

        # Transform matches to [n_matches, 2]
        n_lines = len(matches)
        matches_t = np.stack(
            [np.arange(n_lines).repeat(topk), matches.flatten()], axis=1
        )
        return matches_t
