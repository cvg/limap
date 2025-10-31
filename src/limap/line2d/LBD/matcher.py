import numpy as np
import pytlbd

from ..base_matcher import (
    BaseMatcher,
    DefaultMatcherOptions,
)


class LBDMatcher(BaseMatcher):
    def __init__(self, extractor, options=DefaultMatcherOptions):
        super().__init__(extractor, options)

    def get_module_name(self):
        return "lbd"

    def check_compatibility(self, extractor):
        return extractor.get_module_name() == "lbd"

    def match_pair(self, descinfo1, descinfo2):
        if self.topk == 0:
            return self.match_segs_with_descinfo(descinfo1, descinfo2)
        else:
            return self.match_segs_with_descinfo_topk(
                descinfo1, descinfo2, topk=self.topk
            )

    def match_segs_with_descinfo(self, descinfo1, descinfo2):
        try:
            matches = pytlbd.lbd_matching_multiscale(
                descinfo1["ms_lines"].tolist(),
                descinfo2["ms_lines"].tolist(),
                descinfo1["line_descriptors"].tolist(),
                descinfo2["line_descriptors"].tolist(),
            )
            matches = np.array(matches)[:, :2]
        except RuntimeError:
            matches = np.zeros((0, 2))
        return matches

    def match_segs_with_descinfo_topk(self, descinfo1, descinfo2, topk=10):
        raise NotImplementedError()
