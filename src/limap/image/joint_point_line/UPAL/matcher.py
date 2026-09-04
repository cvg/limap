import numpy as np
import torch

from ...line.base_matcher import (
    BaseMatcher,
    DefaultMatcherOptions,
)


class UPALMatcher(BaseMatcher):
    """Match lines by their two endpoint descriptors.

    Scores a pair by the better of the two endpoint orientations, as
    ``nn_endpoints`` does, but resolves the assignment by mutual nearest
    neighbour instead of SuperGlue's Sinkhorn, whose bin score is tied to
    SuperPoint's 256-d descriptors.
    """

    def __init__(self, extractor, options=DefaultMatcherOptions, device=None):
        super().__init__(extractor, options)
        assert self.extractor.get_module_name() == "upal"
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )

    def get_module_name(self):
        return "upal"

    def _line_scores(self, descinfo1, descinfo2):
        desc1 = torch.tensor(
            descinfo1["endpoints_desc"], dtype=torch.float, device=self.device
        )
        desc2 = torch.tensor(
            descinfo2["endpoints_desc"], dtype=torch.float, device=self.device
        )
        scores = desc1.t() @ desc2
        n_lines1, n_lines2 = scores.shape[0] // 2, scores.shape[1] // 2
        scores = scores.reshape(n_lines1, 2, n_lines2, 2)
        return 0.5 * torch.maximum(
            scores[:, 0, :, 0] + scores[:, 1, :, 1],
            scores[:, 0, :, 1] + scores[:, 1, :, 0],
        )

    def match_pair(self, descinfo1, descinfo2):
        if descinfo1["endpoints_desc"].shape[1] == 0:
            return np.empty((0, 2), dtype=int)
        if descinfo2["endpoints_desc"].shape[1] == 0:
            return np.empty((0, 2), dtype=int)
        with torch.no_grad():
            scores = self._line_scores(descinfo1, descinfo2)
            if self.topk == 0:
                nearest2 = scores.argmax(dim=1)
                nearest1 = scores.argmax(dim=0)
                index1 = torch.arange(len(scores), device=scores.device)
                mutual = nearest1[nearest2] == index1
                matches = torch.stack([index1[mutual], nearest2[mutual]], dim=1)
            else:
                topk = min(self.topk, scores.shape[1])
                nearest2 = torch.argsort(scores, dim=1)[:, -topk:]
                index1 = (
                    torch.arange(len(scores), device=scores.device)
                    .unsqueeze(1)
                    .expand_as(nearest2)
                )
                matches = torch.stack(
                    [index1.reshape(-1), nearest2.reshape(-1)], dim=1
                )
        return matches.cpu().numpy().astype(int)
