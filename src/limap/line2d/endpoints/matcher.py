import numpy as np
import torch

from limap.point2d.superglue import SuperGlue

from ..base_matcher import (
    BaseMatcher,
    DefaultMatcherOptions,
)


class NNEndpointsMatcher(BaseMatcher):
    def __init__(self, extractor, options=DefaultMatcherOptions, device=None):
        super().__init__(extractor, options)
        assert self.extractor.get_module_name() == "superpoint_endpoints"
        self.device = "cuda" if device is None else device
        self.sg = (
            SuperGlue({"weight_path": self.weight_path}).eval().to(self.device)
        )

    def get_module_name(self):
        return "nn_endpoints"

    def match_pair(self, descinfo1, descinfo2):
        if self.topk == 0:
            return self.match_segs_with_descinfo(descinfo1, descinfo2)
        else:
            return self.match_segs_with_descinfo_topk(
                descinfo1, descinfo2, topk=self.topk
            )

    def match_segs_with_descinfo(self, descinfo1, descinfo2):
        desc1 = (
            torch.tensor(
                descinfo1["endpoints_desc"],
                dtype=torch.float,
                device=self.device,
            ),
        )
        desc2 = (
            torch.tensor(
                descinfo2["endpoints_desc"],
                dtype=torch.float,
                device=self.device,
            ),
        )

        with torch.no_grad():
            # Run the point matching
            scores = desc1[0].t() @ desc2[0]

            # Retrieve the best matching score of the line endpoints
            n_lines1 = scores.shape[0] // 2
            n_lines2 = scores.shape[1] // 2
            scores = scores.reshape(n_lines1, 2, n_lines2, 2)
            scores = 0.5 * torch.maximum(
                scores[:, 0, :, 0] + scores[:, 1, :, 1],
                scores[:, 0, :, 1] + scores[:, 1, :, 0],
            )

            # Run the Sinkhorn algorithm and get the line matches
            scores = self.sg._solve_optimal_transport(scores[None])
            matches = self.sg._get_matches(scores)[0].cpu().numpy()[0]

        # Transform matches to [n_matches, 2]
        id_list_1 = np.arange(0, matches.shape[0])[matches != -1]
        id_list_2 = matches[matches != -1]
        matches_t = np.stack([id_list_1, id_list_2], 1)
        return matches_t

    def match_segs_with_descinfo_topk(self, descinfo1, descinfo2, topk=10):
        desc1 = (
            torch.tensor(
                descinfo1["endpoints_desc"],
                dtype=torch.float,
                device=self.device,
            ),
        )
        desc2 = (
            torch.tensor(
                descinfo2["endpoints_desc"],
                dtype=torch.float,
                device=self.device,
            ),
        )

        with torch.no_grad():
            # Run the point matching
            scores = desc1[0].t() @ desc2[0]

            # Retrieve the best matching score of the line endpoints
            n_lines1 = scores.shape[0] // 2
            n_lines2 = scores.shape[1] // 2
            scores = scores.reshape(n_lines1, 2, n_lines2, 2)
            scores = 0.5 * torch.maximum(
                scores[:, 0, :, 0] + scores[:, 1, :, 1],
                scores[:, 0, :, 1] + scores[:, 1, :, 0],
            )

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


class SuperGlueEndpointsMatcher(BaseMatcher):
    def __init__(
        self,
        extractor,
        options=DefaultMatcherOptions,
        weights="outdoor",
        device=None,
    ):
        super().__init__(extractor, options)
        assert self.extractor.get_module_name() == "superpoint_endpoints"
        self.device = "cuda" if device is None else device
        self.sg = SuperGlue({"weights": weights}).eval().to(self.device)

    def get_module_name(self):
        return "superglue_endpoints"

    def match_pair(self, descinfo1, descinfo2):
        if len(descinfo1["lines"]) == 0 or len(descinfo1["lines"]) == 0:
            return np.empty((0, 2))
        if self.topk == 0:
            return self.match_segs_with_descinfo(descinfo1, descinfo2)
        else:
            return self.match_segs_with_descinfo_topk(
                descinfo1, descinfo2, topk=self.topk
            )

    def match_segs_with_descinfo(self, descinfo1, descinfo2):
        # Setup the inputs for SuperGlue
        inputs = {
            "image_shape0": descinfo1["image_shape"],
            "image_shape1": descinfo2["image_shape"],
            "keypoints0": torch.tensor(
                descinfo1["lines"][None], dtype=torch.float, device=self.device
            ),
            "keypoints1": torch.tensor(
                descinfo2["lines"][None], dtype=torch.float, device=self.device
            ),
            "scores0": torch.tensor(
                descinfo1["lines_score"].repeat(2)[None],
                dtype=torch.float,
                device=self.device,
            ),
            "scores1": torch.tensor(
                descinfo2["lines_score"].repeat(2)[None],
                dtype=torch.float,
                device=self.device,
            ),
            "descriptors0": torch.tensor(
                descinfo1["endpoints_desc"][None],
                dtype=torch.float,
                device=self.device,
            ),
            "descriptors1": torch.tensor(
                descinfo2["endpoints_desc"][None],
                dtype=torch.float,
                device=self.device,
            ),
        }

        with torch.no_grad():
            # Run the point matching
            out = self.sg(inputs)

            # Retrieve the best matching score of the line endpoints
            n_lines1 = len(descinfo1["lines"]) // 2
            n_lines2 = len(descinfo2["lines"]) // 2
            scores = out["scores"].reshape(n_lines1, 2, n_lines2, 2)
            scores = 0.5 * torch.maximum(
                scores[:, 0, :, 0] + scores[:, 1, :, 1],
                scores[:, 0, :, 1] + scores[:, 1, :, 0],
            )

            # Run the Sinkhorn algorithm and get the line matches
            scores = self.sg._solve_optimal_transport(scores[None])
            matches = self.sg._get_matches(scores)[0].cpu().numpy()[0]

        # Transform matches to [n_matches, 2]
        id_list_1 = np.arange(0, matches.shape[0])[matches != -1]
        id_list_2 = matches[matches != -1]
        matches_t = np.stack([id_list_1, id_list_2], 1)
        return matches_t

    def match_segs_with_descinfo_topk(self, descinfo1, descinfo2, topk=10):
        # Setup the inputs for SuperGlue
        inputs = {
            "image_shape0": descinfo1["image_shape"],
            "image_shape1": descinfo2["image_shape"],
            "keypoints0": torch.tensor(
                descinfo1["lines"][None], dtype=torch.float, device=self.device
            ),
            "keypoints1": torch.tensor(
                descinfo2["lines"][None], dtype=torch.float, device=self.device
            ),
            "scores0": torch.tensor(
                descinfo1["lines_score"].repeat(2)[None],
                dtype=torch.float,
                device=self.device,
            ),
            "scores1": torch.tensor(
                descinfo2["lines_score"].repeat(2)[None],
                dtype=torch.float,
                device=self.device,
            ),
            "descriptors0": torch.tensor(
                descinfo1["endpoints_desc"][None],
                dtype=torch.float,
                device=self.device,
            ),
            "descriptors1": torch.tensor(
                descinfo2["endpoints_desc"][None],
                dtype=torch.float,
                device=self.device,
            ),
        }

        with torch.no_grad():
            # Run the point matching
            out = self.sg(inputs)

            # Retrieve the best matching score of the line endpoints
            n_lines1 = len(descinfo1["lines"]) // 2
            n_lines2 = len(descinfo2["lines"]) // 2
            scores = out["scores"].reshape(n_lines1, 2, n_lines2, 2)
            scores = 0.5 * torch.maximum(
                scores[:, 0, :, 0] + scores[:, 1, :, 1],
                scores[:, 0, :, 1] + scores[:, 1, :, 0],
            )

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
