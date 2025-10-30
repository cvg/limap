import os

import numpy as np
import torch
from gluestick.models.gluestick import GlueStick
from pycolmap import logging

from ..base_matcher import (
    BaseMatcher,
    DefaultMatcherOptions,
)


class GlueStickMatcher(BaseMatcher):
    def __init__(self, extractor, options=DefaultMatcherOptions, device=None):
        super().__init__(extractor, options)
        self.device = "cuda" if device is None else device
        self.gs = GlueStick({}).eval().to(self.device)
        if self.weight_path is None:
            ckpt = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "weights/checkpoint_GlueStick_MD.tar",
            )
        else:
            ckpt = os.path.join(
                self.weight_path,
                "line2d",
                "GlueStick",
                "weights/checkpoint_GlueStick_MD.tar",
            )
        if not os.path.isfile(ckpt):
            self.download_model(ckpt)
        ckpt = torch.load(ckpt, map_location="cpu")["model"]
        ckpt = {k[8:]: v for (k, v) in ckpt.items() if k.startswith("matcher")}
        self.gs.load_state_dict(ckpt, strict=True)

    def download_model(self, path):
        import subprocess

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        link = "https://github.com/cvg/GlueStick/releases/download/v0.1_arxiv/checkpoint_GlueStick_MD.tar"
        cmd = ["wget", link, "-O", path]
        logging.info("Downloading GlueStick model...")
        subprocess.run(cmd, check=True)

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
        # Setup the inputs for GlueStick
        inputs = {
            "image_size0": tuple(descinfo1["image_shape"]),
            "image_size1": tuple(descinfo2["image_shape"]),
            "keypoints0": torch.tensor(
                descinfo1["junctions"][None],
                dtype=torch.float,
                device=self.device,
            ),
            "keypoints1": torch.tensor(
                descinfo2["junctions"][None],
                dtype=torch.float,
                device=self.device,
            ),
            "keypoint_scores0": torch.tensor(
                descinfo1["junc_scores"][None],
                dtype=torch.float,
                device=self.device,
            ),
            "keypoint_scores1": torch.tensor(
                descinfo2["junc_scores"][None],
                dtype=torch.float,
                device=self.device,
            ),
            "descriptors0": torch.tensor(
                descinfo1["junc_desc"][None],
                dtype=torch.float,
                device=self.device,
            ),
            "descriptors1": torch.tensor(
                descinfo2["junc_desc"][None],
                dtype=torch.float,
                device=self.device,
            ),
            "lines0": torch.tensor(
                descinfo1["lines"][None], dtype=torch.float, device=self.device
            ),
            "lines1": torch.tensor(
                descinfo2["lines"][None], dtype=torch.float, device=self.device
            ),
            "line_scores0": torch.tensor(
                descinfo1["line_scores"][None],
                dtype=torch.float,
                device=self.device,
            ),
            "line_scores1": torch.tensor(
                descinfo2["line_scores"][None],
                dtype=torch.float,
                device=self.device,
            ),
            "lines_junc_idx0": torch.tensor(
                descinfo1["lines_junc_idx"][None],
                dtype=torch.long,
                device=self.device,
            ),
            "lines_junc_idx1": torch.tensor(
                descinfo2["lines_junc_idx"][None],
                dtype=torch.long,
                device=self.device,
            ),
        }

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
        # Setup the inputs for GlueStick
        inputs = {
            "image_size0": tuple(descinfo1["image_shape"]),
            "image_size1": tuple(descinfo2["image_shape"]),
            "keypoints0": torch.tensor(
                descinfo1["junctions"][None],
                dtype=torch.float,
                device=self.device,
            ),
            "keypoints1": torch.tensor(
                descinfo2["junctions"][None],
                dtype=torch.float,
                device=self.device,
            ),
            "keypoint_scores0": torch.tensor(
                descinfo1["junc_scores"][None],
                dtype=torch.float,
                device=self.device,
            ),
            "keypoint_scores1": torch.tensor(
                descinfo2["junc_scores"][None],
                dtype=torch.float,
                device=self.device,
            ),
            "descriptors0": torch.tensor(
                descinfo1["junc_desc"][None],
                dtype=torch.float,
                device=self.device,
            ),
            "descriptors1": torch.tensor(
                descinfo2["junc_desc"][None],
                dtype=torch.float,
                device=self.device,
            ),
            "lines0": torch.tensor(
                descinfo1["lines"][None], dtype=torch.float, device=self.device
            ),
            "lines1": torch.tensor(
                descinfo2["lines"][None], dtype=torch.float, device=self.device
            ),
            "line_scores0": torch.tensor(
                descinfo1["line_scores"][None],
                dtype=torch.float,
                device=self.device,
            ),
            "line_scores1": torch.tensor(
                descinfo2["line_scores"][None],
                dtype=torch.float,
                device=self.device,
            ),
            "lines_junc_idx0": torch.tensor(
                descinfo1["lines_junc_idx"][None],
                dtype=torch.long,
                device=self.device,
            ),
            "lines_junc_idx1": torch.tensor(
                descinfo2["lines_junc_idx"][None],
                dtype=torch.long,
                device=self.device,
            ),
        }

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
