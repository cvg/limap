import os
import subprocess

import cv2
import numpy as np
import torch
from pycolmap import logging
from skimage.draw import line

from .experiment import load_config
from .model.line_matcher import LineMatcher


class SOLD2LineDetector:
    def __init__(self, device=None, cfg_path=None, weight_path=None):
        nowpath = os.path.dirname(os.path.abspath(__file__))
        if cfg_path is None:
            cfg_path = "config/export_line_features.yaml"
        self.cfg = load_config(os.path.join(nowpath, cfg_path))
        if weight_path is None:
            self.ckpt_path = os.path.join(
                nowpath, "pretrained_models/sold2_wireframe.tar"
            )
        else:
            self.ckpt_path = os.path.join(
                weight_path,
                "line2d",
                "SOLD2",
                "pretrained_models/sold2_wireframe.tar",
            )
        if device is None:
            device = "cuda"
        self.device = device

        # initialize line matcher
        self.initialize_line_matcher()

    def initialize_line_matcher(self):
        if not os.path.isfile(self.ckpt_path):
            if not os.path.exists(os.path.dirname(self.ckpt_path)):
                os.makedirs(os.path.dirname(self.ckpt_path))
            link = "https://cvg-data.inf.ethz.ch/SOLD2/sold2_wireframe.tar"
            cmd = ["wget", link, "-O", self.ckpt_path]
            logging.info("Downloading SOLD2 model...")
            subprocess.run(cmd, check=True)
        self.line_matcher = LineMatcher(
            self.cfg["model_cfg"],
            self.ckpt_path,
            self.device,
            self.cfg["line_detector_cfg"],
            self.cfg["line_matcher_cfg"],
            self.cfg["multiscale_cfg"]["multiscale"],
            self.cfg["multiscale_cfg"]["scales"],
        )

    def sold2segstosegs(self, segs_sold2):
        return np.flip(segs_sold2, axis=2).reshape(len(segs_sold2), 4)

    def segstosold2segs(self, segs):
        return np.flip(segs.reshape(segs.shape[0], 2, 2), axis=2)

    def detect(self, input_image, saliency=False, scale_factor=None):
        if input_image.shape[0] < 80 or input_image.shape[1] < 80:
            return np.array([]), None, None, [np.array([]), np.array([])]
        # Convert to grayscale if necessary
        if len(input_image.shape) == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        input_image = (input_image / 255.0).astype(float)
        input_image = torch.tensor(input_image, dtype=torch.float)[None, None]
        input_image = input_image.to(self.device)

        # forward
        with torch.no_grad():
            if self.cfg["multiscale_cfg"]["multiscale"]:
                net_outputs = self.line_matcher.multiscale_line_detection(
                    input_image, scales=self.cfg["multiscale_cfg"]["scales"]
                )
            else:
                net_outputs = self.line_matcher.line_detection(input_image)
        segs_sold2 = net_outputs["line_segments"]
        descriptor = net_outputs["descriptor"]
        with torch.no_grad():
            descinfo = self.line_matcher.line_matcher.compute_descriptors(
                segs_sold2, descriptor
            )
        descriptor = descriptor.cpu().numpy()
        if len(descinfo) != 0:
            descinfo[0] = descinfo[0].cpu().numpy()
        heatmap = net_outputs["heatmap"]
        segs = self.sold2segstosegs(segs_sold2)

        # get saliencies
        rounded_segs = np.round(segs_sold2).astype(int)
        rounded_segs[..., 0] = np.clip(
            rounded_segs[..., 0], 0, heatmap.shape[-2] - 1
        )
        rounded_segs[..., 1] = np.clip(
            rounded_segs[..., 1], 0, heatmap.shape[-1] - 1
        )
        resulting_saliency = []
        for s in rounded_segs:
            pts = line(s[0, 0], s[0, 1], s[1, 0], s[1, 1])
            sal = heatmap[pts].sum()
            resulting_saliency.append(sal)
        saliencies = np.array(resulting_saliency)
        return (
            np.hstack([segs, saliencies[:, np.newaxis]]),
            descriptor,
            heatmap,
            descinfo,
        )

    def get_heatmap(self, input_image):
        input_image = (input_image / 255.0).astype(float)
        input_image = torch.tensor(input_image, dtype=torch.float)[None, None]
        input_image = input_image.to(self.device)
        with torch.no_grad():
            net_outputs = self.line_matcher.line_detection(input_image)
        heatmap = net_outputs["heatmap"]
        return heatmap

    def match(self, img1, img2):
        img1 = (img1 / 255.0).astype(float)
        img1 = torch.tensor(img1, dtype=torch.float, device=self.device)[
            None, None
        ]
        img2 = (img2 / 255.0).astype(float)
        img2 = torch.tensor(img2, dtype=torch.float, device=self.device)[
            None, None
        ]
        outputs = self.line_matcher([img1, img2])
        matches = outputs["matches"]
        return matches

    def match_segs_with_descriptor(self, segs1, desc1, segs2, desc2):
        if segs1.shape[0] == 0 or segs2.shape[0] == 0:
            return []
        segs1_sold2 = self.segstosold2segs(segs1[:, :4])
        segs2_sold2 = self.segstosold2segs(segs2[:, :4])
        desc1 = torch.tensor(desc1, dtype=torch.float, device=self.device)
        desc2 = torch.tensor(desc2, dtype=torch.float, device=self.device)
        matches = self.line_matcher.line_matcher.forward(
            segs1_sold2, segs2_sold2, desc1, desc2
        )
        return matches

    def match_segs_with_descinfo(self, descinfo1, descinfo2):
        if len(descinfo1) == 0 or len(descinfo2) == 0:
            return []
        descinfo1 = [
            torch.tensor(descinfo1[0], dtype=torch.float, device=self.device),
            descinfo1[1],
        ]
        descinfo2 = [
            torch.tensor(descinfo2[0], dtype=torch.float, device=self.device),
            descinfo2[1],
        ]
        matches = self.line_matcher.line_matcher.compute_matches(
            descinfo1, descinfo2
        )

        # transform matches to [n_matches, 2]
        id_list_1 = np.arange(0, matches.shape[0])[matches != -1]
        id_list_2 = matches[matches != -1]
        matches_t = np.stack([id_list_1, id_list_2], 1)
        return matches_t

    def match_segs_with_descinfo_topk(self, descinfo1, descinfo2, topk=10):
        if len(descinfo1) == 0 or len(descinfo2) == 0:
            return []
        if len(descinfo1[0]) == 0 or len(descinfo2[0]) == 0:
            return []
        descinfo1 = [
            torch.tensor(descinfo1[0], dtype=torch.float, device=self.device),
            descinfo1[1],
        ]
        descinfo2 = [
            torch.tensor(descinfo2[0], dtype=torch.float, device=self.device),
            descinfo2[1],
        ]
        matches = self.line_matcher.line_matcher.compute_matches_topk_gpu(
            descinfo1, descinfo2, topk=topk
        )

        # transform matches to [n_matches, 2]
        n_lines = matches.shape[0]
        topk = matches.shape[1]
        matches_t = []
        for idx in range(topk):
            matches_t_idx = np.stack(
                [np.arange(0, n_lines), matches[:, idx]], 1
            )
            matches_t.append(matches_t_idx)
        matches_t = np.concatenate(matches_t, 0)
        return matches_t

    def compute_descinfo(self, segs, desc):
        if segs.shape[0] == 0:
            return []
        segs_sold2 = self.segstosold2segs(segs[:, :4])
        if desc is None:
            return []
        desc = torch.tensor(desc, dtype=torch.float, device=self.device)
        with torch.no_grad():
            descinfo = self.line_matcher.line_matcher.compute_descriptors(
                segs_sold2, desc
            )
        if len(descinfo) != 0:
            descinfo[0] = descinfo[0].cpu().numpy()
        return descinfo
