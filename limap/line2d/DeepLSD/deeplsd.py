import os

import numpy as np
import torch
from deeplsd.models.deeplsd_inference import DeepLSD
from pycolmap import logging

from ..base_detector import (
    BaseDetector,
    DefaultDetectorOptions,
)


class DeepLSDDetector(BaseDetector):
    def __init__(self, options=DefaultDetectorOptions):
        super().__init__(options)

        conf = {
            "detect_lines": True,
            "line_detection_params": {
                "merge": False,
                "grad_nfa": True,
                "filtering": "normal",
                "grad_thresh": 3,
            },
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.weight_path is None:
            ckpt = os.path.join(os.path.dirname(__file__), "deeplsd_md.tar")
        else:
            ckpt = os.path.join(
                self.weight_path, "line2d", "DeepLSD", "deeplsd_md.tar"
            )
        if not os.path.isfile(ckpt):
            self.download_model(ckpt)
        ckpt = torch.load(ckpt, map_location="cpu")
        self.net = DeepLSD(conf).eval()
        self.net.load_state_dict(ckpt["model"])
        self.net = self.net.to(self.device)

    def download_model(self, path):
        import subprocess

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        link = "https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_md.tar"
        cmd = ["wget", link, "-O", path]
        logging.info("Downloading DeepLSD model...")
        subprocess.run(cmd, check=True)

    def get_module_name(self):
        return "deeplsd"

    def detect(self, camview):
        img = camview.read_image(set_gray=True)
        img = (
            torch.tensor(img[None, None], dtype=torch.float, device=self.device)
            / 255
        )
        with torch.no_grad():
            lines = self.net({"image": img})["lines"][0]

        # Use the line length as score
        lines = np.concatenate(
            [
                lines.reshape(-1, 4),
                np.linalg.norm(
                    lines[:, 0] - lines[:, 1], axis=1, keepdims=True
                ),
            ],
            axis=1,
        )
        return lines
