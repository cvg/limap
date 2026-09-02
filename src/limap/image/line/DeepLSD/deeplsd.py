import numpy as np
import cv2
from pathlib import Path
import torch
from deeplsd.models.deeplsd_inference import DeepLSD

from ..base_detector import (
    BaseDetector,
    DefaultDetectorOptions,
)
from limap.util.model_weights import download_weights, resolve_weight_path


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
        ckpt = resolve_weight_path(
            self.weight_path, "line2d", "DeepLSD", "deeplsd_md.tar"
        )
        download_weights(
            "https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_md.tar", ckpt
        )
        ckpt = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.net = DeepLSD(conf).eval()
        self.net.load_state_dict(ckpt["model"])
        self.net = self.net.to(self.device)

    def get_module_name(self):
        return "deeplsd"

    def detect(self, image_path: Path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
