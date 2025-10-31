import os

import cv2
import numpy as np
import torch
from hawp.fsl.config import cfg as model_config
from hawp.ssl.models import MODELS
from pycolmap import logging

from ..base_detector import (
    BaseDetector,
    DefaultDetectorOptions,
)


class HAWPv3Detector(BaseDetector):
    def __init__(self, options=DefaultDetectorOptions):
        super().__init__(options)
        # Load the HAWPv3 model
        if self.weight_path is None:
            ckpt = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "weights/hawpv3_wireframe.pth",
            )
        else:
            ckpt = os.path.join(
                self.weight_path,
                "line2d",
                "HAWPv3",
                "weights/hawpv3_wireframe.pth",
            )
        config = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "hawpv3.yaml"
        )
        model_config.merge_from_file(config)
        self.net = MODELS["HAWP"](model_config, gray_scale=True)
        self.net = self.net.eval().cuda()
        if not os.path.isfile(ckpt):
            self.download_model(ckpt)
        self.net.load_state_dict(torch.load(ckpt))

    def download_model(self, path):
        import subprocess

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        link = "https://github.com/cherubicXN/hawp-torchhub/releases/download/HAWPv3/hawpv3-fdc5487a.pth"
        cmd = ["wget", link, "-O", path]
        logging.info("Downloading HAWPv3 model...")
        subprocess.run(cmd, check=True)

    def get_module_name(self):
        return "hawpv3"

    def detect(self, camview):
        img = camview.read_image(set_gray=True)
        segs = self.detect_hawp(img, self.net)
        return segs

    def detect_hawp(self, img, net, shape=(512, 512), thresh=0.5):
        # Detect the lines with HAWP
        h, w = img.shape[:2]
        if shape is not None:
            np_img = cv2.resize(img, (shape[1], shape[0]))
        else:
            np_img = img.copy()
            shape = (h, w)
        image_tensor = (
            torch.tensor(np_img[None, None], dtype=torch.float, device="cuda")
            / 255.0
        )
        meta = {"filename": "", "height": shape[0], "width": shape[1]}
        with torch.no_grad():
            out = net(image_tensor, [meta])[0]
        sx = w / float(shape[1])
        sy = h / float(shape[0])
        lines = out["lines_pred"].cpu().numpy()
        lines *= np.array([sx, sy, sx, sy]).reshape(-1, 4)
        scores = out["lines_score"].cpu().numpy()
        lines = lines[scores > thresh]
        scores = scores[scores > thresh]
        lines = np.concatenate([lines[:, :4], scores[:, None]], axis=1)

        return lines
