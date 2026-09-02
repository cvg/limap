import cv2
import numpy as np
from pathlib import Path
import torch
from tp_lsd.modeling.TP_Net import Res320
from tp_lsd.utils.reconstruct import TPS_line
from tp_lsd.utils.utils import load_model

from ..base_detector import (
    BaseDetector,
    DefaultDetectorOptions,
)
from limap.util.model_weights import download_weights, resolve_weight_path


class TPLSDDetector(BaseDetector):
    def __init__(self, options=DefaultDetectorOptions):
        super().__init__(options)
        # Load the TP-LSD model
        head = {"center": 1, "dis": 4, "line": 1}
        ckpt = resolve_weight_path(
            self.weight_path, "line2d", "TP_LSD", "pretrained/Res512.pth"
        )
        download_weights(
            "https://github.com/Siyuada7/TP-LSD/blob/master/pretraineds/"
            "Res512.pth?raw=true",
            ckpt,
        )
        self.net = load_model(Res320(head), ckpt)
        self.net = self.net.cuda().eval()

    def get_module_name(self):
        return "tp_lsd"

    def detect(self, image_path: Path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        segs = self.detect_tplsd(img, self.net)
        return segs

    def detect_tplsd(self, img, net):
        H, W = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        imgv0 = hsv[..., 2]
        imgv = cv2.resize(
            imgv0,
            (0, 0),
            fx=1.0 / 4,
            fy=1.0 / 4,
            interpolation=cv2.INTER_LINEAR,
        )
        imgv = cv2.GaussianBlur(imgv, (5, 5), 3)
        imgv = cv2.resize(imgv, (W, H), interpolation=cv2.INTER_LINEAR)
        imgv = cv2.GaussianBlur(imgv, (5, 5), 3)

        imgv1 = imgv0.astype(np.float32) - imgv + 127.5
        imgv1 = np.clip(imgv1, 0, 255).astype(np.uint8)
        hsv[..., 2] = imgv1
        inp = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        inp = inp.astype(np.float32) / 255.0
        inp = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).cuda()
        with torch.no_grad():
            outputs = net(inp)
        lines = TPS_line(outputs[-1], 0.25, 0.5, H, W)[0].reshape(-1, 2, 2)

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
