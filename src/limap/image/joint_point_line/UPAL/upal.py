import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import limap.util.io as limapio
from limap.util.model_weights import download_weights, resolve_weight_path

from ...line.base_detector import (
    BaseDetector,
    DefaultDetectorOptions,
)

# Pinned so the checkpoint stays in step with the inference code it is loaded
# into; upstream ships no versioned release.
_UPAL_COMMIT = "91d79a08a57766b319d144950ab9e9cf9615e429"
_UPAL_WEIGHTS_URL = (
    f"https://raw.githubusercontent.com/francois141/upal/{_UPAL_COMMIT}"
    "/weights/upal.tar"
)


class UPALDetector(BaseDetector):
    """Joint point-line network, used here for its line half.

    Segments and their endpoint descriptors both come out of a single encoder
    pass (:meth:`detect_and_extract`). Going through the upstream API instead
    would encode twice, since ``UPAL.describe_keypoints`` re-runs the backbone.
    """

    def __init__(
        self, options=DefaultDetectorOptions, upal_options=None, device=None
    ):
        super().__init__(options)
        from upal import load_model

        if upal_options is None:
            # Imported here so that the registry, which owns the dataclass,
            # stays importable without torch.
            from ...line.register_detector import UPALOptions

            upal_options = UPALOptions()
        self.min_line_length = upal_options.min_line_length
        self.max_mean_distance = upal_options.max_mean_distance
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        ckpt = resolve_weight_path(
            self.weight_path, "line2d", "UPAL", "upal.tar"
        )
        download_weights(_UPAL_WEIGHTS_URL, ckpt)
        self.net = load_model(
            ckpt,
            device=self.device,
            max_num_keypoints=upal_options.max_num_keypoints,
        )

    def get_module_name(self):
        return "upal"

    def get_descinfo_fname(self, descinfo_folder, img_id):
        fname = os.path.join(descinfo_folder, f"descinfo_{img_id}.npz")
        return fname

    def save_descinfo(self, descinfo_folder, img_id, descinfo):
        limapio.check_makedirs(descinfo_folder)
        fname = self.get_descinfo_fname(descinfo_folder, img_id)
        limapio.save_npz(fname, descinfo)

    def read_descinfo(self, descinfo_folder, img_id):
        fname = self.get_descinfo_fname(descinfo_folder, img_id)
        descinfo = limapio.read_npz(fname)
        return descinfo

    def _load_image(self, image_path: Path):
        """Read an image as the ``1 x 3 x H x W`` tensor the network wants."""
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .float()
            .div_(255)
            .unsqueeze(0)
            .to(self.device)
        )
        return img.shape, tensor

    def _encode(self, image, with_points=False):
        """Run the backbone once and return everything both halves need.

        Mirrors ``UPAL.forward`` but keeps the normalized feature map, which
        upstream discards and then recomputes to describe line endpoints.
        ``with_points`` additionally describes the keypoints themselves, for
        the joint point-line path.
        """
        from upal.model import InputPadder

        _, _, height, width = image.shape
        padder = InputPadder(height, width, divisor=32)
        raw_features = self.net.encoder_backbone(padder.pad(image))
        score_map = padder.unpad(
            self.net.keypoint_and_junction_branch(raw_features)
        )
        features = padder.unpad(F.normalize(raw_features, p=2, dim=1))
        distance_field = (
            torch.exp(-self.net.distance_field_branch(features))
            * self.net.line_neighborhood
        )
        keypoints_n, scores, _ = self.net.keypoint_detector(score_map)
        wh = torch.tensor([width - 1, height - 1], device=image.device)
        keypoints = wh * (keypoints_n[0] + 1.0) / 2.0
        points = None
        if with_points:
            points = {
                "keypoints": keypoints.cpu().numpy(),
                "scores": scores[0].cpu().numpy(),
                "descriptors": self.net.descriptor_branch(
                    features, keypoints_n
                )[0]
                .t()
                .cpu()
                .numpy(),
            }
        return features, keypoints, distance_field[0, 0], points

    def _describe(self, features, points):
        """Sample descriptors at pixel coordinates from a cached feature map."""
        if len(points) == 0:
            return np.zeros((self.net.descriptor_branch.sf_conv.in_channels, 0))
        _, _, height, width = features.shape
        wh = points.new_tensor((width - 1, height - 1))
        normalized = 2.0 * points / wh - 1.0
        desc = self.net.descriptor_branch(features, [normalized])[0]
        return desc.t().cpu().numpy()

    def _detect_lines(self, image, distance_field, keypoints):
        """Point-seeded LSD, keeping only line-field-supported segments.

        Ported from ``upal.postprocess.detect_lines`` to return limap's
        ``(N, 5)`` layout with the LSD score, which upstream discards, and to
        leave truncation to ``take_longest_k``.
        """
        import points_lsd

        # Private, but reimplementing them would have to reproduce the exact
        # gradient convention the LSD extension expects.
        from upal.postprocess import _line_seed_points, _lsd_gradients

        weights = image.new_tensor((0.299, 0.587, 0.114)).view(3, 1, 1)
        gray = (image[0] * weights).sum(dim=0).mul(255.0)
        gradients, angles = _lsd_gradients(gray)
        seeds = _line_seed_points(keypoints, gray.shape)
        if len(seeds) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        segments = points_lsd.lsd_from_points(
            np.ascontiguousarray(gray.cpu().numpy(), dtype=np.float64),
            np.ascontiguousarray(seeds.cpu().numpy(), dtype=np.int32),
            1.0,
            0.6,
            0.0,
            np.ascontiguousarray(gradients.cpu().numpy(), dtype=np.float64),
            np.ascontiguousarray(angles.cpu().numpy(), dtype=np.float64),
        )
        if len(segments) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        lines = (
            torch.from_numpy(segments[:, :4]).to(distance_field).reshape(-1, 2, 2)
        )
        scores = torch.from_numpy(segments[:, 4]).to(distance_field)
        lengths = torch.linalg.vector_norm(lines[:, 1] - lines[:, 0], dim=1)
        keep = lengths >= self.min_line_length
        lines, scores = lines[keep], scores[keep]
        if len(lines) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        samples = lines[:, :1] + (lines[:, 1:] - lines[:, :1]) * torch.linspace(
            0, 1, 32, device=lines.device
        ).view(1, -1, 1)
        x = samples[..., 0].round().long().clamp_(0, distance_field.shape[1] - 1)
        y = samples[..., 1].round().long().clamp_(0, distance_field.shape[0] - 1)
        keep = distance_field[y, x].mean(dim=1) <= self.max_mean_distance
        lines, scores = lines[keep], scores[keep]
        return (
            torch.cat([lines.reshape(-1, 4), scores[:, None]], dim=1)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def _compute_descinfo(self, features, image_shape, segs):
        """A descinfo holds, in the ``superpoint_endpoints`` layout:
        - the original image shape (h, w)
        - the 2D endpoints of the lines in shape [N*2, 2] (xy convention)
        - the line score of shape [N] (LSD score * sqrt(line_length),
          normalized to the strongest line in the image)
        - the descriptor of each endpoint of shape [128, N*2]
        """
        dim = self.net.descriptor_branch.sf_conv.in_channels
        if len(segs) == 0:
            return {
                "image_shape": image_shape,
                "lines": np.zeros((0, 2)),
                "lines_score": np.zeros((0,)),
                "endpoints_desc": np.zeros((dim, 0)),
            }
        lines = segs[:, :4].reshape(-1, 2)
        scores = segs[:, -1] * np.sqrt(
            np.linalg.norm(segs[:, :2] - segs[:, 2:4], axis=1)
        )
        scores /= np.amax(scores) + 1e-8
        endpoints = torch.from_numpy(lines).float().to(features.device)
        return {
            "image_shape": image_shape,
            "lines": lines,
            "lines_score": scores,
            "endpoints_desc": self._describe(features, endpoints),
        }

    def detect(self, image_path: Path):
        image_shape, image = self._load_image(image_path)
        del image_shape
        with torch.no_grad():
            _, keypoints, distance_field, _ = self._encode(image)
            return self._detect_lines(image, distance_field, keypoints)

    def extract(self, image_path: Path, segs):
        image_shape, image = self._load_image(image_path)
        with torch.no_grad():
            features, _, _, _ = self._encode(image)
            return self._compute_descinfo(features, image_shape, segs)

    def detect_and_extract(self, image_path: Path):
        segs, descinfo, _ = self.detect_and_extract_joint(
            image_path, with_points=False
        )
        return segs, descinfo

    def detect_and_extract_joint(self, image_path: Path, with_points=True):
        """Detect and describe points and lines from a single encoder pass.

        Returns the usual ``(segs, descinfo)`` plus, when ``with_points``, a
        dict of keypoints / scores / descriptors in the same resolution as the
        segments, for the joint point-line path.
        """
        image_shape, image = self._load_image(image_path)
        with torch.no_grad():
            features, keypoints, distance_field, points = self._encode(
                image, with_points=with_points
            )
            segs = self._detect_lines(image, distance_field, keypoints)
            descinfo = self._compute_descinfo(features, image_shape, segs)
        if points is not None:
            # hloc rebuilds a tensor shape from this, so it must be integral:
            # torch.empty((1,) + tuple(grp["image_size"])[::-1]).
            points["image_size"] = np.array(
                [image_shape[1], image_shape[0]], dtype=np.int64
            )
        return segs, descinfo, points

    def sample_descinfo_by_indexes(self, descinfo, indexes):
        indexes = np.array(indexes, dtype=int)
        endpoints = np.stack([2 * indexes, 2 * indexes + 1], axis=1).reshape(-1)
        return {
            "image_shape": descinfo["image_shape"],
            "lines": descinfo["lines"][endpoints],
            "lines_score": descinfo["lines_score"][indexes],
            "endpoints_desc": descinfo["endpoints_desc"][:, endpoints],
        }
