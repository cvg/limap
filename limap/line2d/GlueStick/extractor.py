import os

import numpy as np
import torch
from gluestick.models.wireframe import lines_to_wireframe
from omegaconf import OmegaConf

import limap.util.io as limapio
from limap.point2d.superpoint import SuperPoint, sample_descriptors

from ..base_detector import (
    BaseDetector,
    DefaultDetectorOptions,
)


class WireframeExtractor(BaseDetector):
    def __init__(self, options=DefaultDetectorOptions, device=None):
        super().__init__(options)
        self.device = "cuda" if device is None else device
        self.sp = (
            SuperPoint({"weight_path": self.weight_path}).eval().to(self.device)
        )
        self.wireframe_params = OmegaConf.create(
            {
                "nms_radius": 3,
                "force_num_junctions": False,
            }
        )

    def get_module_name(self):
        return "wireframe"

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

    def extract(self, camview, segs):
        img = camview.read_image(set_gray=self.set_gray)
        descinfo = self.compute_descinfo(img, segs)
        return descinfo

    def compute_descinfo(self, img, segs):
        """A desc_info is composed of the following tuple / np arrays:
        - the original image shape (h, w)
        - the lines in shape [N, 2, 2] (xy convention)
        - the 2D endpoints of the lines in shape [N*2, 2] (xy convention),
          concatenated with the keypoints [n, 2]
        - the junctions score of shape [N*2 + n]
        - the descriptor of each junction of shape [256, N*2 + n]
        - the index of each line endpoints of shape [N, 2]
        """
        if len(segs) == 0:
            return {
                "image_shape": img.shape,
                "lines": np.empty((0, 2, 2)),
                "line_scores": np.empty((0,)),
                "junctions": np.empty((0, 2)),
                "junc_scores": np.empty((0,)),
                "junc_desc": np.empty((256, 0)),
                "lines_junc_idx": np.empty((0, 2)),
            }
        lines = segs[:, :4].reshape(-1, 2)
        line_scores = segs[:, -1] * np.sqrt(
            np.linalg.norm(segs[:, :2] - segs[:, 2:4], axis=1)
        )
        line_scores /= np.amax(line_scores) + 1e-8
        torch_img = {
            "image": torch.tensor(
                img.astype(np.float32) / 255,
                dtype=torch.float,
                device=self.device,
            )[None, None]
        }
        with torch.no_grad():
            kp, scores, dense_desc = self.sp.compute_dense_descriptor(torch_img)
            kp, scores = torch.stack(kp), torch.stack(scores)
            kp_desc = [
                sample_descriptors(k[None], d[None], 8)[0]
                for k, d in zip(kp, dense_desc)
            ]

        # Remove keypoints that are too close to line endpoints
        line_endpts = torch.tensor(
            lines.reshape(1, -1, 2), dtype=torch.float, device=self.device
        )
        torch_line_scores = torch.tensor(
            line_scores[None], dtype=torch.float, device=self.device
        )
        dist_pt_lines = torch.norm(
            kp[:, :, None] - line_endpts[:, None], dim=-1
        )
        # For each keypoint, mark it as valid or to remove
        pts_to_remove = torch.any(
            dist_pt_lines < self.wireframe_params.nms_radius, dim=2
        )
        kp = kp[0][~pts_to_remove[0]][None]
        scores = scores[0][~pts_to_remove[0]][None]
        kp_desc = kp_desc[0].T[~pts_to_remove[0]].T[None]

        # Connect the lines together to form a wireframe
        # Merge first close-by endpoints to connect lines
        (
            line_points,
            line_pts_scores,
            line_descs,
            _,
            lines,
            lines_junc_idx,
            _,
        ) = lines_to_wireframe(
            line_endpts,
            torch_line_scores,
            dense_desc,
            conf=self.wireframe_params,
        )

        # Add the keypoints to the junctions
        all_points = torch.cat([line_points[0], kp[0]], dim=0).cpu().numpy()
        all_scores = (
            torch.cat([line_pts_scores[0], scores[0]], dim=0).cpu().numpy()
        )
        all_descs = torch.cat([line_descs[0], kp_desc[0]], dim=1).cpu().numpy()
        lines_junc_idx = lines_junc_idx[0].cpu().numpy()

        lines = lines.cpu().numpy().reshape(-1, 2, 2)
        return {
            "image_shape": img.shape,
            "lines": lines,
            "line_scores": line_scores,
            "junctions": all_points,
            "junc_scores": all_scores,
            "junc_desc": all_descs,
            "lines_junc_idx": lines_junc_idx,
        }
