import numpy as np
import torch
from dsfm.trainlib.models.superpoint import SuperPoint
from pytlsd import lsd

from .. import BaseModel
from ..utils.gt_line_matches import (
    UNMATCHED_FEATURE,
    gt_line_matches_from_homography,
    gt_line_matches_from_pose_depth,
)
from ..utils.gt_matches import (
    gt_matches_from_homography,
    gt_matches_from_pose_depth,
)
from .line_transformer import LineTransformer, get_dist_matrix
from .nn_matcher import nn_matcher, nn_matcher_distmat


class LineTrPipeline(BaseModel):
    """Image Matching with SuperPoint & LineTR"""

    default_conf = {
        "auto_min_length": True,
        "nn_threshold": 0.7,
        "max_n_lines": None,  # 300,
        "extractor": {
            "name": "superpoint",
            "nms_radius": 4,
            "detection_threshold": 0.005,
            "max_num_keypoints": None,
            "return_all": True,
        },
        "linetransformer": {
            "max_keylines": -1,
            "min_length": 16,
            "token_distance": 8,
            "nn_threshold": 0.8,
        },
        "ground_truth": {
            "from_pose_depth": False,
            "from_homography": False,
            "th_positive": 3,
            "th_negative": 5,
            "reward_positive": 1,
            "reward_negative": -0.25,
            "is_likelihood_soft": True,
            "p_random_occluders": 0,
            "n_line_sampled_pts": 50,
            "line_perp_dist_th": 5,
            "overlap_th": 0.2,
        },
        "use_lines": True,
        "use_points": True,
    }
    required_data_keys = ["image0", "image1"]

    def _init(self, config):
        self.conf = config
        self.extractor = SuperPoint(config.extractor)
        self.linetransformer = LineTransformer(config.linetransformer)

    def detect_lsd_lines(self, x, max_n_lines=None):
        if max_n_lines is None:
            max_n_lines = self.conf.max_n_lines
        lines = []
        for b in range(len(x)):
            # For each image on batch
            img = (x[b].squeeze().cpu().numpy() * 255).astype(np.uint8)
            if max_n_lines is None:
                b_segs = lsd(img)
            else:
                for s in [0.3, 0.4, 0.5, 0.7, 0.8, 1.0]:
                    b_segs = lsd(img, scale=s)
                    if len(b_segs) >= max_n_lines:
                        break

            segs_length = np.linalg.norm(
                b_segs[:, 2:4] - b_segs[:, 0:2], axis=1
            )
            b_scores = b_segs[:, -1] * np.sqrt(segs_length)
            # Take the most relevant segments with
            indices = np.argsort(-b_scores)
            if max_n_lines is not None:
                indices = indices[:max_n_lines]
            lines.append(
                torch.from_numpy(b_segs[indices, :4].reshape(-1, 2, 2))
            )

        return torch.stack(lines).to(x)

    def _forward(self, data):
        def process_siamese(data, i):
            data_i = {k[:-1]: v for k, v in data.items() if k[-1] == i}
            if self.conf.extractor.name:
                pred_i = self.extractor(data_i)
            else:
                pred_i = {}
                if self.conf.detector.name:
                    pred_i = self.detector(data_i)
                else:
                    for k in [
                        "keypoints",
                        "keypoint_scores",
                        "descriptors",
                        "lines",
                        "line_scores",
                        "line_descriptors",
                        "valid_lines",
                    ]:
                        if k in data_i:
                            pred_i[k] = data_i[k]
                if self.conf.descriptor.name:
                    pred_i = {**pred_i, **self.descriptor({**data_i, **pred_i})}
            return pred_i

        pred0 = process_siamese(data, "0")
        pred1 = process_siamese(data, "1")
        pred = {
            **{k + "0": v for k, v in pred0.items()},
            **{k + "1": v for k, v in pred1.items()},
        }

        if "lines0" not in data:
            image_shape = data["image0"].shape
            if self.conf.auto_min_length:
                self.linetransformer.config["min_length"] = max(
                    16, max(image_shape) / 40
                )
                self.linetransformer.config["token_distance"] = max(
                    8, max(image_shape) / 80
                )

            klines_cv = self.detect_lsd_lines(data["image0"])[0]

            valid_lines0 = data.get("valid_lines0", None)
            klines0 = self.linetransformer.preprocess(
                klines_cv, image_shape, pred0, valid_lines0
            )
            klines0 = self.linetransformer(klines0)
            pred = {**pred, **{k + "0": v for k, v in klines0.items()}}

        if "lines1" not in data:
            image_shape = data["image1"].shape
            if self.conf.auto_min_length:
                self.linetransformer.config["min_length"] = max(
                    16, max(image_shape) / 40
                )
                self.linetransformer.config["token_distance"] = max(
                    8, max(image_shape) / 80
                )

            klines_cv = self.detect_lsd_lines(data["image1"])[0]
            valid_lines1 = data.get("valid_lines1", None)
            klines1 = self.linetransformer.preprocess(
                klines_cv, image_shape, pred1, valid_lines1
            )
            klines1 = self.linetransformer(klines1)
            pred = {**pred, **{k + "1": v for k, v in klines1.items()}}

        if "H" in data:
            if self.conf.use_points:
                assignment, m0, m1 = gt_matches_from_homography(
                    pred["keypoints0"],
                    pred["keypoints1"],
                    **data,
                    pos_th=self.conf.ground_truth.th_positive,
                )
                pred["gt_assignment"] = assignment
                pred["gt_matches0"], pred["gt_matches1"] = m0, m1

            if self.conf.use_lines:
                if (
                    pred["lines0"].shape[2] > 2
                    and self.conf.matcher.supervise_before_nw
                ):
                    # Several points are sampled per line, and we supervise them independently
                    b_size = len(pred["lines0"])
                    (
                        samples_assignment,
                        samples_m0,
                        samples_m1,
                    ) = gt_matches_from_homography(
                        pred["lines0"].reshape(b_size, -1, 2),
                        pred["lines1"].reshape(b_size, -1, 2),
                        **data,
                        pos_th=self.conf.ground_truth.th_positive,
                    )
                    pred["samples_gt_assignment"] = samples_assignment
                    pred["samples_gt_matches0"] = samples_m0
                    pred["samples_gt_matches1"] = samples_m1

                # Compute the GT line association
                (
                    line_assignment,
                    line_m0,
                    line_m1,
                ) = gt_line_matches_from_homography(
                    pred["lines0"],
                    pred["lines1"],
                    pred["valid_lines0"],
                    pred["valid_lines1"],
                    data,
                    self.conf.ground_truth.n_line_sampled_pts,
                    self.conf.ground_truth.line_perp_dist_th,
                    self.conf.ground_truth.overlap_th,
                )
                pred["line_gt_matches0"] = line_m0
                pred["line_gt_matches1"] = line_m1
                pred["line_gt_assignment"] = line_assignment
        elif self.conf.ground_truth.from_pose_depth:
            if self.conf.use_points:
                assignment, m0, m1, d0, d1 = gt_matches_from_pose_depth(
                    pred["keypoints0"],
                    pred["keypoints1"],
                    **data,
                    pos_th=self.conf.ground_truth.th_positive,
                    neg_th=self.conf.ground_truth.th_negative,
                )
                pred["gt_assignment"] = assignment
                pred["gt_matches0"], pred["gt_matches1"] = m0, m1
                pred["gt_depth_keypoints0"], pred["gt_depth_keypoints1"] = (
                    d0,
                    d1,
                )

            if self.conf.use_lines:
                if (
                    pred["lines0"].shape[2] > 2
                    and self.conf.matcher.supervise_before_nw
                ):
                    # Several points are sampled per line, and we supervise them independently
                    b_size = len(pred["lines0"])
                    (
                        samples_assignment,
                        samples_m0,
                        samples_m1,
                    ) = gt_matches_from_pose_depth(
                        pred["lines0"].reshape(b_size, -1, 2),
                        pred["lines1"].reshape(b_size, -1, 2),
                        **data,
                        pos_th=self.conf.ground_truth.th_positive,
                        neg_th=self.conf.ground_truth.th_negative,
                    )[:3]
                    pred["samples_gt_assignment"] = samples_assignment
                    pred["samples_gt_matches0"] = samples_m0
                    pred["samples_gt_matches1"] = samples_m1

                # Compute the GT line association
                (
                    line_assignment,
                    line_m0,
                    line_m1,
                ) = gt_line_matches_from_pose_depth(
                    pred["lines0"],
                    pred["lines1"],
                    pred["valid_lines0"],
                    pred["valid_lines1"],
                    data,
                    self.conf.ground_truth.n_line_sampled_pts,
                    self.conf.ground_truth.line_perp_dist_th,
                    self.conf.ground_truth.overlap_th,
                )
                pred["line_gt_matches0"] = line_m0
                pred["line_gt_matches1"] = line_m1
                pred["line_gt_assignment"] = line_assignment

        data = {**data, **pred}

        # for k in data:
        #     if isinstance(data[k], (list, tuple)) and len(data[k]) > 0 and isinstance(data[k][0], torch.Tensor):
        #         data[k] = torch.stack(data[k])

        ## Feature Point Matching using Nearest Neighbor
        desc0_pnt = data["descriptors0"][0].cpu().numpy()
        desc1_pnt = data["descriptors1"][0].cpu().numpy()
        match_mat_pnt, dist_mat_pnt = nn_matcher(
            desc0_pnt, desc1_pnt, self.conf.nn_threshold, is_mutual_NN=True
        )

        pred["matches_p"] = torch.from_numpy(match_mat_pnt)
        pred["matching_scores_p"] = torch.from_numpy(dist_mat_pnt)

        ## Feature Line Matching using Nearest Neighbor
        desc_slines0 = data["line_descriptors0"].cpu().numpy()
        desc_slines1 = data["line_descriptors1"].cpu().numpy()
        distance_sublines = get_dist_matrix(desc_slines0, desc_slines1)[0]
        distance_matrix = self.linetransformer.subline2keyline(
            distance_sublines,
            data["mat_klines2sublines0"][0],
            data["mat_klines2sublines1"][0],
        )
        match_mat = nn_matcher_distmat(
            distance_matrix,
            self.linetransformer.config["nn_threshold"],
            is_mutual_NN=True,
        )

        pred["matches_l"] = torch.from_numpy(match_mat)
        pred["line_log_assignment"] = torch.from_numpy(distance_matrix)

        assert match_mat.shape[0] == 1
        bool_match_mat = match_mat[0] > 0
        pred["line_matches0"] = np.argmax(bool_match_mat, axis=1)
        pred["line_matches0"][~np.any(bool_match_mat, axis=1)] = (
            UNMATCHED_FEATURE
        )
        pred["line_matches1"] = np.argmax(bool_match_mat, axis=0)
        pred["line_matches1"][~np.any(bool_match_mat, axis=0)] = (
            UNMATCHED_FEATURE
        )
        pred["line_matches0"] = torch.from_numpy(pred["line_matches0"])[None]
        pred["line_matches1"] = torch.from_numpy(pred["line_matches1"])[None]
        lmatch_scores = torch.from_numpy(
            distance_matrix[(0,) + np.where(match_mat[0] > 0)]
        )
        pred["line_match_scores0"] = pred[
            "line_match_scores1"
        ] = -lmatch_scores[None]
        return pred

    def loss(self, pred, data):
        pass

    def metrics(self, pred, data):
        pass
