import os
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

import limap.util.io as limapio

from ..base_matcher import BaseMatcher, BaseMatcherOptions
from .dense_matcher import BaseDenseMatcher


class BaseDenseLineMatcherOptions(NamedTuple):
    n_samples: int = 21
    segment_percentage_th: float = 0.2
    device = "cuda"
    pixel_th: float = 10.0


class BaseDenseLineMatcher(BaseMatcher):
    def __init__(
        self,
        extractor,
        dense_matcher,
        dense_options=BaseDenseLineMatcherOptions(),
        options=BaseMatcherOptions(),
    ):
        super().__init__(extractor, options)
        assert self.extractor.get_module_name() == "dense_naive"
        assert dense_options.n_samples >= 2
        self.dense_options = dense_options
        assert isinstance(dense_matcher, BaseDenseMatcher)
        self.dense_matcher = dense_matcher

    def get_module_name(self):
        raise NotImplementedError

    def match_pair(self, descinfo1, descinfo2):
        if self.topk == 0:
            return self.match_segs_with_descinfo(descinfo1, descinfo2)
        else:
            return self.match_segs_with_descinfo_topk(
                descinfo1, descinfo2, topk=self.topk
            )

    def compute_distance_one_direction(
        self, descinfo1, descinfo2, warp_1to2, cert_1to2
    ):
        # get point samples along lines
        segs1 = torch.from_numpy(descinfo1["lines"]).to(
            self.dense_options.device
        )
        n_segs1 = segs1.shape[0]
        ratio = torch.linspace(
            0, 1, self.dense_options.n_samples, device=self.dense_options.device
        )
        ratio = ratio[:, None].repeat(1, 2)
        coords_1 = ratio * segs1[:, [0]].repeat(
            1, self.dense_options.n_samples, 1
        ) + (1 - ratio) * segs1[:, [1]].repeat(
            1, self.dense_options.n_samples, 1
        )
        coords_1 = coords_1.reshape(-1, 2)
        coords = self.dense_matcher.to_normalized_coordinates(
            coords_1, descinfo1["image_shape"][0], descinfo1["image_shape"][1]
        )
        coords_to_2 = F.grid_sample(
            warp_1to2.permute(2, 0, 1)[None],
            coords[None, None],
            align_corners=False,
            mode="bilinear",
        )[0, :, 0].mT
        coords_to_2 = self.dense_matcher.to_unnormalized_coordinates(
            coords_to_2,
            descinfo2["image_shape"][0],
            descinfo2["image_shape"][1],
        )
        cert_to_2 = F.grid_sample(
            cert_1to2[None, None],
            coords[None, None],
            align_corners=False,
            mode="bilinear",
        )[0, 0, 0]
        cert_to_2 = cert_to_2.reshape(-1, self.dense_options.n_samples)

        # get projections
        segs2 = torch.from_numpy(descinfo2["lines"]).to(
            self.dense_options.device
        )
        n_segs2 = segs2.shape[0]
        starts2, ends2 = segs2[:, 0], segs2[:, 1]
        directions = ends2 - starts2
        directions /= torch.norm(directions, dim=1, keepdim=True)
        starts2_proj = (starts2 * directions).sum(1)
        ends2_proj = (ends2 * directions).sum(1)

        # get line equations
        starts_homo = torch.cat([starts2, torch.ones_like(segs2[:, [0], 0])], 1)
        ends_homo = torch.cat([ends2, torch.ones_like(segs2[:, [0], 0])], 1)
        lines2_homo = torch.cross(starts_homo, ends_homo)
        lines2_homo /= torch.norm(lines2_homo[:, :2], dim=1)[:, None].repeat(
            1, 3
        )

        # compute distance
        coords_to_2_homo = torch.cat(
            [coords_to_2, torch.ones_like(coords_to_2[:, [0]])], 1
        )
        coords_proj = torch.matmul(coords_to_2, directions.T)
        dists = torch.abs(torch.matmul(coords_to_2_homo, lines2_homo.T))
        has_overlap = torch.where(
            coords_proj > starts2_proj,
            torch.ones_like(dists),
            torch.zeros_like(dists),
        )
        has_overlap = torch.where(
            coords_proj < ends2_proj, has_overlap, torch.zeros_like(dists)
        )
        dists = dists.reshape(
            n_segs1, self.dense_options.n_samples, n_segs2
        ).permute(0, 2, 1)
        has_overlap = (
            has_overlap.reshape(n_segs1, self.dense_options.n_samples, n_segs2)
            .permute(0, 2, 1)
            .to(torch.bool)
        )

        # get active lines for each target
        sample_thresh = self.dense_matcher.get_sample_thresh()
        good_sample = cert_to_2 > sample_thresh
        good_sample = torch.logical_and(
            good_sample[:, None].repeat(1, has_overlap.shape[1], 1),
            has_overlap,
        )
        sample_weight = good_sample.to(torch.float)
        sample_weight_sum = sample_weight.sum(2)
        overlap = sample_weight_sum / self.dense_options.n_samples
        sample_weight[sample_weight_sum > 0] /= sample_weight_sum[
            sample_weight_sum > 0
        ][:, None].repeat(1, sample_weight.shape[2])

        # get weighted dists
        weighted_dists = (dists * sample_weight).sum(2)
        weighted_dists[overlap < self.dense_options.segment_percentage_th] = (
            10000.0  # ensure that there is overlap
        )
        return weighted_dists, overlap

    def match_segs_with_descinfo(self, descinfo1, descinfo2):
        img1 = descinfo1["camview"].read_image()
        img2 = descinfo2["camview"].read_image()
        (
            warp_1to2,
            cert_1to2,
            warp_2to1,
            cert_2to1,
        ) = self.dense_matcher.get_warping_symmetric(img1, img2)

        # compute distance and overlap
        dists_1to2, overlap_1to2 = self.compute_distance_one_direction(
            descinfo1, descinfo2, warp_1to2, cert_1to2
        )
        dists_2to1, overlap_2to1 = self.compute_distance_one_direction(
            descinfo2, descinfo1, warp_2to1, cert_2to1
        )
        # overlap = torch.maximum(overlap_1to2, overlap_2to1.T)
        dists = torch.where(
            overlap_1to2 > overlap_2to1.T, dists_1to2, dists_2to1.T
        )

        # match: one-way nearest neighbor
        # TODO: one-to-many matching
        inds_1, inds_2 = torch.nonzero(
            dists
            == dists.min(dim=-1, keepdim=True).values
            * (dists <= self.dense_options.pixel_th),
            as_tuple=True,
        )
        inds_1 = inds_1.detach().cpu().numpy()
        inds_2 = inds_2.detach().cpu().numpy()
        matches_t = np.stack([inds_1, inds_2], axis=1)
        return matches_t

    def match_segs_with_descinfo_topk(self, descinfo1, descinfo2, topk=10):
        raise NotImplementedError


class RoMaLineMatcher(BaseDenseLineMatcher):
    def __init__(
        self,
        extractor,
        mode="outdoor",
        dense_options=BaseDenseLineMatcherOptions(),
        options=BaseMatcherOptions(),
    ):
        from .dense_matcher import RoMa

        roma_matcher = RoMa(mode=mode, device=dense_options.device)
        super().__init__(
            extractor,
            roma_matcher,
            dense_options=dense_options,
            options=options,
        )

    def get_module_name(self):
        return "dense_roma"
