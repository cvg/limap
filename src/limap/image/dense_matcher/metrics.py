from .base_dense_matcher import DenseMatchingResult

import torch
import torch.nn.functional as F
import numpy as np
from limap.geometry import Line2d


def compute_point_distance_matrix(
    match: DenseMatchingResult,
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    coords_1 = torch.from_numpy(source_points).to(match.device).float()
    coords_2 = torch.from_numpy(target_points).to(match.device).float()
    coords = match.to_normalized_coordinates(
        coords_1, match.source_shape[0], match.source_shape[1]
    )
    coords_to_2 = F.grid_sample(
        match.warp.permute(2, 0, 1)[None],
        coords[None, None],
        align_corners=False,
        mode="bilinear",
    )[0, :, 0].mT
    coords_to_2 = match.to_unnormalized_coordinates(
        coords_to_2,
        match.target_shape[0],
        match.target_shape[1],
    )
    cert_to_2 = F.grid_sample(
        match.certainty[None, None],
        coords[None, None],
        align_corners=False,
        mode="bilinear",
    )[0, 0, 0]
    dists = torch.cdist(coords_to_2, coords_2, p=2)
    bad_sample = cert_to_2 < match.sample_threshold
    dists[bad_sample] = 10000.0
    return dists.cpu().numpy()


def compute_line_distance_matrix(
    match: DenseMatchingResult,
    source_lines: list[Line2d],
    target_lines: list[Line2d],
    n_samples: int = 21,
    min_segment_overlap: float = 0.2,
) -> np.ndarray:
    # Get point samples along lines
    lines1_array = np.array([line.as_array() for line in source_lines])
    segs1 = torch.from_numpy(lines1_array).to(match.device).float()
    n_segs1 = segs1.shape[0]
    ratio = torch.linspace(0, 1, n_samples, device=match.device)
    ratio = ratio[:, None].repeat(1, 2)
    coords_1 = ratio * segs1[:, [0]].repeat(1, n_samples, 1) + (
        1 - ratio
    ) * segs1[:, [1]].repeat(1, n_samples, 1)
    coords_1 = coords_1.reshape(-1, 2)
    coords = match.to_normalized_coordinates(
        coords_1, match.source_shape[0], match.source_shape[1]
    )
    coords_to_2 = F.grid_sample(
        match.warp.permute(2, 0, 1)[None],
        coords[None, None],
        align_corners=False,
        mode="bilinear",
    )[0, :, 0].mT
    coords_to_2 = match.to_unnormalized_coordinates(
        coords_to_2,
        match.target_shape[0],
        match.target_shape[1],
    )
    cert_to_2 = F.grid_sample(
        match.certainty[None, None],
        coords[None, None],
        align_corners=False,
        mode="bilinear",
    )[0, 0, 0]
    cert_to_2 = cert_to_2.reshape(-1, n_samples)

    # Get projections
    lines2_array = np.array([line.as_array() for line in target_lines])
    segs2 = torch.from_numpy(lines2_array).to(match.device).float()
    n_segs2 = segs2.shape[0]
    starts2, ends2 = segs2[:, 0], segs2[:, 1]
    directions = ends2 - starts2
    directions /= torch.norm(directions, dim=1, keepdim=True)
    starts2_proj = (starts2 * directions).sum(1)
    ends2_proj = (ends2 * directions).sum(1)

    # Get line equations
    starts_homo = torch.cat([starts2, torch.ones_like(segs2[:, [0], 0])], 1)
    ends_homo = torch.cat([ends2, torch.ones_like(segs2[:, [0], 0])], 1)
    lines2_homo = torch.cross(starts_homo, ends_homo, dim=1)
    lines2_homo /= torch.norm(lines2_homo[:, :2], dim=1)[:, None].repeat(1, 3)

    # Compute distance
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
    dists = dists.reshape(n_segs1, n_samples, n_segs2).permute(0, 2, 1)
    has_overlap = (
        has_overlap.reshape(n_segs1, n_samples, n_segs2)
        .permute(0, 2, 1)
        .to(torch.bool)
    )

    # Get active lines for each target
    sample_thresh = match.sample_threshold
    good_sample = cert_to_2 >= sample_thresh
    good_sample = torch.logical_and(
        good_sample[:, None].repeat(1, has_overlap.shape[1], 1),
        has_overlap,
    )
    sample_weight = good_sample.to(torch.float)
    sample_weight_sum = sample_weight.sum(2)
    overlap = sample_weight_sum / n_samples
    sample_weight[sample_weight_sum > 0] /= sample_weight_sum[
        sample_weight_sum > 0
    ][:, None].repeat(1, sample_weight.shape[2])

    # Get weighted dists
    weighted_dists = (dists * sample_weight).sum(2)
    weighted_dists[overlap < min_segment_overlap] = (
        10000.0  # ensure that there is overlap
    )
    return weighted_dists.cpu().numpy()


def compute_mask_overlap_matrix(
    match: DenseMatchingResult,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    min_num_pixels: int = 200,
) -> np.ndarray:
    """
    Calculate overlap (intersection size / smaller region size)
    on the source images by inverse warping the target image.
    """
    device = match.device

    # Load masks
    src_mask = torch.from_numpy(source_mask).to(device).long()
    tgt_mask = torch.from_numpy(target_mask).to(device).long()

    Hs, Ws = match.source_shape
    Ht, Wt = match.target_shape

    N1 = int(src_mask.max().item())  # source labels (1..N1)
    N2 = int(tgt_mask.max().item())  # target labels (1..N2)

    if N1 == 0 or N2 == 0:
        return np.zeros((N1, N2), dtype=np.float32)

    # Build source pixel grid
    ys, xs = torch.meshgrid(
        torch.arange(Hs, device=device),
        torch.arange(Ws, device=device),
        indexing="ij",
    )
    src_pix = torch.stack([xs, ys], dim=-1).float()  # (Hs,Ws,2)

    src_norm = match.to_normalized_coordinates(src_pix, Hs, Ws)[None]
    warp_field = match.warp.permute(2, 0, 1)[None]

    # Warp source→target coordinates
    warped_xy = F.grid_sample(
        warp_field,
        src_norm,
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)

    warped_xy = match.to_unnormalized_coordinates(warped_xy, Ht, Wt)
    warped_norm = match.to_normalized_coordinates(warped_xy, Ht, Wt)
    warped_norm = warped_norm[None]

    # Warp *target* mask into *source* frame with bilinear sampling
    #    → this gives SOFT mask values
    warped_tgt_soft = F.grid_sample(
        tgt_mask.float()[None, None],  # (1,1,Ht,Wt)
        warped_norm,  # (1,Hs,Ws,2)
        mode="bilinear",  # soft
        align_corners=False,
    )[0, 0]  # (Hs,Ws)

    # Certainty sampling (same logic as your point function)
    certainty_src = F.grid_sample(
        match.certainty[None, None].float(),
        src_norm,
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    valid_mask = (certainty_src >= match.sample_threshold).float()

    # One-hot source mask (binary for each label)
    # shape: (N1,Hs,Ws)
    src_onehot = F.one_hot(src_mask.clamp(0, N1), N1 + 1)[..., 1:].permute(
        2, 0, 1
    )
    src_onehot = src_onehot.float() * valid_mask  # remove uncertain pixels

    # Soft one-hot for warped target mask
    #    Use triangular kernel: soft[j] = max(1 - |value - j|, 0)
    labels = torch.arange(1, N2 + 1, device=device)[:, None, None]  # (N2,1,1)
    warped_val = warped_tgt_soft[None]  # (1,Hs,Ws)
    tgt_soft_onehot = (1 - (warped_val - labels).abs()).clamp(min=0, max=1)
    tgt_soft_onehot = tgt_soft_onehot * valid_mask  # ignore uncertain regions

    # Flatten masks
    src_f = src_onehot.reshape(N1, -1)  # (N1,HW)
    tgt_f = tgt_soft_onehot.reshape(N2, -1)  # (N2,HW)

    # Compute Overlap matrix
    #    overlap(i,j) = intersection / min(area_src, area_tgt)
    intersection = src_f @ tgt_f.t()  # (N1,N2)

    area_src = src_f.sum(dim=1, keepdim=True)  # (N1,1)
    area_tgt = tgt_f.sum(dim=1)[None, :]  # (1,N2)

    min_area = torch.minimum(area_src, area_tgt)  # (N1,N2)

    overlap = torch.where(
        min_area > 0,
        intersection / min_area,
        torch.zeros_like(intersection),
    )
    overlap = torch.where(
        intersection >= min_num_pixels,
        overlap,
        torch.zeros_like(overlap),
    )
    return overlap.cpu().numpy()
