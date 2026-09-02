import numpy as np
from scipy.ndimage import distance_transform_edt
import limap.geometry
import limap.scene


def convert_plane_mask_to_groups2d(
    plane_mask: np.ndarray,
    points: np.ndarray,
    lines: list[limap.geometry.Line2d],
    min_line_overlap_length: float = 40.0,
    dilation_radius: float = 2.0,
    normal_map: np.ndarray | None = None,
) -> list[limap.scene.Group2d]:
    """
    Convert a plane segmentation mask to Group2d objects.

    Args:
        plane_mask: (H, W) int array with plane labels (0 = background)
        points: (N, 2) array of 2D point coordinates
        lines: list of Line2d objects
        min_line_overlap_length: minimum overlap length to associate a line
        dilation_radius: radius for dilating plane regions
        normal_map: optional (H, W, 3) array of per-pixel normal vectors.
                    If provided, average normals are computed per plane and
                    stored as Group2d params.

    Returns:
        List of Group2d objects, one per plane label (1..num_planes)
    """
    H, W = plane_mask.shape
    num_planes = int(plane_mask.max())

    # Early return if no planes detected
    if num_planes == 0:
        return []

    # Create groups for labels 1..num_planes
    groups = [
        limap.scene.Group2d(limap.geometry.GroupType.PLANE)
        for _ in range(num_planes)
    ]

    # Extract per-plane normals if normal map is provided
    if normal_map is not None:
        for k in range(1, num_planes + 1):
            mask_k = plane_mask == k
            if not np.any(mask_k):
                continue
            # Extract normals for pixels in this plane
            plane_normals = normal_map[mask_k]  # (N, 3)
            # Compute average normal
            avg_normal = np.mean(plane_normals, axis=0)
            norm = np.linalg.norm(avg_normal)
            if norm > 1e-6:
                avg_normal = avg_normal / norm
            else:
                avg_normal = np.array([0.0, 0.0, 1.0])
            # Store as Group2d params (camera-frame unit normal)
            groups[k - 1].params = avg_normal.tolist()

    # -----------------------------------------
    # Precompute per-plane distance transforms
    # dist_maps[k][y, x] = distance in pixels to plane k
    # -----------------------------------------
    if dilation_radius > 0:
        dist_maps = []
        for k in range(1, num_planes + 1):
            mask_k = (plane_mask == k).astype(np.uint8)
            # EDT of inverted mask gives distance to the nearest plane pixel
            dist = distance_transform_edt(1 - mask_k).astype(np.float16)
            dist_maps.append(dist)
        dist_maps = np.stack(dist_maps, axis=0)
    else:
        dist_maps = None

    # ---- Point assignment (vectorized) ----
    if points.ndim < 2 or points.shape[0] == 0:
        points = np.empty((0, 2), dtype=np.float64)
    xy = np.rint(points).astype(int)
    xs, ys = xy[:, 0], xy[:, 1]
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    valid_pids = np.where(valid)[0]

    if len(valid_pids) > 0:
        xs_valid = xs[valid_pids]
        ys_valid = ys[valid_pids]

        if dilation_radius == 0:
            # Vectorized: get labels for all valid points at once
            lbls = plane_mask[ys_valid, xs_valid]
            # Build point lists per plane using bincount-style grouping
            for k in range(1, num_planes + 1):
                pids_for_plane = valid_pids[lbls == k].tolist()
                if pids_for_plane:
                    groups[k - 1].add_points(pids_for_plane)
        else:
            # Vectorized multi-plane: (K, N_valid) distances
            dvals = dist_maps[:, ys_valid, xs_valid]
            close_mask = dvals <= dilation_radius  # (K, N_valid) bool

            for k in range(num_planes):
                pids_for_plane = valid_pids[close_mask[k]].tolist()
                if pids_for_plane:
                    groups[k].add_points(pids_for_plane)

    # ---- Line assignment (batched) ----
    # Precompute line data to avoid repeated attribute access
    num_lines = len(lines)
    if num_lines == 0:
        return groups

    # Collect per-plane line assignments
    plane_line_ids = [[] for _ in range(num_planes)]

    for lid, ln in enumerate(lines):
        x1, y1 = ln.start
        x2, y2 = ln.end

        # geometric length
        L = ln.length()
        if min_line_overlap_length > L:
            continue

        N = int(min(40, max(2, np.ceil(L))))

        xs_line = np.linspace(x1, x2, N, dtype=np.float32)
        ys_line = np.linspace(y1, y2, N, dtype=np.float32)

        xi = np.rint(xs_line).astype(int)
        yi = np.rint(ys_line).astype(int)

        valid_mask = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        xi = xi[valid_mask]
        yi = yi[valid_mask]

        n_valid = len(xi)
        if n_valid == 0:
            continue

        if dilation_radius == 0:
            lbls = plane_mask[yi, xi]
            counts = np.bincount(lbls, minlength=num_planes + 1)
            # Vectorized threshold check for all planes
            overlap_lengths = (counts[1 : num_planes + 1] / n_valid) * L
            for k in np.where(overlap_lengths >= min_line_overlap_length)[0]:
                plane_line_ids[k].append(lid)
        else:
            dvals = dist_maps[:, yi, xi]  # (K, N_valid)
            plane_counts = (dvals <= dilation_radius).sum(axis=1)
            overlap_lengths = (plane_counts / n_valid) * L
            for k in np.where(overlap_lengths >= min_line_overlap_length)[0]:
                plane_line_ids[k].append(lid)

    # Batch add all lines per plane
    for k in range(num_planes):
        if plane_line_ids[k]:
            groups[k].add_lines(plane_line_ids[k])

    return groups
