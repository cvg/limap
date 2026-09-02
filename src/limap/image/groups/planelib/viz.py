from pathlib import Path
import numpy as np

import cv2
import seaborn as sns
from tqdm import tqdm

import limap.scene
import limap.util.io as limapio
import pycolmap
from pycolmap import logging


def visualize_top_components(
    segmentation: np.ndarray,
    top_n: int = 10,
    ignore_label: int = 0,
    colormap: str = "hls",
    background_image: np.ndarray | None = None,
    alpha: float = 0.5,
):
    """
    Visualize top-n largest plane segments with distinct colors.

    Args:
        segmentation: (H, W) segmentation map with plane IDs
        top_n: Number of top planes to show
        ignore_label: Label to treat as background (default 0)
        colormap: Seaborn color palette name
        background_image: Optional (H, W, 3) BGR image to blend with
        alpha: Blending factor for overlay (0=background only, 1=colors only)

    Returns:
        colored_seg: (H, W, 3) BGR image
    """
    # Find top-n largest components using np.unique
    unique_labels, counts = np.unique(segmentation, return_counts=True)

    # Filter out ignore label
    valid_mask = unique_labels != ignore_label
    unique_labels = unique_labels[valid_mask]
    counts = counts[valid_mask]

    # Sort by size and take top-n
    sorted_indices = np.argsort(-counts)
    top_n_labels = unique_labels[sorted_indices[:top_n]]

    # Prepare color palette
    palette = sns.color_palette(colormap, n_colors=top_n)
    palette_rgb = (np.array(palette) * 255).astype(np.uint8)

    # Create visualization
    colored_seg = np.zeros((*segmentation.shape, 3), dtype=np.uint8)

    # Assign colors to top-n planes
    plane_mask = np.zeros(segmentation.shape, dtype=bool)
    for idx, label in enumerate(top_n_labels):
        mask = segmentation == label
        colored_seg[mask] = palette_rgb[idx]
        plane_mask |= mask

    # Convert RGB to BGR for cv2.imwrite compatibility
    colored_seg = cv2.cvtColor(colored_seg, cv2.COLOR_RGB2BGR)

    # Blend with background image if provided
    if background_image is not None:
        blended = background_image.copy()
        blended[plane_mask] = (
            alpha * colored_seg[plane_mask]
            + (1 - alpha) * background_image[plane_mask]
        ).astype(np.uint8)
        return blended

    return colored_seg


def visualize_single_component(
    segmentation: np.ndarray,
    target_label: int,
    color: tuple[int, ...] = (255, 0, 0),
    background_image: bool | None = None,
    alpha: float = 0.6,
):
    """
    visualization of only one connected component.
    segmentation : 2D array of labels
    target_label : label id to highlight
    color        : RGB tuple (default red)
    """
    H, W = segmentation.shape
    color_image = np.zeros((H, W, 3), dtype=np.uint8)

    # mask of the target component
    mask = segmentation == target_label

    # assign color to that component
    color_image[mask] = color

    # convert to RGB to be safe
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    if background_image is not None:
        # Ensure background is RGB
        if background_image.ndim == 2:
            bg_rgb = cv2.cvtColor(
                background_image.astype(np.uint8), cv2.COLOR_GRAY2RGB
            )
        else:
            bg_rgb = background_image.astype(np.uint8)
    blended = bg_rgb.copy()
    blended[mask] = (
        alpha * color_image[mask] + (1 - alpha) * bg_rgb[mask]
    ).astype(np.uint8)
    return blended


def visualize_plane_tracks(
    db_path: Path,
    structure_db_path: Path,
    workspace_path: Path,
    image_dir: Path,
    neighbors: dict,
    recon,
    output_dir: Path,
):
    pycolmap.Database.open(db_path)
    stdb = limap.scene.StructureDatabase.open(structure_db_path)
    stdbc = limap.scene.StructureDatabaseCache.create(stdb)

    group_corr_graph = stdbc.structure_correspondence_graph.group_graph

    # --- Initial component visualizations ---
    for img_id, _neighbor in tqdm(neighbors.items()):
        mask = limapio.read_npy(
            workspace_path
            / "group_description/plane_mask"
            / f"mask_{img_id}.npy"
        ).astype(int)

        seg_vis = visualize_top_components(mask)

        (output_dir / "plane_vis").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(output_dir / "plane_vis" / f"vis_mask_{img_id}.png"), seg_vis
        )

    # --- Detailed correspondence visualization ---
    image_names = {
        img_id: image_dir / img.name for img_id, img in recon.images.items()
    }

    start_ids = limapio.read_npy(
        workspace_path / "group_description/start_ids.npy"
    ).item()

    for counter, (img_id, _neighbor) in enumerate(neighbors.items()):
        logging.info(f"image counter = {counter}")
        if counter >= 10:
            break

        mask = limapio.read_npy(
            workspace_path
            / "group_description/plane_mask"
            / f"mask_{img_id}.npy"
        ).astype(int)

        for base_group_id in np.arange(1, 6):
            out_dir = output_dir / "tracks" / f"corr_{img_id}_{base_group_id}"
            out_dir.mkdir(parents=True, exist_ok=True)

            color_vis = visualize_single_component(
                mask,
                base_group_id,
                background_image=cv2.imread(str(image_names[img_id])),
            )
            cv2.imwrite(str(out_dir / "base.png"), color_vis)

            # Convert base group ID to global group ID
            group_id = base_group_id + (start_ids["PLANE"][img_id] - 1)

            corrs = group_corr_graph.extract_transitive_correspondences(
                img_id, group_id, 1
            )

            for _idx, corr in enumerate(corrs):
                corr_img_id = corr.image_id
                ng_group_id = (
                    corr.point2D_idx - start_ids["PLANE"][corr_img_id] + 1
                )

                ng_mask = limapio.read_npy(
                    workspace_path
                    / "group_description/plane_mask"
                    / f"mask_{corr_img_id}.npy"
                ).astype(int)

                color_vis = visualize_single_component(
                    ng_mask,
                    ng_group_id,
                    background_image=cv2.imread(str(image_names[corr_img_id])),
                )

                cv2.imwrite(
                    str(out_dir / f"corr_{corr_img_id}_{ng_group_id}.png"),
                    color_vis,
                )
