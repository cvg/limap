"""Visualize 2D image features from a structure reconstruction.

Usage:
    python visualize_image_features.py \
        --input_dir <reconstruction_dir> \
        --image_dir <undistorted_images_dir> \
        --output_dir <output_dir> \
        [--group_workspace <group_description_dir>] \
        [--image_ids 28 204 303]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

import limap.util.io as limapio
import limap.visualize as limapvis
from limap.scene import HolisticReconstruction


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize 2D image features")
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        required=True,
        help="Path to structure reconstruction directory",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to undistorted images directory",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--group_workspace",
        type=str,
        default=None,
        help="Path to group_description directory for mask overlay",
    )
    parser.add_argument(
        "--image_ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific image IDs to visualize (default: all)",
    )
    parser.add_argument(
        "--line_thickness",
        type=int,
        default=1,
        help="Line thickness for drawing",
    )
    parser.add_argument(
        "--show_endpoints",
        action="store_true",
        help="Show line endpoints",
    )
    parser.add_argument(
        "--show_masks",
        action="store_true",
        help="Overlay group masks (requires --group_workspace)",
    )
    parser.add_argument(
        "--mask_alpha",
        type=float,
        default=0.3,
        help="Mask overlay transparency (0=invisible, 1=opaque)",
    )
    parser.add_argument(
        "--mask_types",
        type=str,
        nargs="+",
        default=["plane", "cylinder", "sphere"],
        choices=["plane", "cylinder", "sphere"],
        help="Which mask types to show",
    )
    return parser.parse_args()


def get_mask_colors(n_labels: int) -> np.ndarray:
    """Generate distinct colors for mask labels."""
    import seaborn as sns

    colors = sns.color_palette("tab10", n_colors=max(n_labels, 10))
    return (np.array(colors) * 255).astype(np.uint8)


def load_group_mask(
    group_workspace: Path,
    image_id: int,
    mask_type: str,
) -> np.ndarray | None:
    """Load segmentation mask for a specific image and group type.

    Args:
        group_workspace: Path to group_description directory
        image_id: Image ID
        mask_type: One of "plane", "cylinder", "sphere"

    Returns:
        (H, W) mask with label indices, or None if not found
    """
    type_dir = f"{mask_type}_detections"
    mask_path = (
        group_workspace / type_dir / "seg_labels" / f"mask_{image_id}.npy"
    )
    if not mask_path.exists():
        return None
    return limapio.read_npy(mask_path)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.3,
    colors: np.ndarray | None = None,
) -> np.ndarray:
    """Overlay colored mask on image.

    Args:
        image: (H, W, 3) BGR image
        mask: (H, W) label mask (0 = background)
        alpha: Transparency for overlay
        colors: (N, 3) RGB colors for each label

    Returns:
        Image with mask overlay
    """
    result = image.copy()
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]  # Skip background

    if colors is None:
        colors = get_mask_colors(int(unique_labels.max()) + 1)

    for label in unique_labels:
        label_mask = mask == label
        color_idx = (label - 1) % len(colors)
        color_bgr = colors[color_idx][::-1]  # RGB to BGR
        result[label_mask] = (
            (1 - alpha) * result[label_mask] + alpha * color_bgr
        ).astype(np.uint8)

    return result


def visualize_image(
    image_id: int,
    hrecon: HolisticReconstruction,
    image_dir: Path,
    group_workspace: Path | None = None,
    line_thickness: int = 2,
    show_endpoints: bool = True,
    show_masks: bool = False,
    mask_alpha: float = 0.3,
    mask_types: list[str] = None,
) -> np.ndarray | None:
    """Visualize features for a single image.

    Args:
        image_id: Image ID to visualize
        hrecon: HolisticReconstruction instance
        image_dir: Path to images directory
        group_workspace: Optional path to group_description for masks
        line_thickness: Line drawing thickness
        show_endpoints: Whether to draw endpoints
        show_masks: Whether to overlay masks
        mask_alpha: Mask transparency
        mask_types: Which mask types to show

    Returns:
        Visualized image or None if image not found
    """
    point_recon = hrecon.point_recon
    structure_recon = hrecon.structure_recon

    if image_id not in point_recon.images:
        print(f"Image ID {image_id} not found in reconstruction")
        return None

    image_info = point_recon.images[image_id]
    image_name = image_info.name
    image_path = image_dir / image_name

    if not image_path.exists():
        print(f"Image file not found: {image_path}")
        return None

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    # Overlay masks if requested
    if show_masks and group_workspace is not None:
        if mask_types is None:
            mask_types = ["plane", "cylinder", "sphere"]
        for mask_type in mask_types:
            mask = load_group_mask(group_workspace, image_id, mask_type)
            if mask is not None:
                n_labels = int(mask.max())
                if n_labels > 0:
                    img = overlay_mask(img, mask, alpha=mask_alpha)

    # Draw 2D lines if available
    if image_id in structure_recon.structures2d:
        structure2d = structure_recon.structures2d[image_id]
        lines = structure2d.lines
        if len(lines) > 0:
            img = limapvis.draw_2d_lines(
                img,
                lines,
                color=(0, 255, 0),  # Green for lines
                thickness=line_thickness,
                endpoints=show_endpoints,
            )

    # Draw 2D points (from COLMAP)
    points_2d = [p.xy for p in image_info.points2D]
    if points_2d:
        points_array = np.array(points_2d)
        img = limapvis.draw_2d_points(
            img,
            points_array,
            color=(255, 0, 0),  # Blue for points (BGR)
            thickness=1,
        )

    return img


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    group_workspace = (
        Path(args.group_workspace) if args.group_workspace else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reconstruction
    print(f"Loading reconstruction from {input_dir}")
    hrecon = HolisticReconstruction(str(input_dir))
    point_recon = hrecon.point_recon
    structure_recon = hrecon.structure_recon

    print(f"  Images: {len(point_recon.images)}")
    print(f"  Points: {len(point_recon.points3D)}")
    print(f"  Lines: {len(structure_recon.lines3D)}")
    print(f"  Structures2D: {len(structure_recon.structures2d)}")

    # Determine which images to visualize
    image_ids = args.image_ids or list(point_recon.images.keys())

    print(f"\nVisualizing {len(image_ids)} images to {output_dir}")

    for image_id in image_ids:
        print(f"Processing image {image_id}...", end=" ")
        img = visualize_image(
            image_id,
            hrecon,
            image_dir,
            group_workspace=group_workspace,
            line_thickness=args.line_thickness,
            show_endpoints=args.show_endpoints,
            show_masks=args.show_masks,
            mask_alpha=args.mask_alpha,
            mask_types=args.mask_types,
        )

        if img is None:
            print("skipped")
            continue

        # Save to file
        image_name = point_recon.images[image_id].name
        out_name = Path(image_name).stem + "_features.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), img)
        print(f"saved to {out_name}")

    print(f"\nDone! Saved {len(image_ids)} visualizations to {output_dir}")


if __name__ == "__main__":
    main()
