"""
Runner for undistorting Hypersim images to pinhole cameras.

This is the first step of the structure triangulation pipeline. It:
1. Reads a Hypersim scene and creates a COLMAP reconstruction
2. Undistorts images (copies images since Hypersim is already pinhole)
3. Optionally resizes images to a maximum dimension

Usage:
    python runners/hypersim/undistort_images.py \
        --data_dir <path_to_hypersim_data> \
        --scene_id ai_001_001 \
        --output_dir <output_directory>
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Hypersim import Hypersim
from loader import read_scene_hypersim

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import limap.runners
import limap.util.config as cfgutils
import limap.util.io as limapio


def run_undistort_hypersim(cfg, dataset, scene_id, cam_id=0):
    """Read Hypersim scene and undistort images to pinhole.

    Args:
        cfg: Configuration dictionary with keys:
            - output_dir: Output directory path
            - max_image_dim: (Optional) Maximum image dimension for resizing
        dataset: Hypersim dataset instance
        scene_id: Scene identifier (e.g., "ai_001_001")
        cam_id: Camera ID (default 0)

    Returns:
        Tuple of (undistorted_image_dir, undistorted_model_dir)
    """
    # Read scene and create reconstruction
    image_dir, recon, _ = read_scene_hypersim(
        cfg, dataset, scene_id, cam_id=cam_id, load_depth=False
    )

    output_dir = Path(cfg["output_dir"])
    init_model_dir = output_dir / "init_model"
    limapio.check_makedirs(init_model_dir)

    # Write initial model
    recon.write(init_model_dir)
    print(f"Initial model written to: {init_model_dir}")
    print(f"  - {recon.num_cameras()} cameras")
    print(f"  - {recon.num_images()} images")

    # Undistort images (for pinhole this essentially copies images)
    undistort_dir = output_dir / "undistorted"
    limapio.check_makedirs(undistort_dir)

    undistorted_image_dir, undistorted_model_dir = (
        limap.runners.undistort_images(image_dir, init_model_dir, undistort_dir)
    )
    print(f"Undistorted images written to: {undistorted_image_dir}")
    print(f"Undistorted model written to: {undistorted_model_dir}")

    # Optionally resize images
    if cfg.get("max_image_dim") is not None:
        limap.runners.resize_images_to_max_dim(
            undistorted_image_dir, undistorted_model_dir, cfg["max_image_dim"]
        )
        print(f"Images resized to max dimension: {cfg['max_image_dim']}")

    return undistorted_image_dir, undistorted_model_dir


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="Undistort Hypersim images to pinhole cameras"
    )
    arg_parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default=None,
        help="config file (optional)",
    )
    arg_parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="path to Hypersim data directory",
    )
    arg_parser.add_argument(
        "--scene_id",
        type=str,
        default="ai_001_001",
        help="scene identifier",
    )
    arg_parser.add_argument(
        "--cam_id",
        type=int,
        default=0,
        help="camera ID",
    )
    arg_parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="output directory",
    )
    arg_parser.add_argument(
        "--max_image_dim",
        type=int,
        default=None,
        help="maximum image dimension (optional)",
    )
    arg_parser.add_argument(
        "--input_n_views",
        type=int,
        default=100,
        help="number of views to process",
    )
    arg_parser.add_argument(
        "--input_stride",
        type=int,
        default=1,
        help="stride for selecting views",
    )

    args = arg_parser.parse_args()

    # Build config from args
    if args.config_file is not None:
        cfg = cfgutils.load_config(args.config_file)
    else:
        cfg = {}

    # Override with command line arguments
    cfg["data_dir"] = args.data_dir
    cfg["scene_id"] = args.scene_id
    cfg["cam_id"] = args.cam_id
    cfg["output_dir"] = args.output_dir
    cfg["input_n_views"] = args.input_n_views
    cfg["input_stride"] = args.input_stride
    if args.max_image_dim is not None:
        cfg["max_image_dim"] = args.max_image_dim

    return cfg


def main():
    cfg = parse_args()
    dataset = Hypersim(cfg["data_dir"])
    undistorted_image_dir, undistorted_model_dir = run_undistort_hypersim(
        cfg, dataset, cfg["scene_id"], cam_id=cfg["cam_id"]
    )
    print("\nDone!")
    print(f"Undistorted images: {undistorted_image_dir}")
    print(f"Undistorted model: {undistorted_model_dir}")


if __name__ == "__main__":
    main()
