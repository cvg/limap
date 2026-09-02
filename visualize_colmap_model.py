import numpy as np
import pycolmap

import open3d as o3d
import limap.visualize as limapvis


def parse_args():
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="visualize colmap model using Open3D backend"
    )
    arg_parser.add_argument(
        "-i", "--input_dir", type=str, required=True, help="input colmap folder"
    )
    arg_parser.add_argument(
        "--disable_robust_ranges",
        action="store_true",
        help="whether to use computed robust ranges",
    )
    arg_parser.add_argument(
        "--point_size",
        type=float,
        default=2.0,
        help="Point size",
    )
    arg_parser.add_argument(
        "--line_width",
        type=float,
        default=2.0,
        help="Line width for camera frustums",
    )
    arg_parser.add_argument(
        "--cam_scale",
        type=float,
        default=0.1,
        help="scale of the camera geometry",
    )
    arg_parser.add_argument(
        "--reproj_error_thresh",
        type=float,
        default=2.0,
        help="reprojection error threshold",
    )
    arg_parser.add_argument(
        "--min_track_length",
        type=int,
        default=2,
        help="Minimum track length (number of observations) for points.",
    )
    arg_parser.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="Save screenshot to this path (PNG). Press 'S' to save.",
    )
    args = arg_parser.parse_args()
    return args


def vis_colmap_reconstruction(
    args, recon: pycolmap.Reconstruction, ranges=None
):
    # Use legacy o3d.visualization.Visualizer for better visual quality.
    # This provides more predictable point_size and line_width behavior
    # compared to the newer Filament-based draw() API.
    # Filter points by track length and reprojection error
    points3D = [
        p
        for p in recon.points3D.values()
        if p.track.length() >= args.min_track_length
    ]
    pts = np.array([p.xyz for p in points3D], dtype=np.float32)
    if args.reproj_error_thresh > 0.0:
        errs = np.array([p.error for p in points3D], dtype=np.float32)
        mask = errs <= args.reproj_error_thresh
        pts = pts[mask]
    print(f"Number of valid points for visualization: {pts.shape[0]}")
    pcd = limapvis.open3d_get_3d_points(pts, ranges=ranges)
    camera_set = limapvis.open3d_get_camera_frustums(
        recon,
        ranges=ranges,
        scale_cam_geometry=args.cam_scale,
    )

    # Use VisualizerWithKeyCallback if screenshot is needed
    if args.screenshot:
        vis = o3d.visualization.VisualizerWithKeyCallback()
    else:
        vis = o3d.visualization.Visualizer()
    vis.create_window(height=1080, width=1920)
    vis.add_geometry(pcd)
    vis.add_geometry(camera_set)

    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.line_width = args.line_width

    # Register key callback for screenshot (press 'S')
    if args.screenshot:

        def save_screenshot_callback(vis):
            vis.capture_screen_image(args.screenshot)
            print(f"Screenshot saved to {args.screenshot}")
            return False

        vis.register_key_callback(ord("S"), save_screenshot_callback)
        print(f"Press 'S' to save screenshot to {args.screenshot}")

    vis.run()
    vis.destroy_window()


def main(args):
    recon = pycolmap.Reconstruction(args.input_dir)
    ranges = None
    if not args.disable_robust_ranges:
        points = np.array([point.xyz for _, point in recon.points3D.items()])
        ranges = limapvis.compute_robust_range_points(points)
    vis_colmap_reconstruction(args, recon, ranges=ranges)


if __name__ == "__main__":
    args = parse_args()
    main(args)
