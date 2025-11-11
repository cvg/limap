import numpy as np
import open3d as o3d
import pycolmap

import limap.pointsfm as pointsfm
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
        "--use_robust_ranges",
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
        "--cam_scale",
        type=float,
        default=1.0,
        help="scale of the camera geometry",
    )
    arg_parser.add_argument(
        "--reproj_error_thresh",
        type=float,
        default=2.0,
        help="reprojection error threshold",
    )
    args = arg_parser.parse_args()
    return args


def vis_colmap_reconstruction(
    args, recon: pycolmap.Reconstruction, ranges=None
):
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=1080, width=1920)
    pts = np.array([p.xyz for p in recon.points3D.values()], dtype=np.float32)
    if args.reproj_error_thresh > 0.0:
        errs = np.array(
            [p.error for p in recon.points3D.values()], dtype=np.float32
        )
        mask = errs <= args.reproj_error_thresh  # e.g. keep colored, low-error
        pts = pts[mask]
    print(f"Number of valid points for visualization: {pts.shape[0]}")
    pcd = limapvis.open3d_get_points(pts, ranges=ranges)
    vis.add_geometry(pcd)
    imagecols = pointsfm.convert_colmap_to_imagecols(recon)
    camera_set = limapvis.open3d_get_cameras(
        imagecols,
        ranges=ranges,
        scale_cam_geometry=args.cam_scale,
    )
    vis.add_geometry(camera_set)

    opt = vis.get_render_option()
    opt.point_size = args.point_size

    vis.run()
    vis.destroy_window()


def main(args):
    recon = pycolmap.Reconstruction(args.input_dir)
    ranges = None
    if args.use_robust_ranges:
        points = np.array([point.xyz for _, point in recon.points3D.items()])
        ranges = limapvis.compute_robust_range_points(points)
    vis_colmap_reconstruction(args, recon, ranges=ranges)


if __name__ == "__main__":
    args = parse_args()
    main(args)
