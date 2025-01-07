import pycolmap
import numpy as np
import open3d as o3d
import limap.pointsfm as pointsfm
import limap.visualize as limapvis


def parse_args():
    import argparse

    arg_parser = argparse.ArgumentParser(description="visualize colmap model using Open3D backend")
    arg_parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="input colmap folder"
    )
    arg_parser.add_argument(
        "--use_robust_ranges",
        action="store_true",
        help="whether to use computed robust ranges",
    )
    arg_parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="scaling both the lines and the camera geometry",
    )
    arg_parser.add_argument(
        "--cam_scale",
        type=float,
        default=1.0,
        help="scale of the camera geometry",
    )
    args = arg_parser.parse_args()
    return args

def vis_colmap_reconstruction(recon: pycolmap.Reconstruction, ranges=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=1080, width=1920)
    points = np.array([point.xyz for _, point in recon.points3D.items()])
    pcd = limapvis.open3d_get_points(points, ranges=ranges)
    vis.add_geometry(pcd)
    imagecols = pointsfm.convert_colmap_to_imagecols(recon)
    camera_set = limapvis.open3d_get_cameras(
        imagecols,
        ranges=ranges,
        scale_cam_geometry=args.scale * args.cam_scale,
        scale=args.scale,
    )
    vis.add_geometry(camera_set)
    vis.run()
    vis.destroy_window()

def main(args):
    recon = pycolmap.Reconstruction(args.input_dir)
    ranges = None
    if args.use_robust_ranges:
        points = np.array([point.xyz for _, point in recon.points3D.items()])
        ranges = limapvis.compute_robust_range_points(points)
    vis_colmap_reconstruction(recon, ranges=ranges)

if __name__ == "__main__":
    args = parse_args()
    main(args)
