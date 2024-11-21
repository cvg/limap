import os

import limap.base as base
import limap.util.io as limapio
import limap.visualize as limapvis


def parse_args():
    import argparse

    arg_parser = argparse.ArgumentParser(description="visualize 3d lines")
    arg_parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="input line file. Format supported now: \
              .obj, .npy, linetrack folder.",
    )
    arg_parser.add_argument(
        "-nv",
        "--n_visible_views",
        type=int,
        default=2,
        help="number of visible views",
    )
    arg_parser.add_argument(
        "--imagecols", type=str, default=None, help=".npy file for imagecols"
    )
    arg_parser.add_argument(
        "--metainfos",
        type=str,
        default=None,
        help=".txt file for neighbors and ranges",
    )
    arg_parser.add_argument(
        "--mode", type=str, default="open3d", help="[pyvista, open3d]"
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
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="if set, save the scaled lines in obj format",
    )
    args = arg_parser.parse_args()
    return args


def vis_3d_lines(lines, mode="open3d", ranges=None, scale=1.0):
    if mode == "pyvista":
        limapvis.pyvista_vis_3d_lines(lines, ranges=ranges, scale=scale)
    elif mode == "open3d":
        limapvis.open3d_vis_3d_lines(lines, ranges=ranges, scale=scale)
    else:
        raise NotImplementedError


def vis_reconstruction(
    linetracks,
    imagecols,
    mode="open3d",
    n_visible_views=4,
    ranges=None,
    scale=1.0,
    cam_scale=1.0,
):
    if mode == "open3d":
        VisTrack = limapvis.Open3DTrackVisualizer(linetracks)
    else:
        raise ValueError(
            "Error! Visualization with cameras is only supported with open3d."
        )
    VisTrack.report()
    VisTrack.vis_reconstruction(
        imagecols,
        n_visible_views=n_visible_views,
        ranges=ranges,
        scale=scale,
        cam_scale=cam_scale,
    )


def main(args):
    lines, linetracks = limapio.read_lines_from_input(args.input_dir)
    ranges = None
    if args.metainfos is not None:
        _, ranges = limapio.read_txt_metainfos(args.metainfos)
    if args.use_robust_ranges:
        ranges = limapvis.compute_robust_range_lines(lines)
    if args.n_visible_views > 2 and linetracks is None:
        raise ValueError("Error! Track information is not available.")
    if args.imagecols is None:
        vis_3d_lines(lines, mode=args.mode, ranges=ranges, scale=args.scale)
    else:
        if (not os.path.exists(args.imagecols)) or (
            not args.imagecols.endswith(".npy")
        ):
            raise ValueError(f"Error! Input file {args.imagecols} is not valid")
        imagecols = base.ImageCollection(
            limapio.read_npy(args.imagecols).item()
        )
        vis_reconstruction(
            linetracks,
            imagecols,
            mode=args.mode,
            n_visible_views=args.n_visible_views,
            ranges=ranges,
            scale=args.scale,
            cam_scale=args.cam_scale,
        )
    if args.output_dir is not None:
        limapio.save_obj(args.output_dir, lines)


if __name__ == "__main__":
    args = parse_args()
    main(args)
