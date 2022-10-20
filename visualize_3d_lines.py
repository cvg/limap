import os, sys
import numpy as np

import limap.base as _base
import limap.util.io as limapio
import limap.visualize as limapvis

def parse_args():
    import argparse
    arg_parser = argparse.ArgumentParser(description='visualize 3d lines')
    arg_parser.add_argument('-i', '--input_dir', type=str, required=True, help='input line file. Format supported now: .obj, .npy, linetrack folder.')
    arg_parser.add_argument('-nv', '--n_visible_views', type=int, default=2, help='number of visible views')
    arg_parser.add_argument('--imagecols', type=str, default=None, help=".npy file for imagecols")
    arg_parser.add_argument("--metainfos", type=str, default=None, help=".txt file for neighbors and ranges")
    arg_parser.add_argument('--mode', type=str, default="open3d", help="[pyvista, open3d]")
    args = arg_parser.parse_args()
    return args

def vis_3d_lines(lines, mode="open3d", ranges=None):
    if ranges is not None:
        lines = [line for line in lines if limapvis.test_line_inside_ranges(line, ranges)]
    if mode == "pyvista":
        limapvis.pyvista_vis_3d_lines(lines)
    elif mode == "open3d":
        limapvis.open3d_vis_3d_lines(lines)
    else:
        raise NotImplementedError

def vis_reconstruction(linetracks, imagecols, mode="open3d", n_visible_views=4, ranges=None):
    if ranges is not None:
        linetracks = [track for track in linetracks if limapvis.test_line_inside_ranges(track.line, ranges)]
    if mode == "pyvista":
        VisTrack = limapvis.PyVistaTrackVisualizer(linetracks)
    elif mode == "open3d":
        VisTrack = limapvis.Open3DTrackVisualizer(linetracks)
    else:
        raise NotImplementedError
    VisTrack.report()
    VisTrack.vis_reconstruction(imagecols, n_visible_views=n_visible_views)

def main(args):
    lines, linetracks = limapio.read_lines_from_input(args.input_dir)
    ranges = None
    if args.metainfos is not None:
        _, ranges = limapio.read_txt_metainfos(args.metainfos)
    if args.n_visible_views > 2 and linetracks is None:
        raise ValueError("Error! Track information is not available.")
    if args.imagecols is None:
        vis_3d_lines(lines, mode=args.mode, ranges=ranges)
    else:
        if (not os.path.exists(args.imagecols)) or (not args.imagecols.endswith('.npy')):
            raise ValueError("Error! Input file {0} is not valid".format(args.imagecols))
        imagecols = _base.ImageCollection(limapio.read_npy(args.imagecols).item())
        vis_reconstruction(linetracks, imagecols, mode=args.mode, n_visible_views=args.n_visible_views, ranges=ranges)

if __name__ == '__main__':
    args = parse_args()
    main(args)

