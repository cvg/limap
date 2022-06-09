import os, sys
import numpy as np

import limap.base as _base
import limap.util.io_utils as limapio
import limap.visualize as limapvis

def parse_args():
    import argparse
    arg_parser = argparse.ArgumentParser(description='visualize 3d lines')
    arg_parser.add_argument('-i', '--input_dir', type=str, default='obj', help='input line npy file')
    # arg_parser.add_argument('-nv', '--n_visible_views', type=int, default=4, help='number of visible views')
    arg_parser.add_argument('--imagecols', type=str, default=None, help=".npy file for imagecols")
    arg_parser.add_argument('--mode', type=str, default="pyvista", help="[pyvista, open3d]")
    args = arg_parser.parse_args()
    return args

def vis_3d_lines(lines, mode="pyvista"):
    if mode == "pyvista":
        limapvis.pyvista_vis_3d_lines(lines)
    elif mode == "open3d":
        limapvis.open3d_vis_3d_lines(lines)
    else:
        raise NotImplementedError

def vis_reconstruction(linetracks, imagecols, mode="pyvista"):
    if mode == "pyvista":
        VisTrack = limapvis.PyVistaTrackVisualizer(linetracks)
    elif mode == "open3d":
        VisTrack = limapvis.Open3DTrackVisualizer(linetracks)
    else:
        raise NotImplementedError
    VisTrack.vis_reconstruction(imagecols)

def main(args):
    lines, linetracks = limapio.read_lines_from_input(args.input_dir)
    if args.imagecols is None:
        vis_3d_lines(lines, mode=args.mode)
    else:
        if (not os.path.exists(args.imagecols)) or (not args.imagecols.endswith('.npy')):
            raise ValueError("Error! Input file {0} is not valid".format(args.imagecols))
        imagecols = _base.ImageCollection(limapio.read_npy(args.imagecols).item())
        vis_reconstruction(linetracks, imagecols, mode=args.mode)

if __name__ == '__main__':
    args = parse_args()
    main(args)

