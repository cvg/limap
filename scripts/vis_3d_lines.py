import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from core.visualize import vis_3d_lines, save_obj
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eval_hypersim import read_lines_from_input

def parse_args():
    import argparse
    arg_parser = argparse.ArgumentParser(description='visualize 3d lines')
    arg_parser.add_argument('-nv', '--n_visible_views', type=int, default=4, help='number of visible views')
    arg_parser.add_argument('-i', '--input_dir', type=str, default='lines_to_vis.npy', help='input line npy file')
    arg_parser.add_argument('--save_obj', action='store_true', help='whether to save obj')
    args = arg_parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    img_hw = (600, 800)
    lines, _, _ = read_lines_from_input(args.input_dir, n_visible_views=args.n_visible_views)
    lines = np.array([line.as_array() for line in lines])
    counts = np.array([args.n_visible_views for line in lines])
    vis_3d_lines(lines, img_hw, counts=counts, n_visible_views=args.n_visible_views)
    if args.save_obj:
        save_obj('output.obj', lines, counts, n_visible_views = args.n_visible_views)

