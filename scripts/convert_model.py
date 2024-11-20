import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import limap.base as base
import limap.util.io as limapio
from limap.pointsfm import model_converter as model_converter

if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="model conversion")
    arg_parser.add_argument(
        "-i", "--input_path", required=True, type=str, help="input path"
    )
    arg_parser.add_argument(
        "-o", "--output_path", required=True, type=str, help="output path"
    )
    arg_parser.add_argument(
        "--type", type=str, default="imagecols2colmap", help="conversion type"
    )
    args = arg_parser.parse_args()

    if args.type == "imagecols2colmap":
        imagecols = limapio.read_npy(args.input_path).item()
        if isinstance(imagecols, dict):
            imagecols = base.ImageCollection(imagecols)
        model_converter.convert_imagecols_to_colmap(imagecols, args.output_path)
    elif args.type == "colmap2vsfm":
        model_converter.convert_colmap_to_visualsfm(
            args.input_path, args.output_path
        )
    else:
        raise NotImplementedError
