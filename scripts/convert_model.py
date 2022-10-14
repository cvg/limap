import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import limap.pointsfm as _psfm

if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser(description="model conversion")
    arg_parser.add_argument("-i", "--input_path", required=True, type=str, help="input path")
    arg_parser.add_argument("-o", "--output_path", required=True, type=str, help="output path")
    arg_parser.add_argument("--type", type=str, default="colmap2vsfm", help="conversion type")
    args = arg_parser.parse_args()

    if args.type == "colmap2vsfm":
        _psfm.convert_colmap_to_visualsfm(args.input_path, args.output_path)
    else:
        raise NotImplementedError


