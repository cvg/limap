import limap.base as base
import limap.line2d
import limap.util.io as limapio


def extract_heatmaps_sold2(output_dir, imagecols, skip_exists=False):
    """
    Extract sold2 heatmaps from base.ImageCollection object
    """
    # detect heatmaps
    detector_cfg = {}
    detector_cfg["method"] = "sold2"
    detector = limap.line2d.get_detector(detector_cfg)
    detector.extract_heatmaps_all_images(
        output_dir, imagecols, skip_exists=skip_exists
    )


def parse_config():
    import argparse

    arg_parser = argparse.ArgumentParser(description="extract sold2 heatmaps")
    arg_parser.add_argument(
        "-i", "--input", type=str, required=True, help="imagecols.npy"
    )
    arg_parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="output folder"
    )
    arg_parser.add_argument(
        "--skip_exists", action="store_true", help="skip exists"
    )
    args = arg_parser.parse_args()
    return args


def main(args):
    if not args.input.endswith(".npy"):
        raise ValueError("input file should be with the .npy extension")
    imagecols = base.ImageCollection(limapio.read_npy(args.input).item())
    extract_heatmaps_sold2(
        args.output_dir, imagecols, skip_exists=args.skip_exists
    )


if __name__ == "__main__":
    args = parse_config()
    main(args)
