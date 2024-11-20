import os
import time

import numpy as np
from tqdm import tqdm

import limap.features as limap_features
import limap.util.io as limapio


def extract_feature_s2dnet(feature_extractor, camview):
    import torch

    img = camview.read_image(set_gray=False)
    input_image = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)[
        None, ...
    ]
    input_image = input_image.to("cuda")
    featuremap = feature_extractor.extract_featuremaps(input_image)
    feature = featuremap[0][0].detach().cpu().numpy().astype(np.float16)
    feature = feature.transpose(1, 2, 0)
    return feature


def extract_track_patches_s2dnet(
    cfg, tracks, imagecols, output_dir, skip_exists=False
):
    """
    Extract line patches (S2DNet) for each track
    """
    feature_extractor = limap_features.load_extractor("s2dnet", "cuda")
    extractor = limap_features.get_line_patch_extractor(
        cfg["patch"], cfg["channels"]
    )
    if not skip_exists:
        limapio.delete_folder(output_dir)
    limapio.check_makedirs(output_dir)

    # map supporting line from each track to images
    line2d_collections = {}
    for img_id in imagecols.get_img_ids():
        line2d_collections[img_id] = []
    for track_id, track in enumerate(tqdm(tracks)):
        sorted_ids = track.GetSortedImageIds()
        for img_id in sorted_ids:
            camview = imagecols.camview(img_id)
            line2d_range = extractor.GetLine2DRange(track, img_id, camview)
            line2d_collections[img_id].append([line2d_range, track_id])
        limapio.check_makedirs(os.path.join(output_dir, f"track{track_id}"))

    # extract line patches for each image
    for img_id in tqdm(imagecols.get_img_ids()):
        # extract s2dnet line patches
        feature = extract_feature_s2dnet(
            feature_extractor, imagecols.camview(img_id)
        )

        # extract line patches
        line2ds_info = line2d_collections[img_id]
        line2ds = [k[0] for k in line2ds_info]
        track_ids = [k[1] for k in line2ds_info]
        patches = extractor.ExtractLinePatches(line2ds, feature)
        for patch, track_id in zip(patches, track_ids):
            fname = os.path.join(
                output_dir,
                f"track{track_id}",
                f"track{track_id}_img{img_id}.npy",
            )
            if skip_exists and os.path.exists(fname):
                continue
            limap_features.write_patch(fname, patch, dtype=cfg["dtype"])
            time.sleep(0.001)


def parse_config():
    import argparse

    arg_parser = argparse.ArgumentParser(description="extract sold2 heatmaps")
    arg_parser.add_argument(
        "-i", "--input_dir", type=str, required=True, help="track folder"
    )
    arg_parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="output folder"
    )
    arg_parser.add_argument(
        "--patch_folder", type=str, default="track_patches", help="patch folder"
    )
    arg_parser.add_argument(
        "--skip_exists", action="store_true", help="skip exists"
    )
    # patch config
    arg_parser.add_argument(
        "-c", "--channels", type=int, default=128, help="feature channels"
    )
    arg_parser.add_argument(
        "--k_stretch", type=float, default=1.0, help="k stretch"
    )
    arg_parser.add_argument(
        "--t_stretch", type=float, default=10, help="t stretch in pixels"
    )
    arg_parser.add_argument(
        "--range_perp", type=float, default=16, help="range perp in pixels"
    )
    arg_parser.add_argument(
        "--dtype", type=str, default="float16", help="float16 or float32"
    )
    args = arg_parser.parse_args()
    # generate patch config
    cfg = {}
    cfg["channels"] = args.channels
    cfg["patch"] = {}
    cfg["patch"]["k_stretch"] = args.k_stretch
    cfg["patch"]["t_stretch"] = args.t_stretch
    cfg["patch"]["range_perp"] = args.range_perp
    cfg["dtype"] = args.dtype
    return args, cfg


def main():
    args, cfg = parse_config()
    tracks, _, imagecols, _ = limapio.read_folder_linetracks_with_info(
        args.input_dir
    )
    patches_dir = os.path.join(args.output_dir, args.patch_folder)
    extract_track_patches_s2dnet(
        cfg, tracks, imagecols, patches_dir, skip_exists=args.skip_exists
    )


if __name__ == "__main__":
    main()
