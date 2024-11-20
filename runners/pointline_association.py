import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math

import numpy as np

import limap.base as base
import limap.optimize as optimize
import limap.pointsfm as pointsfm
import limap.runners
import limap.structures as structures
import limap.util.config as cfgutils
import limap.util.io as limapio
import limap.visualize as limapvis
import limap.vplib as vplib


def report_vp(vpresults, vptracks, print_pairs=False):
    n_pairs_parallel, n_pairs_orthogonal = 0, 0
    for i in range(len(vptracks) - 1):
        for j in np.arange(i + 1, len(vptracks)):
            cosine = abs(vptracks[i].direction @ vptracks[j].direction)
            angle = math.acos(min(cosine, 1.0)) * 180.0 / math.pi
            if angle <= 1.0:
                n_pairs_parallel += 1
                if print_pairs:
                    print(
                        f"[LOG] Parallel pair detected: {i} / {j}, \
                          angle = {angle:.2f}"
                    )
            if angle >= 87.0:
                n_pairs_orthogonal += 1
                if print_pairs:
                    print(
                        f"[LOG] Orthogonal pair detected: {i} / {j}, \
                          angle = {angle:.2f}"
                    )
    print(f"[LOG] number of VP tracks: {len(vptracks)}")
    print("[LOG]", [track.length() for track in vptracks])
    print(
        f"[LOG] parallel pairs: {n_pairs_parallel}, \
          orthogonal pairs: {n_pairs_orthogonal}"
    )


def pointline_association(cfg, input_folder, output_folder, colmap_folder):
    """
    optimization to associate points and lines
    """
    (
        linetracks,
        cfg_info,
        imagecols,
        all_2d_segs,
    ) = limapio.read_folder_linetracks_with_info(input_folder)
    all_2d_lines = base.get_all_lines_2d(all_2d_segs)

    ############################################################
    # Point-line bipartite
    ############################################################
    # initiate point-line bipartites on 2d for each image
    reconstruction = pointsfm.PyReadCOLMAP(colmap_folder)
    pointtracks = pointsfm.ReadPointTracks(reconstruction)
    all_bpt2ds, _ = limap.runners.compute_2d_bipartites_from_colmap(
        reconstruction, imagecols, all_2d_lines
    )

    ############################################################
    # VP-line bipartite
    ############################################################
    if cfg["global_pl_association"]["use_vp"]:
        # detect vp
        vpdetector = vplib.get_vp_detector(
            cfg["global_pl_association"]["vpdet"],
            n_jobs=cfg["global_pl_association"]["vpdet"]["n_jobs"],
        )
        vpresults = vpdetector.detect_vp_all_images(
            all_2d_lines, imagecols.get_map_camviews()
        )

        # build vanishing point tracks
        vptrack_constructor = vplib.GlobalVPTrackConstructor()
        vptrack_constructor.Init(vpresults)
        vptracks = vptrack_constructor.ClusterLineTracks(linetracks, imagecols)
        all_bpt2ds_vp = structures.GetAllBipartites_VPLine2d(
            all_2d_lines, vpresults, vptracks
        )

    ############################################################
    # Optimization
    ############################################################
    # optimize association # 1
    cfg_associator = optimize.GlobalAssociatorConfig(
        cfg["global_pl_association"]
    )
    associator = optimize.GlobalAssociator(cfg_associator)
    associator.InitImagecols(imagecols)
    associator.InitPointTracks(pointtracks)
    associator.InitLineTracks(linetracks)
    associator.Init2DBipartites_PointLine(all_bpt2ds)
    # associator.ReassociateJunctions()
    if cfg["global_pl_association"]["use_vp"]:
        associator.InitVPTracks(vptracks)
        associator.Init2DBipartites_VPLine(all_bpt2ds_vp)
    associator.SetUp()
    associator.Solve()

    # iterate optimization until there is no parallel pairs
    if cfg["global_pl_association"]["use_vp"]:
        n_iters = 0
        while n_iters <= 5:
            n_iters += 1
            # update vps
            vptracks_opt_map = associator.GetOutputVPTracks()
            vptracks_opt = [
                vptrack for (idx, vptrack) in vptracks_opt_map.items()
            ]
            vptracks_opt_merged = vplib.MergeVPTracksByDirection(
                vptracks_opt, 1.0
            )
            if len(vptracks_opt_merged) == len(vptracks_opt):
                break

            # run optimization on the merged vptracks
            all_bpt2ds_vp_opt = structures.GetAllBipartites_VPLine2d(
                all_2d_lines, vpresults, vptracks_opt_merged
            )
            associator.InitVPTracks(vptracks_opt_merged)
            associator.Init2DBipartites_VPLine(all_bpt2ds_vp_opt)

            # optimize again
            associator.SetUp()
            associator.Solve()
    new_imagecols = associator.GetOutputImagecols()
    bpt3d = associator.GetBipartite3d_PointLine()
    bpt3d_vp = associator.GetBipartite3d_VPLine()

    ############################################################
    # IO & visualization
    ############################################################
    # report vp tracks
    vptracks = None
    if cfg["global_pl_association"]["use_vp"]:
        vptracks = bpt3d_vp.get_all_points()
    # save line tracks
    newtracks = bpt3d.get_all_lines()
    limapio.save_folder_linetracks_with_info(
        output_folder,
        newtracks,
        config=cfg_info,
        imagecols=imagecols,
        all_2d_segs=all_2d_segs,
    )

    # visualize
    if cfg["visualize"]:
        if cfg["global_pl_association"]["use_vp"]:
            report_vp(vpresults, vptracks, print_pairs=True)
        import pdb

        pdb.set_trace()
        neighbors, ranges = limapio.read_txt_metainfos(
            os.path.join(cfg["input_folder"], "../metainfos.txt")
        )
        limapvis.open3d_draw_bipartite3d_pointline(bpt3d, ranges=ranges)
        pdb.set_trace()
        if cfg["global_pl_association"]["use_vp"]:
            limapvis.open3d_draw_bipartite3d_vpline(bpt3d_vp, ranges=ranges)
            pdb.set_trace()
    return new_imagecols, newtracks, vptracks


def parse_config():
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="refinement with pixelwise optimization"
    )
    arg_parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        required=True,
        help="input folder for line tracks and infos",
    )
    arg_parser.add_argument(
        "--colmap_folder",
        type=str,
        default=None,
        help="colmap folder storing point tracks",
    )
    arg_parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="cfgs/global_pl_association/default.yaml",
        help="config file",
    )
    arg_parser.add_argument(
        "--default_config_file",
        type=str,
        default="cfgs/triangulation/default.yaml",
        help="default config file",
    )
    arg_parser.add_argument(
        "--output_dir", type=str, default=None, help="folder to save"
    )
    arg_parser.add_argument(
        "--output_folder",
        type=str,
        default="associated_tracks",
        help="output filename",
    )

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(
        args.config_file, default_path=args.default_config_file
    )
    shortcuts = dict()
    shortcuts["-nv"] = "--n_visible_views"
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["input_folder"] = args.input_folder.strip("/")
    if args.colmap_folder is not None:
        cfg["colmap_folder"] = args.colmap_folder.strip("/")
    cfg["output_dir"] = args.output_dir
    if cfg["output_dir"] is None:
        cfg["output_dir"] = os.path.dirname(cfg["input_folder"])
    cfg["output_folder"] = args.output_folder
    return cfg


if __name__ == "__main__":
    cfg = parse_config()
    input_folder = cfg["input_folder"]
    output_folder = os.path.join(cfg["output_dir"], cfg["output_folder"])
    colmap_folder = cfg["colmap_folder"]
    new_imagecols, newtracks, vptracks = pointline_association(
        cfg, input_folder, output_folder, colmap_folder
    )
