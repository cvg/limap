import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import limap.base as _base
import limap.vpdetection as _vpdet
import limap.util.io as limapio
import limap.util.config as cfgutils
import limap.optimize

def one_by_one_refinement(cfg):
    '''
    One by one refinement
    '''
    linetracks, cfg_info, imagecols, all_2d_segs = limapio.read_folder_linetracks_with_info(cfg["input_folder"])
    cfg_info = cfg_info.item()

    # vp
    vpresults = None
    if cfg["refinement"]["use_vp"]:
        all_2d_lines = _base.get_all_lines_2d(all_2d_segs)
        vpresults = _vpdet.AssociateVPsParallel(all_2d_lines)

    # one-by-one refinement
    newtracks = limap.optimize.line_refinement(cfg["refinement"], linetracks, imagecols, cfg["heatmap_folder"], cfg["patch_folder"], cfg["featuremap_folder"], vpresults=vpresults)

    # write
    newlines = np.array([track.line.as_array() for track in newtracks if track.count_images() >= cfg_info["n_visible_views"]])
    limapio.save_obj(os.path.join(cfg["output_dir"], "lines_refined.obj"), newlines)
    final_output_dir = os.path.join(cfg["output_dir"], cfg["output_folder"])
    limapio.delete_folder(final_output_dir)
    limapio.save_folder_linetracks_with_info(final_output_dir, newtracks, config=cfg_info, imagecols=imagecols, all_2d_segs=all_2d_segs)

def joint_refinement(cfg):
    '''
    Joint refinement
    '''
    linetracks, cfg_info, imagecols, all_2d_segs = limapio.read_folder_linetracks_with_info(cfg["input_folder"])
    cfg_info = cfg_info.item()

    # vp
    vpresults = None
    if cfg["refinement"]["use_vp"]:
        all_2d_lines = _base.get_all_lines_2d(all_2d_segs)
        vpresults = _vpdet.AssociateVPsParallel(all_2d_lines)

    # joint refinement
    reconstruction = _base.LineReconstruction(linetracks, imagecols)
    lineba_engine = limap.optimize.solve_line_bundle_adjustment(cfg["refinement"], reconstruction, vpresults=vpresults, max_num_iterations=200)
    new_reconstruction = lineba_engine.GetOutputReconstruction()
    newtracks = new_reconstruction.GetTracks(num_outliers=cfg["refinement"]["num_outliers_aggregator"])
    imagecols_output = new_reconstruction.GetImagecols()

    # write
    newlines = np.array([track.line.as_array() for track in newtracks if track.count_images() >= cfg_info["n_visible_views"]])
    limapio.save_obj(os.path.join(cfg["output_dir"], "lines_refined.obj"), newlines)
    final_output_dir = os.path.join(cfg["output_dir"], cfg["output_folder"])
    limapio.delete_folder(final_output_dir)
    limapio.save_folder_linetracks_with_info(final_output_dir, newtracks, config=cfg_info, imagecols=imagecols, all_2d_segs=all_2d_segs)

def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(description='refinement with pixelwise optimization')
    arg_parser.add_argument('-i', '--input_folder', type=str, required=True, help='input folder for tracks and infos')
    arg_parser.add_argument('-ht', '--heatmap_folder', type=str, default=None, help='heatmap_folder')
    arg_parser.add_argument('-pt', '--patch_folder', type=str, default=None, help='patch_folder')
    arg_parser.add_argument('-ft', '--featuremap_folder', type=str, default=None, help='featuremap_folder')
    arg_parser.add_argument('-c', '--config_file', type=str, default='cfgs/refinement/default.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str, default='cfgs/triangulation/default.yaml', help='default config file')

    arg_parser.add_argument('--mesh_dir', type=str, default=None, help='path to the .obj file')
    arg_parser.add_argument('--visualize', action='store_true', help='whether to do the visualization')
    arg_parser.add_argument('--output_dir', type=str, default=None, help='folder to save')
    arg_parser.add_argument('--output_folder', type=str, default='newtracks', help='output filename')

    arg_parser.add_argument('--method', type=str, default='one-by-one', help='["one-by-one", "joint"]')

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(args.config_file, default_path=args.default_config_file)
    shortcuts = dict()
    shortcuts['-nv'] = '--n_visible_views'
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["input_folder"] = args.input_folder.strip('/')
    cfg["output_dir"] = args.output_dir
    if cfg["output_dir"] is None:
        cfg["output_dir"] = os.path.dirname(cfg["input_folder"])
    cfg["output_folder"] = args.output_folder
    cfg["heatmap_folder"] = args.heatmap_folder
    cfg["patch_folder"] = args.patch_folder
    cfg["featuremap_folder"] = args.featuremap_folder
    cfg["refinement"]["visualize"] = args.visualize
    cfg["refinement"]["mesh_dir"] = args.mesh_dir
    cfg["method"] = args.method

    # check
    if cfg["refinement"]["use_heatmap"] and cfg["heatmap_folder"] is None:
        raise ValueError("Path to the heatmap should be given. [-ht]")
    if cfg["refinement"]["use_feature"] and (cfg["patch_folder"] is None and cfg["featuremap_folder"] is None):
        raise ValueError("Path to the patch/featuremap should be given. [-pt] / [-ft]")
    return cfg

if __name__ == '__main__':
    cfg = parse_config()
    if cfg["method"] == "one-by-one":
        one_by_one_refinement(cfg)
    elif cfg["method"] == "joint":
        joint_refinement(cfg)
    else:
        raise NotImplementedError

