import os, sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import limap.base as _base
import limap.features as _features
import limap.refinement as _refinement
import limap.vpdetection as _vpdet

import limap.evaluation as _eval
import limap.util.io_utils as limapio
import limap.visualize as limapvis

def line_refinement(cfg, tracks, imagecols, heatmap_dir=None, patch_dir=None, featuremap_dir=None, vpresults=None, n_visible_views=4):
    '''
    refine each line one by one with fixed cameras
    '''
    evaluator = None
    if cfg["mesh_dir"] is not None:
        MPAU = 0.02539999969303608 # hypersim
        evaluator = _eval.MeshEvaluator(cfg["mesh_dir"], MPAU)

    ids = [k for k in range(len(tracks)) if tracks[k].count_images() >= n_visible_views]
    opttracks = []
    for t_id, track_id in enumerate(tqdm(ids)):
        # initialize data
        track = tracks[track_id]
        sorted_ids = track.GetSortedImageIds()
        p_cameras = []
        p_vpresults = None
        p_heatmaps, p_patches, p_features = None, None, None
        if cfg["use_vp"]: p_vpresults = []
        if cfg["use_heatmap"]: p_heatmaps = []
        if cfg["use_feature"]:
            if patch_dir is not None:
                p_patches = []
            else:
                p_features = []
        for img_id in sorted_ids: # add data for each supporting image
            p_cameras.append(cameras[img_id])
            if cfg["use_vp"]:
                p_vpresults.append(vpresults[img_id])
            if cfg["use_heatmap"]:
                heatmap = limapio.read_npy(os.path.join(heatmap_dir, "heatmap_{0}.npy".format(img_id)))
                p_heatmaps.append(heatmap)
            if cfg["use_feature"]:
                if patch_dir is not None:
                    fname = os.path.join(patch_dir, "track{0}".format(track_id), "track{0}_img{1}.npy".format(track_id, img_id))
                    patch = _features.load_patch(fname, dtype=cfg["dtype"])
                    p_patches.append(patch)
                else:
                    with open(os.path.join(featuremap_dir, "feature_{}.npy").format(img_id), 'rb') as f:
                        featuremap = np.load(f, allow_pickle=True)
                    p_features.append(featuremap.transpose(1,2,0))

        # refine
        rf_engine = _refinement.solve(cfg, track, p_cameras, p_vpresults=p_vpresults, p_heatmaps=p_heatmaps, p_patches=p_patches, p_features=p_features, dtype=cfg["dtype"])
        newtrack = _base.LineTrack(track)
        if rf_engine is None:
            opttracks.append(newtrack)
            continue
        newtrack.line = rf_engine.GetLine3d()
        opttracks.append(newtrack)

        # evaluation
        if evaluator is not None:
            dist = evaluator.ComputeDistLine(track.line, n_samples=1000)
            ratio = evaluator.ComputeInlierRatio(track.line, 0.001)
            newdist = evaluator.ComputeDistLine(newtrack.line, n_samples=1000)
            newratio = evaluator.ComputeInlierRatio(newtrack.line, 0.001)
            if newdist > dist and newratio < ratio:
                print("[DEBUG] t_id = {0}, original: dist = {1:.4f}, ratio = {2:.4f}".format(t_id, dist * 1000, ratio))
                print("[DEBUG] t_id = {0}, optimized: dist = {1:.4f}, ratio = {2:.4f}".format(t_id, newdist * 1000, newratio))

    # output
    newtracks = []
    counter = 0
    for idx, track in enumerate(tracks):
        if track.count_images() < n_visible_views:
            newtracks.append(track)
        else:
            newtracks.append(opttracks[counter])
            counter += 1

    # debug
    if cfg["visualize"]:
        def report_track(track_id):
            limapvis.visualize_line_track(imname_list, tracks[track_id], max_image_dim=-1, cameras=cameras, prefix="track.{0}".format(track_id))
        def report_newtrack(track_id):
            limapvis.visualize_line_track(imname_list, opttracks[track_id], max_image_dim=-1, cameras=cameras, prefix="newtrack.{0}".format(track_id))
        import pdb
        pdb.set_trace()
        VisTrack = limapvis.PyVistaTrackVisualizer(newtracks, visualize=True)
        VisTrack.vis_all_lines(n_visible_views=n_visible_views, width=2)
    return newtracks

def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(description='refinement with pixelwise optimization')
    arg_parser.add_argument('-i', '--input_folder', type=str, required=True, help='input folder for tracks and infos')
    arg_parser.add_argument('-ht', '--heatmap_folder', type=str, default=None, help='heatmap_folder')
    arg_parser.add_argument('-pt', '--patch_folder', type=str, default=None, help='featuremap_folder')
    arg_parser.add_argument('-ft', '--featuremap_folder', type=str, default=None, help='featuremap_folder')
    arg_parser.add_argument('-c', '--config_file', type=str, default='cfgs/refinement/default.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str, default='cfgs/triangulation/default.yaml', help='default config file')

    arg_parser.add_argument('--mesh_dir', type=str, default=None, help='path to the .obj file')
    arg_parser.add_argument('--visualize', action='store_true', help='whether to do the visualization')
    arg_parser.add_argument('--output_dir', type=str, default='tmp', help='folder to save')
    arg_parser.add_argument('--output_folder', type=str, default='newtracks', help='output filename')

    args, unknown = arg_parser.parse_known_args()
    cfg = utils.load_config(args.config_file, default_path=args.default_config_file)
    shortcuts = dict()
    shortcuts['-nv'] = '--n_visible_views'
    cfg = utils.update_config(cfg, unknown, shortcuts)
    cfg["input_folder"] = args.input_folder
    cfg["heatmap_folder"] = args.heatmap_folder
    cfg["patch_folder"] = args.patch_folder
    cfg["featuremap_folder"] = args.featuremap_folder
    cfg["refinement"]["visualize"] = args.visualize
    cfg["refinement"]["mesh_dir"] = args.mesh_dir
    cfg["refinement"]["output_dir"] = args.output_dir
    if not os.path.join(cfg["refinement"]["output_dir"]):
        os.makedirs(cfg["refinement"]["output_dir"])
    cfg["refinement"]["output_folder"] = args.output_folder

    # check
    if cfg["refinement"]["use_heatmap"] and cfg["heatmap_folder"] is None:
        raise ValueError("Path to the heatmap should be given. [-ht]")
    if cfg["refinement"]["use_feature"] and (cfg["patch_folder"] is None and cfg["featuremap_folder"] is None):
        raise ValueError("Path to the patch/featuremap should be given. [-pt] / [-ft]")
    return cfg

def main():
    cfg = parse_config()
    linetracks, cfg_info, imagecols, all_2d_segs = limapio.read_folder_linetracks_with_info(cfg["input_folder"])

    # vp
    vpresults = None
    if cfg["refinement"]["use_vp"]:
        all_2d_lines = _base.GetAllLines2D(all_2d_segs)
        vpresults = _vpdet.AssociateVPsParallel(all_2d_lines)

    # refine
    newtracks = line_refinement(cfg["refinement"], linetracks, imagecols, cfg["heatmap_folder"], cfg["patch_folder"], cfg["featuremap_folder"], vpresults=vpresults)

    # write
    newlines = np.array([track.line.as_array() for track in newtracks])
    newline_counts = np.array([track.count_images() for track in newtracks])
    limapio.save_obj(os.path.join(cfg["output_dir"], "lines_refined.obj"), newlines, counts=newline_counts)
    final_output_dir = os.path.join(cfg["output_dir"], cfg["output_folder"])
    limapio.save_folder_linetracks_with_info(final_output_dir, newtracks, config=cfg_info, imagecols=imagecols, all_2d_segs=all_2d_segs)

if __name__ == '__main__':
    main()

