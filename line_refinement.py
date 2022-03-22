import os, sys
import numpy as np
import cv2
from tqdm import tqdm
import core.utils as utils
import core.visualize as vis
import limap.base as _base
import limap.features as _features
import limap.refinement as _refinement
import limap.evaluation as _eval
import limap.triangulation as _tri
import limap.vpdetection as _vpdet
import pdb

# hypersim
MPAU = 0.02539999969303608

def visualize_heatmap_intersections(prefix, imname_list, image_ids, p_heatmaps, ht_intersections, max_image_dim=None):
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    cNorm  = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap="viridis")

    path = os.path.dirname(prefix)
    if not os.path.exists(path):
        os.makedirs(path)
    for img_id, heatmap, intersections in zip(image_ids, p_heatmaps, ht_intersections):
        imname = imname_list[img_id]

        # visualize image
        img = utils.read_image(imname, max_image_dim=max_image_dim, set_gray=False)
        img = vis.draw_points(img, intersections, (255, 0, 0), 2)
        fname_out = prefix + '_img{0}.png'.format(img_id)
        cv2.imwrite(fname_out, img)

        # visualize heatmap
        heatmap_img = (scalarMap.to_rgba(heatmap)[:,:,:3] * 255).astype(np.uint8)
        heatmap_img = vis.draw_points(heatmap_img, intersections, (255, 0, 0), 2)
        fname_out_heatmap = prefix + '_heatmap{0}.png'.format(img_id)
        cv2.imwrite(fname_out_heatmap, heatmap_img)

def visualize_fconsis_intersections(prefix, imname_list, image_ids, fc_intersections, max_image_dim=None, n_samples_vis=-1):
    if n_samples_vis != -1:
        fc_intersections = fc_intersections[:n_samples_vis]
    path = os.path.dirname(prefix)
    if not os.path.exists(path):
        os.makedirs(path)
    for sample_id, intersections in enumerate(tqdm(fc_intersections)):
        imgs = []
        for data in intersections:
            img_id, point = image_ids[data[0]], data[1]
            img = utils.read_image(imname_list[img_id], max_image_dim=max_image_dim, set_gray=False)
            vis.draw_points(img, [point], (0, 0, 255), 1)
            img = vis.crop_to_patch(img, point, patch_size=100)
            imgs.append(img)
        bigimg = vis.make_bigimage(imgs, pad=20)
        fname_out = prefix + '_sample{0}.png'.format(sample_id, img_id)
        cv2.imwrite(fname_out, bigimg)

def visualize_heatmaps(imname_list, heatmap_dir):
    heatmaps = []
    for img_id, imname in enumerate(tqdm(imname_list)):
        with open(os.path.join(heatmap_dir, 'heatmap_{}.npy'.format(img_id)), 'rb') as f:
            data = np.load(f, allow_pickle=True)
            heatmap = data["data"]
        heatmaps.append(heatmap)
    vis.tmp_visualize_heatmaps(heatmaps)

def unit_test_add_noise_to_track(track):
    # for unit test
    tmptrack = _base.LineTrack(track)
    start = track.line.start + (np.random.rand(3) - 0.5) * 1e-1
    end = track.line.end + (np.random.rand(3) - 0.5) * 1e-1
    tmpline = _base.Line3d(start, end)
    tmptrack.line = tmpline
    return tmptrack

def line_refinement(cfg, imname_list, tracks, cameras, heatmap_dir, patch_dir, featuremap_dir, vpresults=None, max_image_dim=None):
    '''
    refine each line one by one with fixed cameras
    '''
    evaluator = None
    if cfg["mesh_dir"] is not None:
        evaluator = _eval.MeshEvaluator(cfg["mesh_dir"], MPAU)

    ids = [k for k in range(len(tracks)) if tracks[k].count_images() >= 4]
    newtracks = []
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

        # add data for each supporting image
        for img_id in sorted_ids:
            p_cameras.append(cameras[img_id])
            if cfg["use_vp"]:
                p_vpresults.append(vpresults[img_id])
            if cfg["use_heatmap"]:
                with open(os.path.join(heatmap_dir, "heatmap_{}.npy".format(img_id)), 'rb') as f:
                    data = np.load(f, allow_pickle=True)
                    heatmap = data["data"]
                p_heatmaps.append(heatmap)
            if cfg["use_feature"]:
                if patch_dir is not None:
                    folder = 'track{0}'.format(track_id)
                    npy_fname = 'track{0}_img{1}.npy'.format(track_id, img_id)
                    if os.path.exists(os.path.join(patch_dir, folder)):
                        fname = os.path.join(patch_dir, folder, npy_fname)
                    else:
                        fname = os.path.join(patch_dir, npy_fname)
                    patch = _features.load_patch(fname, dtype=cfg["dtype"])
                    p_patches.append(patch)
                else:
                    with open(os.path.join(featuremap_dir, "feature_{}.npy").format(img_id), 'rb') as f:
                        featuremap = np.load(f, allow_pickle=True)
                    p_features.append(featuremap.transpose(1,2,0))

        # # unit test: add noise here
        # track = unit_test_add_noise_to_track(track)

        rf_engine = _refinement.solve(cfg, track, p_cameras, p_vpresults=p_vpresults, p_heatmaps=p_heatmaps, p_patches=p_patches, p_features=p_features, dtype=cfg["dtype"])
        newtrack = _base.LineTrack(track)
        if rf_engine is None:
            newtracks.append(newtrack)
            continue
        newtrack.line = rf_engine.GetLine3d()
        newtracks.append(newtrack)

        # evaluation
        if evaluator is not None:
            dist = evaluator.ComputeDistLine(track.line, n_samples=1000)
            ratio = evaluator.ComputeInlierRatio(track.line, 0.001)
            newdist = evaluator.ComputeDistLine(newtrack.line, n_samples=1000)
            newratio = evaluator.ComputeInlierRatio(newtrack.line, 0.001)
            if newdist > dist and newratio < ratio:
                print("[DEBUG] t_id = {0}, original: dist = {1:.4f}, ratio = {2:.4f}".format(t_id, dist * 1000, ratio))
                print("[DEBUG] t_id = {0}, optimized: dist = {1:.4f}, ratio = {2:.4f}".format(t_id, newdist * 1000, newratio))

        # visualization
        if cfg["visualize_maps"]:
            print("Start visualization...")
            states = rf_engine.GetAllStates()
            for state_id, state in enumerate(tqdm(states)):
                if state_id != 0 and state_id != len(states) - 1:
                    continue
                ht_intersections = rf_engine.GetHeatmapIntersections(state)
                prefix = os.path.join(cfg["vis_folder"], 'track{0}/heatmap/state{1}/vis'.format(track_id, state_id))
                visualize_heatmap_intersections(prefix, imname_list, sorted_ids, p_heatmaps, ht_intersections, max_image_dim=max_image_dim)
                prefix = os.path.join(cfg["vis_folder"], 'track{0}/feature/state{1}/vis'.format(track_id, state_id))
                fc_intersections = rf_engine.GetFConsistencyIntersections(state)
                visualize_fconsis_intersections(prefix, imname_list, sorted_ids, fc_intersections, max_image_dim=max_image_dim)
    # TODO: hard-coded here
    newnewtracks = []
    counter = 0
    for idx, track in enumerate(tracks):
        if track.count_images() < 4:
            newnewtracks.append(track)
        else:
            newnewtracks.append(newtracks[counter])
            counter += 1
    newtracks = newnewtracks
    lines = np.array([track.line.as_array() for track in tracks])
    newlines = np.array([track.line.as_array() for track in newtracks])
    newline_counts = np.array([track.count_images() for track in newtracks])
    final_output_dir = os.path.join(cfg["folder_to_save"], cfg["output_fname"])
    vis.save_linetracks_to_folder(newtracks, final_output_dir)
    vis.save_obj(os.path.join(cfg["folder_to_save"], "lines_refined.obj"), newlines, counts=newline_counts)
    if cfg["visualize"]:
        def report_track(track_id):
            vis.visualize_line_track(imname_list, tracks[track_id], max_image_dim=-1, cameras=cameras, prefix="track.{0}".format(track_id))
        def report_newtrack(track_id):
            vis.visualize_line_track(imname_list, newtracks[track_id], max_image_dim=-1, cameras=cameras, prefix="newtrack.{0}".format(track_id))
        pdb.set_trace()
        img_hw = (cameras[0].h, cameras[0].w)
        vis.vis_3d_lines(newlines, img_hw)
    return newtracks

def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(description='refinement with pixelwise optimization')
    arg_parser.add_argument('-i', '--input_folder', type=str, required=True, help='input folder for tracks and infos')
    arg_parser.add_argument('-ht', '--heatmap_folder', type=str, default=None, help='heatmap_folder')
    arg_parser.add_argument('-pt', '--patch_folder', type=str, default=None, help='featuremap_folder')
    arg_parser.add_argument('-ft', '--featuremap_folder', type=str, default=None, help='featuremap_folder')
    arg_parser.add_argument('-c', '--config_file', type=str, default='cfgs/refinement/default_refinement.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str, default='cfgs/triangulation/default_triangulation.yaml', help='default config file')
    arg_parser.add_argument('--visualize_maps', action='store_true', help='whether to do the visualization on heatmap and featuremetric')
    arg_parser.add_argument('--visualize', action='store_true', help='whether to do the visualization')
    arg_parser.add_argument('--vis_folder', type=str, default='tmp/vis_refinement', help="folder to save visualization")
    arg_parser.add_argument('--visualize_heatmaps', action='store_true', help='whether to do the visualization')
    arg_parser.add_argument('--mesh_dir', type=str, default=None, help='path to the .obj file')
    arg_parser.add_argument('--folder_to_save', type=str, default='tmp', help='folder to save')
    arg_parser.add_argument('--output_fname', type=str, default='newtracks', help='output filename')

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
    cfg["refinement"]["vis_folder"] = args.vis_folder
    cfg["refinement"]["mesh_dir"] = args.mesh_dir
    cfg["refinement"]["visualize_maps"] = args.visualize_maps
    cfg["refinement"]["folder_to_save"] = args.folder_to_save
    if not os.path.join(cfg["refinement"]["folder_to_save"]):
        os.makedirs(cfg["refinement"]["folder_to_save"])
    cfg["refinement"]["output_fname"] = args.output_fname
    cfg["visualize_heatmaps"] = args.visualize_heatmaps

    # check
    if cfg["visualize_heatmaps"] and cfg["heatmap_folder"] is None:
        raise ValueError("Path to the heatmap should be given. [-ht]")
    if cfg["refinement"]["use_heatmap"] and cfg["heatmap_folder"] is None:
        raise ValueError("Path to the heatmap should be given. [-ht]")
    if cfg["refinement"]["use_feature"] and (cfg["patch_folder"] is None and cfg["featuremap_folder"] is None):
        raise ValueError("Path to the patch/featuremap should be given. [-pt] / [-ft]")
    return cfg

def init_workspace():
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

def main():
    cfg = parse_config()
    init_workspace()
    info_path = os.path.join(cfg["input_folder"], "all_infos.npy")
    with open(info_path, 'rb') as f:
        data = np.load(f, allow_pickle=True)
        imname_list, all_2d_segs, cameras_np, cfg_info, n_tracks = data["imname_list"], data["all_2d_segs"], data["cameras_np"], data["cfg"].item(), data["n_tracks"].item()
    # print(cfg)

    if cfg["visualize_heatmaps"]:
        visualize_heatmaps(imname_list, cfg["heatmap_folder"])
        return

    # vp
    vpresults = None
    if cfg["refinement"]["use_vp"]:
        all_2d_lines = _tri.GetAllLines2D(all_2d_segs)
        vpresults = _vpdet.AssociateVPsParallel(all_2d_lines)
    cameras = [_base.Camera(cam[0], cam[1], cam[2][:, 0], cam[3]) for cam in cameras_np]
    linetracks = [_base.LineTrack() for i in range(n_tracks)]
    linetracks = vis.load_linetracks_from_folder(linetracks, cfg["input_folder"])
    newtracks = line_refinement(cfg["refinement"], imname_list, linetracks, cameras, cfg["heatmap_folder"], cfg["patch_folder"], cfg["featuremap_folder"], vpresults=vpresults, max_image_dim=cfg["max_image_dim"])

    final_output_dir = os.path.join(cfg["refinement"]["folder_to_save"], cfg["refinement"]["output_fname"])
    with open(os.path.join(final_output_dir, "all_infos.npy"), "wb") as f:
        np.savez(f, imname_list=imname_list, all_2d_segs=all_2d_segs, cameras_np=cameras_np, cfg=cfg, n_tracks=len(newtracks))

if __name__ == '__main__':
    main()

