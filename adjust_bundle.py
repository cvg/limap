import os, sys
import numpy as np
import cv2
import time
from tqdm import tqdm
import core.utils as utils
import core.visualize as vis
import limap.base as _base
import limap.features as _features
import limap.lineBA as _lineBA
import pdb

def extract_patches(cfg, tracks, cameras, featuremap_dir, output_dir, imname_list=None, max_image_dim=None):
    '''
    extract patches for each supporting image in each track
    '''
    feature_extractor = _features.S2DNetExtractor('cuda')
    extractor = _features.get_extractor(cfg["patch"], cfg["channels"])
    line2d_collections = [[] for _ in cameras]
    for track_id, track in enumerate(tqdm(tracks)):
        sorted_ids = track.GetSortedImageIds()
        for img_id in sorted_ids:
            camera = cameras[img_id]
            line2d_range = extractor.GetLine2DRange(track, img_id, camera)
            line2d_collections[img_id].append([line2d_range, track_id])
        folder = os.path.join(output_dir, 'track{0}'.format(track_id))
        if not os.path.exists(folder):
            os.makedirs(folder)
    for img_id in tqdm(range(len(cameras))):
        if imname_list is not None:
            import torch
            img = utils.read_image(imname_list[img_id], max_image_dim=max_image_dim, set_gray=False)
            input_image = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)[None,...]
            input_image = input_image.to('cuda')
            featuremap = feature_extractor.extract_featuremaps(input_image)
            feature = featuremap[0][0].detach().cpu().numpy().astype(np.float16)
        else:
            with open(os.path.join(featuremap_dir, "feature_{}.npy").format(img_id), 'rb') as f:
                feature = np.load(f, allow_pickle=True)
        feature  = feature.transpose(1,2,0)
        line2ds_info = line2d_collections[img_id]
        line2ds = [k[0] for k in line2ds_info]
        track_ids = [k[1] for k in line2ds_info]
        patches = extractor.ExtractLinePatches(line2ds, feature)
        for patch, track_id in zip(patches, track_ids):
            fname = os.path.join(output_dir, 'track{0}'.format(track_id), "track{0}_img{1}.npy".format(track_id, img_id))
            _features.write_patch(fname, patch, dtype=cfg["dtype"])
            time.sleep(0.001)

def unit_test_add_noise_to_cameras(cameras):
    newcameras = []
    for cam in cameras:
        newcam = _base.Camera(cam)
        quad = utils.rotmat2quaternion(cam.R)
        quad_noise = (np.random.rand(4) - 0.5) * 0.001
        quad = np.array(quad) + quad_noise
        quad = quad / np.linalg.norm(quad)
        newcam.R = utils.quaternion2rotmat(quad)
        t_noise = (np.random.rand(3) - 0.5) * 0.001
        newcam.T = cam.T + t_noise
        newcameras.append(newcam)
    return newcameras

def compute_errors_extrinsics(cameras, cameras_gt):
    errs_R, errs_T = [], []
    for cam, cam_gt in zip(cameras, cameras_gt):
        err_R = np.linalg.norm(cam.R - cam_gt.R)
        errs_R.append(err_R)
        err_T = np.linalg.norm(cam.T - cam_gt.T)
        errs_T.append(err_T)
    print("R error: {0}, T error: {1}".format(np.mean(errs_R), np.mean(errs_T)))
    return errs_R, errs_T

def visualize_heatmap_intersections(prefix, imname_list, heatmaps, ht_intersections, max_image_dim=None):
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    cNorm  = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap="viridis")

    path = os.path.dirname(prefix)
    if not os.path.exists(path):
        os.makedirs(path)
    for img_id, (imname, heatmap, intersections) in enumerate(tqdm(zip(imname_list, heatmaps, ht_intersections))):
        # visualize image
        img = utils.read_image(imname, max_image_dim=max_image_dim, set_gray=False)
        img = vis.draw_points(img, intersections, (255, 0, 0), 1)
        fname_out = prefix + '_img{0}.png'.format(img_id)
        cv2.imwrite(fname_out, img)

        # visualize heatmap
        heatmap_img = (scalarMap.to_rgba(heatmap)[:,:,:3] * 255).astype(np.uint8)
        heatmap_img = vis.draw_points(heatmap_img, intersections, (255, 0, 0), 1)
        fname_out_heatmap = prefix + '_heatmap{0}.png'.format(img_id)
        cv2.imwrite(fname_out_heatmap, heatmap_img)

def line_bundle_adjustment(cfg, imname_list, tracks, cameras, heatmap_dir, patch_dir, max_image_dim=None):
    '''
    line-based bundle adjustment
    '''
    # load heatmaps
    input_heatmaps = None
    if cfg["use_heatmap"]:
        heatmaps = []
        for img_id in range(len(cameras)):
            with open(os.path.join(heatmap_dir, "heatmap_{}.npy".format(img_id)), 'rb') as f:
                data = np.load(f, allow_pickle=True)
                heatmap = data["data"]
            heatmaps.append(heatmap)
        input_heatmaps = heatmaps

    # load patches
    patches_list = None
    if cfg["use_feature"]:
        patches_list = []
        for track_id, track in enumerate(tqdm(tracks)):
            sorted_ids = track.GetSortedImageIds()
            patches = []
            for img_id in sorted_ids:
                fname = os.path.join(patch_dir, "track{0}_img{1}.npy".format(track_id, img_id))
                patch = _features.load_patch(fname, dtype=cfg["dtype"])
                patches.append(patch)
            patches_list.append(patches)

    # unit test
    # newcameras = unit_test_add_noise_to_cameras(cameras)
    newcameras = cameras

    # optimization
    reconstruction = _base.LineReconstruction(tracks, newcameras)
    lineba_engine = _lineBA.solve(cfg, reconstruction, heatmaps=input_heatmaps, patches_list=patches_list, dtype=cfg["dtype"])
    new_reconstruction = lineba_engine.GetOutputReconstruction()

    # visualization and report
    errs_R, errs_T = compute_errors_extrinsics(reconstruction.GetCameras(), cameras)
    errs_R_opt, errs_T_opt = compute_errors_extrinsics(new_reconstruction.GetCameras(), cameras)
    if cfg["visualize"]:
        ht_intersections = lineba_engine.GetHeatmapIntersections(reconstruction)
        prefix = os.path.join(cfg["vis_folder"], 'heatmap/init/vis')
        visualize_heatmap_intersections(prefix, imname_list, heatmaps, ht_intersections, max_image_dim=max_image_dim)
        ht_intersections = lineba_engine.GetHeatmapIntersections(new_reconstruction)
        prefix = os.path.join(cfg["vis_folder"], 'heatmap/opt/vis')
        visualize_heatmap_intersections(prefix, imname_list, heatmaps, ht_intersections, max_image_dim=max_image_dim)
    pdb.set_trace()

def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(description='refinement with pixelwise optimization')
    arg_parser.add_argument('-i', '--input_folder', type=str, required=True, help='input folder for tracks and infos')
    arg_parser.add_argument('-ht', '--heatmap_folder', type=str, default=None, help='heatmap_folder')
    arg_parser.add_argument('-pt', '--patch_folder', type=str, default=None, help='featuremap_folder')
    arg_parser.add_argument('-c', '--config_file', type=str, default='cfgs/refinement/default.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str, default='cfgs/triangulation/default.yaml', help='default config file')
    arg_parser.add_argument('--extract_patches', action='store_true', help='whether to extract patches')
    arg_parser.add_argument('-ft', '--featuremap_folder', type=str, default=None, help='featuremap_folder')
    arg_parser.add_argument('-o', '--output_folder', type=str, default='tmp/track_patches', help='folder to save the extracted patch')
    arg_parser.add_argument('--visualize', action='store_true', help='whether to do the visualization')
    arg_parser.add_argument('--vis_folder', type=str, default='tmp/vis_lineba', help="folder to save visualization")
    arg_parser.add_argument('--image_list', type=str, default=None, help='imname_list.npy or txt')

    args, unknown = arg_parser.parse_known_args()
    cfg = utils.load_config(args.config_file, default_path=args.default_config_file)
    shortcuts = dict()
    shortcuts['-nv'] = '--n_visible_views'
    cfg = utils.update_config(cfg, unknown, shortcuts)
    cfg["input_folder"] = args.input_folder
    cfg["heatmap_folder"] = args.heatmap_folder
    cfg["patch_folder"] = args.patch_folder
    cfg["extract_patches"] = args.extract_patches
    cfg["featuremap_folder"] = args.featuremap_folder
    cfg["output_folder"] = args.output_folder
    cfg["refinement"]["visualize"] = args.visualize
    cfg["refinement"]["vis_folder"] = args.vis_folder
    cfg["image_list"] = None
    if args.image_list is not None:
        cfg["image_list"] = load_imname_list(args.image_list)

    # check
    if cfg["extract_patches"] and (cfg["featuremap_folder"] is None) and (cfg["image_list"] is None):
        raise ValueError('path to the featuremap should be given. [-ft]')
    else:
        if cfg["refinement"]["use_heatmap"] and cfg["heatmap_folder"] is None:
            raise ValueError("Path to the heatmap should be given. [-ht]")
        if cfg["refinement"]["use_feature"] and cfg["patch_folder"] is None:
            raise ValueError("Path to the patches should be given. [-pt]")
    return cfg

def init_workspace():
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

def load_imname_list(fname):
    if fname.endswith('.txt'):
        with open(fname, 'r') as f:
            lines = f.readlines()
        n_images = int(lines[0].strip('\n'))
        assert n_images == len(lines) - 1
        imname_list = [line.strip('\n').split()[1] for line in lines[1:]]
        return imname_list
    elif fname.endswith('.npy'):
        with open(fname, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            imname_list = data["imname_list"]
        return imname_list
    else:
        raise NotImplementedError

def main():
    cfg = parse_config()
    init_workspace()
    info_path = os.path.join(cfg["input_folder"], "all_infos.npy")
    with open(info_path, 'rb') as f:
        data = np.load(f, allow_pickle=True)
        imname_list, all_2d_segs, cameras_np, cfg_info, n_tracks = data["imname_list"], data["all_2d_segs"], data["cameras_np"], data["cfg"].item(), data["n_tracks"].item()
    # print(cfg)

    cameras = [_base.Camera(cam[0], cam[1], cam[2][:, 0], cam[3]) for cam in cameras_np]
    linetracks = [_base.LineTrack() for i in range(n_tracks)]
    linetracks = vis.load_linetracks_from_folder(linetracks, cfg["input_folder"])
    if cfg["extract_patches"]:
        if not os.path.exists(cfg["output_folder"]):
            os.makedirs(cfg["output_folder"])
        extract_patches(cfg["refinement"], linetracks, cameras, cfg["featuremap_folder"], cfg["output_folder"], imname_list=cfg["image_list"], max_image_dim=cfg["max_image_dim"])
    else:
        line_bundle_adjustment(cfg["refinement"], imname_list, linetracks, cameras, cfg["heatmap_folder"], cfg["patch_folder"], max_image_dim=cfg["max_image_dim"])

if __name__ == '__main__':
    main()

