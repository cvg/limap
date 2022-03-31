import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import core.detector.LSD as lsd
import core.detector.SOLD2 as sold2
import core.visualize as vis
import core.utils as utils

def setup(cfg, camviews):
    print("number of images: {0}".format(len(camviews)))
    folder_to_save = cfg["folder_to_save"]
    if cfg["folder_to_save"] is None:
        folder_to_save = 'tmp'
    if not os.path.exists(folder_to_save): os.makedirs(folder_to_save)
    folder_to_load = cfg["folder_to_load"]
    if cfg["use_tmp"]: folder_to_load = "tmp"
    cfg["dir_save"] = folder_to_save
    cfg["dir_load"] = folder_to_load
    return cfg

def compute_sfminfos(cfg, camviews, fname="sfm_metainfos.npy"):
    import limap.pointsfm as _psfm
    if not cfg["load_meta"]:
        # run colmap sfm and compute neighbors, ranges
        colmap_output_path = cfg["sfm"]["colmap_output_path"]
        if not cfg["sfm"]["reuse"]:
            _psfm.run_colmap_sfm_with_known_poses(cfg["sfm"], camviews, output_path=colmap_output_path, use_cuda=cfg["use_cuda"])
        model = _psfm.SfmModel()
        model.ReadFromCOLMAP(colmap_output_path, "sparse", "images")
        neighbors = _psfm.ComputeNeighborsSorted(model, cfg["n_neighbors"], min_triangulation_angle=cfg["sfm"]["min_triangulation_angle"], neighbor_type=cfg["sfm"]["neighbor_type"])
        ranges = model.ComputeRanges(cfg["sfm"]["ranges"]["range_robust"], cfg["sfm"]["ranges"]["k_stretch"])
        with open(os.path.join(cfg["dir_save"], fname), 'wb') as f: np.savez(f, neighbors=neighbors, ranges=ranges)
    else:
        fname_load = os.path.join(cfg["dir_load"], fname)
        with open(fname_load, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            neighbors = data['neighbors']
            neighbors = [neighbor[:cfg["n_neighbors"]] for neighbor in neighbors]
            ranges = data['ranges']
    return neighbors, ranges

def compute_2d_segs(cfg, camviews, compute_descinfo=True):
    descinfo_folder = None
    compute_descinfo = (compute_descinfo and (not cfg["load_match"]) and (not cfg["load_det"])) or cfg["line2d"]["compute_descinfo"]
    image_names = [camview.image_name() for camview in camviews]
    if not cfg["load_det"]:
        descinfo_folder = os.path.join(cfg["dir_save"], "{0}_descinfos".format(cfg["line2d"]["detector"]))
        heatmap_dir = os.path.join(cfg["dir_save"], 'sold2_heatmaps')
        if cfg["line2d"]["detector"] == "sold2":
            all_2d_segs, descinfos = sold2.sold2_detect_2d_segs_on_images(camviews, heatmap_dir=heatmap_dir, max_num_2d_segs=cfg["line2d"]["max_num_2d_segs"])
            vis.save_datalist_to_folder(descinfo_folder, 'descinfo', image_names, descinfos, is_descinfo=True)
            del descinfos
        elif cfg["line2d"]["detector"] == "lsd":
            all_2d_segs = lsd.lsd_detect_2d_segs_on_images(camviews, max_num_2d_segs=cfg["line2d"]["max_num_2d_segs"])
        with open(os.path.join(cfg["dir_save"], '{0}_all_2d_segs.npy'.format(cfg["line2d"]["detector"])), 'wb') as f: np.savez(f, all_2d_segs=all_2d_segs)
        if cfg["line2d"]["detector"] != "sold2" and compute_descinfo:
            # we use the sold2 descriptors for all detectors for now
            sold2.sold2_compute_descinfos(camviews, all_2d_segs, descinfo_dir=descinfo_folder)
    else:
        descinfo_folder = os.path.join(cfg["dir_load"], "{0}_descinfos".format(cfg["line2d"]["detector"]))
        fname_all_2d_segs = os.path.join(cfg["dir_load"], "{0}_all_2d_segs.npy".format(cfg["line2d"]["detector"]))
        print("Loading {0}...".format(fname_all_2d_segs))
        with open(fname_all_2d_segs, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            all_2d_segs = data['all_2d_segs']
        if compute_descinfo:
            descinfo_folder = os.path.join(cfg["dir_save"], "{0}_descinfos".format(cfg["line2d"]["detector"]))
            sold2.sold2_compute_descinfos(camviews, all_2d_segs, descinfo_dir=descinfo_folder)
    # visualize
    if cfg["line2d"]["visualize"]:
        vis.tmp_visualize_2d_segs(camviews, all_2d_segs)
    if cfg["line2d"]["save_l3dpp"]:
        img_hw = [camviews[0].h(), camviews[0].w()]
        vis.tmp_save_all_2d_segs_for_l3dpp(image_names, all_2d_segs, img_hw, folder=os.path.join(cfg["dir_save"], "l3dpp"))
    return all_2d_segs, descinfo_folder

def compute_matches(cfg, camviews, descinfo_folder, neighbors):
    fname_all_matches = '{0}_all_matches_n{1}_top{2}.npy'.format(cfg["line2d"]["detector"], cfg["n_neighbors"], cfg["line2d"]["topk"])
    matches_dir = '{0}_all_matches_n{1}_top{2}'.format(cfg["line2d"]["detector"], cfg["n_neighbors"], cfg["line2d"]["topk"])
    if not cfg['load_match']:
        if descinfo_folder is None:
            descinfo_folder = os.path.join(cfg["dir_load"], "{0}_descinfos".format(cfg["line2d"]["detector"]))
        matches_folder = os.path.join(cfg["dir_save"], matches_dir)
        if cfg["line2d"]["topk"] == 0:
            all_matches = sold2.sold2_match_2d_segs_with_descinfo_by_folder(descinfo_folder, neighbors, n_jobs=cfg["line2d"]["n_jobs"], matches_dir=matches_folder)
        else:
            all_matches = sold2.sold2_match_2d_segs_with_descinfo_topk_by_folder(descinfo_folder, neighbors, topk=cfg["line2d"]["topk"], n_jobs=cfg["line2d"]["n_jobs"], matches_dir=matches_folder)
        return matches_folder
    else:
        folder = os.path.join(cfg["dir_load"], matches_dir)
        if not os.path.exists(folder):
            raise ValueError("Folder {0} not found.".format(folder))
        return folder


