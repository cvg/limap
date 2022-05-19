import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import limap.util.io_utils as limapio

def setup(cfg):
    folder_save = cfg["output_dir"]
    if folder_save is None:
        folder_save = 'tmp'
    limapio.check_makedirs(folder_save)
    folder_load = cfg["load_dir"]
    if cfg["use_tmp"]: folder_load = "tmp"
    if folder_load is None:
        folder_load = folder_save
    cfg["dir_save"] = folder_save
    cfg["dir_load"] = folder_load
    print("output dir: {0}".format(cfg["dir_save"]))
    print("loading dir: {0}".format(cfg["dir_load"]))
    return cfg

def compute_sfminfos(cfg, imagecols, fname="metainfos.txt"):
    import limap.pointsfm as _psfm
    if not cfg["load_meta"]:
        # run colmap sfm and compute neighbors, ranges
        colmap_output_path = os.path.join(cfg["dir_save"], cfg["sfm"]["colmap_output_path"])
        if not cfg["sfm"]["reuse"]:
            _psfm.run_colmap_sfm_with_known_poses(cfg["sfm"], imagecols, output_path=colmap_output_path, use_cuda=cfg["use_cuda"])
        model = _psfm.SfmModel()
        model.ReadFromCOLMAP(colmap_output_path, "sparse", "images")
        neighbors = _psfm.ComputeNeighborsSorted(model, cfg["n_neighbors"], min_triangulation_angle=cfg["sfm"]["min_triangulation_angle"], neighbor_type=cfg["sfm"]["neighbor_type"])
        ranges = model.ComputeRanges(cfg["sfm"]["ranges"]["range_robust"], cfg["sfm"]["ranges"]["k_stretch"])
        fname_save = os.path.join(cfg["dir_save"], fname)
        limapio.save_txt_metainfos(fname_save, neighbors, ranges)
    else:
        # load from precomputed info
        limapio.check_path(cfg["dir_load"])
        fname_load = os.path.join(cfg["dir_load"], fname)
        neighbors, ranges = limapio.read_txt_metainfos(fname_load)
        neighbors = [neighbor[:cfg["n_neighbors"]] for neighbor in neighbors]
    return neighbors, ranges

def compute_2d_segs(cfg, imagecols, compute_descinfo=True):
    import limap.line2d
    basedir = os.path.join("line_detections", cfg["line2d"]["detector"]["method"])
    folder_save = os.path.join(cfg["dir_save"], basedir)
    descinfo_folder = None
    se_det = cfg["skip_exists"] or cfg["line2d"]["detector"]["skip_exists"]
    se_ext = cfg["skip_exists"] or cfg["line2d"]["extractor"]["skip_exists"]
    detector = limap.line2d.get_detector(cfg["line2d"]["detector"], max_num_2d_segs=cfg["line2d"]["max_num_2d_segs"])
    if not cfg["load_det"]:
        if compute_descinfo and cfg["line2d"]["detector"]["method"] == cfg["line2d"]["extractor"]["method"]:
            all_2d_segs, descinfo_folder = detector.detect_and_extract_all_images(folder_save, imagecols, skip_exists=(se_det and se_ext))
        else:
            all_2d_segs = detector.detect_all_images(folder_save, imagecols, skip_exists=se_det)
            if compute_descinfo:
                extractor = limap.line2d.get_extractor(cfg["line2d"]["extractor"])
                descinfo_folder = extractor.extract_all_images(folder_save, imagecols, all_2d_segs, skip_exists=se_ext)
    else:
        folder_load = os.path.join(cfg["dir_load"], basedir)
        all_2d_segs = limapio.read_all_segments_from_folder(detector.get_segments_folder(folder_load))
        extractor = limap.line2d.get_extractor(cfg["line2d"]["extractor"])
        if compute_descinfo:
            descinfo_folder = extractor.extract_all_images(folder_save, imagecols, all_2d_segs, skip_exists=se_ext)
        else:
            descinfo_folder = extractor.get_descinfo_folder(folder_load)
    if cfg["line2d"]["visualize"]:
        detector.visualize_segs(folder_save, imagecols, first_k=10)
    if cfg["line2d"]["save_l3dpp"]:
        limapio.save_l3dpp(os.path.join(folder_save, "l3dpp_format"), imagecols, all_2d_segs)
    return all_2d_segs, descinfo_folder

def compute_matches(cfg, descinfo_folder, neighbors):
    import limap.line2d
    basedir = os.path.join("line_matchings", cfg["line2d"]["detector"]["method"], "feats_{0}".format(cfg["line2d"]["extractor"]["method"]))
    extractor = limap.line2d.get_extractor(cfg["line2d"]["extractor"])
    se_match = cfg["skip_exists"] or cfg["line2d"]["matcher"]["skip_exists"]
    matcher = limap.line2d.get_matcher(cfg["line2d"]["matcher"], extractor, n_neighbors=cfg["n_neighbors"])
    if not cfg["load_match"]:
        folder_save = os.path.join(cfg["dir_save"], basedir)
        matches_folder = matcher.match_all_neighbors(folder_save, neighbors, descinfo_folder, skip_exists=se_match)
    else:
        folder_load = os.path.join(cfg["dir_load"], basedir)
        matches_folder = matcher.get_matchings_folder(folder_load)
    return matches_folder

