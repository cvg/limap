import os
import numpy as np
import warnings
from tqdm import tqdm
import limap.util.io as limapio

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
    print("[LOG] Output dir: {0}".format(cfg["dir_save"]))
    print("[LOG] Loading dir: {0}".format(cfg["dir_load"]))
    if "weight_path" in cfg and cfg["weight_path"] is not None:
        cfg["weight_path"] = os.path.expanduser(cfg["weight_path"])
        print("[LOG] weight dir: {0}".format(cfg["weight_path"]))
    return cfg

def undistort_images(imagecols, output_dir, fname="image_collection_undistorted.npy", load_undistort=False, n_jobs=-1):
    import limap.base as _base

    loaded_ids = []
    unload_ids = imagecols.get_img_ids()
    if load_undistort:
        print("[LOG] Loading undistorted images (n_images = {0})...".format(imagecols.NumImages()))
        fname_in = os.path.join(output_dir, fname)
        if os.path.isfile(fname_in):
            data = limapio.read_npy(fname_in).item()
            loaded_imagecols = _base.ImageCollection(data)
            unload_ids = []
            for id in imagecols.get_img_ids():
                if id in loaded_imagecols.get_img_ids():
                    loaded_ids.append(id)
                else:
                    unload_ids.append(id)
            loaded_imagecols = loaded_imagecols.subset_by_image_ids(loaded_ids)
            if len(unload_ids) == 0:
                return loaded_imagecols

    # start undistortion
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    print("[LOG] Start undistorting images (n_images = {0})...".format(len(unload_ids)))
    import limap.undistortion as _undist
    import cv2, imagesize
    import joblib
    # limapio.delete_folder(output_dir)
    limapio.check_makedirs(output_dir)

    # multi-process undistortion
    def process(imagecols, img_id):
        cam_id = imagecols.camimage(img_id).cam_id
        cam = imagecols.cam(cam_id)
        imname_in = imagecols.camimage(img_id).image_name()
        imname_out = os.path.join(output_dir, "image{0:08d}.png".format(img_id))
        # save image if resizing is needed
        width, height = imagesize.get(imname_in)
        if height != cam.h() or width != cam.w():
            img = imagecols.read_image(img_id)
            cv2.imwrite(imname_out, img)
            imname_in = imname_out
        cam_undistorted =  _undist.UndistortImageCamera(cam, imname_in, imname_out)
        cam_undistorted.set_cam_id(cam_id)
        return cam_undistorted
    outputs = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(process)(imagecols, img_id) for img_id in tqdm(unload_ids))

    # update new imagecols
    imagecols_undistorted = _base.ImageCollection(imagecols)
    cam_dict = {}
    for idx, img_id in enumerate(unload_ids):
        imname_out = os.path.join(output_dir, "image{0:08d}.png".format(img_id))
        cam_undistorted = outputs[idx]
        cam_id = cam_undistorted.cam_id()
        if cam_id not in cam_dict:
            cam_dict[cam_id] = cam_undistorted
            imagecols_undistorted.change_camera(cam_id, cam_undistorted)
        imagecols_undistorted.change_image_name(img_id, imname_out)
    for idx, img_id in enumerate(loaded_ids):
        imname_out = os.path.join(output_dir, "image{0:08d}.png".format(img_id))
        cam_id = loaded_imagecols.camimage(img_id).cam_id
        cam_undistorted = loaded_imagecols.cam(cam_id)
        if cam_id not in cam_dict:
            cam_dict[cam_id] = cam_undistorted
            imagecols_undistorted.change_camera(cam_id, cam_undistorted)
        imagecols_undistorted.change_image_name(img_id, imname_out)

    # save info
    limapio.save_txt_imname_dict(os.path.join(output_dir, 'original_image_list.txt'), imagecols.get_image_name_dict())
    limapio.save_npy(os.path.join(output_dir, fname), imagecols_undistorted.as_dict())
    return imagecols_undistorted

def compute_sfminfos(cfg, imagecols, fname="metainfos.txt"):
    import limap.pointsfm as _psfm
    if not cfg["load_meta"]:
        # run colmap sfm and compute neighbors, ranges
        colmap_output_path = os.path.join(cfg["dir_save"], cfg["sfm"]["colmap_output_path"])
        if not cfg["sfm"]["reuse"]:
            _psfm.run_colmap_sfm_with_known_poses(cfg["sfm"], imagecols, output_path=colmap_output_path, skip_exists=cfg["skip_exists"])
        model = _psfm.SfmModel()
        model.ReadFromCOLMAP(colmap_output_path, "sparse", "images")
        neighbors, ranges = _psfm.compute_metainfos(cfg["sfm"], model, n_neighbors=cfg["n_neighbors"])
        fname_save = os.path.join(cfg["dir_save"], fname)
        limapio.save_txt_metainfos(fname_save, neighbors, ranges)
    else:
        # load from precomputed info
        colmap_output_path = os.path.join(cfg["dir_load"], cfg["sfm"]["colmap_output_path"])
        limapio.check_path(cfg["dir_load"])
        fname_load = os.path.join(cfg["dir_load"], fname)
        neighbors, ranges = limapio.read_txt_metainfos(fname_load)
        for img_id, neighbor in neighbors.items():
            neighbors[img_id] = neighbors[img_id][:cfg["n_neighbors"]]
    return colmap_output_path, neighbors, ranges

def compute_2d_segs(cfg, imagecols, compute_descinfo=True):
    weight_path = None if "weight_path" not in cfg else cfg["weight_path"]
    if "extractor" in cfg["line2d"]:
        print("[LOG] Start 2D line detection and description (detector = {0}, extractor = {1}, n_images = {2})...".format(cfg["line2d"]["detector"]["method"], cfg["line2d"]["extractor"]["method"], imagecols.NumImages()))
    else:
        print("[LOG] Start 2D line detection and description (detector = {0}, n_images = {1})...".format(cfg["line2d"]["detector"]["method"], imagecols.NumImages()))
    import limap.line2d
    if not imagecols.IsUndistorted():
        warnings.warn("The input images are distorted!")
    basedir = os.path.join("line_detections", cfg["line2d"]["detector"]["method"])
    folder_save = os.path.join(cfg["dir_save"], basedir)
    descinfo_folder = None
    se_det = cfg["skip_exists"] or cfg["line2d"]["detector"]["skip_exists"]
    if compute_descinfo:
        se_ext = cfg["skip_exists"] or cfg["line2d"]["extractor"]["skip_exists"]
    detector = limap.line2d.get_detector(cfg["line2d"]["detector"], max_num_2d_segs=cfg["line2d"]["max_num_2d_segs"], do_merge_lines=cfg["line2d"]["do_merge_lines"], visualize=cfg["line2d"]["visualize"], weight_path=weight_path)
    if not cfg["load_det"]:
        if compute_descinfo and cfg["line2d"]["detector"]["method"] == cfg["line2d"]["extractor"]["method"] and (not cfg["line2d"]["do_merge_lines"]):
            all_2d_segs, descinfo_folder = detector.detect_and_extract_all_images(folder_save, imagecols, skip_exists=(se_det and se_ext))
        else:
            all_2d_segs = detector.detect_all_images(folder_save, imagecols, skip_exists=se_det)
            if compute_descinfo:
                extractor = limap.line2d.get_extractor(cfg["line2d"]["extractor"], weight_path=weight_path)
                descinfo_folder = extractor.extract_all_images(folder_save, imagecols, all_2d_segs, skip_exists=se_ext)
    else:
        folder_load = os.path.join(cfg["dir_load"], basedir)
        all_2d_segs = limapio.read_all_segments_from_folder(detector.get_segments_folder(folder_load))
        all_2d_segs = {id: all_2d_segs[id] for id in imagecols.get_img_ids()}
        descinfo_folder = None
        if compute_descinfo:
            extractor = limap.line2d.get_extractor(cfg["line2d"]["extractor"], weight_path=weight_path)
            descinfo_folder = extractor.extract_all_images(folder_save, imagecols, all_2d_segs, skip_exists=se_ext)
    if cfg["line2d"]["save_l3dpp"]:
        limapio.save_l3dpp(os.path.join(folder_save, "l3dpp_format"), imagecols, all_2d_segs)
    return all_2d_segs, descinfo_folder

def compute_matches(cfg, descinfo_folder, image_ids, neighbors):
    weight_path = None if "weight_path" not in cfg else cfg["weight_path"]
    print("[LOG] Start matching 2D lines... (extractor = {0}, matcher = {1}, n_images = {2}, n_neighbors = {3})".format(cfg["line2d"]["extractor"]["method"], cfg["line2d"]["matcher"]["method"], len(image_ids), cfg["n_neighbors"]))
    import limap.line2d
    basedir = os.path.join("line_matchings", cfg["line2d"]["detector"]["method"], "feats_{0}".format(cfg["line2d"]["extractor"]["method"]))
    extractor = limap.line2d.get_extractor(cfg["line2d"]["extractor"], weight_path=weight_path)
    se_match = cfg["skip_exists"] or cfg["line2d"]["matcher"]["skip_exists"]
    matcher = limap.line2d.get_matcher(cfg["line2d"]["matcher"], extractor, n_neighbors=cfg["n_neighbors"], weight_path=weight_path)
    if not cfg["load_match"]:
        folder_save = os.path.join(cfg["dir_save"], basedir)
        matches_folder = matcher.match_all_neighbors(folder_save, image_ids, neighbors, descinfo_folder, skip_exists=se_match)
    else:
        folder_load = os.path.join(cfg["dir_load"], basedir)
        matches_folder = matcher.get_matches_folder(folder_load)
    return matches_folder

def compute_exhausive_matches(cfg, descinfo_folder, image_ids):
    print("[LOG] Start exhausive matching 2D lines... (extractor = {0}, matcher = {1}, n_images = {2})".format(cfg["line2d"]["extractor"]["method"], cfg["line2d"]["matcher"]["method"], len(image_ids)))
    import limap.line2d
    basedir = os.path.join("line_matchings", cfg["line2d"]["detector"]["method"], "feats_{0}".format(cfg["line2d"]["extractor"]["method"]))
    extractor = limap.line2d.get_extractor(cfg["line2d"]["extractor"])
    se_match = cfg["skip_exists"] or cfg["line2d"]["matcher"]["skip_exists"]
    matcher = limap.line2d.get_matcher(cfg["line2d"]["matcher"], extractor, n_neighbors=cfg["n_neighbors"])
    if not cfg["load_match"]:
        folder_save = os.path.join(cfg["dir_save"], basedir)
        matches_folder = matcher.match_all_exhaustive_pairs(folder_save, image_ids, descinfo_folder, skip_exists=se_match)
    else:
        folder_load = os.path.join(cfg["dir_load"], basedir)
        matches_folder = matcher.get_matches_folder(folder_load)
    return matches_folder

