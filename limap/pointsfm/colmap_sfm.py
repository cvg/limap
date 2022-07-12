import os, sys
import shutil
import numpy as np
import cv2
import subprocess
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import read_write_model as colmap_utils
import database
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def run_colmap_sift_matches(image_path, db_path, use_cuda=False):
    ### [COLMAP] feature extraction and matching
    # feature extraction
    cmd = ['colmap', 'feature_extractor',
            '--database_path', db_path,
            '--image_path', image_path,
            '--SiftExtraction.use_gpu', str(int(use_cuda))]
    subprocess.run(cmd, check=True)
    # matching
    cmd = ['colmap', 'exhaustive_matcher',
           '--database_path', db_path,
           '--SiftMatching.use_gpu', str(int(use_cuda))]
    subprocess.run(cmd, check=True)

def run_hloc_matches(cfg, image_path, db_path):
    image_path = Path(image_path)
    from hloc import extract_features, match_features, reconstruction, triangulation
    outputs = Path(os.path.join(os.path.dirname(db_path), 'hloc_outputs'))
    sfm_pairs = Path(os.path.join(outputs, "pairs-exhaustive.txt"))
    sfm_dir = Path(os.path.join(outputs, "sfm"))
    feature_conf = extract_features.confs[cfg["descriptor"]]
    matcher_conf = match_features.confs[cfg["matcher"]]

    # run superpoint
    feature_path = extract_features.main(feature_conf, image_path, outputs)
    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf["output"], outputs, exhaustive=True)

    sfm_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.create_empty_db(db_path)
    reconstruction.import_images("colmap", sfm_dir, image_path, db_path, False)
    image_ids = reconstruction.get_image_ids(db_path)
    reconstruction.import_features(image_ids, db_path, feature_path)
    reconstruction.import_matches(image_ids, db_path, sfm_pairs, match_path, None, None)
    triangulation.geometric_verification("colmap", db_path, sfm_pairs)

def run_colmap_sfm_with_known_poses(cfg, imagecols, output_path='tmp/tmp_colmap', use_cuda=False, skip_exists=False, map_to_original_image_names=False):
    ### initialize sparse folder
    if skip_exists and os.path.exists(output_path):
        return
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    db_path = os.path.join(output_path, 'db.db')
    image_path = os.path.join(output_path, 'images')
    model_path = os.path.join(output_path, 'sparse', 'model')
    point_triangulation_path = os.path.join(output_path, 'sparse')

    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    ### copy images to tmp folder
    for img_id in imagecols.get_img_ids():
        img = imagecols.read_image(img_id)
        fname_to_save = os.path.join(image_path, 'image{0:08d}.png'.format(img_id))
        cv2.imwrite(fname_to_save, img)

    # sift feature extraction and matching
    if cfg["fbase"] == "sift":
        run_colmap_sift_matches(image_path, db_path, use_cuda=use_cuda)
    elif cfg["fbase"] == "hloc":
        assert (use_cuda == True)
        run_hloc_matches(cfg["hloc"], image_path, db_path)

    ### write cameras.txt
    colmap_cameras = {}
    for cam_id in range(imagecols.NumCameras()):
        cam = imagecols.cam(cam_id)
        model_id = cam.model_id()
        model_name = None
        if model_id == 0:
            model_name = "SIMPLE_PINHOLE"
        elif model_id == 1:
            model_name = "PINHOLE"
        else:
            raise ValueError("The provided camera model should be without distortion.")
        colmap_cameras[cam_id] = colmap_utils.Camera(id=cam_id, model=model_name, width=cam.w(), height=cam.h(), params=cam.params())
    fname = os.path.join(model_path, 'cameras.txt')
    colmap_utils.write_cameras_text(colmap_cameras, fname)

    ### write images.txt
    # [IMPORTANT] get image id from the database
    db = database.COLMAPDatabase(db_path)
    rows = db.execute("SELECT * from images")
    colmap_images = {}
    for idx, img_id in enumerate(imagecols.get_img_ids()):
        kk = next(rows)
        assert (kk[0] == idx + 1)
        imname = kk[1]
        img_id_orig = int(imname[5:-4])
        camimage = imagecols.camimage(img_id_orig)
        cam_id = camimage.cam_id
        qvec = camimage.pose.qvec
        tvec = camimage.pose.tvec
        colmap_images[idx+1] = colmap_utils.Image(id=idx+1, qvec=qvec, tvec=tvec,
                                                  camera_id=cam_id, name=imname,
                                                  xys=[], point3D_ids=[])
    fname = os.path.join(model_path, 'images.txt')
    colmap_utils.write_images_text(colmap_images, fname)

    ### write empty points3D.txt
    fname = os.path.join(model_path, 'points3D.txt')
    colmap_utils.write_points3D_text({}, fname)

    ### [COLMAP] point triangulation
    # point triangulation
    cmd = ['colmap', 'point_triangulator',
           '--database_path', db_path,
           '--image_path', image_path,
           '--input_path', model_path,
           '--output_path', point_triangulation_path]
    subprocess.run(cmd, check=True)

    # map to original image names
    if map_to_original_image_names:
        fname_images_bin = os.path.join(point_triangulation_path, "images.bin")
        colmap_images = colmap_utils.read_images_binary(fname_images_bin)
        for idx, img_id in enumerate(imagecols.get_img_ids()):
            colmap_images[idx + 1] = colmap_images[idx + 1]._replace(name = imagecols.image_name(img_id))
        colmap_utils.write_images_binary(colmap_images, fname_images_bin)

