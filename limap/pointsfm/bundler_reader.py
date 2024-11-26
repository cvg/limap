import os

import imagesize
import numpy as np
from _limap import _base, _pointsfm
from pycolmap import logging
from tqdm import tqdm


def ReadModelBundler(bundler_path, list_path, model_path):
    ################################
    # read imname_list
    ################################
    list_path = os.path.join(bundler_path, list_path)
    logging.info(f"Loading bundler list file {list_path}...")
    with open(list_path) as f:
        lines = f.readlines()
    image_names = [line.strip("\n").split(" ")[0] for line in lines]
    imname_list = [
        os.path.join(bundler_path, img_name) for img_name in image_names
    ]

    ################################
    # read sfm model
    ################################
    model_path = os.path.join(bundler_path, model_path)
    logging.info(f"Loading bundler model file {model_path}...")
    with open(model_path) as f:
        lines = f.readlines()
    counter = 1  # start from the second line
    line = lines[counter].strip("\n").split(" ")
    n_images, n_points = int(line[0]), int(line[1])
    counter += 1

    # construct SfmModel instance
    cameras, camimages = [], []
    model = _pointsfm.SfmModel()
    # read camviews
    for img_id in tqdm(range(n_images)):
        # read camera
        line = lines[counter].strip("\n").split(" ")
        f, k1, k2 = float(line[0]), float(line[1]), float(line[2])
        counter += 1
        imname = imname_list[img_id]
        if not os.path.exists(imname):
            raise ValueError(f"Error! Image not found: {imname}")
        width, height = imagesize.get(imname)
        img_hw = [height, width]
        # K = np.zeros((3, 3))
        cx = img_hw[1] / 2.0
        cy = img_hw[0] / 2.0
        params = [f, cx, cy, k1, k2]
        camera = _base.Camera("RADIAL", params, cam_id=img_id, hw=img_hw)
        cameras.append(camera)

        # read pose
        R = np.zeros((3, 3))
        for row_id in range(3):
            line = lines[counter].strip("\n").split(" ")
            R[row_id, 0], R[row_id, 1], R[row_id, 2] = (
                float(line[0]),
                float(line[1]),
                float(line[2]),
            )
            counter += 1
        R[1, :] = -R[1, :]  # for bundler format
        R[2, :] = -R[2, :]  # for bundler format
        T = np.zeros(3)
        line = lines[counter].strip("\n").split(" ")
        T[0], T[1], T[2] = float(line[0]), float(line[1]), float(line[2])
        T[1:] = -T[1:]  # for bundler format
        counter += 1
        pose = _base.CameraPose(R, T)
        camimage = _base.CameraImage(img_id, pose, image_name=imname)
        camimages.append(camimage)

        # add image
        image = _pointsfm.SfmImage(
            imname,
            img_hw[1],
            img_hw[0],
            camera.K().reshape(-1).tolist(),
            camimage.pose.R().reshape(-1).tolist(),
            camimage.pose.T().tolist(),
        )
        model.addImage(image)

    # get image collection
    imagecols = _base.ImageCollection(cameras, camimages)

    # read points
    for _ in tqdm(range(n_points)):
        line = lines[counter].strip("\n").split(" ")
        x, y, z = float(line[0]), float(line[1]), float(line[2])
        counter += 1
        counter += 1  # skip color
        line = lines[counter].strip("\n").split(" ")
        n_views = int(line[0])
        subcounter = 1
        track = []
        for _ in range(n_views):
            track.append(int(line[subcounter]))
            subcounter += 4
        model.addPoint(x, y, z, track)
        counter += 1
    return model, imagecols
