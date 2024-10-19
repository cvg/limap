import os

import imagesize
import numpy as np
from _limap import _base, _pointsfm
from tqdm import tqdm

from limap.util.geometry import rotation_from_quaternion


def ReadModelVisualSfM(vsfm_path, nvm_file="reconstruction.nvm"):
    input_file = os.path.join(vsfm_path, nvm_file)
    if not os.path.exists(input_file):
        raise ValueError(f"Error! Input file {input_file} does not exist.")
    with open(input_file) as f:
        txt_lines = f.readlines()

    # read camviews
    counter = 2  # start from the third line
    n_images = int(txt_lines[counter].strip())
    counter += 1
    cameras, camimages = [], []
    model = _pointsfm.SfmModel()
    for img_id in tqdm(range(n_images)):
        line = txt_lines[counter].strip().split()
        counter += 1

        imname = os.path.join(vsfm_path, line[0])
        f = float(line[1])
        qvec = np.array([float(line[k]) for k in np.arange(2, 6).tolist()])
        center_vec = np.array(
            [float(line[k]) for k in np.arange(6, 9).tolist()]
        )
        k1 = -float(line[9])

        # add camera
        if not os.path.exists(imname):
            raise ValueError(f"Error! Image not found: {imname}")
        width, height = imagesize.get(imname)
        img_hw = [height, width]
        cx = img_hw[1] / 2.0
        cy = img_hw[0] / 2.0
        params = [f, cx, cy, k1]
        camera = _base.Camera("SIMPLE_RADIAL", params, cam_id=img_id, hw=img_hw)
        cameras.append(camera)

        # add image
        R = rotation_from_quaternion(qvec)
        T = -R @ center_vec
        pose = _base.CameraPose(R, T)
        camimage = _base.CameraImage(img_id, pose, image_name=imname)
        camimages.append(camimage)

        # add image to model
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
    counter += 1
    n_points = int(txt_lines[counter].strip())
    counter += 1
    for _ in tqdm(range(n_points)):
        line = txt_lines[counter].strip().split()
        counter += 1
        point = np.array([float(line[k]) for k in range(3)])
        # color = np.array([int(line[k]) for k in np.arange(3, 6).tolist()])
        n_views = int(line[6])
        track = []
        subcounter = 7
        for _ in range(n_views):
            track.append(int(line[subcounter]))
            subcounter += 4
        model.addPoint(point[0], point[1], point[2], track)
    return model, imagecols
