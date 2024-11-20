import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from tqdm import tqdm

import limap.base as base
import limap.undistortion as undistortion

data_dir = os.path.expanduser("~/data/Localization/Aachen-1.1")
img_orig_dir = os.path.join(data_dir, "images_upright")
img_undistort_dir = os.path.join(data_dir, "undistorted")
list_file = os.path.join(
    data_dir, "queries", "night_time_queries_with_intrinsics.txt"
)
camerainfos_file = "camerainfos_night_undistorted.txt"


def load_list_file(fname):
    with open(fname) as f:
        lines = f.readlines()
    imname_list, cameras = [], []
    for line in lines:
        line = line.strip("\n")
        k = line.split(" ")
        imname = k[0]
        # Aachen only uses simple radial model
        assert k[1] == "SIMPLE_RADIAL"
        f = float(k[4])
        cx, cy = float(k[5]), float(k[6])
        k1 = float(k[7])
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        camera = base.Camera(
            K, np.eye(3), np.zeros(3), np.array([k1, 0, 0, 0, 0])
        )
        imname_list.append(imname)
        cameras.append(camera)
    return imname_list, cameras


def process(image_list, cameras):
    with open(camerainfos_file, "w") as f:
        n_images = len(image_list)
        for img_id in tqdm(range(n_images)):
            imname, camera = image_list[img_id], cameras[img_id]
            imname_orig = os.path.join(img_orig_dir, imname)
            imname_undist = os.path.join(img_undistort_dir, imname)
            path = os.path.dirname(imname_undist)
            if not os.path.exists(path):
                os.makedirs(path)
            camera_undistorted = undistortion.undistort_image_camera(
                camera, imname_orig, imname_undist
            )
            img = cv2.imread(imname_undist)
            h, w = img.shape[0], img.shape[1]
            assert camera_undistorted.K[0, 0] == camera_undistorted.K[1, 1]
            fx = camera_undistorted.K[0, 0]
            cx, cy = camera_undistorted.K[0, 2], camera_undistorted.K[1, 2]
            f.write(f"{imname_undist} SIMPLE_PINHOLE {w} {h} {fx} {cx} {cy}\n")


if __name__ == "__main__":
    image_list, cameras = load_list_file(list_file)
    process(image_list, cameras)
