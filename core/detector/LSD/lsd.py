import os, sys
import numpy as np
import cv2
import pytlsd
from tqdm import tqdm
import utils

def lsd_detect_2d_segs_on_images(imname_list, resize_hw=None, max_image_dim=None, set_gray=True, max_num_2d_segs=None):
    all_2d_segs = []
    print("Start lsd line detection (n_images = {0}).".format(len(imname_list)))
    for idx, imname in enumerate(tqdm(imname_list)):
        img = utils.read_image(imname, resize_hw=resize_hw, max_image_dim=max_image_dim, set_gray=set_gray)
        segs = pytlsd.lsd(img)
        if segs.shape[0] < max_num_2d_segs:
            lengths_squared = (segs[:,2] - segs[:,0]) ** 2 + (segs[:,3] - segs[:,1]) ** 2
            indexes = np.argsort(lengths_squared)[::-1][:max_num_2d_segs]
            segs = segs[indexes,:]
        all_2d_segs.append(segs)
        # print("Finishing lsd line detection (num_lines={0}) on image (id={1}): {2}.".format(segs.shape[0], idx, imname))
    return all_2d_segs


