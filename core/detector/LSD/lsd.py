import numpy as np
import pytlsd
from tqdm import tqdm
import utils

def lsd_detect_2d_segs_on_images(imagecols, set_gray=True, max_num_2d_segs=None):
    all_2d_segs = []
    print("Start lsd line detection (n_images = {0}).".format(imagecols.NumImages()))
    for img_id in tqdm(range(imagecols.NumImages())):
        img = imagecols.read_image(img_id, set_gray=set_gray)
        segs = pytlsd.lsd(img)
        if segs.shape[0] > max_num_2d_segs:
            lengths_squared = (segs[:,2] - segs[:,0]) ** 2 + (segs[:,3] - segs[:,1]) ** 2
            indexes = np.argsort(lengths_squared)[::-1][:max_num_2d_segs]
            segs = segs[indexes,:]
        all_2d_segs.append(segs)
        # print("Finishing lsd line detection (num_lines={0}) on image (id={1}): {2}.".format(segs.shape[0], img_id, imagecols.camimage(img_id).image_name()))
    return all_2d_segs

