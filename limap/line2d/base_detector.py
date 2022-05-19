import os, sys
import numpy as np
from tqdm import tqdm
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import limap.util.io_utils as limapio
import limap.visualize as limapvis

class BaseDetector():
    def __init__(self, set_gray=True, max_num_2d_segs=3000, n_jobs=1):
        self.set_gray = set_gray
        self.max_num_2d_segs = max_num_2d_segs
        self.n_jobs = n_jobs

    # The functions below are required for detectors
    def get_module_name(self):
        raise NotImplementedError
    def detect(self, output_folder, img_id, camview):
        raise NotImplementedError
    def extract(self, output_folder, img_id, camview, segs):
        raise NotImplementedError
    def detect_and_extract(self, output_folder, img_id, camview):
        raise NotImplementedError
    # The functions below are required for extractors
    def save_descinfo(self, descinfo_folder, idx, descinfo):
        raise NotImplementedError
    def read_descinfo(self, descinfo_folder, idx):
        raise NotImplementedError

    def get_segments_folder(self, output_folder):
        return os.path.join(output_folder, "segments")
    def get_descinfo_folder(self, output_folder):
        return os.path.join(output_folder, "descinfos", self.get_module_name())
    def take_longest_k(self, segs, max_num_2d_segs=3000):
        if max_num_2d_segs is None or max_num_2d_segs == -1:
            pass
        elif segs.shape[0] > max_num_2d_segs:
            lengths_squared = (segs[:,2] - segs[:,0]) ** 2 + (segs[:,3] - segs[:,1]) ** 2
            indexes = np.argsort(lengths_squared)[::-1][:max_num_2d_segs]
            segs = segs[indexes,:]
        return segs
    def save_segs(self, output_folder, idx, segs):
        seg_folder = self.get_segments_folder(output_folder)
        limapio.check_makedirs(seg_folder)
        limapio.save_txt_segments(seg_folder, idx, segs)
    def visualize_segs(self, output_folder, imagecols, first_k=10):
        seg_folder = self.get_segments_folder(output_folder)
        n_vis_images = min(first_k, imagecols.NumImages())
        vis_folder = os.path.join(output_folder, "visualize")
        limapio.check_makedirs(vis_folder)
        for img_id in range(n_vis_images):
            img = imagecols.read_image(img_id)
            segs = limapio.read_txt_segments(seg_folder, img_id)
            img = limapvis.draw_segments(img, segs, (0, 255, 0))
            fname = os.path.join(vis_folder, "img_{0}_det.png".format(img_id))
            cv2.imwrite(fname, img)

    # TODO: multiprocessing
    def detect_all_images(self, output_folder, imagecols):
        limapio.check_makedirs(output_folder)
        for img_id in tqdm(range(imagecols.NumImages())):
            self.detect(output_folder, img_id, imagecols.camview(img_id))
        seg_folder = self.get_segments_folder(output_folder)
        all_2d_segs = limapio.read_all_segments_from_folder(seg_folder)
        return all_2d_segs

    def extract_all_images(self, output_folder, imagecols, all_2d_segs):
        limapio.check_makedirs(output_folder)
        for img_id in tqdm(range(imagecols.NumImages())):
            self.extract(output_folder, img_id, imagecols.camview(img_id), all_2d_segs[img_id])
        return self.get_descinfo_folder(output_folder)

    def detect_and_extract_all_images(self, output_folder, imagecols):
        limapio.check_makedirs(output_folder)
        for img_id in tqdm(range(imagecols.NumImages())):
            self.detect_and_extract(output_folder, img_id, imagecols.camview(img_id))
        seg_folder = self.get_segments_folder(output_folder)
        descinfo_folder = self.get_descinfo_folder(output_folder)
        all_2d_segs = limapio.read_all_segments_from_folder(seg_folder)
        return all_2d_segs, descinfo_folder


