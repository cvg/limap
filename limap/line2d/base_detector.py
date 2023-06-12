import os
import numpy as np
from tqdm import tqdm
import cv2

import limap.util.io as limapio
import limap.visualize as limapvis

from collections import namedtuple
BaseDetectorOptions = namedtuple("BaseDetectorOptions",
                                 ["set_gray", "max_num_2d_segs", "do_merge_lines", "visualize", "weight_path"],
                                 defaults=[True, 3000, False, False, None])

class BaseDetector():
    def __init__(self, options = BaseDetectorOptions()):
        self.set_gray = options.set_gray
        self.max_num_2d_segs = options.max_num_2d_segs
        self.do_merge_lines = options.do_merge_lines
        self.visualize = options.visualize
        self.weight_path = options.weight_path

    # Module name needs to be set
    def get_module_name(self):
        raise NotImplementedError
    # The functions below are required for detectors
    def detect(self, camview):
        raise NotImplementedError
    # The functions below are required for extractors
    def extract(self, camview, segs):
        raise NotImplementedError
    def get_descinfo_fname(self, descinfo_folder, img_id):
        raise NotImplementedError
    def save_descinfo(self, descinfo_folder, img_id, descinfo):
        raise NotImplementedError
    def read_descinfo(self, descinfo_folder, img_id):
        raise NotImplementedError
    # The functions below are required for double-functioning objects
    def detect_and_extract(self, camview):
        raise NotImplementedError
    def sample_descinfo_by_indexes(descinfo):
        raise NotImplementedError

    def get_segments_folder(self, output_folder):
        return os.path.join(output_folder, "segments")
    def get_descinfo_folder(self, output_folder):
        return os.path.join(output_folder, "descinfos", self.get_module_name())

    def merge_lines(self, segs):
        from limap.line2d.line_utils import merge_lines
        segs = segs[:, :4].reshape(-1, 2, 2)
        segs = merge_lines(segs)
        segs = segs.reshape(-1, 4)
        return segs
    def take_longest_k(self, segs, max_num_2d_segs=3000):
        indexes = np.arange(0, segs.shape[0])
        if max_num_2d_segs is None or max_num_2d_segs == -1:
            pass
        elif segs.shape[0] > max_num_2d_segs:
            lengths_squared = (segs[:,2] - segs[:,0]) ** 2 + (segs[:,3] - segs[:,1]) ** 2
            indexes = np.argsort(lengths_squared)[::-1][:max_num_2d_segs]
            segs = segs[indexes,:]
        return segs, indexes

    def visualize_segs(self, output_folder, imagecols, first_k=10):
        seg_folder = self.get_segments_folder(output_folder)
        n_vis_images = min(first_k, imagecols.NumImages())
        vis_folder = os.path.join(output_folder, "visualize")
        limapio.check_makedirs(vis_folder)
        image_ids = imagecols.get_img_ids()[:n_vis_images]
        for img_id in image_ids:
            img = imagecols.read_image(img_id)
            segs = limapio.read_txt_segments(seg_folder, img_id)
            img = limapvis.draw_segments(img, segs, (0, 255, 0))
            fname = os.path.join(vis_folder, "img_{0}_det.png".format(img_id))
            cv2.imwrite(fname, img)

    def detect_all_images(self, output_folder, imagecols, skip_exists=False):
        seg_folder = self.get_segments_folder(output_folder)
        if not skip_exists:
            limapio.delete_folder(seg_folder)
        limapio.check_makedirs(seg_folder)
        if self.visualize:
            vis_folder = os.path.join(output_folder, "visualize")
            limapio.check_makedirs(vis_folder)
        for img_id in tqdm(imagecols.get_img_ids()):
            if skip_exists and limapio.exists_txt_segments(seg_folder, img_id):
                if self.visualize:
                    segs = limapio.read_txt_segments(seg_folder, img_id)
            else:
                segs = self.detect(imagecols.camview(img_id))
                if self.do_merge_lines:
                    segs = self.merge_lines(segs)
                segs, _ = self.take_longest_k(segs, max_num_2d_segs=self.max_num_2d_segs)
                limapio.save_txt_segments(seg_folder, img_id, segs)
            if self.visualize:
                img = imagecols.read_image(img_id)
                img = limapvis.draw_segments(img, segs, (0, 255, 0))
                fname = os.path.join(vis_folder, "img_{0}_det.png".format(img_id))
                cv2.imwrite(fname, img)
        all_2d_segs = limapio.read_all_segments_from_folder(seg_folder)
        all_2d_segs = {id: all_2d_segs[id] for id in imagecols.get_img_ids()}
        return all_2d_segs

    def extract_all_images(self, output_folder, imagecols, all_2d_segs, skip_exists=False):
        descinfo_folder = self.get_descinfo_folder(output_folder)
        if not skip_exists:
            limapio.delete_folder(descinfo_folder)
        limapio.check_makedirs(descinfo_folder)
        for img_id in tqdm(imagecols.get_img_ids()):
            if skip_exists and os.path.exists(self.get_descinfo_fname(descinfo_folder, img_id)):
                continue
            descinfo = self.extract(imagecols.camview(img_id), all_2d_segs[img_id])
            self.save_descinfo(descinfo_folder, img_id, descinfo)
        return descinfo_folder

    def detect_and_extract_all_images(self, output_folder, imagecols, skip_exists=False):
        assert self.do_merge_lines == False
        seg_folder = self.get_segments_folder(output_folder)
        descinfo_folder = self.get_descinfo_folder(output_folder)
        if not skip_exists:
            limapio.delete_folder(seg_folder)
            limapio.delete_folder(descinfo_folder)
        limapio.check_makedirs(seg_folder)
        limapio.check_makedirs(descinfo_folder)
        if self.visualize:
            vis_folder = os.path.join(output_folder, "visualize")
            limapio.check_makedirs(vis_folder)
        for img_id in tqdm(imagecols.get_img_ids()):
            if skip_exists and os.path.exists(self.get_descinfo_fname(descinfo_folder, img_id)) and limapio.exists_txt_segments(seg_folder, img_id):
                if self.visualize:
                    segs = limapio.read_txt_segments(seg_folder, img_id)
            else:
                segs, descinfo = self.detect_and_extract(imagecols.camview(img_id))
                n_segs_orig = segs.shape[0]
                segs, indexes = self.take_longest_k(segs, max_num_2d_segs=self.max_num_2d_segs)
                if indexes.shape[0] < n_segs_orig:
                    descinfo = self.sample_descinfo_by_indexes(descinfo, indexes)
                limapio.save_txt_segments(seg_folder, img_id, segs)
                self.save_descinfo(descinfo_folder, img_id, descinfo)
            if self.visualize:
                img = imagecols.read_image(img_id)
                img = limapvis.draw_segments(img, segs, (0, 255, 0))
                fname = os.path.join(vis_folder, "img_{0}_det.png".format(img_id))
                cv2.imwrite(fname, img)
        all_2d_segs = limapio.read_all_segments_from_folder(seg_folder)
        all_2d_segs = {id: all_2d_segs[id] for id in imagecols.get_img_ids()}
        return all_2d_segs, descinfo_folder


