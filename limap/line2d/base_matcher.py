import os, sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import limap.util.io_utils as limapio

class BaseMatcher():
    def __init__(self, extractor, topk=10, n_neighbors=20, n_jobs=1):
        self.extractor = extractor
        self.topk = topk
        self.n_neighbors = n_neighbors

    # The functions below are required for matchers
    def get_module_name(self):
        raise NotImplementedError
    def match_pair(self, output_folder, img_id, neighbor_id_list, descinfo_folder):
        raise NotImplementedError

    def get_matches_folder(self, output_folder):
        return os.path.join(output_folder, "{0}_n{1}_top{2}".format(self.get_module_name(), self.n_neighbors, self.topk))
    def read_descinfo(self, descinfo_folder, idx):
        return self.extractor.read_descinfo(descinfo_folder, idx)
    def save_match(self, matches_folder, idx, matches):
        fname = os.path.join(matches_folder, "matches_{0}.npy".format(idx))
        limapio.save_npy(fname, matches)
    def read_match(self, matches_folder, idx):
        fname = os.path.join(matches_folder, "matches_{0}.npy".format(idx))
        return limapio.read_npy(fname)

    # TODO: multiprocessing
    def match_all_neighbors(self, output_folder, neighbors, descinfo_folder):
        matches_folder = self.get_matches_folder(output_folder)
        limapio.check_makedirs(matches_folder)
        n_images = len(neighbors)
        for img_id in tqdm(range(n_images)):
            descinfo1 = self.read_descinfo(descinfo_folder, img_id)
            matches_idx = []
            for ng_img_id in neighbors[img_id]:
                descinfo2 = self.read_descinfo(descinfo_folder, ng_img_id)
                matches = self.match_pair(descinfo1, descinfo2)
                matches_idx.append(matches)
            self.save_match(matches_folder, img_id, matches_idx)
        return matches_folder

