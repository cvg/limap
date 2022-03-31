import os
import numpy as np
import cv2
from .sold2 import *
from tqdm import tqdm
import joblib
import utils
import torch

def sold2_detect_2d_segs_on_images(camviews, set_gray=True, heatmap_dir=None, descriptor_dir=None, max_num_2d_segs=3000):
    all_2d_segs, descriptors, heatmaps, descinfos = [], [], [], []
    print("Start sold2 line detection (n_images = {0}).".format(len(camviews)))
    for idx, camview in enumerate(tqdm(camviews)):
        img = utils.imread_camview(camview, set_gray=set_gray)
        segs, descriptor, heatmap, descinfo = sold2_detect(img)
        if segs.shape[0] > max_num_2d_segs:
            lengths_squared = (segs[:,2] - segs[:,0]) ** 2 + (segs[:,3] - segs[:,1]) ** 2
            indexes = np.argsort(lengths_squared)[::-1][:max_num_2d_segs]
            segs = segs[indexes,:]
            descinfo[0] = descinfo[0].reshape(128, -1, 5)[:, indexes,:].reshape(128, -1)
            descinfo[1] = descinfo[1][indexes,:]
        all_2d_segs.append(segs)
        descinfos.append(descinfo)
        if heatmap_dir is not None:
            if not os.path.exists(heatmap_dir):
                os.makedirs(heatmap_dir)
            with open(os.path.join(heatmap_dir, "heatmap_{0}.npy".format(idx)), "wb") as f:
                np.savez(f, data=heatmap)
        if descriptor_dir is not None:
            if not os.path.exists(descriptor_dir):
                os.makedirs(descriptor_dir)
            with open(os.path.join(descriptor_dir, "descriptor_{0}.npy".format(idx)), "wb") as f:
                np.savez(f, data=descriptor)
        # print("Finishing sold2 line detection (num_lines={0}) on image (id={1}): {2}.".format(len(segs), idx, camview.image_name()))
    torch.cuda.empty_cache()
    return all_2d_segs, descinfos

def sold2_compute_descinfos_with_descriptors(all_2d_segs, descriptors):
    # detector = SOLD2LineDetector()
    descinfos = []
    n_images = len(all_2d_segs)
    print("Start computing sold2 desciptors for line segments (n_images = {0}).".format(n_images))
    for idx in tqdm(range(n_images)):
        segs, desc = all_2d_segs[idx], descriptors[idx]
        # descinfo = sold2_compute_descinfo(segs, desc, detector)
        descinfo = sold2_compute_descinfo(segs, desc)
        descinfos.append(descinfo)
        torch.cuda.empty_cache()
    return descinfos

def sold2_compute_descinfos(camviews, all_2d_segs, set_gray=True, descinfo_dir=None):
    detector = SOLD2LineDetector()
    descinfos = []
    n_images = len(all_2d_segs)
    print("Start computing sold2 desciptors for line segments (n_images = {0}).".format(n_images))
    for idx in tqdm(range(n_images)):
        segs = all_2d_segs[idx]
        camview = camviews[idx]
        if descinfo_dir is not None:
            fname = os.path.join(descinfo_dir, "descinfo_{0}.npy".format(idx))
            if os.path.isfile(fname):
                continue
        img = utils.imread_camview(camview, set_gray=set_gray)
        _, descriptor, _, _ = sold2_detect(img, detector)
        descinfo = sold2_compute_descinfo(segs, descriptor, detector)
        if descinfo_dir is not None:
            if not os.path.exists(descinfo_dir):
                os.makedirs(descinfo_dir)
            with open(os.path.join(descinfo_dir, "descinfo_{0}.npy".format(idx)), "wb") as f:
                arr = descinfo
                if len(arr) == 2:
                    arr[1] = arr[1][None,...]
                np.savez(f, data=arr)
                if len(arr) == 2:
                    arr[1] = arr[1][0]
        else:
            descinfos.append(descinfo)
        torch.cuda.empty_cache()
    return descinfos

def sold2_match_2d_segs_with_descriptors(all_2d_segs, descriptors, neighbors):
    detector = SOLD2LineDetector()
    all_matches = []
    n_images = len(all_2d_segs)
    n_neighbor_list = np.array([len(neighbor) for neighbor in neighbors])
    print("Start matching sold2 desciptors (n_images = {0}, n_image_pairs = {1}, max_n_neighbors = {2}).".format(len(neighbors), n_neighbor_list.sum(), n_neighbor_list.max()))
    for idx1 in tqdm(range(n_images)):
        segs1, desc1, neighbor = all_2d_segs[idx1], descriptors[idx1], neighbors[idx1]
        matches_list_idx = []
        for idx2 in neighbor:
            segs2, desc2 = all_2d_segs[idx2], descriptors[idx2]
            matches = sold2_match_segs_with_descriptor(segs1, desc1, segs2, desc2, detector)
            matches_list_idx.append(matches)
            # print("Finishing sold2 line matching ({0} -> {1}, num_matches={2})".format(idx1, idx2, np.where(matches != -1)[0].shape[0]))
        all_matches.append(matches_list_idx)
    torch.cuda.empty_cache()
    return all_matches

def sold2_match_2d_segs_with_descinfo(descinfos, neighbors, n_jobs=6):
    n_images = len(descinfos)
    all_matches = []
    n_neighbor_list = np.array([len(neighbor) for neighbor in neighbors])
    print("Start matching sold2 desciptors (n_images = {0}, n_image_pairs = {1}, max_n_neighbors = {2}).".format(len(neighbors), n_neighbor_list.sum(), n_neighbor_list.max()))
    def process(descinfos, neighbors, idx1):
        detector = SOLD2LineDetector()
        descinfo1, neighbor = descinfos[idx1], neighbors[idx1]
        matches_idx = []
        for idx2 in neighbor:
            descinfo2 = descinfos[idx2]
            matches = sold2_match_segs_with_descinfo(descinfo1, descinfo2, detector=detector)
            matches_idx.append(matches)
        return matches_idx
    all_matches = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(process)(descinfos, neighbors, idx) for idx in tqdm(range(n_images)))
    torch.cuda.empty_cache()
    return all_matches

def sold2_match_2d_segs_with_descinfo_topk(descinfos, neighbors, topk=3, n_jobs=6):
    n_images = len(descinfos)
    all_matches = []
    n_neighbor_list = np.array([len(neighbor) for neighbor in neighbors])
    print("Start matching sold2 desciptors (n_images = {0}, n_image_pairs = {1}, max_n_neighbors = {2}).".format(len(neighbors), n_neighbor_list.sum(), n_neighbor_list.max()))
    def process(descinfos, neighbors, idx1):
        detector = SOLD2LineDetector()
        descinfo1, neighbor = descinfos[idx1], neighbors[idx1]
        matches_idx = []
        for idx2 in neighbor:
            descinfo2 = descinfos[idx2]
            matches = sold2_match_segs_with_descinfo_topk(descinfo1, descinfo2, topk=topk, detector=detector)
            matches_idx.append(matches)
        return matches_idx
    all_matches = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(process)(descinfos, neighbors, idx) for idx in tqdm(range(n_images)))
    torch.cuda.empty_cache()
    return all_matches

def load_descinfo(folder, idx):
    fname = os.path.join(folder, "descinfo_{0}.npy".format(idx))
    with open(fname, 'rb') as f:
        data = np.load(f, allow_pickle=True)
        descinfo = data["data"]
    if len(descinfo) == 2:
        if len(descinfo[1].shape) == 3:
            descinfo[1] = descinfo[1][0]
    return descinfo

def sold2_match_2d_segs_with_descinfo_by_folder(descinfo_folder, neighbors, n_jobs=6, matches_dir=None):
    n_images = len(neighbors)
    all_matches = []
    n_neighbor_list = np.array([len(neighbor) for neighbor in neighbors])
    print("Start matching sold2 desciptors (n_images = {0}, n_image_pairs = {1}, max_n_neighbors = {2}).".format(len(neighbors), n_neighbor_list.sum(), n_neighbor_list.max()))
    def process(descinfo_folder, neighbors, idx1):
        if matches_dir is not None:
            fname = os.path.join(matches_dir, "matches_{0}.npy".format(idx1))
            if os.path.isfile(fname):
                return []
        detector = SOLD2LineDetector()
        descinfo1 = load_descinfo(descinfo_folder, idx1)
        neighbor = neighbors[idx1]
        matches_idx = []
        for idx2 in neighbor:
            descinfo2 = load_descinfo(descinfo_folder, idx2)
            matches = sold2_match_segs_with_descinfo(descinfo1, descinfo2, detector=detector)
            matches_idx.append(matches)
        if matches_dir is not None:
            if not os.path.exists(matches_dir):
                os.makedirs(matches_dir)
            with open(os.path.join(matches_dir, "matches_{0}.npy".format(idx1)), "wb") as f:
                np.savez(f, data=matches_idx)
            return []
        else:
            return matches_idx
    all_matches = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(process)(descinfo_folder, neighbors, idx) for idx in tqdm(range(n_images)))
    torch.cuda.empty_cache()
    return all_matches

def sold2_match_2d_segs_with_descinfo_topk_by_folder(descinfo_folder, neighbors, topk=3, n_jobs=6, matches_dir=None):
    n_images = len(neighbors)
    all_matches = []
    n_neighbor_list = np.array([len(neighbor) for neighbor in neighbors])
    print("Start matching sold2 desciptors (n_images = {0}, n_image_pairs = {1}, max_n_neighbors = {2}).".format(len(neighbors), n_neighbor_list.sum(), n_neighbor_list.max()))
    def process(descinfo_folder, neighbors, idx1):
        if matches_dir is not None:
            fname = os.path.join(matches_dir, "matches_{0}.npy".format(idx1))
            if os.path.isfile(fname):
                return []
        detector = SOLD2LineDetector()
        descinfo1 = load_descinfo(descinfo_folder, idx1)
        neighbor = neighbors[idx1]
        matches_idx = []
        for idx2 in neighbor:
            descinfo2 = load_descinfo(descinfo_folder, idx2)
            matches = sold2_match_segs_with_descinfo_topk(descinfo1, descinfo2, topk=topk, detector=detector)
            matches_idx.append(matches)
        if matches_dir is not None:
            if not os.path.exists(matches_dir):
                os.makedirs(matches_dir)
            with open(os.path.join(matches_dir, "matches_{0}.npy".format(idx1)), "wb") as f:
                np.savez(f, data=matches_idx)
            return []
        else:
            return matches_idx
    all_matches = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(process)(descinfo_folder, neighbors, idx) for idx in tqdm(range(n_images)))
    torch.cuda.empty_cache()
    return all_matches

