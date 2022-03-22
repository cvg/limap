import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tqdm import tqdm
import cv2

import core.visualize as vis
from core.dataset import Rome
import limap.base as _base
import pdb

N_VISIBLE_VIEWS = 4
dataset_dir = os.path.expanduser('~/data/SfmData/Rome16K')
dataset_image_dir = os.path.join(dataset_dir, 'undistorted')
list_file = os.path.join(dataset_dir, 'bundle/list.orig.txt')
component_folder = os.path.join(dataset_dir, 'bundle/components')
track_folder = 'tmp/finaltracks'
comp_dir = 'tmp/comps'
if not os.path.exists(comp_dir):
    os.makedirs(comp_dir)

def compute_robust_ranges(lines_np):
    endpoints = lines_np.reshape(-1, 3)
    N = endpoints.shape[0]
    start, end = int(N*0.05), int(N*0.95)
    x_array = np.sort(endpoints[:,0])
    x_start, x_end = x_array[start], x_array[end]
    y_array = np.sort(endpoints[:,1])
    y_start, y_end = y_array[start], y_array[end]
    z_array = np.sort(endpoints[:,2])
    z_start, z_end = z_array[start], z_array[end]
    ranges = np.array([[x_start, y_start, z_start], [x_end, y_end, z_end]])
    return ranges

def normalize(lines_np):
    ranges = compute_robust_ranges(lines_np)
    robust_mean = (ranges[0,:] + ranges[1,:]) / 2.0
    scale = np.abs(ranges[0,:] - ranges[1,:]).max()

    # center and rescale
    n_lines = lines_np.shape[0]
    endpoints = lines_np.reshape(-1, 3)
    endpoints = (endpoints - robust_mean) * (2.0 / scale)
    lines_final = endpoints.reshape(n_lines, 2, 3)
    return lines_final

def main():
    # initialize component dataset
    dataset = Rome(list_file, component_folder)
    n_comps = dataset.count_components()

    # process tracks
    n_tracks = vis.count_linetracks_from_folder(track_folder)
    linetracks = [_base.LineTrack() for idx in range(n_tracks)]
    linetracks = vis.load_linetracks_from_folder(linetracks, track_folder)
    components = [[] for idx in range(n_comps)]
    for track in linetracks:
        c_id = dataset.get_component_id_for_image_id_list(track.image_id_list)
        components[c_id].append(track)

    for comp_id, comp in enumerate(tqdm(components)):
        VisTrack = vis.TrackVisualizer(comp)
        lines_np, counts_np = VisTrack.get_lines_np(), VisTrack.get_counts_np()
        if lines_np.shape[0] > 10:
            lines_np = normalize(lines_np)
        vis.save_obj(os.path.join(comp_dir, 'comp_{0}.obj'.format(comp_id)), lines_np, counts=counts_np, n_visible_views=N_VISIBLE_VIEWS)
        # save three images
        images = dataset.get_images_in_component(comp_id)
        for img_idx in range(20):
            fname = os.path.join(comp_dir, 'comp_{0}_{1}.png'.format(comp_id, img_idx))
            if img_idx >= len(images):
                continue
            img = cv2.imread(os.path.join(dataset_image_dir, images[img_idx]))
            cv2.imwrite(fname, img)

    track_counts = [len(comp) for comp in components]
    indexes = np.argsort(track_counts)[::-1]
    for index in indexes[:20]:
        comp = components[index]
        VisTrack = vis.TrackVisualizer(comp)
        print(index)
        VisTrack.report()

if __name__ == '__main__':
    main()

