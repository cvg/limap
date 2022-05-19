import os, sys
import numpy as np
from tqdm import tqdm

import core.visualize as vis
import limap.base as _base

import pdb

n_visible_views = 4
basedir = '/media/shaoliu/cvg-hdd-usb-2tb-08/Lines/experiments/limap_tnt_lsd'
expdir = os.path.join(basedir, "Truck")
input_dir = os.path.join(expdir, "finaltracks")

def main():
    # load image_names
    fname = os.path.join(expdir, "localization/image_list.txt")
    with open(fname, 'r') as f:
        txt_lines = f.readlines()
    time_list = [int(txtline.strip().split(' ')[1][-10:-4]) for txtline in txt_lines[1:]]
    orders = np.argsort(time_list)

    # load linetracks
    n_tracks = vis.count_linetracks_from_folder(input_dir)
    linetracks = [_base.LineTrack() for idx in range(n_tracks)]
    linetracks = vis.load_linetracks_from_folder(linetracks, input_dir)

    # build dict
    dict_images = {}
    for idx, time in enumerate(time_list):
        dict_images[idx] = []
    for track_id, track in enumerate(tqdm(linetracks)):
        for img_id in track.image_id_list:
            dict_images[img_id].append(track_id)
    for idx, time in enumerate(time_list):
        dict_images[idx] = list(set(dict_images[idx]))

    # simulate timer
    counters = np.zeros((n_tracks))
    timers = np.ones((n_tracks)) * -1
    for idx, img_id in enumerate(list(orders)):
        flag1 = counters < n_visible_views
        counters[dict_images[img_id]] += 1
        flag2 = counters >= n_visible_views
        flag = np.logical_and(flag1, flag2)
        timers[flag] = idx
    fname_save = os.path.join(expdir, 'timers.npy')
    np.save(fname_save, timers)
    pdb.set_trace()

main()

